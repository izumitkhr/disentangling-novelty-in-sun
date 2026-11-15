"""Pseudo-stable rate of generated structures when the theoretical
Materials Project materials are randomly removed from the reference.

The stability verdict of a generated structure depends on which competing
phases the reference knows about. Here the theoretical MP materials
(``theoretical = True``, no experimental counterpart) are removed at random
from the reference (every MP GGA/GGA+U entry, re-evaluated with
MACE-OMAT-0), and the pseudo-stable rate (below) is tracked as a function
of the removed fraction of the theoretical materials, for MatterGen,
DiffCSP++ and WyFormer, split by the number of elements (binary ...
quinary). MatterGen trained on Alex-MP-20 is computed as well: its E_hull
against the full reference gives its stable / metastable flags in
analysis_3.

Thinning
  * Only theoretical materials are removed; experimental ones are always
    kept, so 100 % removal leaves the experimental-only reference.
  * Whole MP materials are removed (all run types of a material_id together).
  * Theoretical elemental materials are removed too, except for elements
    with no experimental elemental material, so every hull stays defined.
  * Removal is nested within a seed: the materials removed at 20 % include
    those removed at 10 %, so each seed gives a monotone curve.
  * The curve is the mean over seeds; the band spans min-max over seeds.

Pseudo-stable rate: among structures with E_hull <= t against the thinned
reference, the fraction with E_hull > t against the full reference, i.e. how
often a "stable" verdict from an incomplete reference is wrong. Removing
phases can only lower E_hull, so a structure stable against the full
reference stays stable; the rate is 0 at 0 % removal by construction.
The figures are drawn by make_figure3.py, which leaves out points whose
denominator is below MIN_STABLE in any seed (they are kept in the stats CSV,
together with the plain stable fraction).

E_hull of a candidate is its energy per atom minus the lower convex envelope
of the reference at its composition, clipped at 0 (the candidate itself is
not part of the reference). The envelope is found by a linear program over
the reference entries of the candidate's element set and all its subsets.

Inputs (both computed with the same model; see paths.py):
  data/input/mp_mace_omat_0_medium/
      entries.jsonl.gz, manifest.json   MACE energies of the MP entries
  data/input/generated/energies/mace-omat-0-medium/
      gen_crystals_<generator>_energies.csv, manifest.json
Both store MACE energies with MP2020 corrections; on loading, the
corrections are replaced by the OMat24 refit of the same scheme
(scripts/analysis/corrections, OMat24Compatibility.yaml), which matches the
OMat24 DFT settings MACE-OMAT-0 is trained on. All analyses read the
energies through load_reference / load_generated.

Outputs:
  data/analysis_2/ehull_thinning.csv.gz         E_hull per structure, removed
                                                fraction and seed (cache;
                                                ``--from-cache`` reuses it)
  figures/analysis_2/thinning_stats.csv         stable fraction and
                                                pseudo-stable rate with counts

Denominator: generated structures whose MACE energy was computed
(``status == ok``) and whose elements all have an elemental reference
(structures containing Yb are excluded: MP has no GGA/GGA+U energy for
any Yb material).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corrections as cr  # noqa: E402
from generators import EVALUATED, GENERATORS  # noqa: E402,F401
from paths import ENERGIES_DIR, REFERENCE_DIR, ROOT  # noqa: E402,F401
CACHE = ROOT / "data" / "analysis_2" / "ehull_thinning.csv.gz"
OUT_DIR = ROOT / "figures" / "analysis_2"

NELEMENTS = [2, 3, 4, 5]
LABELS = {2: "Binary", 3: "Ternary", 4: "Quaternary", 5: "Quinary"}
LINESTYLES = {2: "-", 3: "-.", 4: "-", 5: "-."}
MARKERS = {2: "o", 3: "s", 4: "^", 5: "D"}
TOL = 1e-9  # eV/atom; E_hull at or below threshold + TOL counts as stable
MIN_STABLE = 10  # smallest denominator drawn (make_figure3.py)
# same line width / transparency as the CDF lines of analysis_1
LINE_KW = {"lw": 1.8, "alpha": 0.8}


# ---- inputs -----------------------------------------------------------------
def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_reference(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Reference in {directory} is not complete")
    path = directory / "entries.jsonl.gz"
    if sha256(path) != manifest["entries_sha256"]:
        raise ValueError("Reference checksum mismatch")
    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f]
    if len(records) != manifest["entry_count"]:
        raise ValueError("Reference entry count mismatch")
    if not manifest["mp2020_applied"]:
        raise ValueError("Reference energies are not MP2020 corrected")
    for r in records:
        # MACE energy + OMat24 corrections (replacing MP2020)
        r["energy"] = r["uncorrected_energy"] + cr.omat24_correction(
            r["energy_adjustments"]
        )
    return manifest, records


def load_generated(directory: Path, reference_manifest: dict) -> pd.DataFrame:
    """Generated structures with MACE energies, verified against the
    reference (same model weights and dtype)."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Energies in {directory} are not complete")
    for key in ("model_sha256", "dtype"):
        if manifest[key] != reference_manifest[key]:
            raise ValueError(
                f"Generated energies and reference differ in {key}"
            )
    frames = []
    for gen in EVALUATED:
        name = f"gen_crystals_{gen}_energies.csv"
        path = directory / name
        if sha256(path) != manifest["files"][name]["csv_sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
        df = pd.read_csv(path)
        df.insert(0, "generator", gen)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # MACE energy + OMat24 corrections (replacing MP2020)
    ok = df.status == "ok"
    df.loc[ok, "energy_total_eV"] = df.loc[
        ok, "uncorrected_energy_total_eV"
    ] + df.loc[ok, "mp2020_adjustments"].map(
        lambda s: cr.omat24_correction(json.loads(s))
    )
    df.loc[ok, "energy_per_atom_eV"] = (
        df.loc[ok, "energy_total_eV"] / df.loc[ok, "nsites"]
    )
    return df


# ---- thinning ---------------------------------------------------------------
class Reference:
    """Reference entries as arrays, indexed by element set."""

    def __init__(self, records: list[dict]):
        self.elements = sorted(
            {el for r in records for el in r["composition"]}
        )
        col = {el: i for i, el in enumerate(self.elements)}
        n = len(records)
        self.frac = np.zeros((n, len(self.elements)))
        self.e_per_atom = np.empty(n)
        material_ids = []
        theoretical = []
        self.by_system = defaultdict(list)
        for i, r in enumerate(records):
            natoms = sum(r["composition"].values())
            for el, amt in r["composition"].items():
                self.frac[i, col[el]] = amt / natoms
            self.e_per_atom[i] = r["energy"] / natoms
            material_ids.append(r["material_id"])
            theoretical.append(bool(r["theoretical"]))
            self.by_system[tuple(sorted(r["composition"]))].append(i)
        self.col = col
        uniq, self.material = np.unique(material_ids, return_inverse=True)
        self.n_materials = len(uniq)
        unary = np.zeros(self.n_materials, bool)
        unary[self.material[self.frac.max(axis=1) > 1 - 1e-12]] = True
        self.unary_material = unary
        self.unary_elements = {
            el for key in self.by_system if len(key) == 1 for el in key
        }
        theo = np.zeros(self.n_materials, bool)
        theo[self.material] = theoretical
        self.theoretical_material = theo
        # an element whose only elemental materials are theoretical keeps
        # them, otherwise its hulls would lose their elemental reference
        unary_entries = np.flatnonzero(unary[self.material])
        element_of = self.frac[unary_entries].argmax(axis=1)
        has_experimental = np.zeros(len(self.elements), bool)
        has_experimental[element_of[~theo[self.material[unary_entries]]]] = (
            True
        )
        protected = np.zeros(self.n_materials, bool)
        protected[
            self.material[unary_entries[~has_experimental[element_of]]]
        ] = True
        self.protected_elements = sorted(
            self.elements[i] for i in np.flatnonzero(~has_experimental)
        )
        self.removable = theo & ~protected

    def entries_for(self, elements: tuple[str, ...]) -> np.ndarray:
        idx = [
            i
            for k in range(1, len(elements) + 1)
            for sub in combinations(elements, k)
            for i in self.by_system.get(sub, ())
        ]
        return np.asarray(idx, dtype=int)

    def removal_rank(self, seed: int) -> np.ndarray:
        """Per material: position in a random order of the removable
        (theoretical) materials, inf for the others. A material is removed
        at fraction p when rank < p * n_removable."""
        rng = np.random.default_rng(seed)
        removable = np.flatnonzero(self.removable)
        rank = np.full(self.n_materials, np.inf)
        rank[rng.permutation(removable)] = np.arange(len(removable))
        return rank


def hull_energy(frac: np.ndarray, e: np.ndarray, x: np.ndarray) -> float:
    """Lower convex envelope of points (frac, e) at composition x:
    min e @ w  s.t.  frac.T @ w = x, w >= 0 (sum w = 1 follows)."""
    res = linprog(e, A_eq=frac.T, b_eq=x, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog failed: {res.message}")
    return float(res.fun)


# Worker state, shared with forked processes.
_REF: Reference | None = None
_RANKS: np.ndarray | None = None  # (n_seeds, n_materials)
_CUTS: np.ndarray | None = None  # number of removed materials per fraction


def _evaluate_system(task):
    """E_hull of every candidate of one element set, for all runs."""
    elements, cand_ids, cand_x, cand_e = task
    ref = _REF
    idx = ref.entries_for(elements)
    cols = [ref.col[el] for el in elements]
    frac = ref.frac[np.ix_(idx, cols)]
    e = ref.e_per_atom[idx]
    mat = ref.material[idx]
    out = []
    for s, rank in enumerate(_RANKS):
        r = rank[mat]
        prev_keep = None
        prev_hull = None
        for f, cut in enumerate(_CUTS):
            keep = r >= cut
            if prev_keep is not None and np.array_equal(keep, prev_keep):
                hulls = prev_hull  # nothing in this system was removed
            else:
                hulls = [hull_energy(frac[keep], e[keep], x) for x in cand_x]
            prev_keep, prev_hull = keep, hulls
            for cid, ce, h in zip(cand_ids, cand_e, hulls):
                out.append((cid, f, s, max(0.0, ce - h)))
    return out


def compute_ehull(
    ref: Reference,
    cands: pd.DataFrame,
    fractions: list[float],
    seeds: list[int],
    workers: int,
) -> pd.DataFrame:
    global _REF, _RANKS, _CUTS
    n_removable = int(ref.removable.sum())
    _REF = ref
    _RANKS = np.stack([ref.removal_rank(s) for s in seeds])
    _CUTS = np.array([round(p * n_removable) for p in fractions])

    tasks = []
    for elements, grp in cands.groupby("elements", sort=False):
        x = np.array(
            [[c[el] for el in elements] for c in grp["composition"]], float
        )
        x /= x.sum(axis=1, keepdims=True)
        tasks.append(
            (
                elements,
                grp["cid"].to_numpy(),
                x,
                grp["energy_per_atom_eV"].to_numpy(),
            )
        )
    # largest systems first for better load balance
    tasks.sort(key=lambda t: -len(ref.entries_for(t[0])) * len(t[1]))
    rows = []
    with Pool(workers) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_evaluate_system, tasks, chunksize=4), 1
        ):
            rows.extend(res)
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} element sets", flush=True)
    out = pd.DataFrame(rows, columns=["cid", "fi", "si", "ehull_eV"])
    out["removed_fraction"] = np.asarray(fractions)[out.fi]
    out["n_removed_materials"] = _CUTS[out.fi]
    out["seed"] = np.asarray(seeds)[out.si]
    meta = cands[["cid", "generator", "index", "nelements"]]
    out = out.merge(meta, on="cid").drop(columns=["cid", "fi", "si"])
    return out.sort_values(
        ["generator", "index", "seed", "removed_fraction"]
    ).reset_index(drop=True)


def prepare_candidates(gen: pd.DataFrame, ref: Reference) -> pd.DataFrame:
    ok = gen[gen.status == "ok"].copy()
    ok["composition"] = ok["composition"].map(json.loads)
    ok["elements"] = ok["composition"].map(lambda c: tuple(sorted(c)))
    has_ref = ok["elements"].map(lambda els: set(els) <= ref.unary_elements)
    for g in EVALUATED:
        sel = gen.generator == g
        print(
            f"  {g}: {sel.sum()} structures, "
            f"{(gen[sel].status != 'ok').sum()} without energy, "
            f"{(~has_ref[ok.generator == g]).sum()} with an element "
            "missing from the reference"
        )
    ok = ok[has_ref].reset_index(drop=True)
    ok["cid"] = np.arange(len(ok))
    return ok


# ---- summary ----------------------------------------------------------------
def summarize(eh: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Per generator, element count, removed fraction and seed: stable
    fraction and pseudo-stable rate, then mean / min / max over seeds."""
    full = eh[eh.removed_fraction == 0].drop_duplicates(["generator", "index"])
    eh = eh.merge(
        full[["generator", "index", "ehull_eV"]].rename(
            columns={"ehull_eV": "ehull_full_eV"}
        ),
        on=["generator", "index"],
    )
    if (eh.ehull_eV > eh.ehull_full_eV + TOL).any():
        raise ValueError("E_hull increased after removing reference phases")
    rows = []
    groups = [("all", eh)] + [(n, eh[eh.nelements == n]) for n in NELEMENTS]
    keys = ["generator", "removed_fraction", "seed"]
    for t in thresholds:
        for n, sub in groups:
            stable = sub.ehull_eV <= t + TOL
            per_seed = (
                pd.DataFrame(
                    {
                        **{k: sub[k] for k in keys},
                        "n": 1,
                        "n_stable": stable,
                        "n_pseudo": stable & (sub.ehull_full_eV > t + TOL),
                    }
                )
                .groupby(keys)
                .sum()
                .reset_index()
            )
            per_seed["frac"] = per_seed.n_stable / per_seed.n
            per_seed["rate"] = per_seed.n_pseudo / per_seed.n_stable.where(
                per_seed.n_stable > 0
            )
            agg = per_seed.groupby(["generator", "removed_fraction"]).agg(
                n_structures=("n", "first"),
                stable_fraction_mean=("frac", "mean"),
                stable_fraction_min=("frac", "min"),
                stable_fraction_max=("frac", "max"),
                stable_fraction_std=("frac", "std"),
                n_stable_mean=("n_stable", "mean"),
                n_stable_min=("n_stable", "min"),
                n_pseudo_mean=("n_pseudo", "mean"),
                pseudo_rate_mean=("rate", "mean"),
                pseudo_rate_min=("rate", "min"),
                pseudo_rate_max=("rate", "max"),
                pseudo_rate_std=("rate", "std"),
                n_seeds=("frac", "size"),
            )
            agg = agg.reset_index()
            agg.insert(1, "nelements", n)
            agg.insert(2, "threshold_eV", t)
            rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    # generators in EVALUATED order; nelements keeps the "all", 2..5 order
    out["gen_order"] = out.generator.map(list(EVALUATED).index)
    return (
        out.sort_values(["threshold_eV", "gen_order"], kind="stable")
        .drop(columns="gen_order")
        .reset_index(drop=True)
    )


def print_counts(stats: pd.DataFrame) -> None:
    """Structure counts per generator and element count (for the caption)."""
    st = stats[stats.removed_fraction == 0].drop_duplicates(
        ["generator", "nelements"]
    )
    print("structures per generator and element count:")
    print(
        st.pivot(index="generator", columns="nelements", values="n_structures")
        .reindex(list(EVALUATED))
        .to_string()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    ap.add_argument("--energies-dir", type=Path, default=ENERGIES_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[round(0.1 * i, 1) for i in range(11)],
        help="removed fractions of the theoretical MP materials",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.1],
        help="stability thresholds in eV/atom",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="reuse the cached E_hull table instead of recomputing",
    )
    args = ap.parse_args()

    if args.from_cache:
        eh = pd.read_csv(args.cache)
    else:
        print("loading reference ...", flush=True)
        manifest, records = load_reference(args.reference_dir)
        ref = Reference(records)
        print(
            f"  {len(records)} entries, {ref.n_materials} materials, "
            f"{int(ref.theoretical_material.sum())} theoretical, "
            f"{int(ref.removable.sum())} removable; elemental theoretical "
            f"materials kept for {ref.protected_elements}"
        )
        gen = load_generated(args.energies_dir, manifest)
        cands = prepare_candidates(gen, ref)
        print(
            f"computing E_hull for {len(cands)} structures x "
            f"{len(args.fractions)} fractions x {len(args.seeds)} seeds ...",
            flush=True,
        )
        eh = compute_ehull(
            ref, cands, args.fractions, args.seeds, args.workers
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        eh.to_csv(args.cache, index=False, float_format="%.8g")
        print("wrote", args.cache)

    stats = summarize(eh, args.thresholds)
    # figures (t = 0 and the t > 0 supplement) are drawn by make_figure3.py
    print_counts(stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.out_dir / "thinning_stats.csv"
    stats.to_csv(stats_path, index=False)
    print("wrote", stats_path)
    st = stats[stats.nelements != "all"]
    for col in ("stable_fraction_mean", "pseudo_rate_mean", "n_stable_mean"):
        view = st.pivot_table(
            index=["threshold_eV", "generator", "nelements"],
            columns="removed_fraction",
            values=col,
            dropna=False,
        )
        scale = 1 if col.startswith("n_") else 100
        print(f"\n{col}" + ("" if scale == 1 else " (%)"))
        print((view * scale).round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
