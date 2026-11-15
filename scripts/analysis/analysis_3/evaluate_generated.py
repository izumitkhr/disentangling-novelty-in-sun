"""Validity, uniqueness, novelty and stability of the generated structures.

Every generated structure (generators.EVALUATED) is evaluated with
the logic of lemat-genbench (external/lemat-genbench) against the Materials
Project (MP) structures of the reference used in analysis_2:

  structural novelty   BAWL hash (lemat-genbench's default fingerprint) not
                       found among the MP structures. lemat-genbench hashes
                       the cell as given, which makes the hash depend on the
                       cell choice (a supercell gets a different hash), so
                       both generated and MP structures are first reduced to
                       the standardized primitive cell (spglib, symprec 0.1).
  prototype novelty    prototype string (prototype_string.py, get_string,
                       symprec 0.1: anonymized formula | Pearson symbol |
                       space group | site symmetries per element) not found
                       among the strings of all MP entries. Element-blind:
                       a new compound in a known structure type is not a new
                       prototype.
  uniqueness           within the structures of one generator, same BAWL
                       hash (standardized primitive cell) as an earlier
                       structure (generation order); the first of each group
                       is unique, as lemat-genbench's SUN picks one
                       representative per group. Structures without a hash
                       count as unique.
  compositional        lemat-genbench charge neutrality (deviation <= 0.1;
  validity             metallicity -> bond valence -> BVAnalyzer -> oxidation
                       state guess with the LeMat+ICSD lists), and SMACT
                       ``smact_validity`` of the composition. lemat-genbench's
                       minimum-distance and physical-plausibility checks and
                       its overall validity (all three) are recorded too.
                       ``smact_valid`` and ``lemat_valid`` also require a
                       shortest interatomic distance of at least 0.7 A, below
                       which no MACE energy was computed
                       (prepare/compute_generated_energies.py); the
                       checks alone are kept as ``smact_composition_valid``
                       and ``lemat_overall_valid``. A valid structure thus
                       always has an E_hull.
  excluded             structures with an element that no MP entry of the
                       reference contains (Yb: MP has no GGA/GGA+U energy
                       for any Yb material), as no E_hull can be computed.
  composition /        new_chemsys: no MP phase has its element set;
  chemsys novelty      new_composition: element set exists but not the
                       reduced formula; known_composition otherwise. Against
                       the MP materials of the reference (with a GGA/GGA+U
                       thermo entry, as in analysis_2).
  stability            E_hull against the full MACE-OMAT-0 MP reference
                       (analysis_2, no thinning); stable E_hull = 0,
                       metastable E_hull <= 0.1 eV/atom (lemat-genbench's
                       thresholds).
  combined             vsun: SMACT valid, unique, novel (BAWL) and stable;
                       vmsun: the same with metastable.
  not evaluated        a flag whose input is missing (no E_hull below
                       0.7 A, no BAWL hash or prototype string) is NaN in
                       evaluation.csv; every fraction of the summary is over
                       all structures (Yb excluded), counting NaN as failing,
                       and n_<flag>_evaluated gives the structures evaluated.

Only the novelty and validity modules of lemat-genbench are imported (its
package ``__init__`` pulls in the MLIP stack); the environment is the one of
pyproject.toml at the repository root.

Stages (each resumable; results are stored per batch under
data/analysis_3/):
  gen      per generated structure: validity checks and BAWL hashes
  mp       BAWL hashes of the MP structures whose reduced formula occurs
           among the generated ones (exact hash matches need it)
  protogen prototype string of each generated structure
  protomp  prototype string of every MP entry
  combine  one table with all flags and E_hull

Outputs:
  data/analysis_3/batches/{gen,mp,protogen,protomp}_*.jsonl.gz
                                                        per-batch results,
                                                        with the sha256 of
                                                        their inputs
                                                        (*.sha256; a batch
                                                        whose inputs changed
                                                        is recomputed)
  data/analysis_3/mp_prototypes.csv.gz                  prototype
                                                        string of every MP
                                                        entry
  data/analysis_3/evaluation.csv.gz                     one row per
                                                        generated structure
  data/analysis_3/manifest.json                         inputs, versions
  figures/analysis_3/evaluation_summary.csv             fractions per model
                                                        (all structures)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import logging
import signal
import subprocess
import os
import sys
import time
import types
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prototype_string  # noqa: E402
from formula import formula_key  # noqa: E402
from generators import EVALUATED  # noqa: E402
from paths import (  # noqa: E402
    ENERGIES_DIR,
    GEN_DIR,
    LEMAT_ROOT,
    MP_DIR,
    ROOT,
)

EHULL_CACHE = ROOT / "data" / "analysis_2" / "ehull_thinning.csv.gz"
OUT_DIR = ROOT / "data" / "analysis_3"
FIG_DIR = ROOT / "figures" / "analysis_3"

SYMPREC = 0.1  # standardized primitive cell before hashing
CHARGE_TOLERANCE = 0.1  # lemat-genbench default
STABLE, METASTABLE = 0.0, 0.1  # eV/atom, lemat-genbench thresholds
TOL = 1e-9
MIN_DISTANCE = 0.7  # A, no MACE energy below this (prepare/)
BATCH = 200  # structures per stored batch
TIMEOUT = 300  # seconds per check and structure


# ---- lemat-genbench (novelty and validity modules only) ---------------------
def import_lemat():
    """Import the needed lemat-genbench modules without running the package
    ``__init__`` files (they import every benchmark and the MLIP stack)."""
    src = LEMAT_ROOT / "src" / "lemat_genbench"
    for name, sub in (
        ("lemat_genbench", src),
        ("lemat_genbench.fingerprinting", src / "fingerprinting"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(sub)]
            sys.modules[name] = module
    logging.disable(logging.WARNING)
    from lemat_genbench.fingerprinting.utils import (
        get_fingerprint,
        get_fingerprinter,
    )
    from lemat_genbench.metrics.validity_metrics import (
        ChargeNeutralityMetric,
        MinimumInteratomicDistanceMetric,
        PhysicalPlausibilityMetric,
    )

    # default-configured metrics; their compute attributes are exactly what
    # lemat-genbench passes to compute_structure (e.g. the radius table of
    # the distance check, a BVAnalyzer for the charge check)
    metrics = {
        "charge": ChargeNeutralityMetric(),
        "distance": MinimumInteratomicDistanceMetric(),
        "plausibility": PhysicalPlausibilityMetric(),
    }
    return types.SimpleNamespace(
        get_fingerprint=get_fingerprint,
        fingerprinter=get_fingerprinter("bawl"),
        **{
            name: (m.compute_structure, m._get_compute_attributes())
            for name, m in metrics.items()
        },
    )


_LEMAT = None


def lemat():
    global _LEMAT
    if _LEMAT is None:
        warnings.filterwarnings("ignore")
        _LEMAT = import_lemat()
    return _LEMAT


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout


def guarded(func, *args, **kwargs):
    """Run one check with a timeout; return (value, error string)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        return func(*args, **kwargs), None
    except Timeout:
        return None, "timeout"
    except Exception as e:  # noqa: BLE001 - record any failure per structure
        return None, f"{type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)


def standardized(structure):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    return SpacegroupAnalyzer(
        structure, symprec=SYMPREC
    ).get_primitive_standard_structure()


def bawl_hash(structure):
    """BAWL hash of the standardized primitive cell."""
    lm = lemat()
    std, err_std = guarded(standardized, structure)
    h_std, err_h = (
        guarded(lm.get_fingerprint, std, lm.fingerprinter)
        if std is not None
        else (None, err_std)
    )
    return {"bawl_std": h_std, "bawl_error": err_h}


# ---- stage workers ----------------------------------------------------------
def _gen_worker(task):
    from pymatgen.core import Structure
    from smact.screening import smact_validity

    lm = lemat()
    gen, index, cif = task
    row = {"generator": gen, "index": index}
    s, err = guarded(Structure.from_str, cif, fmt="cif")
    if s is None:
        row["parse_error"] = err
        return row
    comp = s.composition
    row.update(
        formula=comp.reduced_formula,
        chemsys="-".join(sorted(el.symbol for el in comp.elements)),
        nelements=len(comp.elements),
        nsites=len(s),
    )
    row.update(bawl_hash(s))
    for name, column in (
        ("charge", "charge_deviation"),
        ("distance", "distance_score"),
        ("plausibility", "plausibility_score"),
    ):
        func, kwargs = getattr(lm, name)
        row[column], row[f"{name}_error"] = guarded(func, s, **kwargs)
    row["smact_valid"], row["smact_error"] = guarded(smact_validity, comp)
    return row


def _mp_worker(task):
    from pymatgen.core import Structure

    entry_id, material_id, sdict = task
    return {
        "entry_id": entry_id,
        "material_id": material_id,
        **bawl_hash(Structure.from_dict(sdict)),
    }


def _protogen_worker(task):
    from pymatgen.core import Structure

    gen, index, cif = task
    s, err = guarded(Structure.from_str, cif, fmt="cif")
    proto = None
    if s is not None:
        proto, err = guarded(prototype_string.get_string, s, symprec=0.1)
    return {
        "generator": gen,
        "index": index,
        "prototype": proto,
        "prototype_error": err,
    }


def _protomp_worker(task):
    from pymatgen.core import Structure

    entry_id, material_id, sdict = task
    proto, err = guarded(
        prototype_string.get_string, Structure.from_dict(sdict), symprec=0.1
    )
    return {
        "entry_id": entry_id,
        "material_id": material_id,
        "prototype": proto,
        "prototype_error": err,
    }


# ---- batching ---------------------------------------------------------------
def batch_key(worker, chunk) -> str:
    """sha256 of the worker's source and the tasks of one batch; a stored
    batch is reused only when its key is unchanged (changes inside the
    libraries the worker calls are not detected)."""
    h = hashlib.sha256(inspect.getsource(worker).encode())
    h.update(
        json.dumps(chunk, sort_keys=True, default=lambda o: o.item()).encode()
    )
    return h.hexdigest()


def run_batches(name, tasks, worker, workers):
    """Run ``worker`` over ``tasks`` in batches of BATCH, storing each batch
    with the key of its inputs (``.sha256`` next to it); batches on disk with
    the same key are reused (resume), the others are recomputed."""
    bdir = OUT_DIR / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    n_batches = (len(tasks) + BATCH - 1) // BATCH
    keys = [
        batch_key(worker, tasks[b * BATCH : (b + 1) * BATCH])
        for b in range(n_batches)
    ]

    def stored_key(b):
        path = bdir / f"{name}_{b:05d}.sha256"
        data = bdir / f"{name}_{b:05d}.jsonl.gz"
        if not (path.exists() and data.exists()):
            return None
        return path.read_text().strip()

    todo = [b for b in range(n_batches) if stored_key(b) != keys[b]]
    print(
        f"[{name}] {len(tasks)} tasks, {n_batches} batches, "
        f"{n_batches - len(todo)} reused",
        flush=True,
    )
    t0 = time.time()
    with Pool(workers, maxtasksperchild=50) as pool:
        for i, b in enumerate(todo, 1):
            chunk = tasks[b * BATCH : (b + 1) * BATCH]
            rows = pool.map(worker, chunk, chunksize=1)
            tmp = bdir / f"{name}_{b:05d}.jsonl.gz.tmp"
            with gzip.open(tmp, "wt") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            tmp.rename(bdir / f"{name}_{b:05d}.jsonl.gz")
            # written last: a batch without its key is recomputed
            (bdir / f"{name}_{b:05d}.sha256").write_text(keys[b] + "\n")
            el = time.time() - t0
            print(
                f"[{name}] batch {b + 1}/{n_batches} "
                f"({i}/{len(todo)} this run, {el / 60:.1f} min, "
                f"eta {el / i * (len(todo) - i) / 60:.1f} min)",
                flush=True,
            )
    rows = []
    for b in range(n_batches):
        with gzip.open(bdir / f"{name}_{b:05d}.jsonl.gz", "rt") as f:
            rows += [json.loads(line) for line in f]
    return pd.DataFrame(rows)


# ---- inputs -----------------------------------------------------------------
def load_generated():
    tasks = []
    for gen in EVALUATED:
        df = pd.read_csv(GEN_DIR / f"gen_crystals_{gen}.csv")
        tasks += [(gen, i, cif) for i, cif in enumerate(df["cif"])]
    return tasks


def load_mp_structures():
    """(entry_id, material_id, structure dict) of every MP thermo entry."""
    out = []
    for path in sorted((MP_DIR / "thermo_batches").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                doc = json.loads(line)
                for run_type, entry in doc["entries"].items():
                    out.append(
                        (
                            f"{doc['material_id']}::{run_type}",
                            doc["material_id"],
                            entry["structure"],
                        )
                    )
    return out


def load_mp_summary():
    with gzip.open(MP_DIR / "summary.jsonl.gz", "rt") as f:
        return pd.DataFrame(json.loads(line) for line in f)


def reduced_formula_of(sdict):
    from pymatgen.core import Composition

    counts = defaultdict(float)
    for site in sdict["sites"]:
        for sp in site["species"]:
            counts[sp["element"]] += sp["occu"]
    return Composition(counts).reduced_formula


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ---- combine ----------------------------------------------------------------
def novelty_class(elements, formula, systems, formulas):
    if formula in formulas:
        return "known_composition"
    if elements in systems:
        return "new_composition"
    return "new_chemsys"


def write_mp_prototypes(proto_mp, summary, path):
    """One table of the prototype string of every MP entry, with
    the formula and the experimental / theoretical flag of its material."""
    info = summary[["material_id", "formula_pretty", "chemsys", "theoretical"]]
    out = proto_mp.merge(info, on="material_id", how="left", validate="m:1")
    out = out[
        [
            "material_id",
            "entry_id",
            "formula_pretty",
            "chemsys",
            "theoretical",
            "prototype",
            "prototype_error",
        ]
    ].sort_values(["material_id", "entry_id"])
    out.to_csv(path, index=False)
    return out


def combine(gen, mp, summary, proto_gen, proto_mp):
    from pymatgen.core import Composition

    df = gen.merge(
        proto_gen, on=["generator", "index"], how="left", validate="1:1"
    )

    # drop structures with an element that no reference entry contains
    ref_ids = set(proto_mp.material_id)
    ref_elements = {
        e
        for cs in summary.loc[summary.material_id.isin(ref_ids), "chemsys"]
        for e in cs.split("-")
    }
    missing = df.chemsys.map(
        lambda cs: isinstance(cs, str)
        and not set(cs.split("-")) <= ref_elements
    )
    n_excluded = df[missing].groupby("generator").size()
    excluded = {
        "per_generator": {g: int(n) for g, n in n_excluded.items()},
        "elements": sorted(
            {e for cs in df.chemsys[missing] for e in cs.split("-")}
            - ref_elements
        ),
    }
    print(f"[combine] excluded (element not in reference): {excluded}")
    df = df[~missing]
    known_proto = set(proto_mp.prototype.dropna())
    df["novel_prototype"] = np.where(
        df.prototype.isna(), np.nan, ~df.prototype.isin(known_proto)
    )
    known_std = set(mp.bawl_std.dropna())
    df["novel_bawl"] = np.where(
        df.bawl_std.isna(), np.nan, ~df.bawl_std.isin(known_std)
    )

    # shortest interatomic distance as recorded with the MACE energies
    dmin = pd.concat(
        pd.read_csv(
            ENERGIES_DIR / f"gen_crystals_{g}_energies.csv",
            usecols=["index", "min_distance_angstrom"],
        ).assign(generator=g)
        for g in EVALUATED
    )
    df = df.merge(dmin, on=["generator", "index"], how="left", validate="1:1")
    df["min_distance_ok"] = df.min_distance_angstrom.ge(MIN_DISTANCE)

    df["charge_valid"] = df.charge_deviation.le(CHARGE_TOLERANCE)
    df["distance_valid"] = df.distance_score.eq(1.0)
    df["plausibility_valid"] = df.plausibility_score.eq(1.0)
    df["lemat_overall_valid"] = (
        df.charge_valid & df.distance_valid & df.plausibility_valid
    )
    df["smact_composition_valid"] = df.smact_valid.fillna(False).astype(bool)
    df["lemat_valid"] = df.lemat_overall_valid & df.min_distance_ok
    df["smact_valid"] = df.smact_composition_valid & df.min_distance_ok

    # composition / chemical-system novelty against the MP materials of the
    # reference (those with a GGA/GGA+U thermo entry), as in analysis_2
    summary = summary[
        ~summary.deprecated.astype(bool) & summary.material_id.isin(ref_ids)
    ]
    comp = summary.formula_pretty.map(lambda f: Composition(f))
    summary = summary.assign(
        rformula=comp.map(formula_key),
        system=comp.map(lambda c: tuple(sorted(e.symbol for e in c.elements))),
    )
    systems, formulas = set(summary.system), set(summary.rformula)
    df["novelty_class_vs_mp"] = [
        novelty_class(tuple(cs.split("-")), formula_key(f), systems, formulas)
        if isinstance(f, str)
        else None
        for cs, f in zip(df.chemsys, df.formula)
    ]

    # stability from analysis_2 (full reference, no thinning)
    eh = pd.read_csv(EHULL_CACHE)
    eh = eh[eh.removed_fraction == 0].drop_duplicates(["generator", "index"])
    df = df.merge(
        eh[["generator", "index", "ehull_eV"]],
        on=["generator", "index"],
        how="left",
    )
    df["stable"] = np.where(
        df.ehull_eV.isna(), np.nan, df.ehull_eV <= STABLE + TOL
    )
    df["metastable"] = np.where(
        df.ehull_eV.isna(), np.nan, df.ehull_eV <= METASTABLE + TOL
    )
    for col in ("smact_valid", "lemat_valid"):
        if df.loc[df[col], "ehull_eV"].isna().any():
            raise ValueError(f"{col} structures without E_hull")

    # uniqueness within each generator (first of each BAWL group)
    order = df.generator.map(list(EVALUATED).index) * 10**6 + df["index"]
    df = df.assign(_order=order).sort_values("_order")
    dup = df.duplicated(["generator", "bawl_std"]) & df.bawl_std.notna()
    df["unique"] = ~dup
    vun = df.smact_valid & df.unique & df.novel_bawl.eq(1)
    df["vsun"] = vun & df.stable.eq(1)
    df["vmsun"] = vun & df.metastable.eq(1)
    df = df.drop(columns="_order")
    df = df.sort_values(
        ["generator", "index"],
        key=lambda s: (
            s.map(list(EVALUATED).index) if s.name == "generator" else s
        ),
    ).reset_index(drop=True)
    return df, excluded


def summarize(df):
    rows = []
    flags = [
        "novel_bawl",
        "charge_valid",
        "smact_composition_valid",
        "min_distance_ok",
        "smact_valid",
        "distance_valid",
        "plausibility_valid",
        "lemat_overall_valid",
        "lemat_valid",
        "stable",
        "metastable",
        "unique",
        "novel_prototype",
        "vsun",
        "vmsun",
    ]
    for gen, g in df.groupby("generator", sort=False):
        row = {"generator": gen, "n_structures": len(g)}
        for f in flags:
            v = pd.to_numeric(g[f], errors="coerce")
            row[f"n_{f}_evaluated"] = int(v.notna().sum())
            # over all structures: not evaluated (NaN) counts as failing
            row[f"fraction_{f}"] = float(v.eq(1).mean())
        counts = g.novelty_class_vs_mp.value_counts(normalize=True)
        for c in ("known_composition", "new_composition", "new_chemsys"):
            row[f"fraction_{c}_vs_mp"] = float(counts.get(c, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def git_commit(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument(
        "--limit",
        type=int,
        help="only the first N structures per generator (smoke test; "
        "writes to data/analysis_3/test/)",
    )
    args = ap.parse_args()

    global OUT_DIR, FIG_DIR
    if args.limit:
        OUT_DIR = OUT_DIR / "test"
        FIG_DIR = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()

    tasks = load_generated()
    if args.limit:
        tasks = [t for t in tasks if t[1] < args.limit]
    gen = run_batches("gen", tasks, _gen_worker, args.workers)

    print("[mp] loading MP structures ...", flush=True)
    mp_all = load_mp_structures()
    # MP structures by formula_key, for those of the generated compositions
    wanted = {formula_key(f) for f in gen.formula.dropna()}
    mp_by_formula = defaultdict(list)
    for entry_id, mid, sdict in mp_all:
        k = formula_key(reduced_formula_of(sdict))
        if k in wanted:
            mp_by_formula[k].append((entry_id, mid, sdict))
    mp_tasks = [t for k in sorted(mp_by_formula) for t in mp_by_formula[k]]
    print(
        f"[mp] {len(mp_all)} MP entries, {len(mp_tasks)} with a reduced "
        f"formula among the generated ones ({len(mp_by_formula)} formulas)",
        flush=True,
    )
    mp = (
        run_batches("mp", mp_tasks, _mp_worker, args.workers)
        if mp_tasks
        else pd.DataFrame(columns=["bawl_std"])
    )

    proto_gen = run_batches("protogen", tasks, _protogen_worker, args.workers)
    proto_mp = run_batches("protomp", mp_all, _protomp_worker, args.workers)
    n_fail = int(proto_mp.prototype.isna().sum())
    print(f"[protomp] {n_fail} MP entries without a prototype", flush=True)

    summary_mp = load_mp_summary()
    write_mp_prototypes(proto_mp, summary_mp, OUT_DIR / "mp_prototypes.csv.gz")
    df, excluded = combine(gen, mp, summary_mp, proto_gen, proto_mp)
    out = OUT_DIR / "evaluation.csv.gz"
    df.to_csv(out, index=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(df)
    summary.to_csv(FIG_DIR / "evaluation_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "limit": args.limit,
        "lemat_genbench_commit": git_commit(LEMAT_ROOT),
        "fingerprint": "BAWL (lemat-genbench get_fingerprinter('bawl')) "
        f"on the standardized primitive cell (symprec {SYMPREC})",
        "prototype": "prototype_string.get_string (symprec 0.1) vs all MP "
        "entries",
        "mp_entries_without_prototype": n_fail,
        "charge_tolerance": CHARGE_TOLERANCE,
        "min_distance_angstrom": MIN_DISTANCE,
        "excluded_element_not_in_reference": excluded,
        "thresholds_eV": {"stable": STABLE, "metastable": METASTABLE},
        "mp_entries": len(mp_all),
        "mp_entries_hashed": len(mp_tasks),
        "mp_summary_sha256": sha256(MP_DIR / "summary.jsonl.gz"),
        "ehull_cache_sha256": sha256(EHULL_CACHE),
        "packages": {
            p: version(p)
            for p in ("pymatgen", "smact", "spglib", "material-hasher")
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out, FIG_DIR / "evaluation_summary.csv", flush=True)
    print(summary.T.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
