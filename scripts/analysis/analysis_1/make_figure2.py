"""Figure 2: decomposition energy of experimentally observed Materials
Project compounds against the MACE-OMAT-0 hull, by number of elements.

Candidates are the MP materials with an experimental counterpart
(``theoretical = False``) with 2-5 elements; each is represented by its
lowest-energy entry. The reference is the same self-consistent MACE-OMAT-0
MP reference (all GGA/GGA+U entries, MACE energies + OMat24 corrections,
scripts/analysis/corrections) and the same lower-convex-envelope linear
program as in analysis_2 - analysis_5,
so the values are directly comparable with the generated structures.

  E_hull  distance from the hull of the full reference (>= 0)
  E_d     signed distance from the hull of the reference without the
          candidate's own material (all its run types); other materials of
          the same composition (polymorphs) are kept. This is the quantity
          used for the generated structures, which are not in the
          reference: negative means below the hull of the other phases.
          (E_hull = max(E_d, 0).)

Panel (a): histogram of E_d (20 meV/atom bins, 0 as an edge) per element
count, as % of the compounds of that element count, offset vertically.
Panel (b): cumulative distribution of E_d; circles mark E_d = 0 (on the
hull) and triangles 100 meV/atom (metastable).

Inputs:  data/input/mp_mace_omat_0_medium/ (read as in analysis_2)
Outputs: data/analysis_1/ed_experimental.csv.gz   per compound
         figures/analysis_1/figure2.{pdf,png}
         figures/analysis_1/figure2_stats.csv      per element count
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "analysis_2"))
import plot_pseudo_stable_thinning as a2  # noqa: E402
import plot_style as ps  # noqa: E402

ROOT = HERE.parents[2]
CACHE = ROOT / "data" / "analysis_1" / "ed_experimental.csv.gz"
OUT = ROOT / "figures" / "analysis_1" / "figure2"
NELEMENTS = [2, 3, 4, 5]
LABELS = {2: "Binary", 3: "Ternary", 4: "Quaternary", 5: "Quinary"}
# dark coolwarm tones (pale centre avoided), as in analysis_2
COLORS = {n: ps.CMAP(v) for n, v in zip(NELEMENTS, (0.0, 0.22, 0.78, 1.0))}
LINE_KW = {"lw": 1.8, "alpha": 0.8}
# solid / dash-dot alternate so that neighbouring element counts are told
# apart where the curves overlap
LINESTYLES = {2: "-", 3: "-.", 4: "-", 5: "-."}
X_RANGE = (-300.0, 400.0)  # meV/atom
BIN_WIDTH = 20.0  # meV/atom
THRESHOLD = 100.0  # meV/atom, metastability
RIDGE_STEP = 28.0  # % per ridge
TOL = 1e-9

_REF = None


def _evaluate(task):
    """E_hull and E_d (eV/atom) of the candidates of one element set."""
    elements, cands = task
    ref = _REF
    idx = ref.entries_for(elements)
    cols = [ref.col[el] for el in elements]
    frac = ref.frac[np.ix_(idx, cols)]
    e = ref.e_per_atom[idx]
    mat = ref.material[idx]
    out = []
    for m, i in cands:
        x = ref.frac[i, cols]
        ei = ref.e_per_atom[i]
        h_all = a2.hull_energy(frac, e, x)
        keep = mat != m
        h_mat = a2.hull_energy(frac[keep], e[keep], x)
        out.append((m, i, max(ei - h_all, 0.0), ei - h_mat))
    return out


def compute(workers: int) -> pd.DataFrame:
    global _REF
    _, records = a2.load_reference(a2.REFERENCE_DIR)
    _REF = ref = a2.Reference(records)
    theo = np.array([bool(r["theoretical"]) for r in records])
    # lowest-energy entry of every experimental material with 2-5 elements
    best = {}
    for i, r in enumerate(records):
        if theo[i] or len(r["composition"]) not in NELEMENTS:
            continue
        m = ref.material[i]
        if m not in best or ref.e_per_atom[i] < ref.e_per_atom[best[m]]:
            best[m] = i
    groups = defaultdict(list)
    for m, i in best.items():
        groups[tuple(sorted(records[i]["composition"]))].append((m, i))
    tasks = sorted(
        groups.items(), key=lambda t: -len(ref.entries_for(t[0])) * len(t[1])
    )
    with Pool(workers) as pool:
        rows = [
            r
            for rs in pool.imap_unordered(_evaluate, tasks, chunksize=8)
            for r in rs
        ]
    df = pd.DataFrame(
        rows, columns=["material", "entry", "ehull_eV", "ed_eV"]
    )
    df["material_id"] = [records[i]["material_id"] for i in df.entry]
    df["entry_id"] = [records[i]["entry_id"] for i in df.entry]
    df["nelements"] = [len(records[i]["composition"]) for i in df.entry]
    df["formula"] = [
        "".join(f"{k}{v:g}" for k, v in records[i]["composition"].items())
        for i in df.entry
    ]
    if (np.maximum(df.ed_eV, 0) - df.ehull_eV).abs().max() > 1e-6:
        raise ValueError("E_hull != max(E_d, 0)")
    cols = [
        "material_id",
        "entry_id",
        "formula",
        "nelements",
        "ehull_eV",
        "ed_eV",
    ]
    return df[cols].sort_values("material_id").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in ["all"] + NELEMENTS:
        d = df if n == "all" else df[df.nelements == n]
        ed = d.ed_eV * 1000
        rows.append(
            {
                "nelements": n,
                "n_compounds": len(d),
                "fraction_on_hull": (d.ehull_eV <= TOL).mean(),
                "fraction_ehull_le_100": (
                    d.ehull_eV <= THRESHOLD / 1000 + TOL
                ).mean(),
                "fraction_ed_negative": (ed < -1e-6).mean(),
                "median_ed_meV": ed.median(),
                "q10_ed_meV": ed.quantile(0.1),
                "q90_ed_meV": ed.quantile(0.9),
                "fraction_below_range": (ed < X_RANGE[0]).mean(),
                "fraction_above_range": (ed > X_RANGE[1]).mean(),
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> list[Path]:
    lo, hi = X_RANGE
    edges = np.arange(lo, hi + BIN_WIDTH / 2, BIN_WIDTH)
    grid = np.linspace(lo, hi, 2001)
    fig, (ax_r, ax_c) = plt.subplots(
        2,
        1,
        figsize=ps.figsize("single", height=5.8),
        sharex=True,
        height_ratios=[1.8, 1],
    )
    k = len(NELEMENTS)
    for j, n in enumerate(NELEMENTS):
        v = df.loc[df.nelements == n, "ed_eV"].to_numpy() * 1000
        c = COLORS[n]
        # (a) ridge: binary on top
        base = (k - 1 - j) * RIDGE_STEP
        counts, _ = np.histogram(v, edges)
        frac = counts / len(v) * 100
        if frac.max() > RIDGE_STEP:
            raise ValueError(f"{LABELS[n]} bin of {frac.max():.1f} % overlaps")
        ax_r.stairs(
            frac + base,
            edges,
            baseline=base,
            fill=True,
            color=c,
            alpha=0.45,
            zorder=2 * j + 1,
        )
        ax_r.stairs(
            frac + base,
            edges,
            baseline=None,
            color=c,
            lw=LINE_KW["lw"],
            zorder=2 * j + 2,
        )
        ax_r.axhline(base, color="0.3", lw=0.8, zorder=2 * j + 1)
        for g in (10, 20):
            ax_r.axhline(base + g, color="0.88", lw=0.5, zorder=0)
        ax_r.text(
            0.02,
            base + 1.5,
            LABELS[n],
            transform=ax_r.get_yaxis_transform(),
            ha="left",
            va="bottom",
            color="0.15",
            zorder=10,
        )
        # (b) cumulative distribution
        s = np.sort(v)
        y = np.searchsorted(s, grid, side="right") / len(s) * 100
        ax_c.plot(
            grid, y, color=c, ls=LINESTYLES[n], label=LABELS[n], **LINE_KW
        )
        for x0, marker, ms in ((0.0, "o", 6.0), (THRESHOLD, "^", 7.5)):
            ax_c.plot(
                x0,
                (v <= x0 + 1e-6).mean() * 100,
                marker,
                color=c,
                ms=ms,
                mec="white",
                mew=0.8,
                zorder=8,
            )
    for ax in (ax_r, ax_c):
        ax.axvline(0, color="0.4", lw=1.0, ls="--", zorder=0)
        ax.axvline(THRESHOLD, color="0.4", lw=1.0, ls=":", zorder=0)
    ax_r.set_xlim(lo, hi)
    ax_r.set_ylim(0, (k - 1) * RIDGE_STEP + 24)
    ticks = [i * RIDGE_STEP + t for i in range(k) for t in (0, 10, 20)]
    ax_r.set_yticks(ticks, ["0", "10", "20"] * k)
    ax_r.set_ylabel("Fraction of compounds (%)")  # per 20 meV/atom bin
    ax_c.set_ylim(0, 100)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.grid(axis="y", color="0.88", lw=0.5)
    ax_c.set_axisbelow(True)
    ax_c.set_ylabel("Cumulative fraction (%)")
    ax_c.set_xlabel("Decomposition energy (meV/atom)")
    ax_c.legend(
        loc="upper left",
        frameon=True,
        handlelength=1.8,
        borderpad=0.3,
        labelspacing=0.2,
    )
    ps.label_panels((ax_r, ax_c), x=-0.2, y=1.02)
    return ps.savefig(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        df = pd.read_csv(args.cache)
    else:
        df = compute(args.workers)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.cache, index=False, float_format="%.8g")
    ps.apply_style()
    paths = plot(df, args.out)
    stats = summarize(df)
    stats_path = args.out.with_name(args.out.name + "_stats.csv")
    stats.to_csv(stats_path, index=False)
    print("wrote", *[str(p) for p in paths], str(stats_path), sep="\n  ")
    print(stats.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
