"""Pseudo-stable rate of generated structures by compositional novelty
under reference thinning.

Each generated structure is classified against the reference it was judged
with (the thinned reference of plot_pseudo_stable_thinning.py, for every
removed fraction and seed):

  new_chemsys        no reference phase has exactly its element set
  new_composition    its element set exists, but no phase with the same
                     reduced formula
  known_composition  a phase with the same reduced formula exists

A structure is pseudo-stable at threshold t when E_hull <= t against the
thinned reference and E_hull > t against the full reference (as in
plot_pseudo_stable_thinning.py). Thinning moves structures between classes (a chemical system
whose phases were all removed becomes new), so the classes are recomputed
for every removed fraction and seed.

The stats table holds, per class, the number of pseudo-stable structures
and the pseudo-stable rate, i.e. among the structures of a class judged
stable against the thinned reference, the fraction unstable against the full
reference (mean, min and max over seeds). The figures are drawn by
make_figure3.py.

Only binary to quinary structures are counted, as in
plot_pseudo_stable_thinning.py.

Inputs:
  data/analysis_2/ehull_thinning.csv.gz        E_hull per structure, removed
                                               fraction and seed (from
                                               plot_pseudo_stable_thinning.py)
  the MACE reference and generated-structure energies (the removal order is
  reproduced from the same seeds)

Outputs:
  data/analysis_2/novelty_thinning.csv.gz      class per structure, removed
                                               fraction and seed
  figures/analysis_2/novelty_stats.csv         counts and rates per class
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_pseudo_stable_thinning as a2  # noqa: E402
from formula import formula_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "analysis_2" / "novelty_thinning.csv.gz"
OUT_DIR = ROOT / "figures" / "analysis_2"

GENERATORS = a2.EVALUATED  # stats tables cover every evaluated generator
NELEMENTS = a2.NELEMENTS
TOL = a2.TOL
CATEGORIES = ["known_composition", "new_composition", "new_chemsys"]
# short labels so that the legend fits inside panel (a)
CAT_LABELS = {
    "known_composition": "Known comp.",
    "new_composition": "New comp.",
    "new_chemsys": "New chemsys",
}
# categorical (not ordered) classes: Okabe-Ito colors, colour-blind safe
# and distinct from the coolwarm element-count palette; the same values are
# in analysis_3/common.py
CAT_COLORS = {
    "known_composition": "#009E73",  # bluish green
    "new_composition": "#0072B2",  # blue
    "new_chemsys": "#D55E00",  # vermillion
}
CAT_LINESTYLES = {  # all solid; classes differ by color and marker
    "known_composition": "-",
    "new_composition": "-",
    "new_chemsys": "-",
}
CAT_MARKERS = {
    "known_composition": "o",
    "new_composition": "s",
    "new_chemsys": "D",
}


# ---- classification ---------------------------------------------------------
def classify(eh: pd.DataFrame) -> pd.DataFrame:
    """Class of every structure for every removed fraction and seed.

    A material is kept at a removed fraction when its removal rank is at
    least the number of removed materials, so a class is present in the
    thinned reference when the largest rank among its materials reaches
    that number (elemental and experimental materials have rank inf).
    """
    manifest, records = a2.load_reference(a2.REFERENCE_DIR)
    ref = a2.Reference(records)
    gen = a2.load_generated(a2.ENERGIES_DIR, manifest)
    cands = a2.prepare_candidates(gen, ref)

    by_system, by_formula = defaultdict(set), defaultdict(set)
    seen = set()
    for i, r in enumerate(records):
        m = ref.material[i]
        if m in seen:
            continue
        seen.add(m)
        by_system[tuple(sorted(r["composition"]))].add(m)
        by_formula[formula_key(r["composition"])].add(m)
    formula = cands.composition.map(formula_key)

    rows = []
    for seed in sorted(eh.seed.unique()):
        rank = ref.removal_rank(int(seed))

        def max_rank(materials, rank=rank):
            return max((rank[m] for m in materials), default=-1.0)

        rows.append(
            pd.DataFrame(
                {
                    "generator": cands.generator,
                    "index": cands["index"],
                    "seed": seed,
                    "max_rank_chemsys": [
                        max_rank(by_system.get(e, ())) for e in cands.elements
                    ],
                    "max_rank_formula": [
                        max_rank(by_formula.get(f, ())) for f in formula
                    ],
                }
            )
        )
    ranks = pd.concat(rows, ignore_index=True)
    out = eh.merge(ranks, on=["generator", "index", "seed"], validate="m:1")
    if len(out) != len(eh):
        raise ValueError("Structures in the E_hull cache and inputs differ")
    cut = out.n_removed_materials
    out["category"] = np.select(
        [out.max_rank_formula >= cut, out.max_rank_chemsys >= cut],
        ["known_composition", "new_composition"],
        "new_chemsys",
    )
    return out.drop(columns=["max_rank_chemsys", "max_rank_formula"])


# ---- summary and plot -------------------------------------------------------
def summarize(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Per generator, threshold, removed fraction and class: structures,
    stable and pseudo-stable counts and the pseudo-stable rate, as mean /
    min / max over seeds."""
    full = df[df.removed_fraction == 0][["generator", "index", "seed"]]
    full = full.assign(
        ehull_full_eV=df.loc[df.removed_fraction == 0, "ehull_eV"]
    )
    df = df.merge(full, on=["generator", "index", "seed"], validate="m:1")
    if (df.ehull_eV > df.ehull_full_eV + TOL).any():
        raise ValueError("E_hull increased after removing reference phases")
    keys = ["generator", "removed_fraction", "seed", "category"]
    rows = []
    for t in thresholds:
        stable = df.ehull_eV <= t + TOL
        per_seed = (
            pd.DataFrame(
                {
                    **{k: df[k] for k in keys},
                    "n": 1,
                    "n_stable": stable,
                    "n_pseudo": stable & (df.ehull_full_eV > t + TOL),
                }
            )
            .groupby(keys)
            .sum()
            # classes absent in a seed count as zero structures
            .unstack("category", fill_value=0)
            .stack("category")
            .reset_index()
        )
        per_seed["rate"] = per_seed.n_pseudo / per_seed.n_stable.where(
            per_seed.n_stable > 0
        )
        total = per_seed.groupby(keys[:3]).n_pseudo.transform("sum")
        per_seed["share"] = per_seed.n_pseudo / total.where(total > 0)
        agg = (
            per_seed.groupby(["generator", "removed_fraction", "category"])
            .agg(
                n_structures_mean=("n", "mean"),
                n_stable_mean=("n_stable", "mean"),
                n_stable_min=("n_stable", "min"),
                n_pseudo_mean=("n_pseudo", "mean"),
                n_pseudo_min=("n_pseudo", "min"),
                n_pseudo_max=("n_pseudo", "max"),
                pseudo_rate_mean=("rate", "mean"),
                pseudo_rate_min=("rate", "min"),
                pseudo_rate_max=("rate", "max"),
                pseudo_share_mean=("share", "mean"),
                n_seeds=("n", "size"),
            )
            .reset_index()
        )
        agg.insert(1, "threshold_eV", t)
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out["gen_order"] = out.generator.map(list(GENERATORS).index)
    out["cat_order"] = out.category.map(CATEGORIES.index)
    return (
        out.sort_values(
            ["threshold_eV", "gen_order", "cat_order", "removed_fraction"]
        )
        .drop(columns=["gen_order", "cat_order"])
        .reset_index(drop=True)
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ehull-cache", type=Path, default=a2.CACHE)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.1],
        help="stability thresholds in eV/atom",
    )
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="reuse the cached classes instead of recomputing",
    )
    args = ap.parse_args()

    if args.from_cache:
        df = pd.read_csv(args.cache)
    else:
        eh = pd.read_csv(args.ehull_cache)
        df = classify(eh)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.cache, index=False, float_format="%.8g")
        print("wrote", args.cache)
    df = df[df.nelements.isin(NELEMENTS)]

    stats = summarize(df, args.thresholds)
    # figures (t = 0 and the t > 0 supplement) are drawn by make_figure3.py
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.out_dir / "novelty_stats.csv"
    stats.to_csv(stats_path, index=False)
    print("wrote", stats_path)
    for col, scale in (
        ("n_pseudo_mean", 1),
        ("pseudo_rate_mean", 100),
        ("n_stable_mean", 1),
    ):
        view = stats.pivot_table(
            index=["threshold_eV", "generator", "category"],
            columns="removed_fraction",
            values=col,
            dropna=False,
        )
        print(f"\n{col}" + (" (%)" if scale == 100 else ""))
        print((view * scale).round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
