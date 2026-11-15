"""Figure 3: pseudo-stability of generated structures under reference
thinning.

Rows: generators (MatterGen, DiffCSP++, WyFormer). Columns, all against the
fraction of theoretical MP materials removed from the reference (mean over
5 seeds, band min-max):
  left    pseudo-stable rate per number of elements
          (plot_pseudo_stable_thinning.py)
  middle  number of pseudo-stable structures per novelty class, stacked
          (plot_pseudo_stable_by_novelty.py)
  right   pseudo-stable rate per novelty class
          (plot_pseudo_stable_by_novelty.py)
Threshold t = 0 (stable). Points whose denominator is below 10 stable
structures in any seed are not drawn (quinary everywhere).

Inputs:  data/analysis_2/ehull_thinning.csv.gz
         data/analysis_2/novelty_thinning.csv.gz
Run plot_pseudo_stable_thinning.py and plot_pseudo_stable_by_novelty.py first
(they write the caches).

Outputs: figures/analysis_2/figure3.{pdf,png}            t = 0 (stable), rate
                                                       axes 0-100 %
         figures/analysis_2/figure_s1.{pdf,png}         t = 100 meV/atom
                                                       (metastable), rate
                                                       axes 0-20 %
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import plot_pseudo_stable_by_novelty as nov  # noqa: E402
import plot_pseudo_stable_thinning as a2  # noqa: E402
import plot_style as ps  # noqa: E402

OUT_DIR = nov.OUT_DIR
# threshold (eV/atom) -> (output name, rate y max in %, count y max or
# None for automatic)
FIGURES = {
    0.0: ("figure3", 100.0, 300.0),
    0.1: ("figure_s1", 20.0, None),
}
NEL_COLORS = {
    n: ps.CMAP(v) for n, v in zip(a2.NELEMENTS, (0.0, 0.22, 0.78, 1.0))
}


def band_line(ax, d, color, ls, marker, label):
    x = d.index * 100
    ax.fill_between(
        x,
        d.pseudo_rate_min * 100,
        d.pseudo_rate_max * 100,
        color=color,
        alpha=0.25,
        lw=0,
    )
    (line,) = ax.plot(
        x,
        d.pseudo_rate_mean * 100,
        color=color,
        ls=ls,
        marker=marker,
        ms=3.5,
        mec="white",
        mew=0.5,
        label=label,
        # points on y = 0 are drawn whole over the axes frame
        clip_on=False,
        zorder=4,
        **a2.LINE_KW,
    )
    return line


def draw(nel, cls, t, out, ymax, min_stable, count_ymax=None):
    """One row per generator, 3 columns, at threshold ``t`` (eV/atom); rate
    axes 0-``ymax`` %."""
    generators = a2.GENERATORS
    nel = nel[np.isclose(nel.threshold_eV, t)]
    cls = cls[
        np.isclose(cls.threshold_eV, t) & cls.generator.isin(generators)
    ]
    word = "Pseudo-stable" if t == 0 else "Pseudo-metastable"
    fig, grid = plt.subplots(
        len(generators),
        3,
        # 5.6 in for the three generators of Figure 3
        figsize=ps.figsize("double", height=1.6 * len(generators) + 0.8),
        sharex=True,
        squeeze=False,
    )
    count_max = (
        cls.groupby(["generator", "removed_fraction"])
        .n_pseudo_mean.sum()
        .max()
    )
    nel_handles, cls_handles = {}, {}
    for (ax_e, ax_n, ax_c), (gen, label) in zip(grid, generators.items()):
        # left: rate per number of elements
        for n in a2.NELEMENTS:
            d = nel[(nel.generator == gen) & (nel.nelements == n)]
            d = d.set_index("removed_fraction")
            d = d.where(d.n_stable_min >= min_stable)
            line = band_line(
                ax_e,
                d,
                NEL_COLORS[n],
                a2.LINESTYLES[n],
                a2.MARKERS[n],
                a2.LABELS[n],
            )
            if d.pseudo_rate_mean.notna().any():
                nel_handles.setdefault(n, line)
        # middle: stacked counts, right: rate per class
        g = cls[cls.generator == gen]
        bottom = None
        for cat in nov.CATEGORIES:
            d = g[g.category == cat].set_index("removed_fraction")
            c = nov.CAT_COLORS[cat]
            h = d.n_pseudo_mean
            ax_n.bar(
                d.index * 100,
                h,
                width=8,
                bottom=bottom,
                color=c,
                alpha=a2.LINE_KW["alpha"],
                lw=0,
            )
            bottom = h if bottom is None else bottom + h
            r = d.where(d.n_stable_min >= min_stable)
            line = band_line(
                ax_c,
                r,
                c,
                nov.CAT_LINESTYLES[cat],
                nov.CAT_MARKERS[cat],
                nov.CAT_LABELS[cat],
            )
            cls_handles.setdefault(cat, line)
        for ax in (ax_e, ax_c):
            top = max(
                (
                    np.nanmax(p.get_paths()[0].vertices[:, 1])
                    for p in ax.collections
                    if len(p.get_paths())
                ),
                default=0,
            )
            if top > ymax + 1e-9:
                raise ValueError(f"{gen}: band up to {top:.1f} % > {ymax}")
            ax.set_ylim(0, ymax)
            step = 25 if ymax == 100 else 5
            ax.set_yticks(np.arange(0, ymax + 1e-9, step))
        if count_ymax is None:
            ax_n.set_ylim(0, count_max * 1.08)
        else:
            if count_max > count_ymax:
                raise ValueError(
                    f"{count_max:.0f} structures exceed the count y range"
                )
            ax_n.set_ylim(0, count_ymax)
            ax_n.set_yticks(np.arange(0, count_ymax + 1e-9, 100))
        for ax in (ax_e, ax_n, ax_c):
            ax.set_xlim(-5, 105)
            ax.set_xticks([0, 50, 100])
            ax.set_xticks([25, 75], minor=True)
            ax.grid(axis="y", color="0.88", lw=0.5)
            ax.set_axisbelow(True)
        ax_e.set_ylabel(f"{label}\nRate (%)")
        ax_n.set_ylabel("Number of structures")
        ax_c.set_ylabel("Rate (%)")
    # "Pseudo-metastable" titles are too wide for one line per column
    # two lines per column title: "<word> <what> by" / "<grouping>"
    for ax, (what, by) in zip(
        grid[0],
        (
            ("rate", "number of constituent elements"),
            ("count", "compositional novelty"),
            ("rate", "compositional novelty"),
        ),
    ):
        ax.set_title(f"{word} {what} by\n{by}", pad=16)
    grid[0, 0].legend(
        [nel_handles[n] for n in a2.NELEMENTS if n in nel_handles],
        [a2.LABELS[n] for n in a2.NELEMENTS if n in nel_handles],
        loc="upper left",
        frameon=True,
        handlelength=2.4,
        borderpad=0.3,
        labelspacing=0.2,
    )
    # middle column: same class colors as patches (stacked bars)
    grid[0, 1].legend(
        handles=[
            Patch(
                color=nov.CAT_COLORS[c],
                alpha=a2.LINE_KW["alpha"],
                label=nov.CAT_LABELS[c],
            )
            for c in nov.CATEGORIES  # same order as the right column
        ],
        loc="upper left",
        frameon=True,
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.2,
    )
    grid[0, 2].legend(
        [cls_handles[c] for c in nov.CATEGORIES],
        [nov.CAT_LABELS[c] for c in nov.CATEGORIES],
        loc="upper left",
        frameon=True,
        handlelength=1.8,
        borderpad=0.3,
        labelspacing=0.2,
    )
    fig.supxlabel(
        "Theoretical MP materials removed (%)", fontsize=ps.FONT_SIZE
    )
    ps.label_panels(grid.ravel(), x=0.0, y=1.02)
    paths = ps.savefig(fig, out)
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--min-stable", type=int, default=a2.MIN_STABLE)
    args = ap.parse_args()

    thresholds = list(FIGURES)
    nel = a2.summarize(pd.read_csv(a2.CACHE), thresholds)
    nel = nel[nel.nelements != "all"]
    df = pd.read_csv(nov.CACHE)
    cls = nov.summarize(df[df.nelements.isin(nov.NELEMENTS)], thresholds)

    ps.apply_style()
    paths = []
    for t, (name, ymax, count_ymax) in FIGURES.items():
        paths += draw(
            nel,
            cls,
            t,
            args.out_dir / name,
            ymax,
            args.min_stable,
            count_ymax,
        )
    print("wrote", *[str(p) for p in paths], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
