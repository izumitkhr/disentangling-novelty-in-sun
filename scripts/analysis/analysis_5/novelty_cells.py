"""Stability of the generated structures across novelty cells.

Structures: valid (SMACT), unique and novel (BAWL) generated structures
with 2-5 elements and an E_hull (analysis_3; the same structures as the
E_d distributions of Figures 4 and 5), split into the novelty cells of
analysis_3/common.py: composition / chemical-system class (color) x
prototype known (filled marker, solid bar) / new (open marker, white
diagonal lines).

Per generator and cell: the number of structures and the stable
(E_hull = 0) and metastable (E_hull <= 0.1 eV/atom) fractions with 95 %
Wilson intervals; and the composition of the V-SUN (valid, unique, novel,
stable) and V-MSUN (metastable) structures by cell. E_hull against the
full MACE-OMAT-0 MP reference (analysis_2).

The panels of Figure 6 (cell shares, stable and metastable fractions) are
drawn by draw(), which analysis_6/make_figure6.py uses.

Input:  data/analysis_3/evaluation.csv.gz (analysis_3/evaluate_generated.py)
Output: figures/analysis_5/novelty_cells_stats.csv  rates with intervals
                                                    per cell, and V-SUN /
                                                    V-MSUN composition
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
A3 = HERE.parent / "analysis_3"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(A3))
import common as cm  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = cm.ROOT / "figures" / "analysis_5" / "novelty_cells_stats.csv"
# panel title: what is counted; the x axis is the fraction of structures
METRICS = {
    "stable": r"Stable ($E_\mathrm{hull} = 0$)",
    "metastable": r"Metastable ($E_\mathrm{hull} \leq 100$ meV/atom)",
}
XMAX = {"stable": 100, "metastable": 100}
SETS = {"stable": "V-SUN", "metastable": "V-MSUN"}
PROTO_OFFSET = {"known": 0.14, "new": -0.14}


Z = 1.959964  # 95 %


def wilson(k: int, n: int) -> tuple[float, float]:
    """95 % Wilson score interval of k successes in n trials."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + Z**2 / n
    mid = (p + Z**2 / (2 * n)) / den
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return mid - half, mid + half


def rates(ev: pd.DataFrame, generators: dict = cm.GENERATORS) -> pd.DataFrame:
    """Stable / metastable fractions with Wilson intervals per generator,
    class and prototype, among the valid, unique, novel structures."""
    ev = ev[
        ev.ehull_eV.notna()
        & ev.novel_prototype.notna()
        & ev.nelements.isin(cm.ED_NELEMENTS)
    ].copy()
    ev["proto"] = np.where(ev.novel_prototype == 1, "new", "known")
    d = ev[cm.masks(ev)["novel"]]
    rows = []
    for gen in generators:
        for cls in cm.CLASSES:
            for proto in ("known", "new"):
                s = d[
                    (d.generator == gen)
                    & (d.novelty_class_vs_mp == cls)
                    & (d.proto == proto)
                ]
                row = {
                    "generator": gen,
                    "novelty_class_vs_mp": cls,
                    "prototype": proto,
                    "n_structures": len(s),
                }
                for m in METRICS:
                    k = int(s[m].eq(1).sum())
                    lo, hi = wilson(k, len(s))
                    row.update(
                        {
                            f"n_{m}": k,
                            f"fraction_{m}": k / len(s) if len(s) else np.nan,
                            f"fraction_{m}_low": lo,
                            f"fraction_{m}_high": hi,
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def load(path: Path) -> pd.DataFrame:
    ev = cm.read_table(path)
    ev = ev[
        ev.ehull_eV.notna()
        & ev.novel_prototype.notna()
        & ev.nelements.isin(cm.ED_NELEMENTS)
    ].copy()
    ev = ev[cm.masks(ev)["novel"]]  # valid, unique, novel
    ev["cell"] = cm.assign_cells(ev)
    return ev


def composition(vun: pd.DataFrame) -> pd.DataFrame:
    """Novelty cells of the V-SUN and V-MSUN structures per generator."""
    rows = []
    for m, label in SETS.items():
        for gen in cm.GENERATORS:
            s = vun[(vun.generator == gen) & vun[m].eq(1)]
            for cell, share in cm.shares(s).items():
                if cell == cm.NOT_NOVEL:
                    continue
                rows.append(
                    {
                        "set": label,
                        "generator": gen,
                        "cell": cm.cell_label(cell),
                        "n_structures": int((s.cell == cell).sum()),
                        "n_set": len(s),
                        "share": share,
                    }
                )
    return pd.DataFrame(rows)


def dot_panel(ax, st, m, ypos, rows):
    for yi, (g, c) in zip(ypos, rows):
        col = cm.CLASS_COLORS[c]
        for p in ("known", "new"):
            x = st.loc[(g, c, p), f"fraction_{m}"] * 100
            lo = st.loc[(g, c, p), f"fraction_{m}_low"] * 100
            hi = st.loc[(g, c, p), f"fraction_{m}_high"] * 100
            eb = ax.errorbar(
                x,
                yi + PROTO_OFFSET[p],
                # Wilson bound at k = 0 can exceed 0 by rounding
                xerr=[[max(x - lo, 0.0)], [max(hi - x, 0.0)]],
                fmt="o",
                ms=4.5,
                color=col,
                mfc=col if p == "known" else "white",
                mec=col,
                mew=1.2,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                zorder=4,
            )
            # markers and bars at 0 % are drawn whole over the axes frame
            for art in eb.get_children():
                art.set_clip_on(False)
    ax.set_xlim(0, XMAX[m])
    ax.set_xticks(np.arange(0, XMAX[m] + 1e-9, 25))
    ax.set_title(METRICS[m])
    ax.set_xlabel("Fraction of structures (%)")
    ax.grid(axis="x", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", length=0)
    top = st[f"fraction_{m}_high"].max() * 100
    if top > XMAX[m]:
        raise ValueError(f"{m}: interval up to {top:.1f} % > {XMAX[m]}")


def legend(fig):
    """One row on top of ``fig``: class colors, then known / new prototype
    (marker and bar)."""
    class_handles = [
        Patch(facecolor=cm.CLASS_COLORS[c], label=cm.CLASS_LABELS[c])
        for c in cm.CLASSES
    ]
    marker_known = Line2D([], [], ls="none", marker="o", ms=4.5, color="0.35")
    marker_new = Line2D(
        [],
        [],
        ls="none",
        marker="o",
        ms=4.5,
        mfc="white",
        mec="0.35",
        mew=1.2,
    )
    bar_known = Patch(**cm.style("0.35", True))
    bar_new = Patch(**cm.style("0.35", False))
    handles = class_handles + [
        (marker_known, bar_known),
        (marker_new, bar_new),
    ]
    labels = [h.get_label() for h in class_handles] + [
        "Known prototype",
        "New prototype",
    ]
    leg = fig.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        loc="outside upper center",
        ncol=len(handles),
        handlelength=2.2,
        handleheight=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.canvas.draw()
    if leg.get_window_extent().width > fig.bbox.width:
        raise ValueError("Legend is wider than the figure")


def draw(
    fig,
    vun: pd.DataFrame,
    rates: pd.DataFrame,
    generators: dict = cm.GENERATORS,
    panel_labels: list[str] | None = None,
    name_offset: float = 82,
) -> tuple:
    """Draw (a)-(c) of Figure 6 on ``fig`` (a Figure or a SubFigure),
    without a legend (legend() adds it); returns the axes of (a), (b), (c).
    ``name_offset``: distance (pt) of the generator names left of (b)."""
    st = rates.set_index(["generator", "novelty_class_vs_mp", "prototype"])
    # the empty middle row separates (a) from the titles of (b), (c)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.75, 0.1, 2.3])
    gens = list(generators)

    # (a) novelty cells of the valid, unique, novel structures
    ax_a = fig.add_subplot(gs[0, :])
    sel = [vun[vun.generator == g] for g in gens]
    cm.stacked(
        ax_a,
        [cm.shares(s) for s in sel],
        [generators[g] for g in gens],
        [f"{len(s)}" for s in sel],
    )
    ax_a.set_xlabel("Share of VUN structures (%)")

    # (b), (c) stable / metastable fraction per cell
    rows = [(g, c) for g in gens for c in cm.CLASSES]
    ypos, y = [], 0.0
    for i, (g, _) in enumerate(rows):
        if i and rows[i - 1][0] != g:
            y += 0.7
        ypos.append(y)
        y += 1.0
    ypos = -np.array(ypos)
    ax_b = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[2, 1], sharey=ax_b)
    for ax, m in zip((ax_b, ax_c), METRICS):
        dot_panel(ax, st, m, ypos, rows)
        # light separators between the generators
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                ax.axhline(
                    (ypos[i] + ypos[i - 1]) / 2,
                    color="0.8",
                    lw=0.6,
                    zorder=0,
                )
    ax_b.set_yticks(ypos, [cm.CLASS_LABELS[c] for _, c in rows])
    # keep the row labels clear of markers drawn on the frame at 0 %
    ax_b.tick_params(axis="y", pad=7)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    for g in gens:
        ys = [yi for yi, (gg, _) in zip(ypos, rows) if gg == g]
        ax_b.annotate(
            generators[g],
            xy=(0, np.mean(ys)),
            xycoords=("axes fraction", "data"),
            xytext=(-name_offset, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            rotation=90,
        )
    ps.label_panels((ax_a, ax_b, ax_c), panel_labels, x=0.0, y=1.02)
    return ax_a, ax_b, ax_c


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--evaluation", type=Path, default=cm.EVALUATION)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ev = cm.read_table(args.evaluation)
    rate_table = rates(ev)
    comp = composition(load(args.evaluation))
    stats = pd.concat(
        [rate_table.assign(table="rate"), comp.assign(table="composition")],
        ignore_index=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out, index=False)
    print("wrote", args.out)
    print(
        comp.pivot_table(
            index="cell",
            columns=["set", "generator"],
            values="share",
            sort=False,
        )
        .mul(100)
        .round(1)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
