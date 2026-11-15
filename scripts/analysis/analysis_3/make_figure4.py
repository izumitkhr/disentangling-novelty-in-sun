"""Figure 4: the generated structures by composition / chemical-system
novelty.

Top row (a-c), one panel per generator: filtering of a V-SUN style
benchmark. The structures pass the filters one after another: Generated ->
Valid (SMACT) -> + Unique (BAWL) -> + Novel (BAWL, against MP) -> +
Metastable (E_hull <= 0.1 eV/atom) -> + Stable (E_hull = 0). Each bar is
the class mix (known composition / new composition / new chemsys, against
all MP materials) of the structures left after the step, as 100 %; the
number on the right is the share of the generated structures left (the last
row is the V-SUN rate).

Middle row (d-f): class mix of the valid, unique, novel structures with
2-5 elements per number of elements (100 % per bar); the number on the
right is the share of these structures with that element count.

Bottom row (g-i): cumulative distribution of the decomposition energy E_d
of the valid, unique, novel structures with 2-5 elements, per class; grey:
the MP experimental compounds of Figure 2. Dashed line E_d = 0 (stable),
dotted line 100 meV/atom (metastable). E_d against the full MACE-OMAT-0 MP
reference (compute_ed.py; analysis_1 for MP).

With ``--validity lemat`` the Valid step and the E_d structures use
lemat-genbench's overall validity instead of SMACT (supplement).

Inputs:  data/analysis_3/evaluation.csv.gz, data/analysis_3/ed_full.csv.gz,
         data/analysis_1/ed_experimental.csv.gz
Outputs: figures/analysis_3/figure4.{pdf,png}
         figures/analysis_3/figure_s2.{pdf,png}  (--validity lemat;
                                                 Fig. S2 with --no-nelements)
         figures/analysis_3/figure4_stats.csv   class shares per step and per
                                                element count, E_d quantiles
                                                per class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = cm.FIG_DIR / "figure4"
MIN_LABEL = 0.10  # write the share inside segments at least this wide
BAR_TEXT_SIZE = 9  # pt, slightly below the 10 pt text so it sits in the bars
LINE_KW = {"lw": 1.8, "alpha": 0.85}


def stacked_panel(ax, st, gen, rows, labels, left_labels):
    """100 % stacked class bars for ``rows`` (index level "step"), with the
    share of the generated structures on the right."""
    steps = list(rows)
    y = np.arange(len(steps))[::-1]
    left = np.zeros(len(steps))
    for cls in cm.CLASSES:
        w = np.array([st.loc[(gen, s, cls), "share_of_step"] for s in steps])
        ax.barh(
            y,
            w * 100,
            left=left * 100,
            height=0.72,
            **cm.style(cm.CLASS_COLORS[cls], True),
        )
        for yi, li, wi in zip(y, left, w):
            if wi >= MIN_LABEL:
                ax.text(
                    (li + wi / 2) * 100,
                    yi,
                    f"{wi * 100:.0f}",
                    ha="center",
                    va="center",
                    color=cm.CLASS_TEXT[cls],
                    fontsize=BAR_TEXT_SIZE,
                )
        left += w
    for yi, s in zip(y, steps):
        frac = st.loc[(gen, s, cm.CLASSES[0]), "fraction_of_generated"]
        txt = f"{frac * 100:.0f}%" if frac >= 0.1 else f"{frac * 100:.1f}%"
        ax.text(102, yi, txt, ha="left", va="center")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticks([25, 75], minor=True)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_yticks(y, [labels[s] for s in steps] if left_labels else [])


def summarize_nelements(
    d: pd.DataFrame, generators=cm.GENERATORS
) -> pd.DataFrame:
    """Class mix of the E_d structures per number of elements."""
    rows = []
    for gen in generators:
        g = d[d.generator == gen]
        for nel in cm.NEL_LABELS:
            s = g[g.nelements == nel]
            for cls in cm.CLASSES:
                n = int((s.novelty_class_vs_mp == cls).sum())
                rows.append(
                    {
                        "generator": gen,
                        "step": nel,
                        "novelty_class_vs_mp": cls,
                        "n_structures": n,
                        "n_step": len(s),
                        "share_of_step": n / len(s) if len(s) else np.nan,
                        "fraction_of_generated": len(s) / len(g),
                    }
                )
    return pd.DataFrame(rows)


def ed_stats(
    d: pd.DataFrame, mp: np.ndarray, generators=cm.GENERATORS
) -> pd.DataFrame:
    rows = [{"generator": "mp_experimental", "group": "all", "values": mp}] + [
        {
            "generator": g,
            "group": c,
            "values": d.loc[
                (d.generator == g) & (d.novelty_class_vs_mp == c), "ed_meV"
            ].to_numpy(),
        }
        for g in generators
        for c in cm.CLASSES
    ]
    out = []
    for r in rows:
        v = r.pop("values")
        out.append(
            {
                **r,
                "n_structures": len(v),
                "median_ed_meV": np.median(v),
                "fraction_ed_le_0": (v <= 1e-6).mean(),
                "fraction_ed_le_100": (v <= cm.ED_THRESHOLD + 1e-6).mean(),
            }
        )
    return pd.DataFrame(out)


def plot(
    steps: pd.DataFrame, nel, d, mp, out: Path, nelements: bool = True
) -> list[Path]:
    """With ``nelements=False`` the middle row (class mix per number of
    elements) is left out and the E_d row becomes (d-f)."""
    key = ["generator", "step", "novelty_class_vs_mp"]
    st = steps.set_index(key)
    sn = nel.set_index(key)
    fig, grid = plt.subplots(
        3 if nelements else 2,
        len(cm.GENERATORS),
        figsize=ps.figsize("double", height=5.8 if nelements else 4.5),
        height_ratios=[1.05, 0.6, 1] if nelements else [1.05, 1],
    )
    ed_row = 2 if nelements else 1
    for j, (gen, label) in enumerate(cm.GENERATORS.items()):
        ax = grid[0, j]
        stacked_panel(ax, st, gen, cm.STEPS, cm.STEPS, j == 0)
        ax.set_title(label)
        if nelements:
            stacked_panel(
                grid[1, j], sn, gen, cm.NEL_LABELS, cm.NEL_LABELS, j == 0
            )
        ax = grid[ed_row, j]
        cm.ed_cdf(ax, mp, color=cm.MP_COLOR, **LINE_KW)
        g = d[d.generator == gen]
        for c in cm.CLASSES:
            cm.ed_cdf(
                ax,
                g.loc[g.novelty_class_vs_mp == c, "ed_meV"],
                color=cm.CLASS_COLORS[c],
                **LINE_KW,
            )
        cm.ed_axes(ax, ylabel=j == 0)
    grid[0, 1].set_xlabel("Compositional novelty class share (%)")
    if nelements:
        grid[1, 1].set_xlabel(
            "Compositional novelty class share of VUN structures (%)"
        )
    grid[ed_row, 0].set_ylabel("Cumulative fraction (%)")
    grid[ed_row, 1].set_xlabel("Decomposition energy (meV/atom)")
    handles = [
        Patch(**cm.style(cm.CLASS_COLORS[c], True), label=cm.CLASS_LABELS[c])
        for c in cm.CLASSES
    ] + [Line2D([], [], color=cm.MP_COLOR, label=cm.MP_LABEL, **LINE_KW)]
    legend = fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles),
        handlelength=1.6,
        handleheight=1.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    ps.label_panels(grid.ravel(), x=0.0, y=1.02)
    fig.canvas.draw()
    if legend.get_window_extent().width > fig.bbox.width:
        raise ValueError("Legend is wider than the figure")
    return ps.savefig(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--evaluation", type=Path, default=cm.EVALUATION)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--validity", choices=["smact", "lemat"], default="smact")
    ap.add_argument(
        "--no-nelements",
        action="store_true",
        help="leave out the middle row (class mix per number of elements)",
    )
    args = ap.parse_args()
    if args.validity != "smact":
        args.out = args.out.with_name("figure_s2")

    steps = cm.summarize_steps(
        cm.read_table(args.evaluation), args.validity
    )
    d = cm.load_vun_ed(args.evaluation, args.validity)
    nel = summarize_nelements(d)
    mp = cm.load_mp_ed()
    ps.apply_style()
    paths = plot(steps, nel, d, mp, args.out, not args.no_nelements)
    eds = ed_stats(d, mp)
    stats = pd.concat(
        [
            steps.assign(table="funnel"),
            nel.assign(table="nelements"),
            eds.assign(table="ed"),
        ],
        ignore_index=True,
    )
    stats_path = args.out.with_name(args.out.name + "_stats.csv")
    stats.to_csv(stats_path, index=False)
    print("wrote", *[str(p) for p in paths], str(stats_path), sep="\n  ")
    print(eds.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
