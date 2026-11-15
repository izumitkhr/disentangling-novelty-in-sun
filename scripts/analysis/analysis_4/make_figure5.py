"""Figure 5: the generated structures by prototype novelty.

The prototype is the element-blind structure-type string
(analysis_3/evaluate_generated.py, ``novel_prototype``: not found among the
strings of all MP entries).

Top row (a-c), one panel per generator: the same V-SUN style filtering as
Figure 4 (Generated -> Valid -> + Unique -> + Novel -> + Metastable ->
+ Stable). Each bar is the share of known (grey) and new (pink) prototypes
among the structures left after the step, as 100 %; the number on the
right is the share of the generated structures left.

Middle row (d-f): prototype mix of the valid, unique, novel structures with
2-5 elements per number of elements (100 % per bar); the number on the
right is the share of these structures with that element count.

Bottom row (g-i): cumulative distribution of the decomposition energy E_d
of the valid, unique, novel structures with 2-5 elements, per prototype
novelty; light grey: the MP experimental compounds of Figure 2. Dashed
line E_d = 0 (stable), dotted line 100 meV/atom (metastable).

Structures whose prototype string could not be computed (a few) are left
out. With ``--split-p1`` both the known and the new prototypes
are split into those in space group P1 (number 1 in the prototype string,
spglib symprec 0.1), i.e. without any symmetry found, and the others; the
E_d curves of the P1 groups (dashed) are drawn only with at least MIN_CURVE
structures.

Inputs:  data/analysis_3/evaluation.csv.gz, data/analysis_3/ed_full.csv.gz,
         data/analysis_1/ed_experimental.csv.gz
Outputs: figures/analysis_4/figure5.{pdf,png}
         figures/analysis_4/figure5_p1.{pdf,png}  (--split-p1)
         figures/analysis_4/figure5_stats.csv     prototype shares per step
                                                  and per element count,
                                                  E_d quantiles
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
sys.path.insert(0, str(HERE.parent / "analysis_3"))
import common as cm  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = cm.ROOT / "figures" / "analysis_4" / "figure5"
PROTO = {"known": "Known prototype", "new": "New prototype"}
PROTO_COLORS = {"known": "0.25", "new": "#CC79A7"}
# --split-p1: prototypes in P1 (no symmetry) apart from the others
PROTO_P1 = {
    "known": "Known prototype (not P1)",
    "known_p1": "Known prototype (P1)",
    "new": "New prototype (not P1)",
    "new_p1": "New prototype (P1)",
}
PROTO_P1_COLORS = {**PROTO_COLORS, "known_p1": "0.68", "new_p1": "#E6B9D3"}
# the P1 groups are drawn dashed in the color of their non-P1 group
P1_BASE = {"known_p1": "known", "new_p1": "new"}
TEXT_COLORS = {
    "known": "white",
    "known_p1": "0.15",
    "new": "white",
    "new_p1": "0.15",
}
MIN_CURVE = 20  # smallest group drawn as an E_d curve (P1 only)
LINE_KW = {"lw": 1.8, "alpha": 0.85}
MIN_LABEL = 0.10  # write the share inside segments at least this wide
BAR_TEXT_SIZE = 9  # pt, slightly below the 10 pt text so it sits in the bars


def categorize(df: pd.DataFrame, split_p1: bool) -> pd.Series:
    """known / new (/ known_p1 / new_p1) per structure with a prototype
    string."""
    new = df.novel_prototype == 1
    out = np.where(new, "new", "known")
    if split_p1:
        p1 = df.prototype.str.split("|").str[3].astype(int) == 1
        out = np.where(p1, np.where(new, "new_p1", "known_p1"), out)
    return pd.Series(out, index=df.index)


def summarize_steps(ev: pd.DataFrame, proto: dict = PROTO) -> pd.DataFrame:
    """Prototype novelty shares of the structures left after each
    filtering step."""
    rows = []
    for gen in cm.GENERATORS:
        g = ev[ev.generator == gen]
        n_gen = len(g)
        g = g[g.novel_prototype.notna()].copy()
        g["proto"] = categorize(g, "new_p1" in proto)
        for step, m in cm.masks(g).items():
            s = g[m]
            for p in proto:
                n = int((s.proto == p).sum())
                rows.append(
                    {
                        "generator": gen,
                        "step": step,
                        "prototype": p,
                        "n_structures": n,
                        "n_step": len(s),
                        "share_of_step": n / len(s) if len(s) else np.nan,
                        "fraction_of_generated": len(s) / n_gen,
                    }
                )
    return pd.DataFrame(rows)


def summarize_nelements(d: pd.DataFrame, proto: dict = PROTO) -> pd.DataFrame:
    """Prototype novelty shares of the E_d structures per number of
    elements."""
    rows = []
    for gen in cm.GENERATORS:
        g = d[d.generator == gen]
        for nel in cm.NEL_LABELS:
            s = g[g.nelements == nel]
            for p in proto:
                n = int((s.proto == p).sum())
                rows.append(
                    {
                        "generator": gen,
                        "step": nel,
                        "prototype": p,
                        "n_structures": n,
                        "n_step": len(s),
                        "share_of_step": n / len(s) if len(s) else np.nan,
                        "fraction_of_generated": len(s) / len(g),
                    }
                )
    return pd.DataFrame(rows)


def ed_stats(d: pd.DataFrame, mp: np.ndarray, proto: dict = PROTO) -> pd.DataFrame:
    groups = [("mp_experimental", "all", mp)] + [
        (g, p, d.loc[(d.generator == g) & (d.proto == p), "ed_meV"].to_numpy())
        for g in cm.GENERATORS
        for p in proto
    ]
    return pd.DataFrame(
        [
            {
                "generator": g,
                "prototype": p,
                "n_structures": len(v),
                "median_ed_meV": np.median(v) if len(v) else np.nan,
                "fraction_ed_le_0": (v <= 1e-6).mean(),
                "fraction_ed_le_100": (v <= cm.ED_THRESHOLD + 1e-6).mean(),
            }
            for g, p, v in groups
        ]
    )


def funnel_panel(ax, st, gen, left_labels, rows=cm.STEPS, proto=PROTO):
    colors = PROTO_P1_COLORS if "new_p1" in proto else PROTO_COLORS
    steps = list(rows)
    y = np.arange(len(steps))[::-1]
    left = np.zeros(len(steps))
    for p in proto:
        w = np.array([st.loc[(gen, s, p), "share_of_step"] for s in steps])
        ax.barh(
            y,
            w * 100,
            left=left * 100,
            height=0.72,
            facecolor=colors[p],
            edgecolor="white",
            lw=0.6,
        )
        for yi, li, wi in zip(y, left, w):
            if wi >= MIN_LABEL:
                ax.text(
                    (li + wi / 2) * 100,
                    yi,
                    f"{wi * 100:.0f}",
                    ha="center",
                    va="center",
                    color=TEXT_COLORS[p],
                    fontsize=BAR_TEXT_SIZE,
                )
        left += w
    for yi, s in zip(y, steps):
        frac = st.loc[(gen, s, next(iter(proto))), "fraction_of_generated"]
        txt = f"{frac * 100:.0f}%" if frac >= 0.1 else f"{frac * 100:.1f}%"
        ax.text(102, yi, txt, ha="left", va="center")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticks([25, 75], minor=True)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_yticks(y, [rows[s] for s in steps] if left_labels else [])


def plot(steps, nel, d, mp, out: Path, proto: dict = PROTO) -> list[Path]:
    split = "new_p1" in proto
    colors = PROTO_P1_COLORS if split else PROTO_COLORS
    st = steps.set_index(["generator", "step", "prototype"])
    sn = nel.set_index(["generator", "step", "prototype"])
    fig, grid = plt.subplots(
        3,
        len(cm.GENERATORS),
        figsize=ps.figsize("double", height=5.8),
        height_ratios=[1.05, 0.6, 1],
    )
    for j, (gen, label) in enumerate(cm.GENERATORS.items()):
        funnel_panel(grid[0, j], st, gen, j == 0, proto=proto)
        grid[0, j].set_title(label)
        funnel_panel(grid[1, j], sn, gen, j == 0, cm.NEL_LABELS, proto)
        ax = grid[2, j]
        cm.ed_cdf(ax, mp, color=cm.MP_COLOR, **LINE_KW)
        g = d[d.generator == gen]
        for p in proto:
            v = g.loc[g.proto == p, "ed_meV"]
            if p in P1_BASE:
                if len(v) < MIN_CURVE:
                    continue
                # dashed in the full color so that it reads on white
                cm.ed_cdf(
                    ax, v, color=colors[P1_BASE[p]], ls="--", **LINE_KW
                )
            else:
                cm.ed_cdf(ax, v, color=colors[p], **LINE_KW)
        cm.ed_axes(ax, ylabel=j == 0)
    # x labels under the middle column
    grid[0, 1].set_xlabel("Prototype novelty share (%)")
    grid[1, 1].set_xlabel("Prototype novelty share of VUN structures (%)")
    grid[2, 1].set_xlabel("Decomposition energy (meV/atom)")
    grid[2, 0].set_ylabel("Cumulative fraction (%)")
    handles = [
        Patch(facecolor=colors[p], label=label) for p, label in proto.items()
    ]
    if split:
        handles.append(
            Line2D([], [], color="0.4", ls="--", label="P1 ($E_d$)", **LINE_KW)
        )
    handles.append(
        Line2D([], [], color=cm.MP_COLOR, label=cm.MP_LABEL, **LINE_KW)
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles) if len(handles) <= 4 else 3,
        # with --split-p1: two rows, known and new prototypes in columns
        handlelength=1.6,
        handleheight=1.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    ps.label_panels(grid.ravel(), x=0.0, y=1.02)
    return ps.savefig(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--evaluation", type=Path, default=cm.EVALUATION)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--split-p1",
        action="store_true",
        help="split the known and the new prototypes into P1 and the others",
    )
    args = ap.parse_args()
    if args.split_p1:
        args.out = args.out.with_name(args.out.name + "_p1")
    proto = PROTO_P1 if args.split_p1 else PROTO

    ev = cm.read_table(args.evaluation)
    steps = summarize_steps(ev, proto)
    d = cm.load_vun_ed(args.evaluation)
    d = d[d.novel_prototype.notna()].copy()
    d["proto"] = categorize(d, args.split_p1)
    nel = summarize_nelements(d, proto)
    mp = cm.load_mp_ed()
    ps.apply_style()
    paths = plot(steps, nel, d, mp, args.out, proto)
    eds = ed_stats(d, mp, proto)
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
