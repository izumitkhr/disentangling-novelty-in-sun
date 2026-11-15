"""Figure 6: stability of the generated structures across novelty cells,
and its dependence on the training data of MatterGen.

Structures: valid (SMACT), unique and novel (BAWL) generated structures
with 2-5 elements and an E_hull, split into the novelty cells of
analysis_3/common.py: compositional novelty class (color) x prototype
known (filled marker, solid bar) / new (open marker, white diagonal lines).
The panels are drawn by analysis_5/novelty_cells.py (draw()).

Left, one row of bars / markers per generator (MatterGen, DiffCSP++,
WyFormer; all trained on MP-20):
  (a) cell shares of the valid, unique, novel structures; the number on
      the right is the number of structures
  (b) stable fraction (E_hull = 0) per cell, 95 % Wilson interval
  (c) metastable fraction (E_hull <= 0.1 eV/atom) per cell
Right, (d)-(f): the same for MatterGen trained on Alex-MP-20
(``mattergen_base``) and on MP-20 (``mattergen_mp_20_base``, the MatterGen
of the other figures).

Input:   data/analysis_3/evaluation.csv.gz
Outputs: figures/analysis_6/figure6.{pdf,png}
         figures/analysis_6/figure6_stats.csv  stable / metastable fractions
                                               with intervals per cell, for
                                               (b, c) and (e, f)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "analysis_3"))
sys.path.insert(0, str(HERE.parent / "analysis_5"))
import common as cm  # noqa: E402
import novelty_cells as nc  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = cm.ROOT / "figures" / "analysis_6" / "figure6"
MODELS = {
    "mattergen_base": "MatterGen (Alex-MP-20)",
    "mattergen_mp_20_base": "MatterGen (MP-20)",
}
# rotated generator labels of the right half, on two lines
MODELS_2LINE = {g: label.replace(" (", "\n(") for g, label in MODELS.items()}
COMBINED_SCALE = 0.62  # figure size relative to 2 x the single-half figure
NAME_OFFSET = 72  # pt, generator names left of (b), (e)
COMBINED_ASPECT = 1.45  # extra height (taller panels)


def plot(vun, rates_all, rates_mattergen, out: Path) -> list[Path]:
    """All generators on the left, (a)-(c), and the two MatterGen models on
    the right, (d)-(f); one shared legend on top."""
    # smaller than two full-size halves side by side, so that the text is
    # larger relative to the figure when it is scaled to the text width
    w, h = ps.figsize("double", height=5.85)
    fig = plt.figure(
        figsize=(2 * w * COMBINED_SCALE, h * COMBINED_SCALE * COMBINED_ASPECT)
    )
    left, right = fig.subfigures(1, 2, wspace=0.015)
    axes = nc.draw(
        left,
        vun,
        rates_all,
        cm.GENERATORS,
        panel_labels=["(a)", "(b)", "(c)"],
        name_offset=NAME_OFFSET,
    )
    axes += nc.draw(
        right,
        vun[vun.generator.isin(MODELS)],
        rates_mattergen,
        MODELS_2LINE,
        panel_labels=["(d)", "(e)", "(f)"],
        name_offset=NAME_OFFSET,
    )
    # titles of the rate panels on two lines, the panel label moved up to
    # the first title line
    for ax in axes:
        title = ax.get_title()
        if not title:
            continue
        ax.set_title(title.replace(" (", "\n(", 1))
        (label,) = [t for t in ax.texts if t.get_text().startswith("(")]
        label.remove()
        ax.annotate(
            label.get_text(),
            xy=(0, 1),
            xycoords=("axes fraction", ax.title),
            ha="left",
            va="top",
            fontsize=label.get_fontsize(),
            fontweight=label.get_fontweight(),
        )
        # x labels of neighbouring panels would run into each other
        ax.set_xlabel(ax.get_xlabel().replace(" of ", " of\n", 1))
    # room between the neighbouring panels for the "100" and "0" ticks
    fig.get_layout_engine().set(wspace=0.0, w_pad=0.01)
    nc.legend(fig)
    paths = ps.savefig(fig, out)
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--evaluation", type=Path, default=cm.EVALUATION)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ev = cm.read_table(args.evaluation)
    vun = nc.load(args.evaluation)
    rates_all = nc.rates(ev)
    rates_mattergen = nc.rates(ev, MODELS)
    ps.apply_style()
    paths = plot(vun, rates_all, rates_mattergen, args.out)
    stats = pd.concat(
        [
            rates_all.assign(panels="b, c"),
            rates_mattergen.assign(panels="e, f"),
        ],
        ignore_index=True,
    )
    stats_path = args.out.with_name(args.out.name + "_stats.csv")
    stats.to_csv(stats_path, index=False)
    print("wrote", *[str(p) for p in paths], str(stats_path), sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
