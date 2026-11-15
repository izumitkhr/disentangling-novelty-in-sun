"""Shared matplotlib style for the two-column paper figures.

Conventions
-----------
* Font: Arial, 10 pt for all text (readable at print size in a
  two-column layout).
* Colors: ``coolwarm`` colormap for the number of elements and the
  default color cycle; the novelty classes use Okabe-Ito colors
  (analysis_3/common.py).
* Ticks: direction "in" on all four sides.
* Figure widths follow typical two-column journals (89 mm single,
  183 mm double).
* Text in PDF/SVG output stays editable (fonttype 42).

Usage from an analysis script (``scripts/analysis/analysis_N/*.py``)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import plot_style as ps

    ps.apply_style()
    fig, ax = plt.subplots(figsize=ps.figsize("single"))
    ...
    ps.label_panels([ax])
    ps.savefig(fig, "figures/analysis_N/name")  # writes name.pdf and .png
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
from cycler import cycler

# ---- paper geometry ---------------------------------------------------------
MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 89.0  # one column of a two-column page
DOUBLE_COLUMN_MM = 183.0  # full page width
COLUMN_WIDTH_IN = {
    "single": SINGLE_COLUMN_MM / MM_PER_INCH,  # 3.50 in
    "double": DOUBLE_COLUMN_MM / MM_PER_INCH,  # 7.20 in
}
GOLDEN_RATIO = (1 + 5**0.5) / 2

# ---- colors -----------------------------------------------------------------
CMAP = mpl.colormaps["coolwarm"]
CMAP_NAME = "coolwarm"

# ---- typography -------------------------------------------------------------
FONT_FAMILY = "Arial"
FONT_SIZE = 10
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10


def diverging_colors(n: int) -> list[tuple]:
    """Colors avoiding the pale center of coolwarm (good for ≤ 4 classes)."""
    if n <= 2:
        return [CMAP(0.0), CMAP(1.0)][:n]
    half = n // 2
    blues = [CMAP(v) for v in np.linspace(0.0, 0.35, half)]
    reds = [CMAP(v) for v in np.linspace(0.65, 1.0, n - half)]
    return blues + reds


def figsize(
    width: str | float = "single",
    aspect: float = 1 / GOLDEN_RATIO,
    *,
    height: float | None = None,
) -> tuple[float, float]:
    """Figure size in inches.

    Parameters
    ----------
    width : "single", "double", or a width in inches.
    aspect : height / width (default: inverse golden ratio).
    height : explicit height in inches (overrides ``aspect``).
    """
    w = (
        COLUMN_WIDTH_IN.get(width, width)
        if isinstance(width, str)
        else float(width)
    )
    h = height if height is not None else w * aspect
    return (w, h)


def rc_params(n_cycle_colors: int = 6) -> dict:
    """The rcParams dictionary applied by :func:`apply_style`.

    The categorical color cycle skips the pale center of coolwarm (which is
    nearly invisible on white) and alternates blue and red shades so that
    consecutive series contrast with each other.
    """
    half = diverging_colors(n_cycle_colors)
    blues, reds = half[: n_cycle_colors // 2], half[n_cycle_colors // 2 :]
    cycle = [c for pair in zip(blues, reds) for c in pair] + (
        reds[len(blues) :] if len(reds) > len(blues) else []
    )
    return {
        # font
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        # colors
        "axes.prop_cycle": cycler(color=cycle),
        "image.cmap": CMAP_NAME,
        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        # axes / lines
        "axes.linewidth": 0.6,
        "axes.labelpad": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.6,
        "errorbar.capsize": 2.0,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.3,
        # layout / output
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 500,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def apply_style(n_cycle_colors: int = 6) -> None:
    """Apply the paper style globally (call once at the top of a script)."""
    mpl.rcParams.update(rc_params(n_cycle_colors))


def savefig(
    fig: mpl.figure.Figure,
    stem: str | Path,
    formats: Sequence[str] = ("pdf", "png"),
    **kwargs,
) -> list[Path]:
    """Save ``fig`` as ``<stem>.<fmt>`` per format, creating the directory.
    PDFs carry no creation date, so an unchanged figure gives the same
    file."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        kw = {"metadata": {"CreationDate": None}} if fmt == "pdf" else {}
        fig.savefig(p, format=fmt, **{**kw, **kwargs})
        paths.append(p)
    return paths


def label_panels(
    axes: Iterable[mpl.axes.Axes],
    labels: Sequence[str] | None = None,
    *,
    x: float = -0.18,
    y: float = 1.02,
    weight: str = "bold",
) -> None:
    """Put (a), (b), ... in the top-left corner outside each panel."""
    axes = list(axes)
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,
            fontsize=FONT_SIZE,
            fontweight=weight,
            va="bottom",
            ha="left",
        )
