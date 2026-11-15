"""Shared definitions of the figures of analysis_3 - analysis_6: the
evaluation table, novelty classes and cells and their colors, the bar
styles, the V-SUN style filtering steps and the E_d distributions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generators import EVALUATED, GENERATORS  # noqa: E402,F401
from tables import read_table  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[3]
EVALUATION = ROOT / "data" / "analysis_3" / "evaluation.csv.gz"
FIG_DIR = ROOT / "figures" / "analysis_3"

CLASSES = ["known_composition", "new_composition", "new_chemsys"]
CLASS_LABELS = {
    "known_composition": "Known comp.",
    "new_composition": "New comp.",
    "new_chemsys": "New chemsys",
}
# same Okabe-Ito colors as analysis_2
CLASS_COLORS = {
    "known_composition": "#009E73",
    "new_composition": "#0072B2",
    "new_chemsys": "#D55E00",
}
# text drawn on top of a class color
CLASS_TEXT = {c: "white" for c in CLASS_COLORS}
HATCH = "////"  # diagonal lines mark invalid structures

# V-SUN style filtering, applied cumulatively
STEPS = {
    "generated": "Generated",
    "valid": "Valid",
    "unique": "+ Unique",
    "novel": "+ Novel",
    "metastable": "+ Metastable",
    "stable": "+ Stable",
}


# Novelty cells of a valid, unique structure: structure known to MP (BAWL),
# or composition / chemsys class x prototype known / new. A new
# prototype implies a new structure, so "not novel" has no prototype split.
NOT_NOVEL = "not_novel"
NOT_NOVEL_COLOR = "0.6"
CELLS = [NOT_NOVEL] + [(c, p) for c in CLASSES for p in ("known", "new")]


def cell_label(cell) -> str:
    if cell == NOT_NOVEL:
        return "Not novel"
    cls, proto = cell
    return f"{CLASS_LABELS[cls]}, {proto} prototype"


def cell_style(cell) -> dict:
    """Bar style of a novelty cell: grey for not novel, class color, white
    diagonal lines for a new prototype."""
    if cell == NOT_NOVEL:
        return {"facecolor": NOT_NOVEL_COLOR, "edgecolor": "white", "lw": 0.6}
    cls, proto = cell
    return style(CLASS_COLORS[cls], proto == "known")


def assign_cells(df: pd.DataFrame) -> pd.Series:
    """Novelty cell per structure (None if BAWL or prototype is missing)."""
    out = []
    for nb, cls, proto in zip(
        df.novel_bawl, df.novelty_class_vs_mp, df.novel_prototype
    ):
        if nb == 0:
            out.append(NOT_NOVEL)
        elif nb == 1 and proto in (0, 1) and isinstance(cls, str):
            out.append((cls, "new" if proto == 1 else "known"))
        else:
            out.append(None)
    return pd.Series(out, index=df.index, dtype=object)


def shares(d: pd.DataFrame) -> dict:
    """Share of each novelty cell (column ``cell``) among the rows of d."""
    counts = d.cell.value_counts()
    n = len(d)
    return {c: counts.get(c, 0) / n if n else np.nan for c in CELLS}


CELL_MIN_LABEL = 0.10  # write the share inside cell segments this wide


def stacked(ax, rows: list[dict], labels: list[str], right: list[str]):
    """Horizontal 100 % stacked bars of novelty cells, top to bottom."""
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for cell in CELLS:
        w = np.array([r[cell] for r in rows])
        st = cell_style(cell)
        ax.barh(y, w * 100, left=left * 100, height=0.7, **st)
        # solid patch behind the number on hatched (new-prototype) segments
        box = (
            {"fc": st["facecolor"], "ec": "none", "pad": 0.6}
            if "hatch" in st
            else {"fc": "none", "ec": "none", "pad": 0}
        )
        for yi, li, wi in zip(y, left, w):
            if wi >= CELL_MIN_LABEL:
                ax.text(
                    (li + wi / 2) * 100,
                    yi,
                    f"{wi * 100:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    bbox=box,
                )
        left += w
    for yi, txt in zip(y, right):
        ax.text(102, yi, txt, ha="left", va="center")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])


def style(color, valid):
    """Bar style: solid for valid, the same color with white diagonal
    lines for invalid."""
    # thin white outline separates adjacent parts (stacked and paired bars)
    base = {"facecolor": color, "edgecolor": "white", "lw": 0.6}
    if valid:
        return base
    return {**base, "hatch": HATCH, "hatchcolor": "white"}


def masks(df: pd.DataFrame, validity: str = "smact") -> dict[str, pd.Series]:
    """Cumulative selection of each filtering step (validity: "smact" or
    "lemat" (lemat-genbench overall validity), BAWL uniqueness and novelty,
    E_hull thresholds)."""
    m = pd.Series(True, index=df.index)
    out = {"generated": m}
    for step, cond in (
        ("valid", df[f"{validity}_valid"].astype(bool)),
        ("unique", df.unique.astype(bool)),
        ("novel", df.novel_bawl.eq(1)),
        ("metastable", df.metastable.eq(1)),
        ("stable", df.stable.eq(1)),
    ):
        m = m & cond
        out[step] = m
    return out


def summarize_steps(
    df: pd.DataFrame, validity: str = "smact", generators=GENERATORS
) -> pd.DataFrame:
    """Class mix of the structures left after each filtering step."""
    rows = []
    for gen in generators:
        g = df[df.generator == gen]
        for step, m in masks(g, validity).items():
            s = g[m]
            for cls in CLASSES:
                n = int((s.novelty_class_vs_mp == cls).sum())
                rows.append(
                    {
                        "generator": gen,
                        "step": step,
                        "novelty_class_vs_mp": cls,
                        "n_structures": n,
                        "n_step": len(s),
                        "n_generated": len(g),
                        "share_of_step": n / len(s) if len(s) else np.nan,
                        "fraction_of_generated": len(s) / len(g),
                    }
                )
    return pd.DataFrame(rows)


# ---- decomposition energy distributions (Figures 4 and 5) ------------------
ED = ROOT / "data" / "analysis_3" / "ed_full.csv.gz"  # compute_ed.py
ED_MP = ROOT / "data" / "analysis_1" / "ed_experimental.csv.gz"  # Fig. 2
ED_RANGE = (-300.0, 400.0)  # meV/atom, as in Figure 2
ED_THRESHOLD = 100.0  # meV/atom, metastability
ED_NELEMENTS = [2, 3, 4, 5]  # as the MP experimental baseline of Figure 2
MP_LABEL = "MP experimental"
MP_COLOR = "0.72"  # light grey baseline under the generator curves
NEL_LABELS = {2: "Binary", 3: "Ternary", 4: "Quaternary", 5: "Quinary"}


def load_vun_ed(
    evaluation: Path = EVALUATION, validity: str = "smact"
) -> pd.DataFrame:
    """Valid (SMACT or lemat-genbench), unique, novel structures with 2-5
    elements and their signed E_d (meV/atom) against the full MACE-OMAT-0 MP
    reference."""
    ev = read_table(evaluation)
    ed = pd.read_csv(ED)
    d = ev.merge(ed, on=["generator", "index"], how="left", validate="1:1")
    d = d[d.ed_eV.notna() & d.nelements.isin(ED_NELEMENTS)].copy()
    d["ed_meV"] = d.ed_eV * 1000
    return d[masks(d, validity)["novel"]]


def load_mp_ed() -> np.ndarray:
    """E_d (meV/atom) of the MP experimental compounds of Figure 2."""
    return pd.read_csv(ED_MP).ed_eV.to_numpy() * 1000


def ed_cdf(ax, values, **kw):
    """Cumulative distribution (%) of ``values`` on the E_d range."""
    lo, hi = ED_RANGE
    grid = np.linspace(lo, hi, 2001)
    v = np.sort(np.asarray(values))
    y = np.searchsorted(v, grid, side="right") / len(v) * 100
    return ax.plot(grid, y, **kw)[0]


def ed_axes(ax, ylabel: bool):
    lo, hi = ED_RANGE
    ax.axvline(0, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.axvline(ED_THRESHOLD, color="0.4", lw=1.0, ls=":", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_xticks([-200, 0, 200, 400])
    ax.set_xticks([-100, 100, 300], minor=True)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    if not ylabel:
        ax.set_yticklabels([])
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
