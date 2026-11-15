"""Compatibility corrections for MACE-OMAT-0 energies.

MACE-OMAT-0 is trained on OMat24 (VASP PBE/PBE+U, version 54
pseudopotentials), so the anion and GGA/GGA+U mixing corrections should be
the OMat24 refit of the MP2020 scheme (arXiv:2410.12771, Appendix C;
OMat24Compatibility.yaml from fairchem) rather than the MP2020 values.

The scheme itself (which species are corrected, oxidation-state and
peroxide detection, GGA/GGA+U mixing) is unchanged, so a stored MP2020
adjustment "MP2020 ... correction (X)" of value n * c_MP(X) becomes
n * c_OMat24(X). Reference entries keep MP's assignment of the species
(their stored adjustments), generated structures the one pymatgen made;
for the generated ones this equals running pymatgen's
MaterialsProject2020Compatibility with ``config_file`` = the OMat24 yaml.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

OMAT24_YAML = Path(__file__).resolve().parent / "OMat24Compatibility.yaml"
# next to the module that defines the class (pymatgen.entries up to 2025,
# pymatgen.analysis.compatibility from 2026)
MP2020_YAML = (
    Path(inspect.getfile(MaterialsProject2020Compatibility)).resolve().parent
    / "MP2020Compatibility.yaml"
)


def _values(path: Path) -> dict[str, float]:
    """Correction per species (eV per atom of the species)."""
    c = yaml.safe_load(path.read_text())["Corrections"]
    out = dict(c["CompositionCorrections"])
    mixing = c["GGAUMixingCorrections"]
    if mixing["O"] != mixing["F"]:
        raise ValueError(f"{path}: O and F mixing corrections differ")
    out.update(mixing["O"])
    return out


MP2020 = _values(MP2020_YAML)
OMAT24 = _values(OMAT24_YAML)


def species(name: str) -> str:
    """Species of a stored adjustment, e.g. 'MP2020 anion correction
    (oxide)' -> 'oxide'."""
    if not name.startswith("MP2020 ") or not name.endswith(")"):
        raise ValueError(f"unexpected energy adjustment {name!r}")
    return name.rsplit("(", 1)[1][:-1]


def omat24_correction(adjustments: list[dict]) -> float:
    """Total OMat24 correction (eV) for the stored MP2020 adjustments."""
    total = 0.0
    for a in adjustments:
        sp = species(a["name"])
        mp = MP2020[sp]
        if mp == 0:
            if a["value"] != 0 or OMAT24[sp] != 0:
                raise ValueError(f"cannot rescale {a}")
            continue
        n = a["value"] / mp
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"{a} is not a whole number of {sp}")
        total += round(n) * OMAT24[sp]
    return total
