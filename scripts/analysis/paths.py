"""Locations of the inputs of the analyses (see README.md, "Inputs").

The inputs are not distributed with this repository. The scripts in
prepare/ write them to these locations; symbolic links to directories
elsewhere work as well.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "input"

# Materials Project GGA/GGA+U summary and thermo records
# (prepare/download_mp.py)
MP_DIR = INPUT_DIR / "mp_gga"
# MACE-OMAT-0 single points of the Materials Project entries
# (prepare/compute_mp_energies.py)
REFERENCE_DIR = INPUT_DIR / "mp_mace_omat_0_medium"
# generated structures, gen_crystals_<generator>.csv with a ``cif`` column
GEN_DIR = INPUT_DIR / "generated"
# MACE-OMAT-0 single points of the generated structures
# (prepare/compute_generated_energies.py)
ENERGIES_DIR = GEN_DIR / "energies" / "mace-omat-0-medium"

# git submodule
LEMAT_ROOT = ROOT / "external" / "lemat-genbench"
