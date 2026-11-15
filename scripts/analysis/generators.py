"""Generators of the analyses, keyed by the file stem in data/input/generated
(gen_crystals_<key>.csv and energies/<model>/gen_crystals_<key>_energies.csv;
see paths.py).

GENERATORS are the models compared in the figures, all trained on MP-20.
MatterGen trained on Alex-MP-20 is evaluated alongside (E_hull, E_d,
validity, uniqueness, novelty) but only shown in Figure 6 (d)-(f), which
compares the two MatterGen training sets.
"""

GENERATORS = {
    "mattergen_mp_20_base": "MatterGen",
    "diffcsp-pp": "DiffCSP++",
    "wyformer": "WyFormer",
}
EXTRA_GENERATORS = {"mattergen_base": "MatterGen (Alex-MP-20)"}
# every generator whose structures are evaluated, in table order
EVALUATED = {**GENERATORS, **EXTRA_GENERATORS}
