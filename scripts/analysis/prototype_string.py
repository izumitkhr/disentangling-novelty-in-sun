"""Prototype string of a crystal structure.

The string identifies the structure type independently of the chemical
species:

    <anonymized formula>|<Pearson symbol>|<space group symbol>|<space group number>|<site symmetries>

The structure is reduced to its conventional standard cell (spglib through
pymatgen, ``symprec``). Each element is described by its reduced
stoichiometric coefficient and the site symmetries of the Wyckoff positions
it occupies (``<site symmetry>(<count>)``, sorted). The elements are sorted
by (coefficient, site symmetries) and named A, B, C, ... in that order, so
that two structures with the same string differ only by a permutation of
the species. Two structures share a prototype if their strings are equal.

Example: rock-salt NaCl gives ``AB|cF8|Fm-3m|225|m-3m(1)_m-3m(1)``.
"""

from collections import Counter, defaultdict

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_symmetrized_structure(structure, symprec=0.10):
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    struc = sga.get_conventional_standard_structure()

    sga = SpacegroupAnalyzer(struc, symprec=symprec)
    symm = sga.get_symmetrized_structure()

    sg_number = sga.get_space_group_number()
    sg_symbol = sga.get_space_group_symbol()
    pearson_symbol = sga.get_pearson_symbol()
    return sga, symm, sg_number, sg_symbol, pearson_symbol


def get_site_symm_dict(symm, sga):
    dataset = sga.get_symmetry_dataset()
    wyckoffs = dataset.wyckoffs
    site_symmetry = dataset.site_symmetry_symbols
    wyckoff_to_sitesymm = {}
    for w, s in zip(wyckoffs, site_symmetry):
        if w not in wyckoff_to_sitesymm.keys():
            wyckoff_to_sitesymm[w] = s
        else:
            assert wyckoff_to_sitesymm[w] == s

    wyckoff_sites = symm.wyckoff_symbols
    wyckoff_dict = defaultdict(list)
    assert len(symm.equivalent_sites) == len(wyckoff_sites)
    for i, site in enumerate(symm.equivalent_sites):
        elem = site[0].species_string
        wyckoff_label = wyckoff_sites[i][-1]
        wyckoff_dict[elem].append(wyckoff_label)

    compressed_site_symm_dict = {}
    for elem, labels in wyckoff_dict.items():
        site_symm_labels = [wyckoff_to_sitesymm[label] for label in labels]
        counts = Counter(site_symm_labels)
        tmp = []
        for k, v in sorted(counts.items()):
            tmp.append(f"{k}({v})")
        compressed = "".join(tmp)
        compressed_site_symm_dict[elem] = compressed

    return compressed_site_symm_dict


def get_anonymized_formula(symm_struc, site_symm_dict):
    comp_dict = symm_struc.composition.reduced_composition.as_dict()

    elem_site_symm_set = []
    for k in site_symm_dict.keys():
        elem_site_symm_set.append((comp_dict[k], site_symm_dict[k]))
    sorted_elem_site_symm_list = sorted(elem_site_symm_set, key=lambda x: (x[0], x[1]))

    anonymized_formula = ""
    for i, (s, site_symm_str) in enumerate(sorted_elem_site_symm_list):
        letter = LETTERS[i]
        if s > 1:
            anonymized_formula += f"{letter}{int(s)}"
        else:
            anonymized_formula += f"{letter}"
    site_symm_str = "_".join(
        [site_symm_str for s, site_symm_str in sorted_elem_site_symm_list]
    )

    return anonymized_formula, site_symm_str


def get_string(structure, symprec=0.10):
    """Prototype string of ``structure`` (its oxidation states are removed
    in place)."""
    structure.remove_oxidation_states()
    sga, symm, sg_number, sg_symbol, pearson_symbol = get_symmetrized_structure(
        structure=structure, symprec=symprec
    )
    site_symm_dict = get_site_symm_dict(symm, sga)
    anonymized_formula, site_symm_str = get_anonymized_formula(symm, site_symm_dict)

    string = (
        f"{anonymized_formula}|{pearson_symbol}|{sg_symbol}|{sg_number}|{site_symm_str}"
    )

    return string
