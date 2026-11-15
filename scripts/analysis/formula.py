"""Key for comparing compositions."""

from pymatgen.core import Composition


def formula_key(composition) -> str:
    """Reduced formula with the elements in alphabetical order, e.g. Zn2O2
    -> "O1 Zn1" (any input accepted by pymatgen's Composition).

    Composition.reduced_formula orders the elements by electronegativity,
    which pymatgen does not define for He, Ne and Ar, so the same
    composition can give different reduced_formula strings. This key does
    not depend on that order.
    """
    return Composition(composition).reduced_composition.alphabetical_formula
