"""Signed decomposition energy E_d of the generated structures.

E_d is the signed distance of a structure's energy per atom from the hull
of the full MACE-OMAT-0 MP reference at its composition (negative: below the
hull). The generated structure is not part of the reference, so E_hull =
max(E_d, 0); the hull is the same lower convex envelope (linear program) as
in analysis_2/plot_pseudo_stable_thinning.py, and the clipped values are
checked against its E_hull cache. E_d is the input of Figures 4 and 5.

Inputs: data/input/mp_mace_omat_0_medium/, data/input/generated/energies/
        (read with analysis_2/plot_pseudo_stable_thinning.py)
        data/analysis_2/ehull_thinning.csv.gz (check)
Output: data/analysis_3/ed_full.csv.gz   signed E_d per structure
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "analysis_2"))
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402
import plot_pseudo_stable_thinning as a2  # noqa: E402

_REF = None


def _signed_ed(task):
    """Signed E_d (eV/atom) of the candidates of one element set."""
    elements, ids, x, e = task
    idx = _REF.entries_for(elements)
    cols = [_REF.col[el] for el in elements]
    frac = _REF.frac[np.ix_(idx, cols)]
    ref_e = _REF.e_per_atom[idx]
    return [
        (i, ei - a2.hull_energy(frac, ref_e, xi))
        for i, xi, ei in zip(ids, x, e)
    ]


def compute_ed(workers: int) -> pd.DataFrame:
    global _REF
    manifest, records = a2.load_reference(a2.REFERENCE_DIR)
    _REF = a2.Reference(records)
    gen = a2.load_generated(a2.ENERGIES_DIR, manifest)
    cands = a2.prepare_candidates(gen, _REF)
    tasks = []
    for elements, grp in cands.groupby("elements", sort=False):
        x = np.array(
            [[c[el] for el in elements] for c in grp["composition"]], float
        )
        x /= x.sum(axis=1, keepdims=True)
        tasks.append(
            (
                elements,
                grp.cid.to_numpy(),
                x,
                grp.energy_per_atom_eV.to_numpy(),
            )
        )
    with Pool(workers) as pool:
        rows = [r for rs in pool.imap_unordered(_signed_ed, tasks) for r in rs]
    ed = pd.DataFrame(rows, columns=["cid", "ed_eV"])
    out = cands[["cid", "generator", "index"]].merge(ed, on="cid")
    # consistency with analysis_2 (E_hull = max(E_d, 0), no thinning)
    eh = pd.read_csv(a2.CACHE)
    eh = eh[eh.removed_fraction == 0].drop_duplicates(["generator", "index"])
    chk = out.merge(eh, on=["generator", "index"], validate="1:1")
    diff = (np.maximum(chk.ed_eV, 0) - chk.ehull_eV).abs().max()
    if diff > 1e-6:
        raise ValueError(f"E_d disagrees with analysis_2 E_hull ({diff} eV)")
    print(
        f"E_d of {len(out)} structures; max |E_hull - max(E_d,0)| {diff:.1e}"
    )
    return out.drop(columns="cid")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=cm.ED)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    ed = compute_ed(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ed.to_csv(args.out, index=False, float_format="%.8g")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
