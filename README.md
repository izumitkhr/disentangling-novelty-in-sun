Code for reproducing the analyses in *Disentangling compositional and prototype novelty in thermodynamic stability benchmarks for crystal generative models* (Izumi Takahara, Teruyasu Mizoguchi and Peichen Zhong).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone --recurse-submodules https://github.com/izumitkhr/disentangling-novelty-in-sun.git
cd disentangling-novelty-in-sun
uv sync --locked    # analysis environment in .venv
```

[LeMat-GenBench](https://github.com/LeMaterial/lemat-genbench), used for the BAWL fingerprint and the validity checks, is included as a git submodule at commit `fbf1ba4` (`external/lemat-genbench`). The preparation of the inputs (`prepare/`) and the analyses use separate environments.

## Inputs

The inputs are not distributed with this repository. The analyses read them from `data/input/` (set in `scripts/analysis/paths.py`):

| Path | Contents | Produced by |
| --- | --- | --- |
| `data/input/mp_gga/` | Materials Project summary and GGA/GGA+U thermo records of every non-deprecated material | `prepare/download_mp.py` |
| `data/input/mp_mace_omat_0_medium/` | MACE-OMAT-0 energies of the Materials Project entries | `prepare/compute_mp_energies.py` |
| `data/input/generated/gen_crystals_<generator>.csv` | Generated structures, one CIF string per row in a `cif` column | [gen-crystals](https://github.com/izumitkhr/gen-crystals) |
| `data/input/generated/energies/mace-omat-0-medium/` | MACE-OMAT-0 energies of the generated structures | [gen-crystals](https://github.com/izumitkhr/gen-crystals), or `prepare/compute_generated_energies.py` |

### Preparing generated structures

The generated structures of the paper and their MACE-OMAT-0 energies are available in [gen-crystals](https://github.com/izumitkhr/gen-crystals), whose `data/` directory has the layout of `data/input/generated/`:

```sh
git clone https://github.com/izumitkhr/gen-crystals.git ../gen-crystals
mkdir -p data/input
ln -s "$(realpath ../gen-crystals/data)" data/input/generated
```

### Preparing Materials Project entries and MLIP energies

```sh
cd prepare
uv sync --locked --group mlip        # adds torch and mace-torch
export MP_API_KEY=...                # https://next-gen.materialsproject.org/api
uv run --locked python download_mp.py            # data/input/mp_gga (resume with --resume)

mkdir -p ../data/models
curl -L -o ../data/models/mace-omat-0-medium.model \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model

uv run --locked --group mlip python compute_mp_energies.py   # data/input/mp_mace_omat_0_medium
# only to recompute the energies of the generated structures (included in gen-crystals)
uv run --locked --group mlip python compute_generated_energies.py \
  --model ../data/models/mace-omat-0-medium.model             # data/input/generated/energies/
```

- The paper uses Materials Project database version 2026.04.13. `download_mp.py` downloads the current version, which is recorded in its `manifest.json`; a later version gives slightly different numbers.
- All MACE energies are single points without relaxation: the Materials Project entries at their relaxed structures, the generated structures as generated. Structures with two atoms closer than 0.7 Å are skipped. The paper's energies were computed on a CUDA GPU in float64.
- Energies are stored with MP2020 corrections (the adjustments stored by the Materials Project for its entries, and pymatgen's `MaterialsProject2020Compatibility` for the generated structures). The analyses replace them by the OMat24 refit of the same scheme on loading (`scripts/analysis/corrections/`).

## Running

```sh
scripts/run_all.sh
```

This runs every analysis in order (about 25 minutes on 32 CPU cores). The intermediate data (E_hull, the evaluation of the generated structures, E_d) are written to `data/`, and the figures to `figures/` as PDF and PNG, each with its numbers in a `*_stats.csv`:

| Figure | File | Script (in `scripts/analysis/`) |
| --- | --- | --- |
| Fig. 2 | `figures/analysis_1/figure2` | `analysis_1/make_figure2.py` |
| Fig. 3, S1 | `figures/analysis_2/figure3`, `figure_s1` | `analysis_2/make_figure3.py` |
| Fig. 4, S2 | `figures/analysis_3/figure4`, `figure_s2` | `analysis_3/make_figure4.py` |
| Fig. 5 | `figures/analysis_4/figure5` | `analysis_4/make_figure5.py` |
| Fig. 6 | `figures/analysis_6/figure6` | `analysis_6/make_figure6.py` |

Each script describes its inputs and outputs in its docstring.

## License

The code is released under the MIT license (`LICENSE`).

- `scripts/analysis/corrections/OMat24Compatibility.yaml` is taken from [fairchem](https://github.com/facebookresearch/fairchem) (MIT license); its header gives the source commit and the license notice.
- LeMat-GenBench (`external/lemat-genbench`) is distributed under the Apache License 2.0.
- The MACE-OMAT-0 model is distributed under the Academic Software Licence and is not included.
- Materials Project data are licensed under CC BY 4.0.
