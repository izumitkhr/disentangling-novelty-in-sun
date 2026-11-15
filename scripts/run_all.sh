#!/usr/bin/env bash
# Run every analysis in order and draw the figures of the paper.
#
# Needs the environment of pyproject.toml (`uv sync`), the git submodule
# external/lemat-genbench, and the inputs under data/input/ (README.md,
# "Inputs"; scripts/analysis/paths.py).
#
# analysis_3/evaluate_generated.py reuses the per-batch results in
# data/analysis_3/batches/ when present; delete that directory to recompute
# them from scratch.
#
# Run:
#   mkdir -p data/logs
#   nohup scripts/run_all.sh > data/logs/run_all.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=scripts/analysis

step() { echo "=== $(date '+%F %T') $*"; }

step "analysis_1 E_d of MP experimental compounds (Fig. 2)"
$PY $A/analysis_1/make_figure2.py
# E_hull of every generated structure, read by the later steps
step "analysis_2 E_hull under reference thinning"
$PY $A/analysis_2/plot_pseudo_stable_thinning.py
step "analysis_2 novelty classes under thinning"
$PY $A/analysis_2/plot_pseudo_stable_by_novelty.py
step "analysis_2 pseudo-stability (Fig. 3, Fig. S1)"
$PY $A/analysis_2/make_figure3.py

step "analysis_3 validity, uniqueness, novelty, prototypes"
$PY $A/analysis_3/evaluate_generated.py
step "analysis_3 E_d of generated structures"
$PY $A/analysis_3/compute_ed.py
step "analysis_3 compositional novelty (Fig. 4, Fig. S2)"
$PY $A/analysis_3/make_figure4.py
$PY $A/analysis_3/make_figure4.py --validity lemat --no-nelements

step "analysis_4 prototype novelty (Fig. 5)"
$PY $A/analysis_4/make_figure5.py

step "analysis_5 stability per novelty cell"
$PY $A/analysis_5/novelty_cells.py

step "analysis_6 MatterGen training data (Fig. 6)"
$PY $A/analysis_6/make_figure6.py

step "done"
