#!/bin/bash
# Rebuild figures/fig1_strong_weak_scaling.{pdf,png} and figures/captions.txt from data/SCALE-CELLS.tsv.
# Needs matplotlib and nothing else: no cluster, no monoprop build, no run directories. Output is
# byte-reproducible (the PDF creation stamp is suppressed), so a regenerated figure that differs
# from the shipped one means the DATA changed, not the clock.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}   # override for a venv, e.g. PY=/path/to/venv/bin/python
"$PY" make_figure.py "$@"
