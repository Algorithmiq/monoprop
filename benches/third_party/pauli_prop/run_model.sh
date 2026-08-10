#!/usr/bin/env bash
# Run the Python Pauli-propagation engines (monoprop, QuEra ppvm, Qiskit pauli-prop,
# cuPauliProp) back to back, each in its own process, writing results.json.
#
# Each engine runs to completion before the next starts: this keeps the peak-memory measurements
# (see _common.HighWaterMark / run_cupauliprop.py's GpuMemPeakSampler) uncontaminated by another
# engine's allocations, and stops the engines from contending for the same CPU/GPU at the same
# time, which would otherwise skew both the runtime and memory numbers.
#
# Order matters: run_monoprop.py (re)creates results.json from settings.json, and run_qiskit.py
# reads run_monoprop.py's per-step term counts back out of it to set its own term budget.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python run_monoprop.py
uv run python run_ppvm.py
uv run python run_qiskit.py
uv run python run_cupauliprop.py
