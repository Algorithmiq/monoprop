#!/usr/bin/env bash
# Run the Julia (MajoranaPropagation.jl) and monoprop 1D Hubbard benchmarks
# back to back, appending results to their respective JSONL files.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export JULIA_NUM_THREADS=8
export monoprop_NUM_THREADS=8 # noqa: SIM112

echo "Running Julia benchmark (cases 1-33, JULIA_NUM_THREADS=${JULIA_NUM_THREADS})"
for case in $(seq 1 33); do
    julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/julia_hubbard1d_benchmark.jl" --case "$case"
done

echo "Running monoprop benchmark (cases 0-32, monoprop_NUM_THREADS=${monoprop_NUM_THREADS})"
for case in $(seq 0 32); do
    uv run python "$SCRIPT_DIR/monoprop_hubbard1d_benchmark.py" --case "$case"
done
