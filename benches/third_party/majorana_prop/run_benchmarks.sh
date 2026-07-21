#!/usr/bin/env bash
# Run the Julia (MajoranaPropagation.jl) and monoprop 1D Hubbard benchmarks
# back to back, appending results to their respective JSONL files.
set -euo pipefail


export JULIA_NUM_THREADS=8
export monoprop_NUM_THREADS=8 # noqa: SIM112

julia --project=@. -e 'using Pkg; Pkg.instantiate()'
julia --project=@. -e 'using Pkg; Pkg.precompile()'

echo "Running Julia benchmark (cases 1-27, JULIA_NUM_THREADS=${JULIA_NUM_THREADS})"
for case in $(seq 1 27); do
    julia --project=@. julia_hubbard1d_benchmark.jl --case "$case"
done

echo "Running monoprop benchmark (cases 0-26, monoprop_NUM_THREADS=${monoprop_NUM_THREADS})"
for case in $(seq 0 26); do
    uv run python monoprop_hubbard1d_benchmark.py --case "$case"
done
