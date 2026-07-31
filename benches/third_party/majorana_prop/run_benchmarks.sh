#!/usr/bin/env bash
# Run the Julia (MajoranaPropagation.jl) and monoprop 1D Hubbard benchmarks
# back to back, appending results to their respective JSONL files.
set -euo pipefail


export JULIA_NUM_THREADS=24
export monoprop_NUM_THREADS=24


echo "Running monoprop benchmark (monoprop_NUM_THREADS=${monoprop_NUM_THREADS})"
uv run python monoprop_hubbard1d_benchmark.py

julia --project=@. -e 'using Pkg; Pkg.instantiate()'
julia --project=@. -e 'using Pkg; Pkg.precompile()'

echo "Running Julia benchmark (JULIA_NUM_THREADS=${JULIA_NUM_THREADS})"
julia --project=@. julia_hubbard1d_benchmark.jl



uv python plot_results.py
