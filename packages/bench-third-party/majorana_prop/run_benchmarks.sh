#!/usr/bin/env bash
# Run the Julia (MajoranaPropagation.jl) and monoprop 1D Hubbard benchmarks
# back to back, merging both into the shared results.json, then plot them.
set -euo pipefail


# Both backends thread (monoprop via its partition facade, MajoranaPropagation.jl via
# AcceleratedKernels on VectorMajoranaSum), and neither is fastest at the same count, so
# these are overridable per side rather than fixed.
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export monoprop_NUM_THREADS="${monoprop_NUM_THREADS:-8}" # noqa: SIM112
# Keep stray BLAS/OpenMP pools from claiming the whole node next to the threads above.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$monoprop_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$monoprop_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$monoprop_NUM_THREADS}"


echo "Running monoprop benchmark (monoprop_NUM_THREADS=${monoprop_NUM_THREADS})"
uv run python monoprop_hubbard1d_benchmark.py

julia --project=@. -e 'using Pkg; Pkg.instantiate()'
julia --project=@. -e 'using Pkg; Pkg.precompile()'

echo "Running Julia benchmark (JULIA_NUM_THREADS=${JULIA_NUM_THREADS})"
julia --project=@. julia_hubbard1d_benchmark.jl

uv run python plot_results.py
