#!/usr/bin/env bash
# Run the Julia (MajoranaPropagation.jl) and monoprop 1D Hubbard benchmarks
# back to back, appending results to their respective JSONL files.
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

julia --project=@. -e 'using Pkg; Pkg.instantiate()'
julia --project=@. -e 'using Pkg; Pkg.precompile()'

echo "Running Julia benchmark (cases 1-15, JULIA_NUM_THREADS=${JULIA_NUM_THREADS})"
for case in $(seq 1 15); do
    julia --project=@. julia_hubbard1d_benchmark.jl --case "$case"
done

echo "Running monoprop benchmark (cases 0-14, monoprop_NUM_THREADS=${monoprop_NUM_THREADS})"
for case in $(seq 0 14); do
    uv run python monoprop_hubbard1d_benchmark.py --case "$case"
done
