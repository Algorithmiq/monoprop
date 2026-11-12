#!/usr/bin/env bash
# cibuildwheel test-command: the serial suite in cibuildwheel's test environment (no mpi4py), then
# the MPI suite once per shipped ABI family, each against that family's MPI from PyPI.
set -euo pipefail

project="${1:?usage: run-tests.sh <project> <wheel>}"
wheel="${2:?usage: run-tests.sh <project> <wheel>}"

cd "$project"
python -m pytest tests -p no:cacheprovider

for mpi in mpich openmpi; do
  echo "::group::MPI tests against PyPI $mpi"
  PYTHON="$(command -v python)" bash tools/cibw/test-wheel-mpi.sh "$project" "$wheel" wheel "$mpi"
  echo "::endgroup::"
done
