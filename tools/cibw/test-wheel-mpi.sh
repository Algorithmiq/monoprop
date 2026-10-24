#!/usr/bin/env bash
# Installs a built monoprop wheel into a fresh venv with mpi4py and whatever MPI the extra
# requirements or the machine provide, then runs the MPI-marked tests under mpiexec and the
# backend tests serially.
#
# usage: test-wheel-mpi.sh <project dir> <wheel file or directory> <mpi4py: wheel|source> [requirement...]
# env:   PYTHON (default python3), MONOPROP_TEST_VENV (default: a new temporary directory)
set -euo pipefail

project="${1:?usage: test-wheel-mpi.sh <project> <wheels> <wheel|source> [requirement...]}"
wheels="${2:?}"
mpi4py_mode="${3:?}"
shift 3

venv="${MONOPROP_TEST_VENV:-$(mktemp -d)/venv}"
"${PYTHON:-python3}" -m venv --clear "$venv"
python="$venv/bin/python"
"$python" -m pip install --quiet --upgrade pip

# Install monoprop itself from the given wheels only: a PyPI release could outrank a dev version.
if [[ -f "$wheels" ]]; then
  "$python" -m pip install --quiet --no-deps "$wheels"
else
  "$python" -m pip install --quiet --no-deps --no-index --find-links "$wheels" monoprop
fi

mpi4py_args=()
if [[ "$mpi4py_mode" == source ]]; then
  mpi4py_args=(--no-binary mpi4py)
fi
# "monoprop" is already satisfied, so this only adds its dependencies; the suite imports qiskit.
"$python" -m pip install --quiet ${mpi4py_args[@]+"${mpi4py_args[@]}"} "monoprop[qiskit]" "mpi4py==4.1.*" "$@" \
  --group "$project/pyproject.toml:test"
"$python" -m pip list

mpiexec="$venv/bin/mpiexec"
if [[ ! -x "$mpiexec" ]]; then
  mpiexec="$(command -v mpiexec)"
fi
# cibuildwheel's Linux container runs as root; Open MPI refuses that without these.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
# Open MPI's shared-memory PMIx store cannot attach at its base address in some containers.
export PMIX_MCA_gds=hash

cd "$project"
"$python" -c "import monoprop; print('MPI_ABIS', monoprop.MPI_ABIS)"
"$mpiexec" -n 2 "$python" -m pytest tests --with-mpi -m mpi -p no:cacheprovider
"$python" -m pytest tests/test_mpi_backend.py -p no:cacheprovider
