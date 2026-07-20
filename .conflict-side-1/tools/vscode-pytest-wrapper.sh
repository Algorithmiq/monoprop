#!/usr/bin/env bash
set -euo pipefail

# VS Code Python Testing wrapper:
# - Keeps discovery stable by running collect-only directly (no mpiexec).
# - Runs tests through mpiexec so both [comm_self] and [comm_world]
#   variants exercise the correct code paths.
#
# Environment variables:
#   monoprop_VSCODE_MPI_MODE   all (default) | auto | off
#   monoprop_MPI_TEST_PROCS    number of MPI ranks (default 2)
#   monoprop_PYTHON_BIN                 python interpreter
#   monoprop_MPI_ALLOW_RUN_AS_ROOT  set to 1 to add --allow-run-as-root (default 1)

PYTHON_BIN="${monoprop_PYTHON_BIN:-/home/vscode/.venv/bin/python}"
ARGS=("$@")

is_collect_only=0
is_mpi_selection=0
has_with_mpi=0
mpi_mode="${monoprop_VSCODE_MPI_MODE:-all}"

for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--collect-only" ]]; then
    is_collect_only=1
  fi
  if [[ "$arg" == "--with-mpi" ]]; then
    has_with_mpi=1
  fi
  if [[ "$arg" == *"test_mpi.py"* ]]; then
    is_mpi_selection=1
  fi
done

# Detect marker-based selections such as: -m mpi
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "-m" ]] && (( i + 1 < ${#ARGS[@]} )); then
    marker_expr="${ARGS[$((i + 1))]}"
    if [[ "$marker_expr" == *"mpi"* ]]; then
      is_mpi_selection=1
    fi
  fi
done

if (( is_collect_only == 1 )); then
  exec "$PYTHON_BIN" -m pytest "${ARGS[@]}"
fi

if [[ "$mpi_mode" == "off" ]]; then
  exec "$PYTHON_BIN" -m pytest "${ARGS[@]}"
fi

should_run_mpi=0
if [[ "$mpi_mode" == "all" ]]; then
  should_run_mpi=1
elif [[ "$mpi_mode" == "auto" ]] && (( is_mpi_selection == 1 )); then
  should_run_mpi=1
fi

if (( should_run_mpi == 1 )); then
  mpi_procs="${monoprop_MPI_TEST_PROCS:-2}"
  if [[ "$mpi_procs" == *";"* ]]; then
    mpi_procs="${mpi_procs%%;*}"
  fi

  MPIEXEC=(mpiexec)
  if [[ "${monoprop_MPI_ALLOW_RUN_AS_ROOT:-1}" == "1" ]]; then
    MPIEXEC+=(--allow-run-as-root)
  fi
  MPIEXEC+=(-n "$mpi_procs")

  if (( has_with_mpi == 1 )); then
    exec "${MPIEXEC[@]}" "$PYTHON_BIN" -m pytest "${ARGS[@]}"
  fi

  exec "${MPIEXEC[@]}" "$PYTHON_BIN" -m pytest --with-mpi "${ARGS[@]}"
fi

exec "$PYTHON_BIN" -m pytest "${ARGS[@]}"
