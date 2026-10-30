#!/usr/bin/env bash
# cibuildwheel repair-wheel-command: vendor non-system libraries as usual, but never MPI. Every MPI
# binary keeps a bare dependency on its ABI family's soname, resolved at runtime against the MPI
# library mpi4py loads. check_wheel.py then audits the result.
set -euo pipefail

dest_dir="${1:?usage: repair-wheel.sh <dest_dir> <wheel>}"
wheel="${2:?usage: repair-wheel.sh <dest_dir> <wheel>}"

case "$(uname)" in
  Linux)
    auditwheel repair \
      --exclude libmpi.so.12 --exclude libmpi.so.40 --exclude libmpi_abi.so.1 \
      -w "$dest_dir" "$wheel"
    ;;
  Darwin)
    # delocate resolves @rpath names only through each binary's own LC_RPATHs and fails on the MPI
    # libraries (they live in per-ABI prefixes, not on those rpaths) before --exclude applies.
    # check_wheel.py verifies each MPI binary's MPI dependency afterwards.
    # --no-sanitize-rpaths: delocate would otherwise strip the absolute MPI search directories
    # (Homebrew, MacPorts, /usr/local) that the MPI binaries need; check_wheel.py requires them.
    delocate-wheel -v --require-archs "$(uname -m)" \
      --ignore-missing-dependencies --no-sanitize-rpaths \
      --exclude libmpi.12.dylib --exclude libpmpi.12.dylib \
      --exclude libmpi.40.dylib --exclude libmpi_abi.1.dylib \
      -w "$dest_dir" "$wheel"
    ;;
  *)
    echo "repair-wheel.sh: unsupported platform $(uname)" >&2
    exit 1
    ;;
esac

# The ABI list comes from the same variable that selected what to build.
abis="$(printf '%s' "${monoprop_MPI_ABIS:?monoprop_MPI_ABIS is not set}" | tr ';' '\n' | cut -d= -f1 | paste -sd, -)"
# `python` is cibuildwheel's build interpreter; manylinux's system python3 is too old for the script.
python "$(dirname "$0")/check_wheel.py" --expect-abis "$abis" "$dest_dir"/*.whl
