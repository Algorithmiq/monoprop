#!/usr/bin/env bash
# cibuildwheel before-all: installs the MPI implementations the wheel's MPI variants compile against,
# from their PyPI wheels, one prefix each (<prefix>/mpich, <prefix>/openmpi). Only mpi.h and the
# libmpi soname matter; the build uses no compiler wrappers.
set -euo pipefail

prefix="${1:?usage: install-mpi.sh <prefix>}"

# Bumping these changes the headers every MPI variant is compiled against.
mpich_version=5.0.2
openmpi_version=5.0.11

# manylinux's system python3 is too old for manylinux_2_28 wheel tags
python=python3
if [[ -x /opt/python/cp312-cp312/bin/python3 ]]; then
  python=/opt/python/cp312-cp312/bin/python3
fi

for spec in "mpich==$mpich_version" "openmpi==$openmpi_version"; do
  name="${spec%%==*}"
  rm -rf "${prefix:?}/$name"
  # --prefix, not --target: pip drops a wheel's lib/ directory when installing with --target.
  "$python" -m pip install --quiet --no-deps --prefix "$prefix/$name" "$spec"
done

if [[ "$(uname)" == Darwin ]]; then
  # The extensions must record libmpi as @rpath/..., as mpi4py's do, to find the user's MPI.
  for dylib in "$prefix"/*/lib/libmpi*.dylib; do
    [[ -L "$dylib" ]] && continue
    id="$(otool -D "$dylib" | tail -n 1)"
    if [[ "$id" != @rpath/* ]]; then
      install_name_tool -id "@rpath/$(basename "$id")" "$dylib"
    fi
  done
fi

ls "$prefix"/*/include/mpi.h
