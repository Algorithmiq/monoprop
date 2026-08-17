#!/bin/bash -l
# Build monoprop with MPI for the Deucalion x86 partition.
#
#   salloc -A <account> -p dev-x86 -N1 --exclusive -t 3:00:00
#   srun ./hpc/deucalion/build.sh
#
# Run it on a compute node, not the login node: the generated binding TUs are
# RAM-hungry and there are 128 exclusive cores and 240 GB going spare on dev-x86.
# Compute nodes have full outbound network here (verified: GitHub, PyPI and the
# sn02 mirror all reachable), so no cache pre-warming is needed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source hpc/deucalion/env.sh

: "${MONOPROP_MAX_NUM_MODES:=1024}"
: "${BUILD_JOBS:=32}"
# Tag the build directory per arm. pyproject.toml sets `build-dir = "build/{state}/{build_type}"`, and
# NEITHER placeholder encodes the cmake defines or the venv -- so every configuration of a worktree
# (MPI on/off, any MAX_NUM_MODES) shares ONE directory. Two jobs submitted together once raced on it:
# 3 of 4 arms died with `configure_file: No such file or directory`, and ctest then ran against the
# half-written tree and reported a meaningless 262/262 passed. The venv is not the isolation boundary
# it looks like -- UV_PROJECT_ENVIRONMENT separates where the wheel is INSTALLED; the compile happens
# in the shared tree either way.
: "${MONOPROP_BUILD_TAG:=$(basename "$MONOPROP_SRC")}"

# THE PINNED BASELINE IS NOT REBUILDABLE FROM HERE either -- see the same guard in
# sbatch/ctest-worktree.sh. $PROJ/src/mp-main is an ordinary group-writable worktree; the only
# thing making it a baseline is that hpc/deucalion/baseline.md5 pins the _core.so it currently
# holds, and a single `uv sync --reinstall-package` here would re-base every ratio in the
# campaign while __version__ -- which is written at install time and not rewritten by a rebuild
# -- kept reporting the same commit. Keyed on the BINARY, so a copy of the baseline under
# another path is protected too.
if [ -r hpc/deucalion/baseline.md5 ]; then
    _pin=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' hpc/deucalion/baseline.md5)
    mapfile -t _cur < <(ls "$VENV"/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null)
    if [ -n "$_pin" ] && [ "${#_cur[@]}" -eq 1 ] \
       && [ "$(md5sum "${_cur[0]}" | cut -d' ' -f1)" = "$_pin" ] \
       && [ "${ALLOW_BASELINE_REBUILD:-0}" != "1" ]; then
        echo "refusing: $MONOPROP_SRC holds the PINNED baseline _core.so ($_pin)." >&2
        echo "  Rebuilding it voids every ratio measured against it, and the version stamp" >&2
        echo "  will not change to tell you. Re-pin deliberately (ALLOW_BASELINE_REBUILD=1)" >&2
        echo "  and re-run the PRs already measured." >&2
        exit 2
    fi
fi

echo "=== toolchain ==="
g++ --version | head -1
mpirun --version | head -1
cmake --version | head -1
echo "target arch: $(g++ -march=native -Q --help=target 2>/dev/null | awk '/^ +-march= /{print $2}')"
echo "max_num_modes: $MONOPROP_MAX_NUM_MODES   jobs: $BUILD_JOBS   build_tag: $MONOPROP_BUILD_TAG"
echo

# --reinstall-package + --no-cache are mandatory, not belt-and-braces:
# [tool.uv] cache-keys in pyproject.toml does not include SKBUILD_* or
# config-settings, so uv will happily serve a cached MPI-OFF build otherwise.
#
# build.tool-args overrides pyproject.toml's ["-j2"], which caps the native
# build at two jobs and, being an explicit ninja argument, also beats
# CMAKE_BUILD_PARALLEL_LEVEL.
#
# MAX_NUM_MODES=1024 instantiates storage widths 32..1024 in 32-mode blocks,
# batched 8 per translation unit => 4 generated bind_up_to_*.cpp, each carrying
# 8 full engine instantiations at -O3 -march=native. Those four are the critical
# path of the whole build.
# `--group test` is mandatory alongside `--all-extras`, not redundant with it: pytest lives in a
# dependency GROUP and --all-extras covers extras only, so uv prunes pytest out of the venv and the
# next `python -m pytest` fails with "No module named pytest" -- which reads as a venv that was never
# set up, and costs a full --no-cache rebuild to recover.
#
# No longer `exec`: the md5 below has to run afterwards. (It used to be, which meant nothing placed
# after this line in a caller's script ever executed.)
uv sync --all-extras --group test --group bench -v \
    --reinstall-package monoprop --no-cache \
    --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" \
    --config-settings-package="monoprop:cmake.define.monoprop_MAX_NUM_MODES=${MONOPROP_MAX_NUM_MODES}" \
    --config-settings-package="monoprop:build-dir=build/{state}/{build_type}-${MONOPROP_BUILD_TAG}" \
    --config-settings-package="monoprop:build.tool-args=-j${BUILD_JOBS}"

# ARM IDENTITY IS THIS md5 AND NOTHING ELSE. `monoprop.__version__` reports HEAD for every editable
# arm (all venvs resolve the Python layer to one live source tree), and the dist-info stamp is written
# at install time and never rewritten by a later cmake rebuild -- both version identifiers go stale
# TOGETHER, and keying a provenance guard on the version once refused five provably-distinct cells.
#
# Hash what the interpreter will actually load, not the build tree: site-packages holds COPIES of
# _core.so and lib64/libmonoprop.so, and they do NOT update when ninja runs. Gating on the build-dir
# .so once measured the previous day's binary for a whole job.
#
# Hashing the FILE rather than `import`ing it, deliberately. Asking the interpreter
# (`from monoprop import _core; _core.__file__`) is the stronger identification and is what a
# measurement harness should do -- but an import can fail for reasons that have nothing to do with the
# build just completed (without the foss/2025b module loaded it dies on `CXXABI_1.3.15 not found`
# against the system libstdc++), and under `set -e` that would make a SUCCESSFUL build report failure.
# A build script must not be able to fail in the checking.
#
# Exactly one match, never `head -1`: scipy ships optimize/_highspy/_core.cpython-*.so, and a loose
# glob has made two genuinely different arms compare byte-identical.
echo
echo "=== ARM IDENTITY ==="
mapfile -t _sos < <(ls "$VENV"/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null)
if [ "${#_sos[@]}" -ne 1 ]; then
    echo "!! matched ${#_sos[@]} monoprop/_core*.so under $VENV, expected exactly 1" >&2
else
    echo "_core.so    $(md5sum "${_sos[0]}" | cut -d' ' -f1)  ${_sos[0]}"
    # Most of the C++ lives here, not in _core.so -- a rebuild that copies only _core.so leaves engine
    # changes behind, so this half is worth recording too.
    _lib="$(dirname "${_sos[0]}")/lib64/libmonoprop.so"
    if [ -f "$_lib" ]; then
        echo "libmonoprop $(md5sum "$_lib" | cut -d' ' -f1)  $_lib"
    else
        echo "libmonoprop MISSING at $_lib" >&2
    fi
fi

# Explicit, and not decoration: under `set -e` a trailing test that merely returns 1 would become the
# script's exit status, and a SUCCESSFUL build would report FAILED to sacct. The identity block above
# is diagnostics -- it must not be able to fail the build.
exit 0
