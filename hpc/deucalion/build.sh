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

echo "=== toolchain ==="
g++ --version | head -1
mpirun --version | head -1
cmake --version | head -1
echo "target arch: $(g++ -march=native -Q --help=target 2>/dev/null | awk '/^ +-march= /{print $2}')"
echo "max_num_modes: $MONOPROP_MAX_NUM_MODES   jobs: $BUILD_JOBS"
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
exec uv sync --all-extras --group test --group bench -v \
    --reinstall-package monoprop --no-cache \
    --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" \
    --config-settings-package="monoprop:cmake.define.monoprop_MAX_NUM_MODES=${MONOPROP_MAX_NUM_MODES}" \
    --config-settings-package="monoprop:build.tool-args=-j${BUILD_JOBS}"
