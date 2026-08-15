# Deucalion (x86 partition) environment for monoprop.
#
# Source this from a login shell or from an sbatch script whose shebang is
# `#!/bin/bash -l`. Without `-l` the `module` shell function is not defined and
# every `module load` below silently does nothing.
#
#   source hpc/deucalion/env.sh
#
# Site-specific values (project id, Slurm account) live in env.local.sh, which is
# untracked. Copy env.local.sh.example and fill it in.

# ---------------------------------------------------------------------------
# Two Lmod traps on this system, both learned the hard way:
#
#   * NEVER `module purge`. MODULEPATH is set by /etc/profile.d/zz-hpcnow-arch.sh
#     to /eb/$(uname -i)/modules/all plus the HPC-X and OpenHPC trees. `purge`
#     drops all of it and every subsequent load fails with "module(s) unknown".
#   * NEVER re-source /etc/profile.d/lmod.sh or zz-hpcnow-arch.sh by hand. The
#     login profile has already run them; re-running resets Lmod's state and
#     loads then silently no-op.
#
# If you have already purged, start a fresh shell. There is no in-place repair.
# ---------------------------------------------------------------------------

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_here/env.local.sh" ] && . "$_here/env.local.sh"

: "${MONOPROP_PROJECT_DIR:=/projects/EEHPC-DEV-2026D08-260}"
: "${MONOPROP_SLURM_ACCOUNT:=eehpc-dev-2026d08-260x}"

export MONOPROP_PROJECT_DIR MONOPROP_SLURM_ACCOUNT
export PROJ="$MONOPROP_PROJECT_DIR"
# Defaults to the primary checkout, but a caller may point it at a git worktree.
# VENV hangs off this, so overriding it is what keeps a worktree's build and its
# virtualenv beside its own sources instead of writing into the main checkout.
: "${MONOPROP_SRC:=$PROJ/src/monoprop}"
export MONOPROP_SRC
export MONOPROP_RUNS="$PROJ/runs"

# ---------------------------------------------------------------------------
# Toolchain.
#
# foss/2025b = GCC 14.3.0 + OpenMPI 5.0.8 + FlexiBLAS + FFTW + ScaLAPACK.
#
# Why this and not the newest available:
#   * monoprop hard-fails below GCC 14 (cmake/compiler_flags/GNU.CXX.cmake
#     raises FATAL_ERROR). 14.3.0 clears it, and the tree needs no header that
#     GCC 14 lacks -- <print>, <format>, <bit>, <ranges> are all present, and
#     nothing includes <mdspan>, which is the main GCC-15-only header.
#   * There is NO Boost module built against GCC 15.2.0. foss/2026.1 would force
#     building Boost 1.85+ from source via tools/install-deps.sh.
#   * OpenMPI 5.0.8 carries PMIx 5.0.8, which pairs with this Slurm's pmix_v4.
#     foss/2026.1's OpenMPI 5.0.10 pulls PMIx 6.1.0 -- an untested pairing here.
#     If PMIx ever misbehaves, fall back to `mpirun` inside the allocation
#     (PRRTE bypasses Slurm's PMIx server entirely).
# ---------------------------------------------------------------------------
module load foss/2025b
module load Boost/1.88.0-GCC-14.3.0
module load CMake/3.31.8-GCCcore-14.3.0
module load Ninja/1.13.0-GCCcore-14.3.0

# Deliberately NOT setting Boost_DIR. The obvious value,
# $EBROOTBOOST/lib/cmake/Boost-1.88.0, does not exist: EasyBuild puts the config
# under lib64/cmake/ and leaves an empty lib/cmake/ beside it. The Boost module
# already puts $EBROOTBOOST on CMAKE_PREFIX_PATH, and find_package(Boost 1.85
# CONFIG REQUIRED) resolves it correctly from there. Setting a wrong Boost_DIR
# would take precedence and fail confusingly.

# ---------------------------------------------------------------------------
# Python / uv.
#
# Everything goes on Lustre. $HOME is capped at 25 GB AND 25,000 inodes, and a
# uv venv plus a managed CPython plus a build tree is comfortably more inodes
# than the ~16k free. `quotahome` to check.
#
# All three of these are per-architecture and must NOT be shared between the x86 and
# the ARM partitions:
#
#   * $PROJ/tools/uv/bin/uv is an x86-64 ELF, so on an A64FX node build.sh's
#     `exec uv sync` dies with "Exec format error" and nothing else gets a chance to
#     explain why.
#   * $PROJ/caches/uv-python holds the managed CPython. uv does tag those directories
#     with the platform triple (cpython-3.11.15-linux-x86_64-gnu), so a shared dir
#     would not actually collide -- but that is uv's internal layout rather than a
#     promise, and a wrong-architecture interpreter reaching scikit-build-core fails
#     deep inside the compile instead of here.
#   * $PROJ/caches/uv holds the wheels uv builds from sdists, which
#     UV_NO_BINARY_PACKAGE=mpi4py below forces it to do. Keeping the two apart also
#     stops an ARM experiment from evicting the warm x86 cache the daily builds want.
#
# CPM_SOURCE_CACHE further down is deliberately NOT split: it is git checkouts of
# msgpack-cxx sources, which are architecture-neutral.
#
# The selection has to happen in this file, not in the caller: build.sh re-sources it,
# so a uv a caller had prepended to PATH would end up behind the one chosen here.
# ---------------------------------------------------------------------------
_arch="$(uname -m)"
case "$_arch" in
    x86_64) _uv_root="$PROJ/tools/uv" _uv_tag="" ;;
    aarch64) _uv_root="$PROJ/tools/uv-arm" _uv_tag="-arm" ;;
    *) _uv_root="" _uv_tag="" ;;
esac

# Refusing outright, rather than leaving uv off PATH: a missing uv that is merely absent
# from PATH resurfaces much later as `uv: command not found` from inside build.sh, or --
# worse, as some other uv the site profile happens to provide.
if [ -z "$_uv_root" ] || [ ! -x "$_uv_root/bin/uv" ]; then
    echo "env.sh: no uv for architecture $_arch." >&2
    if [ -n "$_uv_root" ]; then
        echo "  Expected an executable at $_uv_root/bin/uv. Install it once from a LOGIN node" \
             "(this is only a download, so the login node's own architecture does not matter):" >&2
        echo "    mkdir -p $_uv_root/bin && cd $_uv_root" >&2
        echo "    curl -fsSLO https://github.com/astral-sh/uv/releases/latest/download/uv-$_arch-unknown-linux-gnu.tar.gz" >&2
        echo "    tar xzf uv-$_arch-unknown-linux-gnu.tar.gz" >&2
        echo "    install -m 0755 uv-$_arch-unknown-linux-gnu/uv uv-$_arch-unknown-linux-gnu/uvx $_uv_root/bin/" >&2
        echo "  Then, from a compute node of THAT architecture, populate its interpreter store:" >&2
        echo "    source hpc/deucalion/env.sh && uv python install 3.11" >&2
    else
        echo "  Only x86_64 and aarch64 exist on this system and \$PROJ/tools has no tree for $_arch." >&2
    fi
    unset _here _arch _uv_root _uv_tag
    return 1
fi

export PATH="$_uv_root/bin:$PATH"
export UV_CACHE_DIR="$PROJ/caches/uv$_uv_tag"
export UV_PYTHON_INSTALL_DIR="$PROJ/caches/uv-python$_uv_tag"

# Deliberately NOT routing uv through the site pip proxy.
#
# /etc/profile.d/proxpi.sh sets PIP_INDEX_URL to an internal proxpi cache at
# sn02:9191. uv ignores PIP_INDEX_URL, so pointing UV_DEFAULT_INDEX at it is the
# obvious move -- and it breaks the build. On a cache MISS the proxy's sdist path
# is flaky: fetching mpi4py-4.1.2.tar.gz from a compute node failed three uv
# retries and then died with "Invalid gzip header", while the very same URL
# served a byte-identical, valid archive on every attempt from a login node once
# warm. Wheels were unaffected; only the source distribution failed, which is
# exactly what UV_NO_BINARY_PACKAGE=mpi4py forces us to fetch.
#
# Compute nodes here have full outbound internet (pypi.org returns 200 from
# dev-x86), so uv's default index is both simpler and more reliable. Set
# UV_DEFAULT_INDEX="$PIP_INDEX_URL" only if the offsite route ever disappears,
# and expect to retry sdists.

# mpi4py must be compiled against the cluster OpenMPI. The generic-ABI wheel
# dlopens whatever libmpi it finds, which is not a safe basis for the
# PyMPIComm_Get() handoff into monoprop's C++ layer.
export UV_NO_BINARY_PACKAGE=mpi4py
export MPICC="$(command -v mpicc)"

# cpp/tests/CMakeLists.txt pulls msgpack-cxx via CPM (a git clone from GitHub).
# Compute nodes cannot reach GitHub, so this cache must be warmed from a login
# node before any compute-node build.
export CPM_SOURCE_CACHE="$PROJ/caches/cpm"

export VENV="$MONOPROP_SRC/.venv"

# ---------------------------------------------------------------------------
# Pin the BLAS thread pools to one thread per process.
#
# monoprop does its own partitioning: S pinned partitions per rank, each owning a
# disjoint core block. numpy's BLAS knows nothing about that, and if it sizes its
# pool from the machine rather than the cgroup, every rank on a node spawns a pool
# across all 128 cores while holding only 16 -- ranks then fight for cores that
# monoprop already assigned, and the benchmark measures oversubscription.
#
# This is not hypothetical here: an unpinned interpreter on the login node dies
# with "OpenBLAS blas_thread_init: pthread_create failed for thread 54 of 64"
# before it can even import numpy. Compute nodes are quieter about it, which is
# worse -- they just run slower.
#
# One thread per process is right because the parallelism is monoprop's to hand
# out, not BLAS's. Set all four: numpy may be linked against any of them, and
# FlexiBLAS (what foss/2025b provides) dispatches to a backend chosen at runtime.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# ---------------------------------------------------------------------------

unset _here _arch _uv_root _uv_tag
