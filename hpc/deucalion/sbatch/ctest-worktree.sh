#!/bin/bash -l
# Rebuild one worktree from its CURRENT source and gate its C++ suite in a STANDALONE build tree.
#
#   TREE=$PROJ/src/mp-candidate TAG=candidate \
#       sbatch -A "$MONOPROP_SLURM_ACCOUNT" hpc/deucalion/sbatch/ctest-worktree.sh
#
# Knobs: TREE (required), TAG (build-dir suffix, default the tree's basename), SYNC (1),
# MONOPROP_MAX_NUM_MODES (1024), BUILD_JOBS (32) -- build.sh's names, so one chain has one name
# per knob. MAX_NUM_MODES / JOBS still work as aliases; setting both to different values aborts.
#
# Complements sbatch/mpi-tests-worktree.sh rather than replacing it. That script runs the MPI layouts
# and the Python suite against `build/editable/Release-$TAG/cpp/tests`, which only exists when
# scikit-build-core has just built the tree and needs no regeneration. This one configures its own
# tree, so it works on any checkout and after any source edit -- which is what an A/B arm needs before
# it is measured.
#
# Why the rebuild is not optional bookkeeping: arm identity in this project is the INSTALLED
# _core.so's md5 and nothing else. `monoprop.__version__` reports HEAD for every editable arm, and the
# dist-info stamp is written at install time and never rewritten by a later cmake rebuild -- both
# version identifiers go stale together. Rebuild, print the md5, and every measurement afterwards
# refers to a binary that matches the source.
#
#SBATCH --job-name=mp-ctest-wt
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

[ -n "${TREE:-}" ] || { echo "refusing: set TREE to the worktree to gate" >&2; exit 2; }
[ -d "$TREE" ] || { echo "refusing: TREE=$TREE is not a directory" >&2; exit 2; }
TAG="${TAG:-$(basename "$TREE")}"
# Captured before the `cd "$TREE"` below, and from the submit dir rather than $BASH_SOURCE:
# sbatch STAGES the batch script, so inside a job $BASH_SOURCE is a spool path and names
# nothing in the checkout.
PIN_FILE="$PWD/hpc/deucalion/baseline.md5"

export MONOPROP_SRC="$TREE"
source hpc/deucalion/env.sh
cd "$TREE"

# ONE NAME PER KNOB, ACROSS THE WHOLE CHAIN. build.sh reads MONOPROP_MAX_NUM_MODES and
# BUILD_JOBS; this script read MAX_NUM_MODES and JOBS for the SAME two cmake settings. Both run
# in one pr-ab.sh chain -- build-worktree.sh -> build.sh, then this script, which `uv sync
# --reinstall-package`s the venv again and therefore produces the binary that actually gets
# measured. So `MONOPROP_MAX_NUM_MODES=250 hpc/deucalion/pr-ab.sh ...` built a 250-mode arm and
# then silently replaced it with a 1024-mode one: both arms agree, ab_summary's
# monoprop_max_num_modes check stays quiet, and the campaign measures a configuration nobody
# asked for. Accept build.sh's name, keep the old one as an alias, and refuse rather than pick
# when the two disagree -- a knob that silently defaults is this project's recurring defect.
for _pair in "MONOPROP_MAX_NUM_MODES MAX_NUM_MODES" "BUILD_JOBS JOBS"; do
    read -r _canon _alias <<<"$_pair"
    if [ -n "${!_canon:-}" ] && [ -n "${!_alias:-}" ] && [ "${!_canon}" != "${!_alias}" ]; then
        echo "refusing: $_canon=${!_canon} but $_alias=${!_alias}; they are one knob" >&2
        echo "  ($_canon is build.sh's name and wins everywhere else -- set only that one.)" >&2
        exit 2
    fi
done
MAX_NUM_MODES="${MONOPROP_MAX_NUM_MODES:-${MAX_NUM_MODES:-1024}}"
JOBS="${BUILD_JOBS:-${JOBS:-32}}"

echo "=== tree   : $TREE ==="
echo "=== commit : $(git log --oneline -1) ==="
echo "=== dirty  : $(git status --porcelain | wc -l) files ==="
echo "=== build  : tag=$TAG max_num_modes=$MAX_NUM_MODES jobs=$JOBS ==="

# --reinstall-package + --no-cache are mandatory: pyproject's [tool.uv] cache-keys does not include
# SKBUILD_* or config-settings, so uv otherwise serves a cached build of a different configuration --
# classically an MPI-OFF one, after which every rank silently holds the whole operator.
#
# --group test is mandatory too: pytest lives in a dependency GROUP, and `--all-extras` covers extras,
# so without it uv prunes pytest out of the venv and the next test run fails with "No module named
# pytest", looking exactly like a venv that was never set up.
#
# build-dir is tagged per arm. pyproject's default `build/{state}/{build_type}` encodes neither the
# cmake defines nor the venv, so every configuration of a worktree shares ONE directory: two
# concurrently submitted jobs once raced on it, 3 of 4 arms died with `configure_file: No such file or
# directory`, and ctest still reported 262/262 passed against the half-written tree.
# THE PINNED BASELINE IS NOT REBUILDABLE FROM HERE. $PROJ/src/mp-main is group-writable like
# every other worktree -- nothing in the filesystem stops `TREE=$PROJ/src/mp-main sbatch
# ctest-worktree.sh` from `uv sync --reinstall-package`ing it -- and pr-ab.sh's own guard lives
# in pr-ab.sh, which this script can be submitted without. One command would re-base every
# ratio in a seven-PR campaign, and the only evidence afterwards is an md5 that no longer
# matches baseline.md5. Check the BINARY, not the path: a copy of the baseline under any other
# name is just as pinned, and __version__ goes stale exactly when the binary changes.
if [ -r "$PIN_FILE" ]; then
    _pin=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$PIN_FILE")
    mapfile -t _cur < <(ls "$TREE"/.venv/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null)
    if [ -n "$_pin" ] && [ "${#_cur[@]}" -eq 1 ] \
       && [ "$(md5sum "${_cur[0]}" | cut -d' ' -f1)" = "$_pin" ] \
       && [ "${ALLOW_BASELINE_REBUILD:-0}" != "1" ]; then
        echo "refusing: $TREE holds the PINNED baseline _core.so ($_pin)." >&2
        echo "  $PIN_FILE pins it, and every ratio in the campaign is measured against it." >&2
        echo "  Rebuilding voids all of them, silently -- the version stamp will not change." >&2
        echo "  Re-pin deliberately and re-run the measured PRs, or gate a different TREE." >&2
        echo "  (ALLOW_BASELINE_REBUILD=1 overrides, once you have decided to re-pin.)" >&2
        exit 2
    fi
fi

if [ "${SYNC:-1}" = 1 ]; then
    uv sync --all-extras --group test --group bench \
        --reinstall-package monoprop --no-cache \
        --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" \
        --config-settings-package="monoprop:cmake.define.monoprop_MAX_NUM_MODES=${MAX_NUM_MODES}" \
        --config-settings-package="monoprop:build-dir=build/{state}/{build_type}-$TAG" \
        --config-settings-package="monoprop:build.tool-args=-j${JOBS}" \
        || { echo "!! uv sync FAILED"; exit 1; }
fi

# Exactly one match, never `head -1`: scipy ships optimize/_highspy/_core.cpython-*.so, and a loose
# glob has made two genuinely different arms compare byte-identical.
mapfile -t sos < <(ls "$TREE"/.venv/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null)
[ "${#sos[@]}" -eq 1 ] || { echo "!! matched ${#sos[@]} monoprop/_core*.so, need exactly 1"; exit 1; }
echo "=== ARM IDENTITY md5=$(md5sum "${sos[0]}" | cut -d' ' -f1) so=${sos[0]} ==="
"$TREE/.venv/bin/python" -c "import monoprop as m; print('has_mpi =', m.has_mpi, ' MAX_NUM_MODES =', m.MAX_NUM_MODES)"

# Three defines a plain `cmake -S . -B ...` does not supply, each failing late and unhelpfully:
#
#   Python_EXECUTABLE            the module env puts /bin/python3 (3.6.8) first on PATH, FindPython
#                                picks it, and the >=3.11 requirement then fails.
#   SKBUILD_PROJECT_VERSION_FULL cpp/CMakeLists.txt passes it to write_basic_package_version_file, and
#                                only scikit-build-core ever sets it. Any value does for a test build;
#                                it only names the generated monopropConfigVersion.cmake.
#   nanobind_DIR                 CMakeLists.txt adds src/monoprop/bindings unconditionally and that
#                                needs nanobind's cmake package. nanobind is a BUILD dependency, so it
#                                is absent from the runtime venv uv sync produces. Installing it there
#                                is safe: it is never imported at run time and does not touch
#                                _core.so, which is what identifies the arm.
#
# Reusing build/editable/Release instead does NOT work: its cache pins the build-isolation interpreter
# to a scikit-build-core temp dir that no longer exists, so ninja fails while regenerating build.ninja.
# The skbuild-* presets only adopt the tree and set no cache variables, so they inherit the failure.
"$TREE/.venv/bin/python" -c "import nanobind" 2>/dev/null \
    || uv pip install --python "$TREE/.venv/bin/python" nanobind \
    || { echo "!! nanobind install FAILED"; exit 1; }
NB_DIR=$("$TREE/.venv/bin/python" -m nanobind --cmake_dir)

BUILD="build/ctest-$TAG"
rm -rf "$BUILD"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$TREE/.venv/bin/python" \
    -DSKBUILD_PROJECT_VERSION_FULL=0.0.0 \
    -Dnanobind_DIR="$NB_DIR" \
    -Dmonoprop_ENABLE_MPI=ON -Dmonoprop_MAX_NUM_MODES="$MAX_NUM_MODES" \
    -Dmonoprop_ENABLE_CXX_UNIT_TESTS=ON \
    || { echo "!! cmake configure FAILED"; exit 1; }
# ABORT on a failed build; never fall through to ctest, which will happily run the previous tree and
# report a meaningless pass.
cmake --build "$BUILD" -j"$JOBS" || { echo "!! cmake build FAILED"; exit 1; }

echo "=== ctest labels: $(ctest --test-dir "$BUILD" --print-labels 2>/dev/null | tr '\n' ' ') ==="
# Both gates. `-L serial` is the gate; `-L unit` is its superset and is run too so numbers stay
# comparable across a campaign. `ctest -L mpi` fails on main as well (shm_comm_oversubscribed aborts at
# 2 ranks) and is NOT a gate -- the MPI cases are covered by mpi-tests-worktree.sh.
#
# A slow run here is MPI_Init, not the tests: it initialises every fabric device present, and there are
# 8 mlx5_* HCAs, which is 8.8 s wall against 0.17 s user PER CASE because ctest runs each Boost case as
# its own process. The tell is wall time with no CPU behind it.
ctest --test-dir "$BUILD" -L unit --output-on-failure
rc_unit=$?
echo "=== ctest unit rc=$rc_unit ==="
ctest --test-dir "$BUILD" -L serial --output-on-failure
rc_serial=$?
echo "=== ctest serial rc=$rc_serial ==="

# ONE FAILURE IS NOT A RATE. At least one case here asserts on allocator behaviour rather than on
# library behaviour -- `lazy_fold_survives_operator_growth` requires a reallocated buffer to land at a
# NEW address, and malloc is free to hand the freed block straight back. The same binary has failed it
# under `-L serial` and passed it under `-L unit` in one job, then passed 180/180 in a controlled
# rerun. Before treating a single failure as a regression, re-run the case with:
#
#   CASES=<case> BUILDS="$TREE/build/ctest-$TAG" sbatch hpc/deucalion/sbatch/ctest-repeat.sh
if [ "$rc_unit" -ne 0 ] || [ "$rc_serial" -ne 0 ]; then
    echo "!! a gate failed -- re-run the failing case with ctest-repeat.sh before calling it a regression"
    exit 1
fi
echo "########## BOTH GATES PASSED ##########"
