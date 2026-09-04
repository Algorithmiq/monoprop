#!/bin/bash -l
# Full correctness sweep for a worktree build: C++ CTest (serial + MPI) and the
# Python MPI suite across four rank/partition layouts.
#
#   sbatch --chdir="$PWD" -N2 hpc/deucalion/sbatch/mpi-tests-worktree.sh
#
# Same as mpi-tests.sh except MONOPROP_SRC points at this worktree, so it exercises
# the worktree's own venv and build rather than the primary checkout's.
#
#SBATCH --job-name=mp-test-wt
#SBATCH --partition=dev-x86
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# See build-worktree.sh for the full note. SLURM_SUBMIT_DIR is the directory sbatch was invoked
# from and --chdir does not change it, so a job submitted with --chdir alone gates the
# SUBMITTER's checkout -- 208 green C++ cases and 625 green Python cases about the wrong code,
# which is the most expensive shape of pass there is. EXPECT_TREE turns that into an abort.
if [ -n "${EXPECT_TREE:-}" ]; then
    _want=$(readlink -f "$EXPECT_TREE" 2>/dev/null || echo "$EXPECT_TREE")
    _got=$(readlink -f "$PWD" 2>/dev/null || echo "$PWD")
    if [ "$_want" != "$_got" ]; then
        echo "refusing: EXPECT_TREE=$_want but this job is running in $_got." >&2
        echo "  --chdir and SLURM_SUBMIT_DIR disagree; submit from INSIDE the tree." >&2
        exit 1
    fi
fi

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

NODES="${SLURM_JOB_NUM_NODES:-2}"
rc_total=0
# WHICH suite failed, not just how many. rc_total counts, and a count cannot distinguish two
# real regressions from two suites that never ran -- which is exactly how the missing-build-dir
# bug above read as "2 SUITE(S) FAILED". Every increment names itself here.
FAILED=()

echo "=== build under test ==="
echo "branch: $(git rev-parse --abbrev-ref HEAD)  commit: $(git log --oneline -1)"
"$VENV/bin/python" -c "import monoprop as m; print('has_mpi =', m.has_mpi, ' MAX_NUM_MODES =', m.MAX_NUM_MODES)"
"$VENV/bin/python" -c "import monoprop as m; assert m.has_mpi, 'built without MPI'" || exit 1

echo
echo "############ C++ CTest ############"
# NOT build/editable/Release -- that has no CTestTestfile.cmake and ctest would
# report "No tests were found!!!" and exit 0, a silent pass. enable_testing() is
# called in cpp/CMakeLists.txt, so the registry is rooted one level down.
#
# The `-$MONOPROP_BUILD_TAG` suffix is NOT optional: build.sh tags the build directory per arm
# (`build/{state}/{build_type}-$TAG`, default tag `basename $MONOPROP_SRC`), so a worktree built by
# build.sh has build/editable/Release-<tag>, never build/editable/Release. Derive the tag exactly the
# way build.sh does or the two disagree.
: "${MONOPROP_BUILD_TAG:=$(basename "$MONOPROP_SRC")}"
CTEST_DIR="build/editable/Release-${MONOPROP_BUILD_TAG}/cpp/tests"
# ABORT on a missing registry; never fall through to `ctest --test-dir <nonexistent>`. That prints
# `Failed to change working directory` to stderr and returns non-zero, which here increments rc_total
# by exactly as much as a genuine test failure does -- so a job that never ran a single C++ test
# reports the same `2 SUITE(S) FAILED` as one with two real regressions. A missing build tree means
# the build never happened or the tag is wrong: that is not a test result, and it must not be
# reported as one.
[ -f "$CTEST_DIR/CTestTestfile.cmake" ] || {
    echo "!! no ctest registry at $CTEST_DIR -- build missing or MONOPROP_BUILD_TAG wrong"
    exit 1
}
# The mpi variants shell out to `mpiexec -n <n>` themselves, so they need the batch
# context (a 1-task srun step exposes one slot) and the OpenMPI *5* spelling of the
# oversubscribe knob; the justfile's OMPI_MCA_rmaps_base_oversubscribe is v4 and
# does nothing here.
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
ctest --test-dir "$CTEST_DIR" --output-on-failure -L serial \
    || { rc_total=$((rc_total + 1)); FAILED+=("ctest -L serial"); }
# COUNTED, and a dependent job chained on `afterok` of this one makes it a chain-stopper.
#
# README section 12 and an older comment here both say `ctest -L mpi` "is NOT a signal" because
# shm_comm_oversubscribed aborts at 2 ranks on main as well. That claim predates the export two
# lines up: with OpenMPI 5's PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe set, inside a
# batch allocation, the label has been observed to pass -- the abort was the mapping policy, not
# the branch. So it stays counted (a gate must not get quieter by accident) and it is NAMED, so
# a red here can be told apart from a serial one at a glance. Check the harness before the
# branch: a failure at this label has never yet been the code under test.
ctest --test-dir "$CTEST_DIR" --output-on-failure -L mpi \
    || { rc_total=$((rc_total + 1)); FAILED+=("ctest -L mpi (known to fail on main independently -- see README 12)"); }

echo
echo "############ Python MPI suite across layouts ############"
# (ranks-per-node, partitions-per-rank).
#
# CORRECTED: an earlier comment here gave the pinning condition as
# `ranks_per_node * S <= 128`. That is not the test. Under --cpu-bind=cores the grant is the
# RANK'S OWN mask, so pinning turns off when `S > cpus-per-task` -- a rank dividing its
# already-confined mask a second time. Getting this wrong was a real bug: layouts B and C ran
# unpinned with barrier_groups=0, at 437 vs 15.5 us/sync. See README section 10 rule 3.
#
# --cpu-bind=cores is kept below deliberately, and as of 2026-08-18 it is also the right setting
# for MEASUREMENT -- the sentence that used to stand here ("any measurement must use
# --cpu-bind=none instead") was inverted. It scored bindings by threads pinned; scored by TIME,
# =none costs 1.45x, because the engine's placement_order then spreads each rank's 16 threads
# over four NUMA domains. This is a correctness suite either way: unpinned ranks exercise the
# same code paths.
LAYOUTS=("1 1" "1 16" "2 8" "8 16")
for layout in "${LAYOUTS[@]}"; do
    read -r RPN S <<<"$layout"
    NTASKS=$((RPN * NODES))
    echo
    echo "###### ranks=$NTASKS (${RPN}/node)  partitions=$S  world=$((NTASKS * S)) ######"
    export monoprop_NUM_THREADS="$S" # identical on every rank, or the job deadlocks
    srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RPN" \
         --cpus-per-task=$((128 / RPN)) \
         --cpu-bind=cores --distribution=block:block \
         "$VENV/bin/python" -m pytest tests --with-mpi -q \
        || { rc_total=$((rc_total + 1)); FAILED+=("pytest --with-mpi at ${RPN}x${S}"); }
done

echo
if [ "$rc_total" -eq 0 ]; then
    echo "########## ALL SUITES PASSED ##########"
else
    echo "########## $rc_total SUITE(S) FAILED ##########"
    for _f in ${FAILED[@]+"${FAILED[@]}"}; do echo "##   FAILED: $_f"; done
fi
exit "$rc_total"
