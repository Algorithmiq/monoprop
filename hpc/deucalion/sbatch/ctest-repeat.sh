#!/bin/bash -l
# Run named ctest case(s) N times across one or more build trees and report pass/fail counts.
#
#   CASES=lazy_fold_survives_operator_growth \
#   BUILDS="$PROJ/src/mp-a/build/ctest-a $PROJ/src/mp-b/build/ctest-b" \
#   REPS=60 sbatch -A "$MONOPROP_SLURM_ACCOUNT" hpc/deucalion/sbatch/ctest-repeat.sh
#
# ONE FAILURE IS NOT A RATE, and this script exists because that distinction was load-bearing.
#
# Job 1826934 ran ONE binary twice: `ctest -L unit` passed 212/212 and `ctest -L serial` failed
# `lazy_fold_survives_operator_growth` on `before != after` with both pointers EQUAL. Two invocations
# of one binary disagreeing means the assertion is on allocator behaviour, not on library behaviour --
# the case wants a reallocated buffer to land at a new address, and malloc is entitled to hand the
# freed block straight back. A rerun here (60 reps x 3 trees) came back 180/180, so the rate is under
# ~1/190 and the failure was NOT attributable to the patch under test.
#
# The general rule: a gate that can fail in the direction it is watching for needs its own
# measurement before it voids a build. Establish the rate, then decide.
#
# Note what this cannot tell you on its own: a pre-patch baseline needs a build tree, and a worktree
# holding only `build/editable` has no CTestTestfile.cmake for ctest to read. Build the baseline with
# sbatch/ctest-worktree.sh first, or say plainly that the comparison is missing.
#
#SBATCH --job-name=mp-ctest-repeat
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/deucalion/env.sh

[ -n "${CASES:-}" ] || { echo "refusing: set CASES to the ctest case name(s)" >&2; exit 2; }
[ -n "${BUILDS:-}" ] || { echo "refusing: set BUILDS to one or more build directories" >&2; exit 2; }
REPS=${REPS:-60}

rc_total=0
for build in $BUILDS; do
    name=$(basename "$build")
    if [ ! -f "$build/CTestTestfile.cmake" ]; then
        # Not a failure of the case under test -- say which it is, so an absent baseline is never read
        # as a clean one.
        echo "SKIP build=$name -- no CTestTestfile.cmake (build it with ctest-worktree.sh first)"
        continue
    fi
    for case_name in $CASES; do
        pass=0
        fail=0
        for ((i = 0; i < REPS; i++)); do
            if ctest --test-dir "$build" -R "^${case_name}\$" >/dev/null 2>&1; then
                pass=$((pass + 1))
            else
                fail=$((fail + 1))
            fi
        done
        echo "REPEAT build=$name case=$case_name pass=$pass fail=$fail of $REPS"
        [ "$fail" -eq 0 ] || rc_total=$((rc_total + 1))
    done
done

# A zero-failure sweep is an upper bound on the rate, not proof the case is deterministic. Quote it as
# "0 of N", never as "does not fail".
echo "=== done: $rc_total case/build combination(s) failed at least once ==="
exit 0
