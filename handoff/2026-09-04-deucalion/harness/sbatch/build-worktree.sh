#!/bin/bash -l
# Build whichever worktree this script is submitted from.
#
#   sbatch --chdir="$PWD" hpc/deucalion/sbatch/build-worktree.sh
#
# The only difference from build.sh is MONOPROP_SRC: a worktree must build against
# its own sources and its own virtualenv, or the venv lands in the primary checkout
# and the two builds overwrite each other's installed extension module.
#
#SBATCH --job-name=mp-build-wt
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# SLURM_SUBMIT_DIR is "the directory from which sbatch was invoked" -- sbatch fills it from
# getcwd() and --chdir does not change it. So `sbatch --chdir=<worktree>` submitted from
# somewhere else lands here and builds the SUBMITTER's checkout, quietly, leaving the worktree
# unbuilt while the log's `=== src ===` line looks plausible. EXPECT_TREE is the caller stating
# which tree it meant; when it is set and disagrees, that is a wasted allocation about to
# happen, not a build. Unset (a hand submission from inside the tree) it checks nothing.
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

echo "=== branch : $(git rev-parse --abbrev-ref HEAD) ==="
echo "=== commit : $(git log --oneline -1) ==="
echo "=== src    : $MONOPROP_SRC ==="
echo "=== venv   : $VENV ==="

exec ./hpc/deucalion/build.sh
