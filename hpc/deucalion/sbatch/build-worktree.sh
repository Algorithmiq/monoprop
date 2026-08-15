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

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

echo "=== branch : $(git rev-parse --abbrev-ref HEAD) ==="
echo "=== commit : $(git log --oneline -1) ==="
echo "=== src    : $MONOPROP_SRC ==="
echo "=== venv   : $VENV ==="

exec ./hpc/deucalion/build.sh
