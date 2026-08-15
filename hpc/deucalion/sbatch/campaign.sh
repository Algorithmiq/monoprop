#!/bin/bash
# Submit the fixed-model campaign, one sbatch per cell, from the checkout under test.
#
#   hpc/deucalion/sbatch/campaign.sh wave1    # anchors + the cheap cutoff rungs
#   hpc/deucalion/sbatch/campaign.sh wave2    # the rest of the sweeps
#   hpc/deucalion/sbatch/campaign.sh ranks    # the multi-node rungs
#   hpc/deucalion/sbatch/campaign.sh serial   # "non-MPI": 1 rank x 16 partitions
#
# Collate whatever has landed with:
#   hpc/deucalion/tools/campaign_summary.py "$MONOPROP_RUNS"/models-*
#
# One-factor-at-a-time around a per-model anchor, not a cross product. The anchors are the
# measured 100M rungs and every sweep runs at the anchor's lower_atol, because at the models'
# own default 1e-4 the cutoff and the system size are BOTH saturated and would sweep a flat
# line. Calibration (serial, 1x16):
#
#   hubbard @ lower_atol=1.25e-05   c6 23,933,586  c8 66,689,918  c10 96,981,051 (ANCHOR)
#                                   c12 105,786,653 -- over the 100M cap, excluded
#   hubbard @ c10, num_sites        20 75,847,560  30 90,154,750  45 96,457,682  60 = anchor
#   pauli   @ lower_atol=5e-05      c8 320,889  c10 4,751,695  c12 29,385,590  c14 91,273,861 (ANCHOR)
#   pauli   @ c14, num_layers       5 691,430  10 70,924,929  20 = anchor  30 91,326,094 (saturated)
#
# A cell that hits its time limit degrades rather than dies: ab_summary pairs on the reps
# BOTH arms ran, so a truncated cell is a cell with fewer reps.
set -uo pipefail

WAVE="${1:?usage: submit.sh <wave1|wave2|ranks|serial>}"
export MONOPROP_SRC="${MONOPROP_SRC:-$PWD}"
source "$MONOPROP_SRC/hpc/deucalion/env.sh"
ACCT="${MONOPROP_SLURM_ACCOUNT:?set MONOPROP_SLURM_ACCOUNT (see hpc/deucalion/env.local.sh.example)}"
SRC="$MONOPROP_SRC"
cd "$SRC" || exit 1

# cell <tag> <nodes> <time> <ranks-per-node> <MODEL=..> [MODEL_ARGS=..]
cell () {
    local tag="$1" nodes="$2" time="$3" rpn="$4"; shift 4
    echo "-> $tag  N=$nodes t=$time rpn=$rpn  $*"
    env "$@" RANKS_PER_NODE="$rpn" RESULTS_TAG="$tag" REPS="${REPS:-6}" \
        sbatch --account="$ACCT" -N "$nodes" -t "$time" \
               --job-name="mpc-$tag" --chdir="$SRC" \
               hpc/deucalion/sbatch/models-ab.sh
}

case "$WAVE" in
    wave1)  # the two anchors, plus the cheap rungs that cost almost nothing to have early
        cell hub-c10-anchor 1 3:30:00 8 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-anchor 1 3:30:00 8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell hub-c6 1 1:30:00 8 MODEL=hubbard CUTOFF=6 LOWER_ATOL=1.25e-05
        cell pau-c10 1 1:00:00 8 MODEL=pauli CUTOFF=10 LOWER_ATOL=5e-05
        cell pau-c12 1 1:30:00 8 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05
        cell pau-l5 1 1:00:00 8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 \
             MODEL_ARGS=--pauli-num-layers=5
        ;;
    wave2)  # the remaining sweep points, sized from wave1's measured wall time
        cell hub-c8 1 3:00:00 8 MODEL=hubbard CUTOFF=8 LOWER_ATOL=1.25e-05
        cell hub-s20 1 3:00:00 8 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05 \
             "MODEL_ARGS=--hubbard-num-sites=20 --hubbard-observable-site=15"
        cell hub-s30 1 3:00:00 8 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05 \
             "MODEL_ARGS=--hubbard-num-sites=30 --hubbard-observable-site=23"
        cell pau-l10 1 3:00:00 8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 \
             MODEL_ARGS=--pauli-num-layers=10
        ;;
    ranks)  # rank scaling at the anchors. Wall time is roughly flat in N -- setup is
            # replicated per rank and barely falls with node count -- so the limit does not
            # shrink as the node count grows.
        cell hub-c10-N2 2 3:30:00 8 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell hub-c10-N4 4 3:30:00 8 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-N2 2 3:30:00 8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell pau-c14-N4 4 3:30:00 8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05
        ;;
    serial)  # "non-MPI": the MPI build at ONE rank x 16 partitions. This is NOT a build
             # without MPI -- conftest still imports mpi4py and bench_comm is COMM_WORLD at
             # size 1 -- and it uses 16 of the node's 128 cores, so it is much slower than
             # the 8x16 cells and needs a longer limit, not a shorter one.
        cell hub-c10-serial 1 3:50:00 1 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-serial 1 3:50:00 1 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell hub-c6-serial 1 2:00:00 1 MODEL=hubbard CUTOFF=6 LOWER_ATOL=1.25e-05
        cell pau-c12-serial 1 2:00:00 1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05
        ;;
    *)
        echo "unknown wave: $WAVE" >&2
        exit 2
        ;;
esac
