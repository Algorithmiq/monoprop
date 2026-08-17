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
#
# Time limits come from the same calibration, which timed `steps x propagate` at 1x16:
#
#   hubbard  c6 24.4s   c8 69.0s   s20 70.6s   s30 84.5s   s45 93.4s   c12 102.2s
#   pauli    c10 2.0s   c12 17.2s  l10 16.0s   l30 83.5s
#
# A cell is 24 invocations (2 arms x 6 reps x 2 cells), so the limit is roughly
# 24 x (build_graph + propagate + ~15s of import and model construction). Note the serial
# rungs are NOT 8x the 8x16 ones: at 4.75M terms 1x16 propagate is 2.0s against the 8x16
# cell's 3.0s, because at these sizes the exchange funnel costs more than the extra cores
# buy. Sizing the serial wave as "8x the ranks means 8x the wall" would have been wrong.
set -uo pipefail

WAVE="${1:?usage: submit.sh <wave1|wave2|ranks|serial>}"
export MONOPROP_SRC="${MONOPROP_SRC:-$PWD}"
source "$MONOPROP_SRC/hpc/deucalion/env.sh"
ACCT="${MONOPROP_SLURM_ACCOUNT:?set MONOPROP_SLURM_ACCOUNT (see hpc/deucalion/env.local.sh.example)}"
SRC="$MONOPROP_SRC"
cd "$SRC" || exit 1

# cell <tag> <nodes> <time> <ranks-per-node> <WORKLOAD=..> [MODEL_ARGS=..]
# WORKLOAD, not MODEL: ab.sh took over from models-ab.sh and drives the random
# problem under the same name, so the axis is the workload rather than the model.
cell () {
    local tag="$1" nodes="$2" time="$3" rpn="$4"; shift 4
    echo "-> $tag  N=$nodes t=$time rpn=$rpn  $*"
    env "$@" RANKS_PER_NODE="$rpn" RESULTS_TAG="$tag" REPS="${REPS:-6}" \
        sbatch --account="$ACCT" -N "$nodes" -t "$time" \
               --job-name="mpc-$tag" --chdir="$SRC" \
               hpc/deucalion/sbatch/ab.sh
}

case "$WAVE" in
    wave1)  # the two anchors, plus the cheap rungs that cost almost nothing to have early
        cell hub-c10-anchor 1 3:30:00 8 WORKLOAD=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-anchor 1 3:30:00 8 WORKLOAD=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell hub-c6 1 1:30:00 8 WORKLOAD=hubbard CUTOFF=6 LOWER_ATOL=1.25e-05
        cell pau-c10 1 1:00:00 8 WORKLOAD=pauli CUTOFF=10 LOWER_ATOL=5e-05
        cell pau-c12 1 1:30:00 8 WORKLOAD=pauli CUTOFF=12 LOWER_ATOL=5e-05
        cell pau-l5 1 1:00:00 8 WORKLOAD=pauli CUTOFF=14 LOWER_ATOL=5e-05 \
             MODEL_ARGS=--pauli-num-layers=5
        ;;
    wave2)  # the remaining sweep points, sized from wave1's measured wall time
        cell hub-c8 1 3:00:00 8 WORKLOAD=hubbard CUTOFF=8 LOWER_ATOL=1.25e-05
        cell hub-s20 1 3:00:00 8 WORKLOAD=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05 \
             "MODEL_ARGS=--hubbard-num-sites=20 --hubbard-observable-site=15"
        cell hub-s30 1 3:00:00 8 WORKLOAD=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05 \
             "MODEL_ARGS=--hubbard-num-sites=30 --hubbard-observable-site=23"
        cell pau-l10 1 3:00:00 8 WORKLOAD=pauli CUTOFF=14 LOWER_ATOL=5e-05 \
             MODEL_ARGS=--pauli-num-layers=10
        ;;
    ranks)  # rank scaling at the anchors. Wall time is roughly flat in N -- setup is
            # replicated per rank and barely falls with node count -- so the limit does not
            # shrink as the node count grows.
        cell hub-c10-N2 2 3:30:00 8 WORKLOAD=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell hub-c10-N4 4 3:30:00 8 WORKLOAD=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-N2 2 3:30:00 8 WORKLOAD=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell pau-c14-N4 4 3:30:00 8 WORKLOAD=pauli CUTOFF=14 LOWER_ATOL=5e-05
        ;;
    serial)  # "non-MPI": the MPI build at ONE rank x 16 partitions. This is NOT a build
             # without MPI -- conftest still imports mpi4py and bench_comm is COMM_WORLD at
             # size 1 -- and it uses 16 of the node's 128 cores, so it is not comparable
             # with the 8x16 cells. The limits are the measured 1x16 propagate times above
             # with about 2x margin, NOT 8x the 8x16 limit: see the note in the header.
        cell hub-c10-serial 1 2:30:00 1 WORKLOAD=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-serial 1 2:30:00 1 WORKLOAD=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell hub-c6-serial 1 1:30:00 1 WORKLOAD=hubbard CUTOFF=6 LOWER_ATOL=1.25e-05
        cell pau-c12-serial 1 1:00:00 1 WORKLOAD=pauli CUTOFF=12 LOWER_ATOL=5e-05
        ;;
    *)
        echo "unknown wave: $WAVE" >&2
        exit 2
        ;;
esac
