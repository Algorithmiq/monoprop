#!/bin/bash
# Submit the fixed-model campaign, one sbatch per cell, from the checkout under test.
#
#   hpc/deucalion/sbatch/campaign.sh wave1    # anchors + the cheap cutoff rungs
#   hpc/deucalion/sbatch/campaign.sh wave2    # the rest of the sweeps
#   hpc/deucalion/sbatch/campaign.sh ranks    # the multi-node rungs
#   hpc/deucalion/sbatch/campaign.sh serial   # "non-MPI": 1 rank x 16 partitions
#   hpc/deucalion/sbatch/campaign.sh geometry # ranks x partitions at fixed model, one node
#   hpc/deucalion/sbatch/campaign.sh layout   # geometry again + a node ladder, for the
#                                             # retained-exchange-layout change
#
# Both arms are venvs, not branches: MAIN_VENV (default $PROJ/src/mp-main/.venv) and PORT_VENV
# (default $PROJ/src/mp-invidx/.venv). `cell` forwards the caller's environment, so pointing an
# arm at another worktree's build needs no edit here:
#   PORT_VENV=$PROJ/src/mp-gws/.venv hpc/deucalion/sbatch/campaign.sh geometry
# benches/ comes from THIS checkout for both arms -- only the compiled extension differs -- so a
# metric the harness records but only one arm's binding emits is recorded for that arm alone.
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

WAVE="${1:?usage: submit.sh <wave1|wave2|ranks|serial|geometry|layout|nocache|sparse|sparse-stack>}"
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
             # size 1 -- and it uses 16 of the node's 128 cores, so it is not comparable
             # with the 8x16 cells. The limits are the measured 1x16 propagate times above
             # with about 2x margin, NOT 8x the 8x16 limit: see the note in the header.
        cell hub-c10-serial 1 2:30:00 1 MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05
        cell pau-c14-serial 1 2:30:00 1 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05
        cell hub-c6-serial 1 1:30:00 1 MODEL=hubbard CUTOFF=6 LOWER_ATOL=1.25e-05
        cell pau-c12-serial 1 1:00:00 1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05
        ;;
    geometry)  # Separate the flat world size P = ranks x partitions from the MPI rank count.
               # Every other wave sweeps ranks at a FIXED 16 partitions, so P and the rank
               # count move together and no cell in this directory can tell which one a cost
               # tracks. These four break the tie: read down the columns, not across.
               #
               #            P = 16        P = 128
               #   1 rank   g1x16         g1x128
               #   8 ranks  g8x2          g8x16
               #
               # The columns are core-matched (16 cores left, 128 right), so the time row is
               # also the test of whether extra partitions inside one rank buy anything.
               # Each rung needs its OWN tag: RESULTS is keyed on (model, tag, nodes) and
               # omits the layout, so two geometries sharing a tag collide and the second
               # refuses. PARTITIONS rides along as an env assignment -- `cell` forwards
               # anything after the 4th positional verbatim -- and rpn x PARTITIONS must stay
               # <= 128 or models-ab.sh refuses (over 128, pinning silently switches off).
               # c12 not c14: the P^2 signal is already 2x between P=16 and P=128 at 29.4M
               # terms, and c12 propagates in half the anchor's time.
        cell gws-c12-g1x16  1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=16
        cell gws-c12-g8x2   1 1:30:00  8 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=2
        cell gws-c12-g1x128 1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=128
        cell gws-c12-g8x16  1 1:30:00  8 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=16
               # The headline rung: same geometry and problem as models-pauli-pau-c14-anchor-N1,
               # so the "before" arm reproduces a measurement that already exists on disk.
        cell gws-c14-g8x16  1 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        ;;
    layout)  # The same four geometries again, plus a node ladder, for the change that stops
             # RETAINING the P-sized exchange layouts. `geometry` measured the record shrink;
             # this measures the layouts, and it has to re-run the four rather than reuse them
             # because the port arm is a different build. Tags are gws2-, not gws-: reusing a
             # tag would make models-ab.sh refuse a non-empty RESULTS dir, and that refusal is
             # the only thing standing between two campaigns and one silently merged table.
        cell gws2-c12-g1x16  1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=16
        cell gws2-c12-g8x2   1 1:30:00  8 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=2
        cell gws2-c12-g1x128 1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=128
        cell gws2-c12-g8x16  1 1:30:00  8 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=16
             # The node ladder at the anchor: P = 128, 256, 512 at a FIXED 8x16 per node, so
             # the only thing moving is the world size. This is the rung that matters most --
             # the retained layouts are P ints per layer per partition, so the saving grows
             # with P while the operator each rank walks does not.
             #
             # One tag across all three: RESULTS keys on (model, tag, NODES), so the node
             # count already separates them and three tags would only make the ladder harder
             # to collate. Time is flat in N, not divided by it -- setup is replicated per
             # rank -- so N=4 gets the same limit as N=1, which the `ranks` wave established.
        cell gws2-c14-g8x16  1 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        cell gws2-c14-g8x16  2 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        cell gws2-c14-g8x16  4 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        ;;
    nocache) # Deriving the RECV layout too, instead of caching the transpose per layer.
             # Deliberately narrower than `layout`: that wave had to separate P from the rank
             # count, which is settled. This buys a second point on a known axis rather than a
             # fifth geometry at one P.
             #
             # Both cells run against the SAME main arm the `layout` wave used, so the increment
             # is readable against gws2's numbers cell for cell.
             #
             # What to expect -- a prediction, not a hope: counted `graph` falls by exactly 80 B
             # per layer core (sizeof(LayerCore) 248 -> 168) and by nothing per slot, because the
             # transpose cache was a diagnostic OUTSIDE total_bytes. The 8 B/slot it did cost has
             # to be read from d_recv_cache_bytes going to zero and from RSS. A headline that
             # barely moves is the correct result here, not a disappointing one.
        cell gws3-c12-g1x128 1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=128
        cell gws3-c14-g8x16  4 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        ;;
    sparse)  # Storing only the world slots that carry traffic.
             #
             # This wave runs against a DIFFERENT baseline from every wave above it: the arm is
             # origin/main plus the memory instrument and nothing else, because origin/main
             # predates graph_memory_breakdown() and records none of the fields the law below is
             # asserted on. Pass it explicitly:
             #
             #   MAIN_VENV=$PROJ/src/mp-slotsbase/.venv \
             #   PORT_VENV=$PROJ/src/mp-slots/.venv \
             #   hpc/deucalion/sbatch/campaign.sh sparse
             #
             # The prediction, stated before the cells run. The baseline record is 32 B per world
             # slot (main stores the D range twice); the port stores 12 B per OCCUPIED slot and
             # gains 24 B per layer core for the world size and the self-slot position:
             #
             #   graph drop  =  32 x slots  -  12 x occupied  -  24 x cores
             #
             # exactly, to the byte. `occupied` must be IDENTICAL on both arms -- occupancy is a
             # property of the traffic, not of the format -- and that is the load-bearing check:
             # if it moves, the layout changed what is being sent, not merely how it is addressed.
             #
             # The P ladder is the same 8x16 geometry at 1, 2 and 4 nodes, so only the node count
             # moves. c12 at 1x16 is included precisely because it is the WORST case for sparsity:
             # occupancy is 31.5% there against 17.5% at P=512, so it is where this could lose.
             #
             # Tagged sp2 because the sp cells already exist and models-ab.sh refuses to write
             # over a populated results dir. Those cells are valid measurements -- the arms were
             # distinct, by md5 -- but they predate monoprop_core_md5, so their own provenance
             # cannot prove it and ab_summary refuses them. sp2 re-runs the same two binaries to
             # get a record that stands on its own; keep sp so the two can be compared.
        cell sp2-c12-g1x16   1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=16
        cell sp2-c12-g1x128  1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=128
        cell sp2-c14-g8x16   1 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        cell sp2-c14-g8x16   2 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        cell sp2-c14-g8x16   4 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        ;;
    sparse-stack)  # The sparse slot records measured on top of the DERIVED exchange layout.
             #
             # Different baseline again, and this is the one that is easy to get wrong: the arm is
             # perf/graph-world-size (PR #237), NOT main. #238 is now stacked on #237, so its own
             # before/after is against its base. Pass both explicitly:
             #
             #   MAIN_VENV=$PROJ/src/mp-gws/.venv \
             #   PORT_VENV=$PROJ/src/mp-slots/.venv \
             #   hpc/deucalion/sbatch/campaign.sh sparse-stack
             #
             # Why it is worth re-measuring rather than reusing the sp2 numbers: on main the dense
             # record was 32 B and the retained exchange layout dominated everything; on #237 the
             # record is 16 B and the exchange layout is gone, so the slot array is ~84% of what is
             # left. The same commit is a much bigger lever here, and the two branches have never
             # been compiled together before -- derive_exchange_layout now scatters over occupied
             # slots instead of probing sin_send_size(r), which is a binary search once sparse.
             #
             # Time matters here more than in sp2: that scatter is on the exchange path.
        cell sp3-c12-g1x16   1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=16
        cell sp3-c12-g1x128  1 1:30:00  1 MODEL=pauli CUTOFF=12 LOWER_ATOL=5e-05 PARTITIONS=128
        cell sp3-c14-g8x16   1 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        cell sp3-c14-g8x16   2 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        cell sp3-c14-g8x16   4 3:30:00  8 MODEL=pauli CUTOFF=14 LOWER_ATOL=5e-05 PARTITIONS=16
        ;;
    *)
        echo "unknown wave: $WAVE" >&2
        exit 2
        ;;
esac
