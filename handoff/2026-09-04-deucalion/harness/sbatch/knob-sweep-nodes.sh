#!/bin/bash -l
# Same N=1 question as knob-sweep-n1.sh, but replicated across nodes to afford more reps.
#
#   sbatch -N3 -A "$MONOPROP_SLURM_ACCOUNT" -t 1:30:00 --chdir="$PWD" \
#          hpc/deucalion/sbatch/knob-sweep-nodes.sh
#
# Why this exists: job 1821340 refuted the barrier-grouping hypothesis (BARRIER_GROUPING=0
# left `propagate` 1.24-2.32x slower than main in all four reps) but could NOT separate
# placement from transport -- `unpinned` came back 0.74, 1.26, 1.37, 1.99, straddling 1.0.
# The limit is the instrument, not the hypothesis: `main`'s own per-rep minimum swung
# 1632 -> 3000 ms with nothing changed. More reps is the only thing that helps, and at
# 244 s per invocation more reps serially is hours.
#
# So each node runs the WHOLE arm set back to back, and the nodes run concurrently:
#
#   node 0:  [main grouped unpinned]  [grouped unpinned main]  [unpinned main grouped]
#   node 1:  [grouped unpinned main]  [unpinned main grouped]  [main grouped unpinned]
#   node 2:  [unpinned main grouped]  [main grouped unpinned]  [grouped unpinned main]
#
# ROUNDS x NODES = 9 observations in the wall time of 3, and -- the point -- every ratio is
# still a WITHIN-NODE pair, because an arm's neighbours in a round share its node and its
# time window. Splitting the arms across nodes instead would have been simpler and wrong: it
# trades the time confound for a node confound and destroys the pairing that makes
# ab_summary's statistics mean anything.
#
# `flat` is dropped: it answered its question at N=1 and costs a third of the budget.
#
#SBATCH --job-name=mp-knob-nodes
#SBATCH --partition=normal-x86
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${MAIN_VENV:=$PROJ/src/mp-main/.venv}"
: "${PORT_VENV:=$PROJ/src/mp-port/.venv}"
: "${ROUNDS:=3}"
: "${RESULTS_TAG:=}"
: "${OBS_TERMS:=24000000}"
: "${NUM_MODES:=250}"
: "${CUTOFF:=6}"
: "${NUM_GENERATORS:=100}"
: "${RANKS_PER_NODE:=8}"
: "${PARTITIONS:=16}"

mapfile -t NODE_LIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES=${#NODE_LIST[@]}
LAYOUT="B_$(printf '%dx%d' "$RANKS_PER_NODE" "$PARTITIONS")"

# arm|venv-var|PARTITION_PINNING|BARRIER_GROUPING
ARMS=(
    "main|MAIN_VENV|1|1"
    "grouped|PORT_VENV|1|1"
    "unpinned|PORT_VENV|0|1"
)
NARMS=${#ARMS[@]}

RESULTS="${MONOPROP_RUNS}/knob-nodes${RESULTS_TAG:+-${RESULTS_TAG}}"
if [ -e "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS already holds results; pass a distinct RESULTS_TAG" >&2
    exit 1
fi
mkdir -p "$RESULTS"

export MALLOC_ARENA_MAX="$PARTITIONS"
export monoprop_NUM_THREADS="$PARTITIONS"
export monoprop_COMM_PROFILE=1

echo "nodes: ${NODE_LIST[*]}   arms: $NARMS   rounds: $ROUNDS"
echo "observations: $((NNODES * ROUNDS)) per arm, in the wall time of $ROUNDS"
echo

for spec in "${ARMS[@]}"; do
    IFS='|' read -r name venv_var _ _ <<<"$spec"
    eval "venv=\$$venv_var"
    [ -x "$venv/bin/python" ] || { echo "missing venv for $name: $venv" >&2; exit 1; }
    echo "=== $name: $venv ==="
    "$venv/bin/python" - <<'PY' || exit 1
import pathlib
import monoprop
import pytest_benchmark  # noqa: F401
assert monoprop.has_mpi, "built without MPI"
print("   version:", monoprop.__version__, " MAX_NUM_MODES:", monoprop.MAX_NUM_MODES)
PY
done
echo

# One node's whole share of the sweep. Runs in a background subshell, so every export below
# is private to it -- with three of these live at once, a global export would race and label
# another node's results.
run_node() {
    local j=$1 node=${NODE_LIST[$j]}
    local round offset idx NAME VENV_VAR PIN GROUP venv REP LABEL OUT

    for round in $(seq 1 "$ROUNDS"); do
        for offset in $(seq 0 $((NARMS - 1))); do
            # Rotate by (round + node + offset): each arm meets each position once per
            # ROUNDS, and the phase differs per node so no arm shares a position with
            # itself across nodes either.
            idx=$(((round - 1 + j + offset) % NARMS))
            IFS='|' read -r NAME VENV_VAR PIN GROUP <<<"${ARMS[$idx]}"
            eval "venv=\$$VENV_VAR"

            # Unique across nodes, so the nine observations do not collide in one directory.
            REP=$((j * ROUNDS + round))
            LABEL="N1_${LAYOUT}_fresh_${NAME}_r${REP}"
            OUT="$RESULTS/${LABEL}.log"
            echo "######## [$node] $LABEL: pinning=$PIN grouping=$GROUP ########"

            export monoprop_BENCH_LABEL="$LABEL"
            export monoprop_BENCH_RESULTS="$RESULTS"
            export monoprop_PARTITION_PINNING="$PIN"
            export monoprop_BARRIER_GROUPING="$GROUP"

            # --nodelist pins this step to one node and --exclusive keeps the three
            # concurrent steps off each other's cores. Without --exclusive Slurm may place
            # two steps on the same node and they would measure each other.
            srun --mpi=pmix --nodes=1 --nodelist="$node" --exclusive \
                 --ntasks="$RANKS_PER_NODE" --ntasks-per-node="$RANKS_PER_NODE" \
                 --cpus-per-task=$((128 / RANKS_PER_NODE)) \
                 --cpu-bind=cores --distribution=block:block \
                 "$venv/bin/python" -m pytest benches/bench_random.py \
                 -o filterwarnings=default -k "heisenberg and (build_graph or propagate)" \
                 --num-modes="$NUM_MODES" --cutoff="$CUTOFF" \
                 --obs-terms="$OBS_TERMS" --num-generators="$NUM_GENERATORS" \
                 --bench-rounds=1 \
                 --benchmark-json="$RESULTS/time-${LABEL}.json" \
                 -q -s -p no:cacheprovider >"$OUT" 2>&1 \
                 || echo "!! $LABEL failed (rc=$?) on $node -- see $OUT"

            if [ "$NAME" != main ]; then
                grep -ao "barrier_groups=[0-9]* pinned=[0-9]*" "$OUT" | sort -u \
                    | sed "s/^/   [$node] state: /"
            fi
        done
    done
}

for j in $(seq 0 $((NNODES - 1))); do
    run_node "$j" &
done
wait

echo
echo "=== sweep summary ==="
"$PORT_VENV/bin/python" hpc/deucalion/tools/knob_summary.py "$RESULTS" 2>&1 \
    | tee "$RESULTS/KNOB-SUMMARY.md"
summary_rc=${PIPESTATUS[0]}
if [ "$summary_rc" -ne 0 ]; then
    echo "!! knob_summary.py exited $summary_rc: KNOB-SUMMARY.md is a diagnostic, NOT a result." >&2
fi
exit "$summary_rc"
