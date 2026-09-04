#!/bin/bash -l
# Which of the branch's mechanisms costs `propagate` its 1.34x at one node?
#
#   sbatch -N1 -A "$MONOPROP_SLURM_ACCOUNT" -t 1:30:00 --chdir="$PWD" \
#          hpc/deucalion/sbatch/knob-sweep-n1.sh
#
# RESULTS-ab-100m.md measured `propagate` 1.34x SLOWER on the branch at N=1, in all four
# reps, while the same operation is 1.15x faster at N=2. The sign flip is the finding that
# blocks calling the branch a win, and branch-vs-main cannot resolve it: the branch changes
# placement, the barrier shape, the shm transport AND the response leg of the query exchange
# (Engine.h's reverse_of_previous) all at once, and `main` emits no COMMPROF to compare with.
#
# So this sweeps runtime knobs on ONE binary instead. Three of the four arms are the same
# build, differing only in environment, which makes their COMMPROF directly comparable --
# per-arm mpi_s / barrier_peers_s attribution that no main-vs-branch run can produce.
#
#   main      baseline build; cannot pin at layout B at all (the 1.34x anchor)
#   grouped   branch as shipped: pinned, two-level barrier
#   flat      monoprop_BARRIER_GROUPING=0 -- pinned, FLAT barrier
#   unpinned  monoprop_PARTITION_PINNING=0 -- no placement
#
# `flat` is the arm that could not exist before. The barrier's domains derive from the
# cpusets, so the only way to get a flat barrier used to be turning pinning off -- which also
# unpins, confounding "grouped vs flat" with "pinned vs unpinned" in every prior before/after,
# including the evidence originally used to justify the second level.
#
# Reading:
#   flat recovers main's time      -> the second barrier level is the cost. Actionable, and
#                                     it meets the bar RESULTS-threading-baseline.md already
#                                     set for deleting it (~270 production + 145 test lines).
#   unpinned recovers, flat not    -> placement/NUMA first-touch. Chase with --mem-bind.
#   neither recovers               -> the transport or Engine.h. Needs targeted builds; out
#                                     of scope here, report it as unresolved rather than
#                                     guessing.
#
#SBATCH --job-name=mp-knob-n1
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${MAIN_VENV:=$PROJ/src/mp-main/.venv}"
: "${PORT_VENV:=$PROJ/src/mp-port/.venv}"
: "${REPS:=4}"
: "${RESULTS_TAG:=}"

# Full size deliberately. `propagate`'s port/main runs 0.79 at 0.8M terms, 0.99 at 25M and
# 1.34 at 100M, so the effect under investigation only exists at the top of that range and a
# cheap rung would show nothing and point the wrong way.
: "${OBS_TERMS:=24000000}"
: "${NUM_MODES:=250}"
: "${CUTOFF:=6}"
: "${NUM_GENERATORS:=100}"
: "${RANKS_PER_NODE:=8}"
: "${PARTITIONS:=16}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$((RANKS_PER_NODE * NODES))
LAYOUT="B_$(printf '%dx%d' "$RANKS_PER_NODE" "$PARTITIONS")"

RESULTS="${MONOPROP_RUNS}/knob-n1${RESULTS_TAG:+-${RESULTS_TAG}}-N${NODES}"
if [ -e "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS already holds results; pass a distinct RESULTS_TAG" >&2
    exit 1
fi
mkdir -p "$RESULTS"

# Identical on every arm. Only the two knobs below may differ between arms, or the sweep
# stops isolating anything.
export MALLOC_ARENA_MAX="$PARTITIONS"
export monoprop_NUM_THREADS="$PARTITIONS"
export monoprop_COMM_PROFILE=1

# arm|venv-var|PARTITION_PINNING|BARRIER_GROUPING
ARMS=(
    "main|MAIN_VENV|1|1"
    "grouped|PORT_VENV|1|1"
    "flat|PORT_VENV|1|0"
    "unpinned|PORT_VENV|0|1"
)

for spec in "${ARMS[@]}"; do
    IFS='|' read -r name venv_var _ _ <<<"$spec"
    eval "venv=\$$venv_var"
    echo "=== $name: $venv ==="
    [ -x "$venv/bin/python" ] || { echo "missing venv: $venv" >&2; exit 1; }
    "$venv/bin/python" - <<'PY' || exit 1
import pathlib
import monoprop
import pytest_benchmark  # noqa: F401  - fails here rather than mid-measurement
assert monoprop.has_mpi, "built without MPI"
print("   module :", pathlib.Path(monoprop.__file__).resolve())
print("   version:", monoprop.__version__, " MAX_NUM_MODES:", monoprop.MAX_NUM_MODES)
PY
done
echo

for rep in $(seq 1 "$REPS"); do
    # Rotate the whole arm order by rep rather than flipping a pair. With four arms a
    # parity bit cannot spread them: it would leave every arm in one of two positions, and
    # a first-vs-last position effect would land on the same arms every time. Rotation gives
    # each arm each position once per four reps. The failure this guards against has shipped
    # here before -- see the parity comment in ab-100m.sh.
    for offset in $(seq 0 $((${#ARMS[@]} - 1))); do
        idx=$(((rep - 1 + offset) % ${#ARMS[@]}))
        IFS='|' read -r NAME VENV_VAR PIN GROUP <<<"${ARMS[$idx]}"
        eval "venv=\$$VENV_VAR"

        LABEL="N${NODES}_${LAYOUT}_fresh_${NAME}_r${rep}"
        OUT="$RESULTS/${LABEL}.log"
        echo "######## $LABEL: pinning=$PIN grouping=$GROUP R=$NTASKS S=$PARTITIONS ########"

        export monoprop_BENCH_LABEL="$LABEL"
        export monoprop_BENCH_RESULTS="$RESULTS"
        export monoprop_PARTITION_PINNING="$PIN"
        export monoprop_BARRIER_GROUPING="$GROUP"

        # -s is load-bearing: COMMPROF is written straight to fd 2 from a transport
        # destructor and pytest's capture is fd-level, so without it the instrument that
        # proves each knob took effect looks like it never fired.
        srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RANKS_PER_NODE" \
             --cpus-per-task=$((128 / RANKS_PER_NODE)) \
             --cpu-bind=cores --distribution=block:block \
             "$venv/bin/python" -m pytest benches/bench_random.py \
             -o filterwarnings=default -k "heisenberg and (build_graph or propagate)" \
             --num-modes="$NUM_MODES" --cutoff="$CUTOFF" \
             --obs-terms="$OBS_TERMS" --num-generators="$NUM_GENERATORS" \
             --bench-rounds=1 \
             --benchmark-json="$RESULTS/time-${LABEL}.json" \
             -q -s -p no:cacheprovider >"$OUT" 2>&1 \
             || echo "!! $LABEL failed (rc=$?), continuing -- see $OUT"

        if [ "$NAME" != main ]; then
            nprof=$(grep -c COMMPROF "$OUT" 2>/dev/null || echo 0)
            [ "$nprof" -ge "$NTASKS" ] \
                || echo "!! COMMPROF: $nprof lines for $NTASKS ranks in $LABEL"
            grep -ao "barrier_groups=[0-9]* pinned=[0-9]*" "$OUT" | sort -u | sed 's/^/   state: /'
        fi
    done
done

echo
echo "=== sweep summary ==="
# $PORT_VENV/bin/python, never a bare python3: the compute nodes carry 3.6.8, and both
# parsers need `from __future__ import annotations` (3.7) and str.removeprefix (3.9). A
# parser that dies here writes its traceback to the .err file nobody reads and leaves a
# header with no rows, which looks exactly like four arms that measured the same.
"$PORT_VENV/bin/python" hpc/deucalion/tools/knob_summary.py "$RESULTS" 2>&1 \
    | tee "$RESULTS/KNOB-SUMMARY.md"
summary_rc=${PIPESTATUS[0]}
if [ "$summary_rc" -ne 0 ]; then
    echo "!! knob_summary.py exited $summary_rc: KNOB-SUMMARY.md is a diagnostic, NOT a result." >&2
fi
exit "$summary_rc"
