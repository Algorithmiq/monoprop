#!/bin/bash -l
# Interleaved time+memory A/B of two builds at ~100M terms, in ONE allocation.
#
#   sbatch -N1 -A "$MONOPROP_SLURM_ACCOUNT" -t 3:30:00 --chdir="$PWD" hpc/deucalion/sbatch/ab-100m.sh
#   sbatch -N2 -A "$MONOPROP_SLURM_ACCOUNT" -t 3:00:00 --chdir="$PWD" hpc/deucalion/sbatch/ab-100m.sh
#
# Why one allocation: two separate jobs get two separate allocations, and any
# per-allocation slowdown (a noisy neighbour on the fabric, a node with a different clock
# ceiling) lands entirely on one side and reads as an effect of the code change. That trap
# has already cost this project a run -- same node, but 13:20 vs 13:22 in different jobs,
# which left one layout looking 1.85x worse on `pare` with no way to tell drift from signal.
#
# Both venvs therefore run inside one allocation, the two sides of a cell run back to back,
# and the order flips every (rep, cell). `benches/` comes from this checkout for both sides,
# so only the compiled extension differs: each venv is an editable install of its own tree.
#
#SBATCH --job-name=mp-ab100m
#SBATCH --partition=normal-x86
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-N%N-%j.out
#SBATCH --error=%x-N%N-%j.err

set -uo pipefail

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

# Both arms are git worktrees under $PROJ, built by the same build-worktree.sh job.
#
# The port default is NOT $PWD/.venv, even though $PWD is usually a port checkout. A venv
# in $HOME sits on a different filesystem from one under $PROJ (Lustre), so making it the
# default puts a per-arm difference in shared-library and .pyc load path into a measurement
# whose whole purpose is to attribute a difference to the compiled extension. $HOME is also
# inode-capped at 25k, which a second build tree can exhaust.
#
# Only the venvs differ: `benches/` is taken from $PWD on both sides, and the branch does
# not touch src/ or benches/ at all (verified: origin/main..HEAD is cpp/, docs and justfile).
: "${MAIN_VENV:=$PROJ/src/mp-main/.venv}"
: "${PORT_VENV:=$PROJ/src/mp-port/.venv}"
: "${REPS:=4}"
: "${RESULTS_TAG:=}"

# ~100M propagated terms. The obs-terms -> propagated-terms multiplier is ~4.1 at cutoff 6
# with 100 generators (measured: 200k -> 813,913 and 7M -> 28,965,342), so 24M lands just
# under 100M. num-modes is 250 and not 256: a MAX_NUM_MODES=250 build rejects 256 outright,
# while 250 is accepted by both a 250- and a 1024-cap build.
: "${OBS_TERMS:=24000000}"
: "${NUM_MODES:=250}"
: "${CUTOFF:=6}"
: "${NUM_GENERATORS:=100}"
: "${RANKS_PER_NODE:=8}"
: "${PARTITIONS:=16}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$((RANKS_PER_NODE * NODES))
LAYOUT="B_$(printf '%dx%d' "$RANKS_PER_NODE" "$PARTITIONS")"

if [ $((RANKS_PER_NODE * PARTITIONS)) -gt 128 ]; then
    echo "refusing: ${RANKS_PER_NODE} ranks x ${PARTITIONS} partitions exceeds 128 cores/node" >&2
    exit 1
fi

RESULTS="${MONOPROP_RUNS}/ab-100m${RESULTS_TAG:+-${RESULTS_TAG}}-N${NODES}"
if [ -e "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS already holds results; pass a distinct RESULTS_TAG" >&2
    exit 1
fi
mkdir -p "$RESULTS"

# Identical on both arms. MALLOC_ARENA_MAX is not tuning: glibc opens arenas per thread, so
# an unplaced build fragments differently from a placed one, and that difference shows up in
# RSS and nowhere in the engine's own byte counts. Pinning it means a memory delta between
# the arms is the operator, not the allocator.
export MALLOC_ARENA_MAX="$PARTITIONS"
export monoprop_NUM_THREADS="$PARTITIONS"
export monoprop_COMM_PROFILE=1

# Provenance before measuring: an A/B is worthless if both sides turn out to be the same
# build, and the checks that cost seconds here would otherwise surface after a multi-minute
# operator build. pytest_benchmark is included because it lives in an optional dependency
# group, and its absence otherwise fails the run at the first benchmark rather than now.
for side in main port; do
    eval "venv=\$${side^^}_VENV"
    echo "=== $side: $venv ==="
    if [ ! -x "$venv/bin/python" ]; then
        echo "missing venv: $venv" >&2
        exit 1
    fi
    "$venv/bin/python" - <<'PY' || exit 1
import pathlib
import monoprop
import pytest_benchmark  # noqa: F401  - fails here rather than mid-measurement
assert monoprop.has_mpi, "built without MPI"
print("   module :", pathlib.Path(monoprop.__file__).resolve())
print("   version:", monoprop.__version__, " variant:", monoprop.__variant__)
print("   has_mpi:", monoprop.has_mpi, " MAX_NUM_MODES:", monoprop.MAX_NUM_MODES)
PY
done
echo

# Two cells, split by whether the operation needs the session-scoped built_graph fixture.
#
# `fresh` never requests built_graph, so no foreign 100M-term graph is resident while
# build_graph and propagate construct their own propagators. Fusing all four into one
# invocation would leave a full graph alive across propagate's setup -- two complete
# operators per rank, which does not fit a node at this size. The split costs one extra
# srun and problem build per (side, rep); it does not cost an extra operator build, because
# both orderings build three.
CELLS=(
    "fresh|heisenberg and (build_graph or propagate)"
    "graph|heisenberg and (energy or gradient)"
)

for rep in $(seq 1 "$REPS"); do
    cell_idx=0
    for cell in "${CELLS[@]}"; do
        NAME="${cell%%|*}"
        SELECT="${cell#*|}"
        cell_idx=$((cell_idx + 1))

        # Flip the order on every (rep, cell), not just every rep: the two sides of one
        # cell then straddle any drift instead of trailing it.
        #
        # The parity MUST include `rep`. A single counter incremented once per cell gives
        # cell i the values i, i+N, i+2N, ... which only alternates when the cell count N is
        # odd. With two cells it never alternates at all, so one cell would have `main`
        # first in every rep -- precisely the confound this script exists to remove, and it
        # fails silently: the summary still prints, and a first-vs-second-position effect
        # reads as an effect of the code. That shipped once already, on three layouts
        # narrowed to two. Deriving the parity from (rep + cell_idx) is independent of how
        # many cells there are.
        if [ $(((rep + cell_idx) % 2)) -eq 0 ]; then
            ORDER="main port"
        else
            ORDER="port main"
        fi

        for side in $ORDER; do
            eval "venv=\$${side^^}_VENV"
            LABEL="N${NODES}_${LAYOUT}_${NAME}_${side}_r${rep}"
            OUT="$RESULTS/${LABEL}.log"
            echo "######## $LABEL: R=$NTASKS S=$PARTITIONS world=$((NTASKS * PARTITIONS)) ########"

            export monoprop_BENCH_LABEL="$LABEL"
            export monoprop_BENCH_RESULTS="$RESULTS"

            # -s is load-bearing: pytest's capture is fd-level, and the COMMPROF line is
            # written straight to fd 2 from a transport destructor, so without it the
            # instrument looks like it never fired. Passed on both arms even though only
            # the port build emits anything, because redirecting fd 2 on one side only is
            # a systematic difference between the arms.
            #
            # --bench-rounds=1 is mandatory, not a default: pedantic builds round k+1's
            # arguments before releasing round k's, so more than one round holds two
            # propagators at once and doubles peak memory. Repetition comes from $REPS.
            srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RANKS_PER_NODE" \
                 --cpus-per-task=$((128 / RANKS_PER_NODE)) \
                 --cpu-bind=cores --distribution=block:block \
                 "$venv/bin/python" -m pytest benches/bench_random.py \
                 -o filterwarnings=default -k "$SELECT" \
                 --num-modes="$NUM_MODES" --cutoff="$CUTOFF" \
                 --obs-terms="$OBS_TERMS" --num-generators="$NUM_GENERATORS" \
                 --bench-rounds=1 \
                 --benchmark-json="$RESULTS/time-${LABEL}.json" \
                 -q -s -p no:cacheprovider >"$OUT" 2>&1 \
                 || echo "!! $LABEL failed (rc=$?), continuing -- see $OUT"

            # Assert the instrument fired, at the point of collection. "Both arms measured
            # the same" and "the profile never emitted" are otherwise the same observation.
            # Only the port build has monoprop_COMM_PROFILE, so main is expected to be
            # silent and its silence carries no information either way.
            if [ "$side" = port ]; then
                nprof=$(grep -c COMMPROF "$OUT" 2>/dev/null || echo 0)
                [ "$nprof" -ge "$NTASKS" ] \
                    || echo "!! COMMPROF: $nprof lines for $NTASKS ranks in $LABEL"
            fi
            "$PORT_VENV/bin/python" - "$RESULTS/${LABEL}.json" <<'PY' || true
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
if path.exists():
    meta = json.loads(path.read_text()).get("meta", {})
    print("   placement:", meta.get("pinning", {}), " threads:", meta.get("monoprop_threads"))
PY
        done
    done
done

echo
echo "=== paired summary ==="
# Both parsers below run under $PORT_VENV/bin/python and never a bare `python3`. The compute
# nodes' /usr/bin/python3 is 3.6.8: ab_summary.py's `from __future__ import annotations`
# needs 3.7, and str.removeprefix needs 3.9. A parser that dies on a compute node writes its
# traceback to stderr -- the .err file, not the .out anyone reads -- and leaves a table
# header with no rows beneath it, which reads exactly like "the two arms are identical".
# That has already happened once on this project.
#
# pipefail does not catch it either: tee exits 0 and creates the file regardless, so the
# python side's status has to be read out of PIPESTATUS. stderr is folded into the pipe so
# the reason lands in the summary a reader is looking at, not in a second file.
"$PORT_VENV/bin/python" hpc/deucalion/tools/ab_summary.py "$RESULTS" 2>&1 \
    | tee "$RESULTS/AB-SUMMARY.md"
summary_rc=${PIPESTATUS[0]}
if [ "$summary_rc" -ne 0 ]; then
    echo "!! ab_summary.py exited $summary_rc: AB-SUMMARY.md is a diagnostic, NOT a result." >&2
    echo "!! Do not read the absence of a difference above as an absence of a difference." >&2
fi

# The summary IS the deliverable of this job, so its failure must be visible in sacct rather
# than only in the log.
exit "$summary_rc"
