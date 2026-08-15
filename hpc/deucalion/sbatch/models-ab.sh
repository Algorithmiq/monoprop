#!/bin/bash -l
# Interleaved time+memory A/B of two builds on a FIXED MODEL, in ONE allocation.
#
#   MODEL=hubbard CUTOFF=10 LOWER_ATOL=1.25e-05 \
#     sbatch -N1 -A "$MONOPROP_SLURM_ACCOUNT" -t 2:00:00 --chdir="$PWD" \
#            hpc/deucalion/sbatch/models-ab.sh
#
# The sibling of ab-100m.sh, which drives the random problem. Everything about the protocol
# is the same and deliberately so -- one allocation, order flipped per (rep, cell), the
# two-cell split, --bench-rounds=1, -s, identical allocator and thread settings on both arms
# -- because those are the parts that were got wrong before. See ab-100m.sh for why each
# one is there; this file only documents what differs.
#
# What differs: it drives benches/bench_models.py rather than bench_random.py, so the size
# knobs are the model's own config fields, and the label carries the model and the real
# layout letter so a campaign of many cells collates into one table.
#
#SBATCH --job-name=mp-models-ab
#SBATCH --partition=normal-x86
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-N%N-%j.out
#SBATCH --error=%x-N%N-%j.err

set -uo pipefail

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${MAIN_VENV:=$PROJ/src/mp-main/.venv}"
: "${PORT_VENV:=$PROJ/src/mp-invidx/.venv}"
: "${REPS:=6}"
: "${RESULTS_TAG:=}"

# The model and its size. Both arms get identical values -- these are the sweep's axes, and
# a cell where they differ is not a cell.
#
# The defaults are the measured 100M rungs. At the models' own default lower_atol=1e-4 the
# length cutoff and the system size are BOTH saturated (Hubbard is flat from cutoff 10 and
# from num_sites 60; Pauli is flat from num_layers 20), so the coefficient tolerance is the
# only axis that reaches 100M terms and the other sweeps have to run at the tolerance that
# gets there. Calibration: hubbard c10/1.25e-05 -> 96,981,051 terms, pauli c14/5e-05 ->
# 91,273,861.
: "${MODEL:=hubbard}"
case "$MODEL" in
    hubbard) : "${CUTOFF:=10}" "${LOWER_ATOL:=1.25e-05}" ;;
    pauli) : "${CUTOFF:=14}" "${LOWER_ATOL:=5e-05}" ;;
    *)
        echo "refusing: unknown MODEL '$MODEL'" >&2
        exit 1
        ;;
esac
# Extra per-model config overrides for the size axis, e.g.
#   MODEL_ARGS="--hubbard-num-sites=30 --hubbard-observable-site=23"
# The Hubbard observable site is a lattice position, not a constant: it has to move with
# num_sites or the sweep also slides the observable out of the middle of the lattice.
: "${MODEL_ARGS:=}"

: "${RANKS_PER_NODE:=8}"
: "${PARTITIONS:=16}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$((RANKS_PER_NODE * NODES))

# The layout letter is the bake-off's, derived rather than hardcoded: ab-100m.sh pinned
# "B_" into the label, so every layout it ran was labelled B and colliding labels collated
# into one table.
# The letter names a (ranks, partitions) PAIR, not a rank count: the bake-off's A is one
# rank over the whole node, and calling the 1x16 "non-MPI" rung A too would claim it used
# 128 cores when it uses 16. A rung that is not one of the three named layouts gets X, which
# still collates on its own -- an honest "not the bake-off's" beats a familiar wrong letter.
case "${RANKS_PER_NODE}x${PARTITIONS}" in
    1x128) LAYOUT_LETTER=A ;;  # 1 rank over the whole node
    8x16) LAYOUT_LETTER=B ;;   # 1 rank per NUMA domain -- the measured winner
    2x64) LAYOUT_LETTER=C ;;   # 1 rank per socket
    *) LAYOUT_LETTER=X ;;      # unnamed, e.g. the 1x16 rung that uses 16 of 128 cores
esac
LAYOUT="${LAYOUT_LETTER}_$(printf '%dx%d' "$RANKS_PER_NODE" "$PARTITIONS")"

if [ $((RANKS_PER_NODE * PARTITIONS)) -gt 128 ]; then
    echo "refusing: ${RANKS_PER_NODE} ranks x ${PARTITIONS} partitions exceeds 128 cores/node" >&2
    exit 1
fi

RESULTS="${MONOPROP_RUNS}/models-${MODEL}${RESULTS_TAG:+-${RESULTS_TAG}}-N${NODES}"
if [ -e "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS already holds results; pass a distinct RESULTS_TAG" >&2
    exit 1
fi
mkdir -p "$RESULTS"

export MALLOC_ARENA_MAX="$PARTITIONS"
export monoprop_NUM_THREADS="$PARTITIONS"
# No monoprop_COMM_PROFILE: it exists on neither arm here, so setting it and then counting
# COMMPROF lines is a check that can only ever report zero -- an alarm every cell, and the
# kind of two-zeros comparison that has produced false diagnoses on this project before.

# GNU time, not the shell builtin -- only the former has -v. Identical on both arms or
# neither, since wrapping one side only is a systematic difference between them.
if [ -x /usr/bin/time ]; then
    TIME_V="/usr/bin/time -v"
else
    TIME_V=""
    echo "!! /usr/bin/time absent: no independent peak-RSS control this run" >&2
fi

echo "=== model  : $MODEL cutoff=$CUTOFF lower_atol=$LOWER_ATOL ${MODEL_ARGS:-(default size)} ==="
echo "=== layout : $LAYOUT  nodes=$NODES ranks=$NTASKS world=$((NTASKS * PARTITIONS)) ==="
echo "=== results: $RESULTS ==="

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

# The same two-cell split as ab-100m.sh, and for the same reason: `fresh` never requests the
# session-scoped graph fixture, so no second full operator is resident while build_graph and
# propagate build their own. `benches/` comes from THIS checkout on both arms -- the new
# per-operation model benchmarks exist only here, and both arms must run the same test code
# for the comparison to be about the compiled extension.
#
# `-k "<model> and (...)"` also excludes the fused test_model[<model>], which would otherwise
# match on the model name alone and re-do the work of the propagate cell.
CELLS=(
    "fresh|${MODEL} and (build_graph or propagate)"
    "graph|${MODEL} and (energy or gradient)"
)

for rep in $(seq 1 "$REPS"); do
    cell_idx=0
    for cell in "${CELLS[@]}"; do
        NAME="${cell%%|*}"
        SELECT="${cell#*|}"
        cell_idx=$((cell_idx + 1))

        # Parity includes `rep`; see ab-100m.sh for the failure this prevents.
        if [ $(((rep + cell_idx) % 2)) -eq 0 ]; then
            ORDER="main port"
        else
            ORDER="port main"
        fi

        for side in $ORDER; do
            eval "venv=\$${side^^}_VENV"
            LABEL="N${NODES}_${LAYOUT}_${MODEL}_${NAME}_${side}_r${rep}"
            OUT="$RESULTS/${LABEL}.log"
            echo "######## $LABEL: R=$NTASKS S=$PARTITIONS world=$((NTASKS * PARTITIONS)) ########"

            export monoprop_BENCH_LABEL="$LABEL"
            export monoprop_BENCH_RESULTS="$RESULTS"

            # /usr/bin/time -v wraps each rank so every cell carries a peak RSS that does NOT
            # come from the suite's own instrumentation. The engine's byte ledger has been
            # wrong by up to 23x in both directions against the kernel, so the rule on this
            # project is that no B/term figure ships without a peak-RSS figure beside it --
            # and a figure produced by the thing under test is not an independent check of
            # it. `memhwm` in the results JSON is the suite's own answer; this is the
            # control. Absent /usr/bin/time the run proceeds unwrapped rather than failing.
            # --cpu-bind=none, NOT =cores. Measured on this branch at 8 ranks x 16
            # partitions: --cpu-bind=cores hands each rank a disjoint 16-core slice, so
            # enumerate_physical_cores reports 16 while the placement policy still asks for
            # ranks_per_node x partitions = 128, refuses, and every rank runs UNPLACED --
            # 0 pinned threads on both arms, which ab_summary rightly refuses as "two
            # unplaced builds". The measured alternatives, at 8x16:
            #
            #   --cpu-bind=cores    affinity 16   0 pinned   UNPLACED
            #   --cpu-bind=threads  affinity 16   0 pinned   UNPLACED
            #   --cpu-bind=none     affinity 128  16 pinned  PLACED, and the 8 ranks take
            #                                                DISJOINT sixteenths covering
            #                                                all 128 cpus exactly once
            #
            # Leaving all 128 visible is what lets the engine divide the node itself. The
            # per-rank-slice fix that would make =cores work lives on another branch.
            srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RANKS_PER_NODE" \
                 --cpus-per-task=$((128 / RANKS_PER_NODE)) \
                 --cpu-bind=none --distribution=block:block \
                 ${TIME_V:-} "$venv/bin/python" -m pytest benches/bench_models.py \
                 -o filterwarnings=default -k "$SELECT" -m slow \
                 "--${MODEL}-cutoff=$CUTOFF" "--${MODEL}-lower-atol=$LOWER_ATOL" \
                 ${MODEL_ARGS:-} \
                 --bench-rounds=1 \
                 --benchmark-json="$RESULTS/time-${LABEL}.json" \
                 -q -s -p no:cacheprovider >"$OUT" 2>&1 \
                 || echo "!! $LABEL failed (rc=$?), continuing -- see $OUT"

            # Assert the instrument fired, at the point of collection: "both arms measured
            # the same" and "nothing emitted" are otherwise the same observation.
            if [ -n "${TIME_V:-}" ]; then
                nrss=$(grep -c "Maximum resident set size" "$OUT" 2>/dev/null || echo 0)
                [ "$nrss" -ge "$NTASKS" ] \
                    || echo "!! time -v: $nrss RSS lines for $NTASKS ranks in $LABEL"
                awk '/Maximum resident set size/ {kb=$NF; s+=kb; if (kb>m) m=kb; n++}
                     END {if (n) printf("   peak RSS: sum %.1f GiB, worst rank %.1f GiB, over %d ranks\n",
                                        s/1048576, m/1048576, n)}' "$OUT"
            fi
            "$PORT_VENV/bin/python" - "$RESULTS/${LABEL}.json" <<'PY' || true
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
if path.exists():
    d = json.loads(path.read_text())
    meta = d.get("meta", {})
    terms = {k: v.get("terms") for k, v in d.get("opsize", {}).items()}
    print("   placement:", meta.get("pinning", {}), " threads:", meta.get("monoprop_threads"))
    print("   terms:", terms)
PY
        done
    done
done

echo
echo "=== paired summary ==="
# $PORT_VENV/bin/python, never a bare python3: the compute nodes' /usr/bin/python3 is 3.6.8,
# and a parser that dies there writes its traceback to the .err file nobody reads while
# leaving a headed table with no rows -- which reads exactly like "the two arms are
# identical". See ab-100m.sh.
# --allow-both-placed, because here both arms placing is the HEALTHY state. That refusal
# exists for a branch where placement itself is the change, so a baseline that also placed
# means a venv is not the build it is labelled as. This branch does not touch placement:
# both arms run the same launcher settings and both must pin, and it is neither arm pinning
# that would void the run. Every other refusal is left armed.
"$PORT_VENV/bin/python" hpc/deucalion/tools/ab_summary.py "$RESULTS" --allow-both-placed 2>&1 \
    | tee "$RESULTS/AB-SUMMARY.md"
summary_rc=${PIPESTATUS[0]}
if [ "$summary_rc" -ne 0 ]; then
    echo "!! ab_summary.py exited $summary_rc: AB-SUMMARY.md is a diagnostic, NOT a result." >&2
    echo "!! Do not read the absence of a difference above as an absence of a difference." >&2
fi

exit "$summary_rc"
