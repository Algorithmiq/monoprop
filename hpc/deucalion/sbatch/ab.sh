#!/bin/bash -l
# Interleaved time+memory A/B of two builds, in ONE allocation, on any of the three workloads.
#
#   WORKLOAD=hubbard sbatch -N1 -A "$MONOPROP_SLURM_ACCOUNT" -t 3:00:00 --chdir="$PWD" \
#            hpc/deucalion/sbatch/ab.sh
#
# This replaces ab-100m.sh (random) and models-ab.sh (fixed models). They were the same
# protocol driving two bench files, and keeping two copies is how their summaries drifted
# into two formats. Every hard-won property of both is preserved verbatim and each is
# commented where it lives, because each of them is a trap that has already cost a run:
#
#   * ONE allocation for both arms. Two jobs get two allocations, and any per-allocation
#     slowdown lands entirely on one side and reads as an effect of the code. That cost this
#     project a run once already -- same node, 13:20 vs 13:22, one layout looking 1.85x worse
#     on `pare` with no way to tell drift from signal.
#   * Order flipped on (rep + cell_idx) % 2, never a running counter. A counter gives cell i
#     the values i, i+N, i+2N, ... which only alternates when the cell count N is odd -- with
#     two cells it never alternates at all, and it fails silently.
#   * The cell names `fresh` and `graph`. ab_summary's LABEL_RE matches exactly those two; a
#     third name makes every file unparseable, which prints an empty table rather than an error.
#   * --bench-rounds=1. pytest-benchmark's pedantic `setup=` builds the next round's problem
#     while the current one is resident, which doubles peak memory.
#   * -s. pytest captures at the fd level, so C++ stderr -- every LAYERPROF, COMMPROF and
#     PROFMEM line -- is discarded without it. That has voided three measurements.
#   * MALLOC_ARENA_MAX=$PARTITIONS and monoprop_NUM_THREADS=$PARTITIONS, identical on both arms.
#   * The summary run under a real venv python. The compute nodes' /usr/bin/python3 is 3.6.8,
#     and a parser that dies there writes a traceback to the .err nobody reads while leaving a
#     headed table with no rows -- which reads exactly like "the two arms are identical".
#   * `benches/` comes from THIS checkout for both arms, so only the compiled extension differs.
#
#SBATCH --job-name=mp-ab
#SBATCH --partition=normal-x86
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-N%N-%j.out
#SBATCH --error=%x-N%N-%j.err

set -uo pipefail

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${MAIN_VENV:=$PROJ/src/ab-baseline-main/.venv}"
: "${PORT_VENV:?PORT_VENV must name the venv under test}"
: "${REPS:=6}"
: "${RESULTS_TAG:=}"

# ---------------------------------------------------------------- the workload

# The defaults are the measured 100M rungs. At the models' own default lower_atol=1e-4 the
# length cutoff and the system size are BOTH saturated (Hubbard is flat from cutoff 10 and
# from num_sites 60; Pauli is flat from num_layers 20), so the coefficient tolerance is the
# only axis that reaches 100M terms. Calibration: hubbard c10/1.25e-05 -> 96,981,051 terms,
# pauli c14/5e-05 -> 91,273,861, random 250/c6/24M obs-terms -> 99,441,369.
: "${WORKLOAD:=hubbard}"
case "$WORKLOAD" in
    hubbard)
        BENCH=benches/bench_models.py
        : "${CUTOFF:=10}" "${LOWER_ATOL:=1.25e-05}"
        SIZE_ARGS=("--hubbard-cutoff=$CUTOFF" "--hubbard-lower-atol=$LOWER_ATOL")
        # `-m slow` and a `-k` that names the model. The bare model name also matches the
        # fused test_model[hubbard], which would re-do the propagate cell's work.
        MARK_ARGS=(-m slow)
        SELECT_PREFIX="hubbard and "
        ;;
    pauli)
        BENCH=benches/bench_models.py
        : "${CUTOFF:=14}" "${LOWER_ATOL:=5e-05}"
        SIZE_ARGS=("--pauli-cutoff=$CUTOFF" "--pauli-lower-atol=$LOWER_ATOL")
        MARK_ARGS=(-m slow)
        SELECT_PREFIX="pauli and "
        ;;
    random)
        BENCH=benches/bench_random.py
        # num-modes is 250 and not 256: a MAX_NUM_MODES=250 build rejects 256 outright, while
        # 250 is accepted by both a 250- and a 1024-cap build. The obs-terms -> propagated-terms
        # multiplier is ~4.1 at cutoff 6 with 100 generators.
        : "${OBS_TERMS:=24000000}" "${NUM_MODES:=250}" "${CUTOFF:=6}" "${NUM_GENERATORS:=100}"
        SIZE_ARGS=("--num-modes=$NUM_MODES" "--cutoff=$CUTOFF"
                   "--obs-terms=$OBS_TERMS" "--num-generators=$NUM_GENERATORS")
        MARK_ARGS=()
        SELECT_PREFIX="heisenberg and "
        ;;
    *)
        echo "refusing: unknown WORKLOAD '$WORKLOAD' (hubbard|pauli|random)" >&2
        exit 1
        ;;
esac

# Extra per-model config overrides for the size axis, e.g.
#   MODEL_ARGS="--hubbard-num-sites=30 --hubbard-observable-site=23"
# The Hubbard observable site is a lattice position, not a constant: it has to move with
# num_sites or the sweep also slides the observable out of the middle of the lattice.
: "${MODEL_ARGS:=}"

# ---------------------------------------------------------------- the geometry

: "${RANKS_PER_NODE:=8}"
: "${PARTITIONS:=16}"

NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$((RANKS_PER_NODE * NODES))

if [ $((RANKS_PER_NODE * PARTITIONS)) -gt 128 ]; then
    echo "refusing: ${RANKS_PER_NODE} ranks x ${PARTITIONS} partitions exceeds 128 cores/node" >&2
    exit 1
fi

# Derived, not hardcoded: ab-100m.sh pinned "B_" into the label, so every layout it ran was
# labelled B and colliding labels collated into one table. The letter names a (ranks,
# partitions) PAIR, not a rank count -- A is one rank over the WHOLE node, so calling a 1x16
# rung A too would claim 128 cores where it uses 16. Anything unnamed gets X, which still
# collates on its own; an honest "not the bake-off's" beats a familiar wrong letter.
case "${RANKS_PER_NODE}x${PARTITIONS}" in
    1x128) LAYOUT_LETTER=A ;;  # 1 rank over the whole node
    8x16) LAYOUT_LETTER=B ;;   # 1 rank per NUMA domain -- the measured default
    2x64) LAYOUT_LETTER=C ;;   # 1 rank per socket
    *) LAYOUT_LETTER=X ;;
esac
LAYOUT="${LAYOUT_LETTER}_$(printf '%dx%d' "$RANKS_PER_NODE" "$PARTITIONS")"

# --cpu-bind=none, NOT =cores, is the DEFAULT and it is a launcher-side workaround for an
# engine bug. Measured at 8 ranks x 16 partitions:
#
#   --cpu-bind=cores    affinity 16   0 pinned   UNPLACED
#   --cpu-bind=threads  affinity 16   0 pinned   UNPLACED
#   --cpu-bind=none     affinity 128  16 pinned  PLACED, and the 8 ranks take disjoint
#                                                sixteenths covering all 128 cpus exactly once
#
# With =cores each rank sees a disjoint 16-core slice, enumerate_physical_cores reports 16
# while the placement policy still asks for ranks_per_node x partitions = 128, refuses, and
# every rank runs unplaced -- on BOTH arms, which reads as a clean null. Three runs behind
# RESULTS-invidx-memory.md were refused for exactly this, and four A/B jobs were voided.
#
# CPU_BIND exists so the placement PR can measure the configuration that IS the defect. On any
# other PR, overriding it is how you void your own run.
: "${CPU_BIND:=none}"

# ---------------------------------------------------------------- provenance

# RESULTS_ROOT, not $MONOPROP_RUNS directly. $PROJ/runs is shared, and a campaign that
# collates by globbing it has already swept another session's A/B into a table and printed a
# 1.17x 6/6 regression that was not its own. Every PR campaign gets its own subdirectory and
# pr_report.py is handed an explicit list of directories rather than a pattern.
: "${RESULTS_ROOT:=$MONOPROP_RUNS}"
mkdir -p "$RESULTS_ROOT"
RESULTS="${RESULTS_ROOT}/ab-${WORKLOAD}${RESULTS_TAG:+-${RESULTS_TAG}}-N${NODES}"
if [ -e "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS already holds results; pass a distinct RESULTS_TAG" >&2
    exit 1
fi
mkdir -p "$RESULTS"

export MALLOC_ARENA_MAX="$PARTITIONS"
export monoprop_NUM_THREADS="$PARTITIONS"

# monoprop_PARTITIONS is set EXPLICITLY, not left to resolve_partition_count_'s auto path.
#
# Unset, the engine computes min(monoprop_NUM_THREADS, enumerate_physical_cores()). That reads
# the visible core count -- which is precisely what --cpu-bind and the cgroup placement fix
# perturb. Under --cpu-bind=cores a rank sees 16 cores, not 128, so an arm that changes what
# the topology reports could silently partition differently from the other one, and the A/B
# would be comparing two different world sizes while the label claimed one. The label says
# RxS; this makes S true by construction on both arms rather than by coincidence.
export monoprop_PARTITIONS="$PARTITIONS"

# EXTRA_ENV is `NAME=value` pairs separated by `;`, applied to BOTH arms. It is how a highlight
# cell reaches a knob (monoprop_PROFILE, monoprop_PARTITIONS) without a second copy of this
# file. Both arms or neither: setting one side only is a systematic difference between them.
if [ -n "${EXTRA_ENV:-}" ]; then
    IFS=';' read -r -a _extra <<< "$EXTRA_ENV"
    for kv in "${_extra[@]}"; do
        [ -z "$kv" ] && continue
        case "$kv" in
            *=*) export "${kv?}" ;;
            *) echo "refusing: EXTRA_ENV entry '$kv' is not NAME=value" >&2; exit 1 ;;
        esac
    done
fi

# GNU time, not the shell builtin -- only the former has -v. Identical on both arms or neither.
if [ -x /usr/bin/time ]; then
    TIME_V="/usr/bin/time -v"
else
    TIME_V=""
    echo "!! /usr/bin/time absent: no independent peak-RSS control this run" >&2
fi

echo "=== workload: $WORKLOAD ${SIZE_ARGS[*]} ${MODEL_ARGS:-} ==="
echo "=== layout  : $LAYOUT  nodes=$NODES ranks=$NTASKS world=$((NTASKS * PARTITIONS)) ==="
echo "=== bind    : --cpu-bind=$CPU_BIND ==="
echo "=== env     : ${EXTRA_ENV:-(none)} ==="
echo "=== results : $RESULTS ==="

for side in main port; do
    eval "venv=\$${side^^}_VENV"
    echo "=== $side: $venv ==="
    if [ ! -x "$venv/bin/python" ]; then
        echo "missing venv: $venv" >&2
        exit 1
    fi
    # The md5 of the loaded .so, printed here as well as recorded per cell. Arm identity is
    # that hash and nothing else: __version__ is a git describe stamped into the dist-info at
    # install time, and a later `cmake --build` + copy into the venv does not rewrite it, so an
    # arm can advertise one commit while serving a binary from several commits later.
    "$venv/bin/python" - <<'PY' || exit 1
import hashlib, pathlib
import monoprop
import pytest_benchmark  # noqa: F401  - fails here rather than mid-measurement
assert monoprop.has_mpi, "built without MPI"
so = pathlib.Path(monoprop._core.__file__)  # noqa: SLF001
print("   module :", pathlib.Path(monoprop.__file__).resolve())
print("   core   :", so, hashlib.md5(so.read_bytes()).hexdigest())  # noqa: S324
print("   version:", monoprop.__version__, " variant:", monoprop.__variant__)
print("   has_mpi:", monoprop.has_mpi, " MAX_NUM_MODES:", monoprop.MAX_NUM_MODES)
PY
done
echo

# ---------------------------------------------------------------- the cells

# Two cells, and the split is not cosmetic. `fresh` never requests the session-scoped graph
# fixture, so no second full operator is resident while build_graph and propagate build their
# own. And `build_graph` EXTENDS the graph rather than replacing it, so hubbard's 29 Trotter
# steps retain 29 layer-sets and OOM a 242 GiB node even at 1.9M terms, while `propagate`
# releases each layer as it contracts and runs the same model to 97M terms in 379 MiB.
CELLS=(
    "fresh|${SELECT_PREFIX}(build_graph or propagate)"
    "graph|${SELECT_PREFIX}(energy or gradient)"
)

# CELL_SPEC overrides that list, as `name|selector` entries separated by `;`. Hubbard at full
# size runs propagate ONLY, for the reason above: fusing the operations means reporting nothing
# for the model at full size, splitting them reports what is actually measurable. A
# propagate-only cell is a `fresh` cell with one operation.
if [ -n "${CELL_SPEC:-}" ]; then
    IFS=';' read -r -a CELLS <<< "$CELL_SPEC"
    for cell in "${CELLS[@]}"; do
        case "${cell%%|*}" in
            fresh | graph) ;;
            *)
                echo "refusing: cell name '${cell%%|*}' is not fresh or graph" >&2
                exit 2
                ;;
        esac
    done
fi

for rep in $(seq 1 "$REPS"); do
    cell_idx=0
    for cell in "${CELLS[@]}"; do
        NAME="${cell%%|*}"
        SELECT="${cell#*|}"
        cell_idx=$((cell_idx + 1))

        # Parity includes `rep` -- see the header. The two sides of one cell then straddle any
        # drift instead of trailing it.
        if [ $(((rep + cell_idx) % 2)) -eq 0 ]; then
            ORDER="main port"
        else
            ORDER="port main"
        fi

        for side in $ORDER; do
            eval "venv=\$${side^^}_VENV"
            LABEL="N${NODES}_${LAYOUT}_${WORKLOAD}_${NAME}_${side}_r${rep}"
            OUT="$RESULTS/${LABEL}.log"
            echo "######## $LABEL: R=$NTASKS S=$PARTITIONS world=$((NTASKS * PARTITIONS)) ########"

            export monoprop_BENCH_LABEL="$LABEL"
            export monoprop_BENCH_RESULTS="$RESULTS"

            # /usr/bin/time -v wraps each rank so every cell carries a peak RSS that does NOT
            # come from the suite's own instrumentation. The engine's byte ledger has been
            # wrong by up to 23x in BOTH directions against the kernel, so the rule here is
            # that no B/term figure ships without a peak-RSS figure beside it -- and a figure
            # produced by the thing under test is not an independent check of it.
            srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RANKS_PER_NODE" \
                 --cpus-per-task=$((128 / RANKS_PER_NODE)) \
                 --cpu-bind="$CPU_BIND" --distribution=block:block \
                 ${TIME_V:-} "$venv/bin/python" -m pytest "$BENCH" \
                 -o filterwarnings=default -k "$SELECT" "${MARK_ARGS[@]}" \
                 "${SIZE_ARGS[@]}" ${MODEL_ARGS:-} \
                 --bench-rounds=1 \
                 --benchmark-json="$RESULTS/time-${LABEL}.json" \
                 -q -s -p no:cacheprovider >"$OUT" 2>&1 \
                 || echo "!! $LABEL failed (rc=$?), continuing -- see $OUT"

            # Assert the instrument fired, at the point of collection: "both arms measured the
            # same" and "nothing emitted" are otherwise the same observation.
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
    print("   core md5 :", meta.get("monoprop_core_md5"))
    print("   terms:", terms)
PY
        done
    done
done

echo
echo "=== paired summary ==="
# --allow-both-placed unless this run IS about placement. Both arms placing is the HEALTHY
# state; that refusal exists for a branch where placement itself is the change, so a baseline
# that also placed means a venv is not the build it is labelled as. Every other refusal stays
# armed. ALLOW_BOTH_PLACED=0 is how the placement PR re-arms it.
: "${ALLOW_BOTH_PLACED:=1}"
SUMMARY_ARGS=()
[ "$ALLOW_BOTH_PLACED" = "1" ] && SUMMARY_ARGS+=(--allow-both-placed)

"$PORT_VENV/bin/python" hpc/deucalion/tools/ab_summary.py "$RESULTS" "${SUMMARY_ARGS[@]}" 2>&1 \
    | tee "$RESULTS/AB-SUMMARY.md"
summary_rc=${PIPESTATUS[0]}
if [ "$summary_rc" -ne 0 ]; then
    echo "!! ab_summary.py exited $summary_rc: AB-SUMMARY.md is a diagnostic, NOT a result." >&2
    echo "!! Do not read the absence of a difference above as an absence of a difference." >&2
fi

exit "$summary_rc"
