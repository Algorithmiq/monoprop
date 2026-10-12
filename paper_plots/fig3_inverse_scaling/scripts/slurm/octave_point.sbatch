#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#
# One work item of the Fig. 3 sweeps: one engine at one or more N, on a whole node.
# Submit from paper_plots/fig3_inverse_scaling/, which supplies --chdir; by hand:
#
#     sbatch --chdir=. -J mono-1024 scripts/slurm/octave_point.sbatch \
#         /path/to/site.env nsweep 7 monoprop 1024
#
# <sweep> is `nsweep` (panels a, c: the time of the whole run) or `perlayer` (panels b, d:
# the time of each propagate call, one per layer, or one per gate group for Hubbard).
#
# Every engine in both sweeps is timed by the same rule (keep_timing in the drivers): up to
# MAX_ROUNDS fresh rounds, the minimum kept; always a second round when it fits, since a
# fresh process's first round can be far slower (1.9x for PauliPropagation.jl at N=32); more
# rounds while under 60 s of timing; none that would end past CEILING_S.
#
# Engines: monoprop, ppvm, julia (PauliPropagation.jl) on the kicked-Ising chain of N qubits;
# monoprop-majorana, julia-majorana (MajoranaPropagation.jl) on the 2D Hubbard lattice of
# N modes, N = 2 * lx * ly, with the lattice taken from LATTICE below.
#
# PARAMETERS ARRIVE AS POSITIONAL ARGUMENTS, NEVER VIA --export. Naming a variable in
# --export makes Slurm rebuild the user environment by running a login shell on the node;
# where compute nodes have no home directory (AWS PCS) the job is then HELD with
# Reason=user_env_retrieval_failed_requeued_held and shows as "PD 0:00" indefinitely.
#
# Every point has CEILING_S of timed propagation, the same for every engine, and a hard
# `timeout` of CEILING_S + SETUP_S (process start, model build, JIT warm-up): a point that
# exceeds it is killed, recorded as a TIMEOUT row (the figure draws it as a lower bound),
# and the job still exits 0. A row whose one round finished past CEILING_S is drawn the same.
set -uo pipefail

SITE="${1:?usage: octave_point.sbatch <site.env> <sweep> <layers> <engine> <N> [N...]}"
SWEEP="${2:?missing <sweep>}"
LAYERS="${3:?missing <layers>}"
ENGINE="${4:?missing <engine>}"
shift 4
[[ "$SWEEP" == nsweep || "$SWEEP" == perlayer ]] || { echo "FATAL: sweep must be nsweep or perlayer" >&2; exit 2; }
NS=("$@")
((${#NS[@]})) || { echo "FATAL: no N values given" >&2; exit 2; }
# shellcheck source=/dev/null
source "$SITE"
: "${PYTHON:?} ${PPVM_PYTHON:?} ${JULIA:?} ${RUN_DIR:?}"

PLOTS="$PWD" # --chdir points at paper_plots/fig3_inverse_scaling/
CUTOFF=6
# The Hubbard support cutoff; the site file may override it while the pilot sizes the model.
HUBBARD_CUTOFF="${HUBBARD_CUTOFF:-4}"
# No coefficient truncation for any engine: the cutoff alone fixes the term set, so term
# counts agree exactly between engines. The Hubbard drivers do not even take a tolerance.
ATOL=0
MAX_ROUNDS="${MAX_ROUNDS:-10}"
# The paper's limit is 15 min of timed propagation per point, the same for every engine.
CEILING_S="${CEILING_S:-900}"
SETUP_S="${SETUP_S:-600}"
# Modes -> the lx x ly lattice of the Hubbard row, as near square as the octave allows.
declare -A LATTICE=([32]="4 4" [64]="8 4" [128]="8 8" [256]="16 8" [512]="16 16" [1024]="32 16")

# One writer per file: concurrent appends from several nodes to one file over NFS can
# interleave. The file is named for this job's own N set; merge afterwards.
SUFFIX="$(
    IFS=-
    echo "${NS[*]}"
)"
OUT="$RUN_DIR/${SWEEP}/${ENGINE}_N${SUFFIX}.jsonl"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "sweep=$SWEEP engine=$ENGINE N=${NS[*]} layers=$LAYERS cutoff=$CUTOFF hubbard_cutoff=$HUBBARD_CUTOFF"
log "host=$(hostname) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} out=$OUT"
mkdir -p "$RUN_DIR/$SWEEP" || { echo "FATAL: cannot create $RUN_DIR/$SWEEP" >&2; exit 1; }
echo "$SWEEP $ENGINE N=${NS[*]} layers=$LAYERS host=$(hostname) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} $(date -u --iso-8601=seconds)" \
    >>"$RUN_DIR/provenance.txt"

if [[ -n "${JOB_HOME:-}" ]]; then
    export HOME="$JOB_HOME"
    export XDG_CACHE_HOME="$HOME/.cache" MPLCONFIGDIR="$HOME/.mpl"
    mkdir -p "$HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
fi
# Per process: concurrent points each remove their own at the end.
export TMPDIR="/tmp/octave.${SLURM_JOB_ID:-0}.$$"
mkdir -p "$TMPDIR"
[[ -n "${JULIA_DEPOT_PATH:-}" ]] && export JULIA_DEPOT_PATH

# The drivers record these verbatim into each row, and the figure refuses anything but 1.
export monoprop_NUM_THREADS=1 RAYON_NUM_THREADS=1 JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

# Pin the timed process to one physical core, away from CPU 0 (boot CPU and default IRQ
# target), and move the batch shell off it. campaign.sbatch, which runs several points at once,
# passes PIN_CPU: one core per NUMA domain, so concurrent points share neither an L3 slice nor
# a memory controller. Slurm may report a flat fake topology, so --hint=nomultithread cannot be
# relied on; taskset inside the batch step's cpuset can, and no srun step has to be scheduled.
if [[ -z "${PIN_CPU:-}" ]]; then
    mapfile -t PRIMARY < <(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '!seen[$2]++ {print $1}')
    NCORES=${#PRIMARY[@]}
    if ((NCORES > 8)); then PIN_CPU=${PRIMARY[8]}; else PIN_CPU=${PRIMARY[$((NCORES / 2))]}; fi
fi
log "pinning to CPU $PIN_CPU (concurrent points: ${CONCURRENCY:-1})"
taskset -cp 1 $$ >/dev/null 2>&1 || true
PIN=(taskset -c "$PIN_CPU")

# A killed point leaves no row of its own, so write one: the figure draws it as a lower bound.
timeout_row() {
    local engine="$1" n="$2" rounds="$3" basis=pauli
    [[ "$engine" == *majorana ]] && basis=majorana
    # The engine field the driver itself would have written.
    case "$engine" in
        julia) engine=julia_pauli ;;
        julia-majorana) engine=julia_majorana ;;
        monoprop-majorana) engine=monoprop ;;
    esac
    printf '{"engine": "%s", "basis": "%s", "timeout": true, "num_qubits": %d, "layers": %d, "cutoff": %d, "max_rounds": %d, "ceiling_seconds": %d, "host": "%s"}\n' \
        "$engine" "$basis" "$n" "$LAYERS" "$4" "$rounds" "$CEILING_S" "$(hostname)" >>"$OUT"
}

TIMEDOUT=0
run_point() {
    local engine="$1" n="$2" t0 rc tmo rounds="$MAX_ROUNDS" lx ly per_layer=() cutoff="$CUTOFF"
    local timing=(--rounds "$rounds" --time-limit "$CEILING_S")
    tmo=$((CEILING_S + SETUP_S))
    [[ "$SWEEP" == perlayer ]] && per_layer=(--per-layer)
    if [[ "$engine" == *majorana ]]; then
        cutoff="$HUBBARD_CUTOFF"
        [[ -n "${LATTICE[$n]:-}" ]] || { log "FATAL no lattice for N=$n"; return 1; }
        read -r lx ly <<<"${LATTICE[$n]}"
    fi
    if [[ -f "$OUT" ]] && grep -Eq "\"num_qubits\": ?${n}([,}])" "$OUT"; then
        log "SKIP  $engine N=$n (already present)"
        return 0
    fi
    log "START $engine N=$n max rounds=$rounds (timed ${CEILING_S}s, kill at ${tmo}s)"
    t0=$SECONDS
    case "$engine" in
        monoprop)
            "${PIN[@]}" timeout "$tmo" "$PYTHON" "$PLOTS/scripts/monoprop_single_layer.py" \
                --basis pauli --num-qubits "$n" --cutoff "$cutoff" --layers "$LAYERS" \
                --lower-atol "$ATOL" "${timing[@]}" --out "$OUT"
            ;;
        monoprop-majorana)
            "${PIN[@]}" timeout "$tmo" "$PYTHON" "$PLOTS/scripts/monoprop_single_layer.py" \
                --basis majorana --model hubbard2d --lx "$lx" --ly "$ly" --num-qubits "$n" \
                --cutoff "$cutoff" --layers "$LAYERS" --lower-atol "$ATOL" "${timing[@]}" \
                --out "$OUT"
            ;;
        ppvm)
            "${PIN[@]}" timeout "$tmo" "$PPVM_PYTHON" "$PLOTS/scripts/ppvm_single_layer.py" \
                --num-qubits "$n" --cutoff "$cutoff" --layers "$LAYERS" \
                --lower-atol "$ATOL" "${timing[@]}" --out "$OUT"
            ;;
        julia)
            # --gcthreads=1,0 is mandatory: Julia defaults GC threads to ncpu/2, all contending
            # for the one pinned CPU, and JULIA_NUM_THREADS does not control them.
            "${PIN[@]}" timeout "$tmo" "$JULIA" --project="$PLOTS/scripts/julia-pauli" \
                --startup-file=no --history-file=no --banner=no --gcthreads=1,0 \
                "$PLOTS/scripts/julia_pauli_single_layer.jl" \
                --num-qubits "$n" --cutoff "$cutoff" --layers "$LAYERS" \
                --lower-atol "$ATOL" "${timing[@]}" "${per_layer[@]}" --out "$OUT"
            ;;
        julia-majorana)
            "${PIN[@]}" timeout "$tmo" "$JULIA" --project="$PLOTS/scripts/julia-majorana" \
                --startup-file=no --history-file=no --banner=no --gcthreads=1,0 \
                "$PLOTS/scripts/julia_majorana_hubbard2d.jl" \
                --lx "$lx" --ly "$ly" --cutoff "$cutoff" --layers "$LAYERS" \
                "${timing[@]}" "${per_layer[@]}" --out "$OUT"
            ;;
        *)
            log "FATAL unknown engine $engine"
            return 1
            ;;
    esac
    rc=$?
    if ((rc == 124)); then
        log "TIMEOUT $engine N=$n after ${tmo}s (lower-bound row written)"
        timeout_row "$engine" "$n" "$rounds" "$cutoff"
        TIMEDOUT=$((TIMEDOUT + 1))
    elif ((rc != 0)); then
        log "FAIL  $engine N=$n rc=$rc"
    else
        log "DONE  $engine N=$n wall=$((SECONDS - t0))s"
    fi
    return 0
}

for n in "${NS[@]}"; do
    run_point "$ENGINE" "$n"
done
rm -rf "$TMPDIR"
log "done (timeouts: $TIMEDOUT)"
