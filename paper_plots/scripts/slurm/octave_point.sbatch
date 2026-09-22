#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#
# One work item of the Fig. 6 octave sweep: one engine at one or more N, on a whole node.
# Submit through submit_octave.sh, which supplies --partition, --chdir (paper_plots/) and the
# log paths; by hand:
#
#     sbatch --chdir=paper_plots -J mono-1024 scripts/slurm/octave_point.sbatch \
#         /path/to/site.env 7 monoprop 1024
#
# PARAMETERS ARRIVE AS POSITIONAL ARGUMENTS, NEVER VIA --export. Naming a variable in
# --export makes Slurm rebuild the user environment by running a login shell on the node;
# where compute nodes have no home directory (AWS PCS) the job is then HELD with
# Reason=user_env_retrieval_failed_requeued_held and shows as "PD 0:00" indefinitely.
#
# Every point runs under a hard `timeout`: at five layers the PauliPropagation.jl N=1024
# point ran for five hours without finishing, ~38x off the trend of its own N <= 512 points,
# and held a sequential sweep hostage. A point that exceeds its ceiling is killed and is
# simply absent; the job still exits 0.
set -uo pipefail

SITE="${1:?usage: octave_point.sbatch <site.env> <layers> <engine> <N> [N...]}"
LAYERS="${2:?missing <layers>}"
ENGINE="${3:?missing <engine>}"
shift 3
NS=("$@")
((${#NS[@]})) || { echo "FATAL: no N values given" >&2; exit 2; }
# shellcheck source=/dev/null
source "$SITE"
: "${PYTHON:?} ${PPVM_PYTHON:?} ${JULIA:?} ${RUN_DIR:?}"

PLOTS="$PWD" # --chdir points at paper_plots/
CUTOFF=6
ATOL=0
ROUNDS_MONOPROP=10
ROUNDS_REF=1

# One writer per file: concurrent appends from several nodes to one file over NFS can
# interleave. The file is named for this job's own N set; merge afterwards.
SUFFIX="$(
    IFS=-
    echo "${NS[*]}"
)"
OUT="$RUN_DIR/${ENGINE}_N${SUFFIX}.jsonl"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "engine=$ENGINE N=${NS[*]} layers=$LAYERS cutoff=$CUTOFF"
log "host=$(hostname) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} out=$OUT"
mkdir -p "$RUN_DIR" || { echo "FATAL: cannot create $RUN_DIR" >&2; exit 1; }
echo "$ENGINE N=${NS[*]} layers=$LAYERS host=$(hostname) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} $(date -u --iso-8601=seconds)" \
    >>"$RUN_DIR/provenance.txt"

if [[ -n "${JOB_HOME:-}" ]]; then
    export HOME="$JOB_HOME"
    export XDG_CACHE_HOME="$HOME/.cache" MPLCONFIGDIR="$HOME/.mpl"
    mkdir -p "$HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
fi
export TMPDIR="/tmp/octave.${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
[[ -n "${JULIA_DEPOT_PATH:-}" ]] && export JULIA_DEPOT_PATH

# The drivers record these verbatim into each row, and the figure refuses anything but 1.
export monoprop_NUM_THREADS=1 RAYON_NUM_THREADS=1 JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

# Pin the timed process to one physical core, away from CPU 0 (boot CPU and default IRQ
# target), and move the batch shell off it. Slurm may report a flat fake topology, so
# --hint=nomultithread cannot be relied on to do this.
mapfile -t PRIMARY < <(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '!seen[$2]++ {print $1}')
NCORES=${#PRIMARY[@]}
if ((NCORES > 8)); then PIN_CPU=${PRIMARY[8]}; else PIN_CPU=${PRIMARY[$((NCORES / 2))]}; fi
log "cores=$NCORES pinning to CPU $PIN_CPU"
taskset -cp 1 $$ >/dev/null 2>&1 || true
PIN=(srun --ntasks=1 --unbuffered "--cpu-bind=map_cpu:${PIN_CPU}")

# Per-point ceiling, ~3x the 7-layer wall time (monoprop's is x10 for its 10 rounds),
# floored at 600 s. Julia's are deliberately loose: it has already left its own trend once.
ceiling() {
    case "$1:$2" in
        monoprop:1024) echo 1800 ;;
        monoprop:512) echo 900 ;;
        monoprop:*) echo 600 ;;
        ppvm:1024) echo 9000 ;;
        ppvm:512) echo 2700 ;;
        ppvm:*) echo 900 ;;
        julia:512) echo 7200 ;;
        julia:*) echo 1800 ;;
        *) echo 600 ;;
    esac
}

TIMEDOUT=0
run_point() {
    local engine="$1" n="$2" t0 rc tmo
    tmo="$(ceiling "$engine" "$n")"
    if [[ -f "$OUT" ]] && grep -Eq "\"num_qubits\": ?${n}([,}])" "$OUT"; then
        log "SKIP  $engine N=$n (already present)"
        return 0
    fi
    log "START $engine N=$n (ceiling ${tmo}s)"
    t0=$SECONDS
    case "$engine" in
        monoprop)
            "${PIN[@]}" timeout "$tmo" "$PYTHON" "$PLOTS/scripts/monoprop_single_layer.py" \
                --basis pauli --num-qubits "$n" --cutoff "$CUTOFF" --layers "$LAYERS" \
                --lower-atol "$ATOL" --rounds "$ROUNDS_MONOPROP" --out "$OUT"
            ;;
        ppvm)
            "${PIN[@]}" timeout "$tmo" "$PPVM_PYTHON" "$PLOTS/scripts/ppvm_single_layer.py" \
                --num-qubits "$n" --cutoff "$CUTOFF" --layers "$LAYERS" \
                --lower-atol "$ATOL" --rounds "$ROUNDS_REF" --out "$OUT"
            ;;
        julia)
            # --gcthreads=1,0 is mandatory: Julia >= 1.10 defaults GC threads to ncpu/2, all
            # contending for the one pinned CPU, and JULIA_NUM_THREADS does not control them.
            "${PIN[@]}" timeout "$tmo" "$JULIA" --project="$PLOTS/scripts" --startup-file=no \
                --history-file=no --banner=no --gcthreads=1,0 \
                "$PLOTS/scripts/julia_pauli_single_layer.jl" \
                --num-qubits "$n" --cutoff "$CUTOFF" --layers "$LAYERS" \
                --lower-atol "$ATOL" --rounds "$ROUNDS_REF" --out "$OUT"
            ;;
        *)
            log "FATAL unknown engine $engine"
            return 1
            ;;
    esac
    rc=$?
    if ((rc == 124)); then
        log "TIMEOUT $engine N=$n after ${tmo}s (point omitted)"
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
