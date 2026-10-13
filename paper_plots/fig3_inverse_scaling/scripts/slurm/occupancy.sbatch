#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
#
# How evenly the operator is spread over the anticommutation index, after every layer, for every
# N of Fig. 3 (scripts/occupancy.py). Untimed, so it may run on any node. One run per (row, N)
# goes to the deepest layer and records every layer on the way. Submit from
# paper_plots/fig3_inverse_scaling/:
#
#     sbatch -A <account> -p <partition> -t 02:00:00 --chdir="$PWD" \
#         scripts/slurm/occupancy.sbatch /path/to/site.env
#
# The export holds ~1 KB per term, so the runs that reach ~1e8 terms (BIG) go one at a time
# beside the rest, and a run stops at the first layer above MAX_TERMS.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SITE="${1:?usage: occupancy.sbatch <site.env>}"
export FIG3_RUN="${FIG3_RUN:-occupancy-${SLURM_JOB_ID:-manual}}"
# shellcheck source=/dev/null
source "$SITE"
if [[ -n "${SITE_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$SITE_SETUP"
fi
: "${PYTHON:?} ${RUN_DIR:?}"

ROWS="${ROWS:-pauli majorana}"
NS="${NS:-32 64 128 256 512 1024}"
PAULI_DEPTH="${PAULI_DEPTH:-13}"
HUBBARD_DEPTH="${HUBBARD_DEPTH:-5}"
HUBBARD_CUTOFF="${HUBBARD_CUTOFF:-3}"
HUBBARD_CUTOFF_TYPE="${HUBBARD_CUTOFF_TYPE:-support}"
HUBBARD_BOUNDARY="${HUBBARD_BOUNDARY:-open}"
MAX_TERMS="${MAX_TERMS:-1.6e8}"
BIG="${BIG:-pauli:256 pauli:512 pauli:1024 majorana:512 majorana:1024}"
SMALL_WORKERS="${SMALL_WORKERS:-3}"
declare -A LATTICE=([32]="4 4" [64]="8 4" [72]="6 6" [128]="8 8" [242]="11 11" [256]="16 8" [512]="16 16" [968]="22 22" [1024]="32 16")

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 monoprop_NUM_THREADS=1 monoprop_PARTITIONS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
mkdir -p "$RUN_DIR/occupancy"

occupancy_point() {
    local row="$1" n="$2" lx ly out t0
    out="$RUN_DIR/occupancy/${row}_N${n}.jsonl"
    if [[ -s "$out" ]]; then
        echo "[occupancy] SKIP $row N=$n (present)"
        return
    fi
    t0=$SECONDS
    echo "[occupancy] START $row N=$n $(date -u +%H:%M:%S)"
    if [[ "$row" == majorana ]]; then
        read -r lx ly <<<"${LATTICE[$n]}"
        "$PYTHON" scripts/occupancy.py --basis majorana --num-qubits "$n" --lx "$lx" --ly "$ly" \
            --cutoff "$HUBBARD_CUTOFF" --cutoff-type "$HUBBARD_CUTOFF_TYPE" --boundary "$HUBBARD_BOUNDARY" --layers "$HUBBARD_DEPTH" --max-terms "$MAX_TERMS" --out "$out"
    else
        "$PYTHON" scripts/occupancy.py --basis pauli --num-qubits "$n" --cutoff 6 \
            --layers "$PAULI_DEPTH" --max-terms "$MAX_TERMS" --out "$out"
    fi
    echo "[occupancy] $([[ $? == 0 ]] && echo DONE || echo FAIL) $row N=$n wall=$((SECONDS - t0))s"
}

big=() small=()
for row in $ROWS; do
    for n in $NS; do
        if [[ " $BIG " == *" $row:$n "* ]]; then big+=("$row $n"); else small+=("$row $n"); fi
    done
done
echo "[occupancy] host=$(hostname) job=${SLURM_JOB_ID:-?} big=${#big[@]} small=${#small[@]} out=$RUN_DIR/occupancy"

(for item in "${big[@]}"; do occupancy_point $item; done) &
printf '%s\n' "${small[@]}" | xargs -P "$SMALL_WORKERS" -I{} bash -c "$(declare -f occupancy_point); $(declare -p LATTICE); RUN_DIR='$RUN_DIR' PYTHON='$PYTHON' HUBBARD_CUTOFF='$HUBBARD_CUTOFF' HUBBARD_CUTOFF_TYPE='$HUBBARD_CUTOFF_TYPE' HUBBARD_BOUNDARY='$HUBBARD_BOUNDARY' PAULI_DEPTH='$PAULI_DEPTH' HUBBARD_DEPTH='$HUBBARD_DEPTH' MAX_TERMS='$MAX_TERMS' occupancy_point {}"
wait
echo "[occupancy] done $(date -u --iso-8601=seconds)"
