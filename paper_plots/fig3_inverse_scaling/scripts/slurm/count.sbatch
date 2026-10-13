#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
#
# The work-per-gate strip of Fig. 3: for every point of the timed sweeps, the terms that
# anticommute with each gate of the last layer (scripts/count_anticommuting.py). Untimed, so it
# may run on any node and need not share the timing campaign's allocation. Submit from
# paper_plots/fig3_inverse_scaling/:
#
#     sbatch -A <account> -p <partition> -t 03:00:00 --chdir="$PWD" \
#         scripts/slurm/count.sbatch /path/to/site.env
#
# The counter holds the exported operator as Python tuples, ~1 KB per term at its peak, so the
# points above ~1.5e7 terms (BIG) run one at a time while the rest share the node. A point above
# ~2e8 terms does not fit a 223 GiB node at all and is left out (SKIP).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SITE="${1:?usage: count.sbatch <site.env>}"
export FIG3_RUN="${FIG3_RUN:-count-${SLURM_JOB_ID:-manual}}"
# shellcheck source=/dev/null
source "$SITE"
if [[ -n "${SITE_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$SITE_SETUP"
fi
: "${PYTHON:?} ${RUN_DIR:?}"

ROWS="${ROWS:-pauli majorana}"
NS="${NS:-32 64 128 256 512 1024}"
PAULI_LAYERS="${PAULI_LAYERS:-5 7 9}"
HUBBARD_LAYERS="${HUBBARD_LAYERS:-2 3 4}"
MAX_N="${MAX_N-}"
# "<row>:<layers>:<N>" points by size: nine kicked-Ising layers hold 7.9e7 terms at N=1024, four
# Hubbard steps 9.3e7.
BIG="${BIG:-pauli:7:1024 pauli:9:256 pauli:9:512 pauli:9:1024 majorana:3:512 majorana:3:1024 majorana:4:512 majorana:4:1024}"
SKIP="${SKIP:-}"
SMALL_WORKERS="${SMALL_WORKERS:-3}"
HUBBARD_CUTOFF="${HUBBARD_CUTOFF:-3}"
declare -A LATTICE=([32]="4 4" [64]="8 4" [128]="8 8" [256]="16 8" [512]="16 16" [1024]="32 16")

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 monoprop_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
mkdir -p "$RUN_DIR/count"

layers_for() {
    case "$1" in
        pauli) echo "$PAULI_LAYERS" ;;
        majorana) echo "$HUBBARD_LAYERS" ;;
    esac
}

max_n() {
    local spec
    for spec in $MAX_N; do
        [[ "${spec%:*}" == "$1:$2" ]] && { echo "${spec##*:}"; return; }
    done
    echo 1000000
}

count_point() {
    local row="$1" layers="$2" n="$3" lx ly out t0
    out="$RUN_DIR/count/${row}_L${layers}_N${n}.jsonl"
    if [[ -s "$out" ]]; then
        echo "[count] SKIP $row L=$layers N=$n (present)"
        return
    fi
    t0=$SECONDS
    echo "[count] START $row L=$layers N=$n $(date -u +%H:%M:%S)"
    if [[ "$row" == majorana ]]; then
        read -r lx ly <<<"${LATTICE[$n]}"
        "$PYTHON" scripts/count_anticommuting.py --basis majorana --num-qubits "$n" \
            --lx "$lx" --ly "$ly" --cutoff "$HUBBARD_CUTOFF" --layers "$layers" --out "$out"
    else
        "$PYTHON" scripts/count_anticommuting.py --basis pauli --num-qubits "$n" \
            --cutoff 6 --layers "$layers" --out "$out"
    fi
    echo "[count] $([[ $? == 0 ]] && echo DONE || echo FAIL) $row L=$layers N=$n wall=$((SECONDS - t0))s"
}

big=() small=()
for row in $ROWS; do
    for layers in $(layers_for "$row"); do
        for n in $NS; do
            ((n <= $(max_n "$row" "$layers"))) || continue
            [[ " $SKIP " == *" $row:$layers:$n "* ]] && continue
            if [[ " $BIG " == *" $row:$layers:$n "* ]]; then big+=("$row $layers $n"); else small+=("$row $layers $n"); fi
        done
    done
done
echo "[count] host=$(hostname) job=${SLURM_JOB_ID:-?} big=${#big[@]} small=${#small[@]} out=$RUN_DIR/count"

# The big points one at a time, beside the small ones.
(for item in "${big[@]}"; do count_point $item; done) &
printf '%s\n' "${small[@]}" | xargs -P "$SMALL_WORKERS" -I{} bash -c "$(declare -f count_point); $(declare -p LATTICE); RUN_DIR='$RUN_DIR' PYTHON='$PYTHON' HUBBARD_CUTOFF='$HUBBARD_CUTOFF' count_point {}"
wait
echo "[count] done $(date -u --iso-8601=seconds)"
