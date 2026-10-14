#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
#
# The work behind the cost panels of Fig. 3: for every point of the timed sweeps, the terms the
# operator holds when each gate is applied, summed over the run (scripts/count_held.py). Untimed,
# so it may run on any node. Submit from paper_plots/fig3_inverse_scaling/:
#
#     sbatch -A <account> -p <partition> -t 02:00:00 --chdir="$PWD" \
#         scripts/slurm/held.sbatch /path/to/site.env
#
# monoprop holds ~160 B per term, so even the largest point (3.7e8 terms) fits beside the others;
# the points run WORKERS at a time, largest first.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SITE="${1:?usage: held.sbatch <site.env>}"
export FIG3_RUN="${FIG3_RUN:-held-${SLURM_JOB_ID:-manual}}"
# shellcheck source=/dev/null
source "$SITE"
if [[ -n "${SITE_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$SITE_SETUP"
fi
: "${PYTHON:?} ${RUN_DIR:?}"

PAULI_NS="${PAULI_NS:-32 64 128 256 512 1024}"
HUBBARD_NS="${HUBBARD_NS:-32 72 128 242 512 968}"
PAULI_LAYERS="${PAULI_LAYERS:-7 10 13}"
HUBBARD_LAYERS="${HUBBARD_LAYERS:-3 4 5}"
WORKERS="${WORKERS:-4}"
HUBBARD_CUTOFF="${HUBBARD_CUTOFF:-3}"
HUBBARD_CUTOFF_TYPE="${HUBBARD_CUTOFF_TYPE:-support}"
HUBBARD_BOUNDARY="${HUBBARD_BOUNDARY:-open}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 monoprop_NUM_THREADS=1 monoprop_PARTITIONS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export RUN_DIR PYTHON HUBBARD_CUTOFF HUBBARD_CUTOFF_TYPE HUBBARD_BOUNDARY
mkdir -p "$RUN_DIR/held"

held_point() {
    local row="$1" layers="$2" n="$3" side out t0
    out="$RUN_DIR/held/${row}_L${layers}_N${n}.jsonl"
    [[ -s "$out" ]] && { echo "[held] SKIP $row L=$layers N=$n (present)"; return; }
    t0=$SECONDS
    echo "[held] START $row L=$layers N=$n $(date -u +%H:%M:%S)"
    if [[ "$row" == majorana ]]; then
        # Square lattices: N = 2 L^2.
        side="$(awk -v n="$n" 'BEGIN {print int(sqrt(n / 2) + 0.5)}')"
        "$PYTHON" scripts/count_held.py --basis majorana --num-qubits "$n" --lx "$side" --ly "$side" \
            --cutoff "$HUBBARD_CUTOFF" --cutoff-type "$HUBBARD_CUTOFF_TYPE" \
            --boundary "$HUBBARD_BOUNDARY" --layers "$layers" --out "$out"
    else
        "$PYTHON" scripts/count_held.py --basis pauli --num-qubits "$n" --cutoff 6 \
            --layers "$layers" --out "$out"
    fi
    echo "[held] $([[ $? == 0 ]] && echo DONE || echo FAIL) $row L=$layers N=$n wall=$((SECONDS - t0))s"
}
export -f held_point

# Largest first: the cost grows with N and steeply with depth.
{
    for l in $PAULI_LAYERS; do for n in $PAULI_NS; do echo "$((n * l * l)) pauli $l $n"; done; done
    for l in $HUBBARD_LAYERS; do for n in $HUBBARD_NS; do echo "$((n * l * l * 4)) majorana $l $n"; done; done
} | sort -k1,1nr | cut -d' ' -f2- >"$RUN_DIR/held/queue.txt"
echo "[held] host=$(hostname) job=${SLURM_JOB_ID:-?} points=$(wc -l <"$RUN_DIR/held/queue.txt") out=$RUN_DIR/held"
xargs -P "$WORKERS" -L1 bash -c 'held_point "$@"' _ <"$RUN_DIR/held/queue.txt"
rm -f "$RUN_DIR/held/queue.txt"
echo "[held] done $(date -u --iso-8601=seconds)"
