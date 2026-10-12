#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
#
# Every Fig. 3 series in ONE allocation on one node, so that every curve, exponent and crossover
# comes from the same machine. Submit from paper_plots/fig3_inverse_scaling/:
#
#     sbatch -A <account> -p <partition> -t <walltime> --chdir="$PWD" \
#         scripts/slurm/campaign.sbatch /path/to/site.env
#
# The site file (see site.env.example) supplies the paths, and optionally SITE_SETUP (a file
# sourced once before the first point, e.g. a module/runtime setup) and the sweep knobs below.
#
# Points run CONCURRENTLY, one per NUMA domain: each worker owns one core, in a different domain
# from every other worker, so concurrent points share neither an L3 slice nor a memory
# controller and allocate node-local memory. The queue is ordered longest-first, so the points
# that run side by side are of similar length and the node is not held by a straggler.
# Concurrency is a measured choice, not an assumption: the pilot times the same point alone and
# beside a full set of neighbours, and the campaign runs packed only if the two agree.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SITE="${1:?usage: campaign.sbatch <site.env>}"
# shellcheck source=/dev/null
source "$SITE"
if [[ -n "${SITE_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$SITE_SETUP"
fi

ROWS="${ROWS:-pauli majorana}"
SWEEPS="${SWEEPS:-nsweep perlayer}"
NSWEEP_NS="${NSWEEP_NS:-32 64 128 256 512 1024}"
PERLAYER_NS="${PERLAYER_NS:-32 128 512}"
PAULI_LAYERS="${PAULI_LAYERS:-7}"
: "${HUBBARD_LAYERS:?set HUBBARD_LAYERS in the site file (sized by the pilot)}"
: "${PERLAYER_PAULI_LAYERS:?set PERLAYER_PAULI_LAYERS in the site file}"
: "${PERLAYER_HUBBARD_LAYERS:?set PERLAYER_HUBBARD_LAYERS in the site file}"

POINT="$PWD/scripts/slurm/octave_point.sbatch"

engines_for() {
    case "$1:$2" in
        pauli:nsweep) echo "monoprop julia ppvm" ;;
        pauli:perlayer) echo "monoprop julia" ;;
        majorana:*) echo "monoprop-majorana julia-majorana" ;;
    esac
}

layers_for() {
    case "$1:$2" in
        pauli:nsweep) echo "$PAULI_LAYERS" ;;
        pauli:perlayer) echo "$PERLAYER_PAULI_LAYERS" ;;
        majorana:nsweep) echo "$HUBBARD_LAYERS" ;;
        majorana:perlayer) echo "$PERLAYER_HUBBARD_LAYERS" ;;
    esac
}

# A rough relative cost, only to order the queue: references pay ~K per gate, monoprop ~K/N,
# and K grows ~N for a fixed depth. Monoprop's points are short, so they fill in at the end.
estimate() {
    local engine="$1" n="$2"
    case "$engine" in
        monoprop*) echo $((n * 10)) ;;
        *) echo $((n * n)) ;;
    esac
}

# One worker core per NUMA domain: the ninth core of each, away from CPU 0 and the domain's
# first CCX (where the batch shell and system daemons land).
mapfile -t SLOTS < <(lscpu -p=CPU,NODE | grep -v '^#' | sort -t, -k2,2n -k1,1n |
    awk -F, '{i[$2]++} i[$2] == 9 {print $1}')
((${#SLOTS[@]})) || SLOTS=(8)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-${#SLOTS[@]}}"
SLOTS=("${SLOTS[@]:0:$MAX_CONCURRENCY}")
export CONCURRENCY=${#SLOTS[@]}

QUEUE="$(mktemp "${TMPDIR:-/tmp}/fig3-queue.XXXXXX")"
for sweep in $SWEEPS; do
    [[ "$sweep" == nsweep ]] && ns="$NSWEEP_NS" || ns="$PERLAYER_NS"
    for row in $ROWS; do
        layers="$(layers_for "$row" "$sweep")"
        for n in $ns; do
            for engine in $(engines_for "$row" "$sweep"); do
                echo "$(estimate "$engine" "$n") $sweep $layers $engine $n"
            done
        done
    done
done | sort -k1,1nr >"$QUEUE"

pop() {
    # Take the head of the queue under a lock; empty output means the queue is drained.
    flock "$QUEUE.lock" bash -c 'head -n1 "$1"; sed -i 1d "$1"' _ "$QUEUE"
}

worker() {
    local cpu="$1" item est sweep layers engine n
    while item="$(pop)" && [[ -n "$item" ]]; do
        read -r est sweep layers engine n <<<"$item"
        PIN_CPU="$cpu" bash "$POINT" "$SITE" "$sweep" "$layers" "$engine" "$n"
    done
}

echo "[campaign] host=$(hostname) job=${SLURM_JOB_ID:-?} workers on CPUs ${SLOTS[*]}" \
    "points=$(wc -l <"$QUEUE") $(date -u --iso-8601=seconds)"
for cpu in "${SLOTS[@]}"; do
    worker "$cpu" &
done
wait
rm -f "$QUEUE" "$QUEUE.lock"
echo "[campaign] done $(date -u --iso-8601=seconds)"
