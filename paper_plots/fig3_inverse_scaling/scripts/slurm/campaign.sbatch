#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --open-mode=append
#
# Every Fig. 3 series in ONE allocation on one node, so that every curve and exponent comes from
# the same machine. Submit from paper_plots/fig3_inverse_scaling/:
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
NS="${NS:-32 64 128 256 512 1024}"
# Depths: small, medium and large, in layers (kicked Ising) or Trotter steps (Hubbard).
PAULI_LAYERS="${PAULI_LAYERS:-5 7 9}"
HUBBARD_LAYERS="${HUBBARD_LAYERS:-2 3 4}"
# "<row>:<layers>:<max N>" caps for a depth that does not fit a node at the largest N; none
# for the figure (set it empty, not unset, to keep none).
MAX_N="${MAX_N-}"
# "<engine>:<layers>:<N>=<reason>" points recorded as timeouts without running, for an engine
# that already exceeded the limit on a shallower circuit of the same N.
IMPLIED="${IMPLIED:-}"
# "<engine>:<layers>:<N><<engine>:<layers>:<N>" the same, decided in flight: the first point
# runs after the second has finished, and only if the second did not time out.
IMPLIED_IF="${IMPLIED_IF:-}"
# Run directories whose rows already settle a point under this limit (settled.py): those
# points are not rerun, and the merge takes their rows from there.
SETTLED_IN="${SETTLED_IN:-}"
export CEILING_S="${CEILING_S:-1800}"

POINT="${POINT:-$PWD/scripts/slurm/octave_point.sbatch}"

engines_for() {
    case "$1" in
        pauli) echo "monoprop julia ppvm" ;;
        majorana) echo "monoprop-majorana julia-majorana" ;;
    esac
}

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

implied() {
    local spec
    for spec in $IMPLIED; do
        [[ "${spec%%=*}" == "$1:$2:$3" ]] && { echo "${spec#*=}"; return; }
    done
}

prerequisite() {
    local spec
    for spec in $IMPLIED_IF; do
        [[ "${spec%%<*}" == "$1:$2:$3" ]] && { echo "${spec#*<}"; return; }
    done
}

# Wait for a queued prerequisite to finish; print the reason to skip if it timed out.
after() {
    local engine layers n out
    IFS=: read -r engine layers n <<<"$1"
    grep -q " $layers $engine $n\$" "$QUEUE.all" || return 0
    out="$RUN_DIR/nsweep/${engine}_L${layers}_N${n}.jsonl"
    # A prerequisite that failed writes nothing; past its own kill time, run the point anyway.
    local deadline=$((SECONDS + CEILING_S + ${SETUP_S:-600} + 600))
    until [[ -s "$out" ]] || ((SECONDS > deadline)); do sleep 30; done
    # Exceeded: killed at the limit, or its one round finished past it.
    if [[ -s "$out" ]] && "$PYTHON" -c 'import json, sys
r = json.loads(open(sys.argv[1]).readline())
sys.exit(0 if r.get("timeout") or r["seconds"] > float(sys.argv[2]) else 1)' "$out" "$CEILING_S"; then
        echo "exceeded-at-depth-${layers}"
    fi
}

# A rough relative cost, only to order the queue: references pay ~K per gate, monoprop ~K/N,
# K grows ~N at fixed depth and steeply with depth.
estimate() {
    local engine="$1" n="$2" layers="$3"
    case "$engine" in
        monoprop*) echo $((n * 10 * layers * layers)) ;;
        *) echo $((n * n * layers * layers)) ;;
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
for row in $ROWS; do
    for layers in $(layers_for "$row"); do
        cap="$(max_n "$row" "$layers")"
        for n in $NS; do
            ((n <= cap)) || continue
            for engine in $(engines_for "$row"); do
                # A conditional point goes last, after the point it waits for.
                est="$(estimate "$engine" "$n" "$layers")"
                [[ -n "$(prerequisite "$engine" "$layers" "$n")" ]] && est=0
                echo "$est $layers $engine $n"
            done
        done
    done
done | "$PYTHON" scripts/slurm/settled.py --ceiling "$CEILING_S" $SETTLED_IN |
    sort -k1,1nr >"$QUEUE"
cp "$QUEUE" "$QUEUE.all"

pop() {
    # Take the head of the queue under a lock; empty output means the queue is drained.
    flock "$QUEUE.lock" bash -c 'head -n1 "$1"; sed -i 1d "$1"' _ "$QUEUE"
}

worker() {
    local cpu="$1" item est layers engine n why
    while item="$(pop)" && [[ -n "$item" ]]; do
        read -r est layers engine n <<<"$item"
        why="$(implied "$engine" "$layers" "$n")"
        if [[ -z "$why" && -n "$(prerequisite "$engine" "$layers" "$n")" ]]; then
            why="$(after "$(prerequisite "$engine" "$layers" "$n")")"
        fi
        IMPLIED_TIMEOUT="$why" PIN_CPU="$cpu" bash "$POINT" "$SITE" nsweep "$layers" "$engine" "$n"
    done
}

echo "[campaign] host=$(hostname) job=${SLURM_JOB_ID:-?} workers on CPUs ${SLOTS[*]}" \
    "points=$(wc -l <"$QUEUE") $(date -u --iso-8601=seconds)"
for cpu in "${SLOTS[@]}"; do
    worker "$cpu" &
done
wait
rm -f "$QUEUE" "$QUEUE.lock" "$QUEUE.all"
echo "[campaign] done $(date -u --iso-8601=seconds)"
