#!/bin/bash -l
# Extend the term-count sequence to S7..S10 for the multi-load scaling grid.
#
# ONE job for all four targets. Term count is invariant to the flat world (measured identical at
# five worlds in this campaign), so every probe can run at N=64 regardless of the size it is
# sizing -- which removes the sequential dependency four separate jobs would have, and puts every
# probe behind one md5 check in one allocation.
#
# Targets are exact powers of two times S6 (6,125,805,627, the measured weak/64 total), NOT
# 2x the previous MEASURED value: that would serialise the sweep for no benefit. The existing
# sequence is x1.90-2.07 rather than exactly x2, so the grid tolerates the drift by normalising
# every figure on MEASURED terms/node.
#
# WHY NOT MODEL IT. The atol->terms local exponent drifts DOWN across the measured range,
# 1.92 -> 1.60, so a single power law fitted anywhere undershoots when extrapolated. Candidates
# below bracket a prediction from the fine end with the exponent nudged down again per octave.
#
# Probes are ordered CHEAPEST FIRST, so a walltime overrun loses the largest target rather than
# all of them.
#
#   cd $PROJ/src/296-on-bench && sbatch -A eehpc-dev-2026d08-260x -p normal-x86 -N64 \
#       -t 1:15:00 --job-name=mp-296-calibseq $PROJ/scratch/pr296bench/calibseq.sh
#
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

PROJ=/projects/EEHPC-DEV-2026D08-260
PROBE="$PROJ/scratch/pr296bench/routing_probe.py"
SO_MD5=${SO_MD5:-1157a5e2b421fb8bd18fc6d16fa39778}
CUTOFF=10
RANKS_PER_NODE=8
PARTITIONS=16
S6=6125805627

# target_label  target_terms  atol_low  atol_high        (ascending size: cheapest probes first)
TARGETS="
S7  12251611254  7.35e-07 7.35e-07
S8  24503222508  4.68e-07 5.05e-07
S9  49006445016  2.94e-07 3.18e-07
S10 98012890032  1.82e-07 1.97e-07
"

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2
export MONOPROP_SRC="$PWD"                 # BEFORE env.sh, or VENV resolves to src/monoprop
source hpc/deucalion/env.sh
unset MALLOC_ARENA_MAX
export PYTHONPATH="$PWD/benches${PYTHONPATH:+:$PYTHONPATH}"
export monoprop_NUM_THREADS="$PARTITIONS" monoprop_PARTITIONS="$PARTITIONS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$((RANKS_PER_NODE * NODES))
TSV="$PROJ/scratch/pr296bench/calibseq-${SLURM_JOB_ID}.tsv"
printf 'target\ttarget_terms\tlower_atol\tterms\tvs_target\tgib_per_node_at_N64\tmax_over_mean\tranks_used\n' > "$TSV"

md5_now=$(md5sum .venv/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null | awk '{print $1}')
echo "=== calibseq  $NODES nodes, R=$NTASKS S=$PARTITIONS (term count is world-invariant)"
echo "=== md5       $md5_now"
if [ "$md5_now" != "$SO_MD5" ]; then
    echo "!! REFUSING: .so md5 is not the pinned arm ($SO_MD5)" >&2
    exit 4
fi
echo "=== S6        $S6 (measured); targets are 2x/4x/8x/16x of it"

probe() {                                  # probe <label> <target> <atol>
    local label=$1 target=$2 atol=$3
    local log="$PROJ/scratch/pr296bench/calibseq-${SLURM_JOB_ID}-${label}-${atol}.log"
    echo "######## $label  atol=$atol  target=$target ########"
    /usr/bin/time -v srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RANKS_PER_NODE" \
        --cpus-per-task=$((128 / RANKS_PER_NODE)) \
        --cpu-bind=cores --distribution=block:block \
        .venv/bin/python "$PROBE" --cutoff "$CUTOFF" --lower-atol "$atol" \
        --tag "calib-$label-$atol" > "$log" 2>&1 < /dev/null
    local rc=$?
    grep -a '^PROBE' "$log" | sed 's/^/   /'
    if [ "$rc" -ne 0 ]; then
        # A failure is a GAP, never a small term count. Record nothing rather than a wrong number.
        echo "!! $label atol=$atol failed rc=$rc" >&2
        grep -aiE 'error|killed|oom|bad_alloc|out of memory' "$log" | head -3 >&2
        return 0
    fi
    local terms mom used
    terms=$(grep -a '^PROBE' "$log" | sed -n 's/.*terms=\([0-9]*\).*/\1/p' | head -1)
    mom=$(grep -a '^PROBE' "$log" | sed -n 's/.*max_over_mean=\([0-9.]*\).*/\1/p' | head -1)
    used=$(grep -a '^PROBE' "$log" | sed -n 's/.*ranks_used=\([0-9]*\).*/\1/p' | head -1)
    [ -z "${terms:-}" ] && { echo "!! $label atol=$atol produced no term count" >&2; return 0; }
    awk -v l="$label" -v g="$target" -v a="$atol" -v t="$terms" -v m="${mom:-NA}" -v u="${used:-NA}" \
        'BEGIN{printf "%s\t%s\t%s\t%s\t%.4f\t%.1f\t%s\t%s\n", l, g, a, t, t/g, t*68/1073741824/64, m, u}' \
        >> "$TSV"
    return 0
}

while read -r label target lo hi <&3; do
    [ -z "${label:-}" ] && continue
    probe "$label" "$target" "$lo"
    # A refined single candidate is written as lo==hi; do not pay for the same probe twice.
    [ "$hi" != "$lo" ] && probe "$label" "$target" "$hi"
done 3<<< "$TARGETS"

echo
echo "=== $TSV ==="
column -t "$TSV" 2>/dev/null || cat "$TSV"

echo
echo "=== exponent check (a local exponent near zero means atol no longer buys terms) ==="
awk -F'\t' 'NR>1 && $4>0 {
    if ($1==p) {
        e = log(pt/$4) / log($3/pa)          # sign-free: works whichever way the pair is ordered
        printf "  %s: atol %s vs %s -> local exponent %.3f%s\n", $1, pa, $3, e,
            (e < 0.5 ? "   << SATURATING, stop the sequence here" : "")
    }
    p=$1; pt=$4; pa=$3
}' "$TSV"

cat <<'NOTE'

ACCEPT the candidate closest to its target and paste its EXACT term count into the rung tables.
Within ~10% of target is fine -- every figure normalises on MEASURED terms/node, so the drift is
reported rather than assumed. What must be exact is the per-rep gate value.

If a target SATURATES, stop the sequence there and shorten the affected curve. Do NOT raise the
cutoff to reach it: cutoff 10/12/14 give 96.98M/105.79M/106.82M terms at atol 1.25e-05, so the
cutoff is already saturated, and a curve at a different cutoff is a different operator.
NOTE
