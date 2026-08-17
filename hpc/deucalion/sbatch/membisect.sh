#!/bin/bash -l
# Attribute a branch's MEMORY across the commits that make it up, and its time in the same runs.
#
# The other half of a time bisection, and the half that is usually missing. Without a memory column
# there is no way to say which commit is worth its time, so a campaign ends up arguing "ship all of it
# or none of it" over what is really a per-commit Pareto choice.
#
#   ARMS_FILE=$PROJ/runs/mycampaign/arms sbatch hpc/deucalion/sbatch/membisect.sh
#
# ARMS_FILE lines are `<arm> <worktree> [KEY=VAL ...]`; `#` comments and blank lines are ignored.
# `<worktree>` may be absolute or relative to $PROJ/src. Trailing KEY=VAL pairs are exported for that
# arm only, which is how one tree provides several arms differing by an env knob:
#
#     main         mp-main
#     a1-index     mp-a1-index
#     candidate    mp-s2
#     p016         mp-profbr   monoprop_INVIDX_BITMAP_PREMIUM=16
#     pinf         mp-profbr   monoprop_INVIDX_BITMAP_PREMIUM=1000000
#
# Env knobs: ARMS_FILE (required), CELLS_FILE, CELLS, REPS, RANKS, PARTITIONS, PROFILE, LABEL, OP.
#
# STATE THE `P`. RANKS x PARTITIONS is the engine's flat world, and results here do NOT transfer
# across it: R=8 once understated a commit's time cost by 3x, and a peak-RSS regression that was
# +7 B/term at R=8 did not exist at R=1 AT ALL -- so a job run at the convenient P measured a fix in a
# regime without the bug. Run both, quote both, and confirm a defect at the P where it was observed.
#
#SBATCH --job-name=mp-membisect
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/deucalion/env.sh

# Resolve siblings relative to the SUBMISSION directory, not to $BASH_SOURCE. sbatch STAGES the batch
# script: under Slurm, ${BASH_SOURCE[0]} is a copy in /var/spool/slurm/d/job<id>/, so anything resolved
# from it lands outside the checkout entirely. This exact bug refused this script on its first run --
# which is the guard working, but the fix is to use the same `hpc/deucalion/...` relative form every
# other script here uses.
HPC="$PWD/hpc/deucalion"
[ -d "$HPC" ] || { echo "refusing: no hpc/deucalion under $PWD -- submit from the harness checkout" >&2; exit 2; }
LABEL="${LABEL:-membisect}"
OUTDIR="$MONOPROP_RUNS/$LABEL-${SLURM_JOB_ID:-manual}"
mkdir -p "$OUTDIR"

RANKS_PER_NODE=${RANKS:-8}
PARTITIONS=${PARTITIONS:-16}
# The ledger is deterministic -- same terms, same allocations -- so a rep is a check on the harness,
# not on variance. Peak RSS and elapsed carry allocator and node jitter, which is what reps are for.
# Budget them against what a sign test can resolve: 3 reps is p=0.25, 4 is p=0.125, 6 is p=0.031.
REPS=${REPS:-2}
OP=${OP:-propagate}

export monoprop_NUM_THREADS=$PARTITIONS
export MALLOC_ARENA_MAX=$PARTITIONS
export OMP_NUM_THREADS=1
export monoprop_MAX_NUM_MODES=1024
unset monoprop_LAYER_PROFILE
unset monoprop_LAYER_GAPS

[ -n "${ARMS_FILE:-}" ] || { echo "refusing: set ARMS_FILE (see the header of this script)" >&2; exit 2; }
[ -r "$ARMS_FILE" ] || { echo "refusing: cannot read ARMS_FILE=$ARMS_FILE" >&2; exit 2; }

ARMS=()
declare -A TREE ENVS
while read -r arm tree rest; do
    case "$arm" in ''|'#'*) continue ;; esac
    [ -n "$tree" ] || { echo "refusing: arm '$arm' has no worktree" >&2; exit 2; }
    case "$tree" in /*) ;; *) tree="$PROJ/src/$tree" ;; esac
    ARMS+=("$arm")
    TREE[$arm]="$tree"
    ENVS[$arm]="$rest"
done < "$ARMS_FILE"
[ "${#ARMS[@]}" -gt 0 ] || { echo "refusing: ARMS_FILE defined no arms" >&2; exit 2; }

# An out-of-range env knob SILENTLY DEFAULTS. parse_positive_int (detail/EnvConfig.h) rejects values
# above 1'000'000 and then does .value_or(<default>), so an arm asking for 1048576 quietly runs the
# SHIPPED rule and comes back byte-identical to the default arm. There is no diagnostic for it in the
# engine; this is the diagnostic.
for arm in "${ARMS[@]}"; do
    for kv in ${ENVS[$arm]}; do
        key=${kv%%=*}; val=${kv#*=}
        case "$key" in monoprop_*) ;; *) continue ;; esac
        case "$val" in
            ''|*[!0-9]*) continue ;;  # not a plain integer, nothing to range-check
            *) [ "$val" -le 1000000 ] || {
                   echo "refusing: $arm sets $key=$val, above parse_positive_int's 1000000 cap --" \
                        "the engine would silently fall back to the default and the arm would" \
                        "measure nothing" >&2; exit 2; } ;;
        esac
    done
done

CELLS_FILE="${CELLS_FILE:-$HPC/cells/100m.cells}"
[ -r "$CELLS_FILE" ] || { echo "refusing: cannot read CELLS_FILE=$CELLS_FILE" >&2; exit 2; }
CELL_LIST=()
declare -A MODEL WANT OVERRIDES
while read -r cell model want rest; do
    case "$cell" in ''|'#'*) continue ;; esac
    CELL_LIST+=("$cell")
    MODEL[$cell]="$model"
    WANT[$cell]="$want"
    OVERRIDES[$cell]="$rest"
done < "$CELLS_FILE"

# Overridable so a wide sweep can run the ONE cell that carries the effect and still fit the walltime.
# Every cell is the default and stays the default: a one-cell reading is a point, not a frontier, and
# two points are not a trend.
if [ -n "${CELLS:-}" ]; then
    IFS=' ' read -r -a CELL_LIST <<< "$CELLS"
fi
for cell in "${CELL_LIST[@]}"; do
    [ -n "${MODEL[$cell]:-}" ] || { echo "refusing: cell '$cell' is not in $CELLS_FILE" >&2; exit 2; }
done

for arm in "${ARMS[@]}"; do
    [ -x "${TREE[$arm]}/.venv/bin/python" ] \
        || { echo "refusing: no venv for arm $arm at ${TREE[$arm]}" >&2; exit 2; }
    [ -d "${TREE[$arm]}/benches" ] \
        || { echo "refusing: arm $arm has no benches/ -- _builders.py supplies MODELS" >&2; exit 2; }
done

# ARM IDENTITY IS THE INSTALLED _core.so's md5 AND NOTHING ELSE.
#
# monoprop.__version__ reports HEAD for every arm (editable installs resolve the Python layer to one
# live source tree), and the dist-info stamp is written at install time and never rewritten by a later
# cmake rebuild -- so both version identifiers go stale TOGETHER. site-packages holds a COPY of
# _core.so, so this hashes what will actually be imported, not the build dir; gating on the build dir
# once measured the previous day's binary for a whole job.
#
# EXACTLY ONE match, not `head -1`: scipy ships optimize/_highspy/_core.cpython-*.so, and a loose glob
# has made two genuinely different arms compare byte-identical -- a false negative in exactly the
# direction this check exists to catch.
declare -A MD5
for arm in "${ARMS[@]}"; do
    mapfile -t sos < <(ls "${TREE[$arm]}"/.venv/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null)
    [ "${#sos[@]}" -eq 1 ] \
        || { echo "refusing: arm $arm matched ${#sos[@]} monoprop/_core*.so files, need exactly 1" >&2; exit 2; }
    MD5[$arm]=$(md5sum "${sos[0]}" | cut -d' ' -f1)
    echo "arm=$arm md5=${MD5[$arm]} tree=${TREE[$arm]} env='${ENVS[$arm]}'"
done

# Arms sharing a worktree MUST be the same binary (they differ by an env knob, so a distinct md5 means
# one is a stale build and the "sweep" is comparing code, not the knob). Arms in different worktrees
# MUST differ (equal md5 means a duplicate arm wearing two labels, or a build that never happened).
for a in "${ARMS[@]}"; do
    for b in "${ARMS[@]}"; do
        [[ "$a" < "$b" ]] || continue
        if [ "${TREE[$a]}" = "${TREE[$b]}" ]; then
            [ "${MD5[$a]}" = "${MD5[$b]}" ] || {
                echo "refusing: $a and $b share $(basename "${TREE[$a]}") but have different md5s --" \
                     "one is a stale build" >&2; exit 2; }
        else
            [ "${MD5[$a]}" != "${MD5[$b]}" ] || {
                echo "refusing: $a and $b are different trees with the SAME md5 -- a duplicate arm," \
                     "or a build that did not happen" >&2; exit 2; }
        fi
    done
done
echo "preflight ok: ${#ARMS[@]} arms, ${#CELL_LIST[@]} cells, reps=$REPS ranks=$RANKS_PER_NODE partitions=$PARTITIONS op=$OP"

for cell in "${CELL_LIST[@]}"; do
    for rep in $(seq 1 "$REPS"); do
        # Rotate the order every rep. One allocation, interleaved arms: the two arms of a rep run back
        # to back on the same node, so a node-state swing hits both and cancels in the per-rep ratio.
        # Taking each side's median first spends the entire point of interleaving.
        order=()
        n=${#ARMS[@]}
        for ((j = 0; j < n; ++j)); do order+=("${ARMS[$(((j + rep) % n))]}"); done
        for arm in "${order[@]}"; do
            out="$OUTDIR/${cell}_rep${rep}_${arm}.log"
            # CALIB_BENCHES points at the arm's OWN benches/: _builders.py is versioned alongside the
            # C++, so another tree's builders would silently profile a different problem.
            env_args=(CALIB_BENCHES="${TREE[$arm]}/benches")
            for kv in ${ENVS[$arm]}; do env_args+=("$kv"); done
            # LAYERPROF goes to stderr and the redirect below merges it into $out. Nothing here is
            # pytest, so there is no fd-level capture to defeat -- that trap belongs to the pytest path,
            # where a live instrument looks exactly like one that never fired without `-s`.
            [ "${PROFILE:-0}" = 1 ] && env_args+=(monoprop_LAYER_PROFILE=1)
            srun --mpi=pmix --ntasks=$RANKS_PER_NODE --ntasks-per-node=$RANKS_PER_NODE \
                 --cpus-per-task=$((128 / RANKS_PER_NODE)) \
                 --cpu-bind=none --distribution=block:block \
                 env "${env_args[@]}" \
                 "${TREE[$arm]}/.venv/bin/python" "$HPC/tools/prof_run.py" \
                 "${MODEL[$cell]}" "$OP" ${OVERRIDES[$cell]} \
                 > "$out" 2>&1 || echo "!! $cell/rep$rep/$arm rc=$? -- see $out"

            run=$(grep -m1 "^PROFRUN " "$out" || echo "PROFRUN MISSING")
            mem=$(grep -m1 "^PROFMEM " "$out" || echo "PROFMEM MISSING")
            echo "cell=$cell rep=$rep arm=$arm env='${ENVS[$arm]}' $run" | tee -a "$OUTDIR/RESULTS"
            echo "cell=$cell rep=$rep arm=$arm $mem" >> "$OUTDIR/RESULTS"

            # A stale driver silently costs the whole job its point: no ledger line, and a collator
            # that skips the file reads exactly like an arm that failed to run.
            grep -q "^PROFMEM terms=" "$out" \
                || echo "!! $cell/rep$rep/$arm: no PROFMEM -- STALE HARNESS, void" | tee -a "$OUTDIR/RESULTS"

            # Assert the SIZE. lower_atol is the size knob and the nominal axes saturate, so a
            # plausible config can land two decades below the cell it is named after. That has happened
            # here: these scripts first ran at 1.9M terms instead of 97M.
            got=$(sed -n 's/.*terms=\([0-9]*\).*/\1/p' <<< "$run" | head -1)
            if [ -n "$got" ] && [ "$got" -gt 0 ]; then
                lo=$((WANT[$cell] * 9 / 10)); hi=$((WANT[$cell] * 11 / 10))
                if [ "$got" -lt "$lo" ] || [ "$got" -gt "$hi" ]; then
                    echo "!! $cell/rep$rep/$arm: terms=$got outside [$lo,$hi] -- WRONG CELL SIZE, void" \
                        | tee -a "$OUTDIR/RESULTS"
                fi
            fi
        done
    done
done

echo "=== done, results in $OUTDIR ==="
echo "collate with:"
echo "  \$VENV/bin/python hpc/deucalion/tools/membisect_summary.py $OUTDIR"
[ "${PROFILE:-0}" = 1 ] && echo "  \$VENV/bin/python hpc/deucalion/tools/layerprof_summary.py --baseline <arm> --arm <arm> $OUTDIR"
exit 0
