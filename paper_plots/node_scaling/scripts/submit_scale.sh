#!/bin/bash
# Submit named rungs of a scaleladder curve. Reads the walltime and partition from the ladder's
# OWN rung table, so a submit line can never disagree with the table the cell will validate
# against. Refuses a CALIB rung rather than queueing a job that will exit 2.
#
#   ./submit_scale.sh <LADDER> <nodes...>          # DRY_RUN=1 to print only
set -uo pipefail
PROJ=/projects/EEHPC-DEV-2026D08-260
TREE="$PROJ/src/296-on-bench"
SELF="$PROJ/scratch/pr296bench/scaleladder.sh"
ACCOUNT=eehpc-dev-2026d08-260x
LADDER=${1:?usage: submit_scale.sh <LADDER> <nodes...>}; shift
REPS_OVERRIDE=${REPS:-}

plan=$(LADDER="$LADDER" bash "$SELF" 2>/dev/null) || { echo "bad LADDER '$LADDER'" >&2; exit 2; }

for n in "$@"; do
    row=$(awk -v n="$n" '$1==n && NF>6 {print}' <<< "$plan")
    if [ -z "$row" ]; then echo "!! $LADDER N=$n is not a rung -- skipped" >&2; continue; fi
    read -r _n _p reps atol terms _mn wall part _rest <<< "$row"
    if [ "$atol" = CALIB ]; then
        echo "!! $LADDER N=$n atol is CALIB -- calibrate first, not queueing" >&2; continue
    fi
    r=${REPS_OVERRIDE:-$reps}
    cmd=(sbatch -A "$ACCOUNT" -p "$part" -N "$n" -t "$wall"
         --export=ALL,LADDER="$LADDER",REPS="$r"
         --job-name="mp-sc-${LADDER}-N${n}-r${r}" "$SELF")
    printf '%-14s N=%-3s r=%-3s %-11s %-9s terms=%s\n' "$LADDER" "$n" "$r" "$part" "$wall" "$terms"
    if [ -n "${DRY_RUN:-}" ]; then continue; fi
    ( cd "$TREE" && "${cmd[@]}" ) || echo "!! submit failed for $LADDER N=$n" >&2
done
