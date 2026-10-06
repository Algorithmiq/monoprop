#!/bin/bash
# Scan every scaleladder job log for failure signatures.
#
# grep MUST be -a: a Slurm .out containing a single NUL is treated as binary and plain grep prints
# nothing at all, which reads exactly like a clean job. Silence here has to mean "no matches", not
# "grep declined to look".
#
# The alternation covers what would actually change a conclusion: OOM kills, MPI aborts, python
# tracebacks, the driver's own refusal codes, and the unpinning signature.
set -uo pipefail
TREE=${TREE:-/projects/EEHPC-DEV-2026D08-260/src/296-on-bench}
PAT='Out of memory|oom-kill|Killed|MPI_ABORT|APPLICATION TERMINATED|Traceback|Segmentation fault|slurmstepd: error|DUE TO TIME LIMIT|CANCELLED|refusing:|GATE FAIL|rc=[1-9]'
bad=0 seen=0
for f in "$TREE"/mp-sc-*.out "$TREE"/mp-sc-*.err; do
    [ -e "$f" ] || continue
    seen=$((seen + 1))
    hits=$(grep -aEn "$PAT" "$f" 2>/dev/null | grep -av 'CANCELLED AT.*DUE TO PREEMPTION' | head -4)
    if [ -n "$hits" ]; then
        bad=$((bad + 1))
        echo "== $(basename "$f")"
        echo "$hits" | sed 's/^/     /'
    fi
done
echo "scanned $seen log files, $bad with failure signatures"
# A scan that found nothing must prove it looked at something.
[ "$seen" -eq 0 ] && { echo "REFUSING: no log files matched -- the scan proved nothing" >&2; exit 2; }
exit 0
