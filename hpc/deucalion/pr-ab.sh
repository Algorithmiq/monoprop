#!/bin/bash -l
# One command per PR: port the ref, build it, gate it, A/B it at one and two nodes against the
# shared main baseline, and collate both into one PR-AB-<pr>.md.
#
#   hpc/deucalion/pr-ab.sh profiling perf/layer-profile
#   hpc/deucalion/pr-ab.sh query-wire stack/4-self-resolve
#   EXTRA_CELLS='random-n1|WORKLOAD=random|-N1' hpc/deucalion/pr-ab.sh query-wire stack/4-...
#
# This is a submitter, not a job: it runs on the login node, submits the whole chain with
# --dependency=afterok, and returns. Nothing needs babysitting and a red gate stops the chain
# before any measurement is taken -- which is the point, because a measurement taken on a tree
# that does not pass its own tests is worse than no measurement.
#
# THE BASELINE IS BUILT ONCE AND RUN EVERY TIME. $BASELINE_TREE (default $PROJ/src/mp-main, the
# tree hpc/deucalion/baseline.md5 was pinned from) is a single worktree
# at origin/main, built and gated once, then reused read-only by every PR. That removes N
# redundant builds and any doubt that two PRs were compared against different baselines; its
# _core.so md5 is recorded in every artifact, and ab_summary's _check_builds refuses a run whose
# two arms share a hash.
#
# But the baseline is still RE-RUN, INTERLEAVED, inside each PR's allocation. The paired per-rep
# ratio is the entire reason these jobs interleave: a baseline measured in a different allocation
# puts any per-allocation drift entirely on one side. That is not hypothetical -- it once printed
# one layout 1.85x worse on `pare`, same node, 13:20 vs 13:22, with no way to tell drift from
# signal.

set -euo pipefail

# DRY_RUN=1 prints the chain instead of submitting it, and creates nothing. It exists because
# validating this file by running it submitted a real four-job chain and left a worktree behind;
# a submitter with no dry mode is one typo away from spending an allocation to read its own help.
: "${DRY_RUN:=0}"
# The counter lives in a file, not a variable: sub() is always called inside $( ), which is a
# subshell, so an incremented variable never reaches the caller and every dry job would print
# the same id -- making the dependency chain, the one thing a dry run exists to check,
# unreadable.
_DRY_COUNT=$(mktemp)
trap 'rm -f "$_DRY_COUNT"' EXIT
echo 0 > "$_DRY_COUNT"
sub() {  # echo a fake job id in dry mode, else sbatch --parsable
    if [ "$DRY_RUN" = "1" ]; then
        local n
        n=$(( $(cat "$_DRY_COUNT") + 1 ))
        echo "$n" > "$_DRY_COUNT"
        printf 'DRY  sbatch %s\n' "$*" >&2
        printf '9%04d\n' "$n"
    else
        sbatch --parsable "$@"
    fi
}
run() {  # side effects that must not happen in dry mode
    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY  %s\n' "$*" >&2
    else
        "$@"
    fi
}
# SUBMIT FROM INSIDE THE TREE, not with --chdir alone. build-worktree.sh and
# mpi-tests-worktree.sh both begin `cd "${SLURM_SUBMIT_DIR:-$PWD}"`, and this Slurm's man page
# defines SLURM_SUBMIT_DIR as "the directory from which sbatch was invoked" -- sbatch fills it
# from getcwd() BEFORE --chdir is applied. So --chdir alone does not reach those scripts: they
# would cd back to this harness checkout and build/gate IT instead of the port tree, with a job
# log whose `=== src ===` line reads plausibly either way. Passing both makes the two mechanisms
# agree, and EXPECT_TREE below makes a future disagreement abort instead of measuring the wrong
# tree. The class of bug this is: the producer (pr-ab.sh) and the consumer (the sbatch script)
# naming the same directory two different ways.
sub_in() {  # $1 directory to submit FROM; rest: sbatch args
    local dir="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY  (cd %s)\n' "$dir" >&2
        sub --chdir="$dir" "$@"
    else
        [ -d "$dir" ] || {
            echo "refusing: cannot submit from $dir -- no such directory" >&2; exit 1; }
        ( cd "$dir" && sbatch --parsable --chdir="$dir" "$@" )
    fi
}

usage() {
    cat >&2 <<'USAGE'
usage: pr-ab.sh <pr-name> <branch-ref> [extra-cells]

  <pr-name>     short slug; names the worktree, the results root and the report
  <branch-ref>  anything git can resolve -- a branch, a tag, a sha

  extra-cells   ';'-separated highlight cells, each `label|ENV=v,ENV=v|sbatch-args`, e.g.
                  'bind-cores|CPU_BIND=cores,ALLOW_BOTH_PLACED=0|-N1'
                  'layoutA|RANKS_PER_NODE=1,PARTITIONS=128|-N1 -N2'
                may also be passed as $EXTRA_CELLS.

env:
  BASELINE_TREE  default $PROJ/src/mp-main (pinned by hpc/deucalion/baseline.md5)
  DRY_RUN=1      print the chain, submit nothing, create nothing
  REPS           default 6
  GRID           default 'hubbard pauli'   (the common grid's workloads)
  SKIP_BUILD=1   reuse an existing port tree as-is (it must already be built and gated)
USAGE
    exit 2
}

# --baseline builds the one shared main arm. It is a separate mode because it must happen
# exactly once: every PR then measures against the same binary, and ab_summary records its
# _core.so md5 in every artifact so a later run can prove it never moved.
if [ "${1:-}" = "--baseline" ]; then
    export MONOPROP_SRC="$PWD"
    source hpc/deucalion/env.sh
    # ONE default, the same one the PR path uses and the one baseline.md5 was pinned from. This
    # used to default to $PROJ/src/ab-baseline-main while the PR path below defaulted to
    # $PROJ/src/mp-main, so `--baseline` built a tree that no PR would ever measure against.
    : "${BASELINE_TREE:=$PROJ/src/mp-main}"
    : "${BASELINE_REF:=origin/main}"
    A="-A $MONOPROP_SLURM_ACCOUNT"
    if [ -d "$BASELINE_TREE" ]; then
        echo "refusing: $BASELINE_TREE exists; remove it to rebuild the baseline" >&2
        echo "  Its _core.so is PINNED in hpc/deucalion/baseline.md5 and every ratio in the" >&2
        echo "  campaign is measured against that binary. Rebuilding it voids all of them." >&2
        exit 1
    fi
    run git -C "$PWD" worktree add --detach "$BASELINE_TREE" "$BASELINE_REF"
    run cp -r "$PWD/hpc" "$BASELINE_TREE/hpc"
    run cp -r "$PWD/benches" "$BASELINE_TREE/benches"
    # TAG is the build-dir suffix, and build.sh derives its own from `basename $MONOPROP_SRC`.
    # Derive it the same way here or the two disagree: ctest-worktree.sh would uv-sync into a
    # SECOND build directory (a cold rebuild inside a 1:30 walltime budgeted for an incremental
    # one) and leave mpi-tests-worktree.sh reading the first one's older binary.
    BASELINE_TAG="$(basename "$BASELINE_TREE")"
    # ALLOW_BASELINE_REBUILD=1 only here. build.sh and ctest-worktree.sh both refuse to touch a
    # tree whose installed _core.so hashes to hpc/deucalion/baseline.md5, because one stray
    # `TREE=$PROJ/src/mp-main sbatch ctest-worktree.sh` re-bases every ratio in the campaign
    # without changing any version string. THIS mode is the one place that is deliberate: it can
    # only run when the tree does not exist yet, and a rebuild that happened to reproduce the
    # pinned bytes exactly must not deadlock it.
    b=$(sub_in "$BASELINE_TREE" $A -J mp-build-baseline \
                --export=ALL,EXPECT_TREE="$BASELINE_TREE",ALLOW_BASELINE_REBUILD=1 \
                "$BASELINE_TREE/hpc/deucalion/sbatch/build-worktree.sh")
    echo "build      $b"
    c=$(sub $A --dependency="afterok:$b" -J mp-ctest-baseline \
                --export=ALL,TREE="$BASELINE_TREE",TAG="$BASELINE_TAG",ALLOW_BASELINE_REBUILD=1 \
                "$BASELINE_TREE/hpc/deucalion/sbatch/ctest-worktree.sh")
    echo "ctest      $c  (after $b)"
    m=$(sub_in "$BASELINE_TREE" $A --dependency="afterok:$c" -N2 \
                -J mp-mpitest-baseline \
                --export=ALL,EXPECT_TREE="$BASELINE_TREE" \
                "$BASELINE_TREE/hpc/deucalion/sbatch/mpi-tests-worktree.sh")
    echo "mpi tests  $m  (after $c)"
    echo
    echo "baseline   $BASELINE_TREE at $BASELINE_REF ($(git rev-parse --short "$BASELINE_REF"))"
    exit 0
fi

[ $# -ge 2 ] || usage
PR_NAME="$1"
BRANCH_REF="$2"
EXTRA_CELLS="${3:-${EXTRA_CELLS:-}}"

case "$PR_NAME" in
    *[!a-z0-9-]*) echo "refusing: pr-name '$PR_NAME' must be [a-z0-9-]" >&2; exit 2 ;;
esac

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${BASELINE_TREE:=$PROJ/src/mp-main}"
: "${REPS:=6}"
: "${GRID:=hubbard pauli}"

HARNESS="$PWD"
PORT_TREE="$PROJ/src/ab-${PR_NAME}-port"
RESULTS_ROOT="$MONOPROP_RUNS/pr-${PR_NAME}"
A="-A $MONOPROP_SLURM_ACCOUNT"

if [ ! -x "$BASELINE_TREE/.venv/bin/python" ]; then
    echo "refusing: no baseline at $BASELINE_TREE/.venv" >&2
    echo "  build it once:  hpc/deucalion/pr-ab.sh --baseline" >&2
    exit 1
fi

# THE BASELINE IS PINNED BY ITS BINARY, not by its branch or its version stamp. The tree is
# shared, so anything could rebuild it between PRs, and a rebuilt baseline silently re-bases
# every ratio in the campaign. __version__ cannot catch that: it is a git describe stamped into
# the dist-info at install time, and a later `cmake --build` + copy into the venv does not
# rewrite it, so the stamp goes stale exactly when the binary changes. Hash the .so.
#
# Hashed here rather than imported: the login node has no toolchain module loaded, so importing
# monoprop fails on a missing CXXABI, while the file is readable either way.
mapfile -t _bsos < <(ls "$BASELINE_TREE"/.venv/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null)
[ "${#_bsos[@]}" -eq 1 ] || {
    echo "refusing: expected exactly one baseline _core.so, found ${#_bsos[@]}" >&2; exit 1; }
BASELINE_MD5_ACTUAL=$(md5sum "${_bsos[0]}" | cut -d' ' -f1)
PIN_FILE="$HARNESS/hpc/deucalion/baseline.md5"
if [ -r "$PIN_FILE" ]; then
    PIN=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$PIN_FILE")
    if [ -n "$PIN" ] && [ "$PIN" != "$BASELINE_MD5_ACTUAL" ]; then
        echo "refusing: baseline binary moved" >&2
        echo "  pinned  $PIN  ($PIN_FILE)" >&2
        echo "  actual  $BASELINE_MD5_ACTUAL  (${_bsos[0]})" >&2
        echo "  every ratio measured against the old one is on a different baseline." >&2
        echo "  Re-pin deliberately, and re-run the PRs already measured." >&2
        exit 1
    fi
fi
echo "baseline   $BASELINE_TREE  _core.so md5 $BASELINE_MD5_ACTUAL"
if [ "$DRY_RUN" != "1" ] && [ -e "$RESULTS_ROOT" ] && [ -n "$(ls -A "$RESULTS_ROOT" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS_ROOT already holds results; move it or pick another pr-name" >&2
    exit 1
fi
run mkdir -p "$RESULTS_ROOT"

# ---------------------------------------------------------------- port the ref

deps=""
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    if [ -d "$PORT_TREE" ] && [ "$DRY_RUN" != "1" ]; then
        echo "refusing: $PORT_TREE exists; remove it or pass SKIP_BUILD=1" >&2
        exit 1
    fi
    # A worktree, not a clone: same object store, so the ref is the ref and there is no
    # fetch to go stale.
    run git -C "$HARNESS" worktree add --detach "$PORT_TREE" "$BRANCH_REF"
else
    [ -x "$PORT_TREE/.venv/bin/python" ] || {
        echo "refusing: SKIP_BUILD=1 but no venv at $PORT_TREE/.venv" >&2; exit 1; }
    echo "build      SKIPPED (reusing $PORT_TREE)"
fi

# Copied OUTSIDE the SKIP_BUILD branch, deliberately. "Both arms are driven by one harness
# revision" is the invariant, and under SKIP_BUILD=1 the copies used to be skipped too -- so a
# reused port tree kept whatever hpc/ and benches/ it had from an earlier run and the two arms
# were driven by two harness revisions, silently. Re-copying is free (nothing here is compiled).
#
# hpc/ is not on the PR branches and never will be -- PRs carry library code only. It is
# copied in so the port tree can run its own gate and its own A/B.
run cp -r "$HARNESS/hpc" "$PORT_TREE/hpc"
# benches/ likewise comes from one revision on both arms, so the only difference between
# them is the compiled extension. A PR that changes benches/ is measured with ITS benches
# on both sides; that is PR 3's business and it says so in its own body.
if [ "${BENCHES_FROM_PORT:-0}" != "1" ]; then
    run cp -r "$HARNESS/benches" "$PORT_TREE/benches"
fi
# Every sbatch below names a file under $PORT_TREE/hpc. Check it exists rather than letting
# sbatch fail one submission at a time in the middle of a chain.
[ "$DRY_RUN" = "1" ] || [ -f "$PORT_TREE/hpc/deucalion/sbatch/ab.sh" ] || {
    echo "refusing: no hpc/deucalion/sbatch/ab.sh under $PORT_TREE -- the harness copy failed" >&2
    exit 1; }

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    # TAG is the build-dir suffix. build.sh derives its own from `basename $MONOPROP_SRC`, and
    # mpi-tests-worktree.sh derives the same, so this MUST be the basename too. It used to be
    # "ab-$PR_NAME", one segment short of the tree name: ctest-worktree.sh then uv-synced into a
    # second, empty build directory -- a cold rebuild inside a 1:30 budget sized for an
    # incremental one -- and reinstalled the venv from it, leaving the mpi gate reading the
    # first build's tree. Same defect class as the Release-vs-Release-<tag> one.
    PORT_TAG="$(basename "$PORT_TREE")"

    b=$(sub_in "$PORT_TREE" $A -J "mp-build-$PR_NAME" \
                --export=ALL,EXPECT_TREE="$PORT_TREE" \
                "$PORT_TREE/hpc/deucalion/sbatch/build-worktree.sh")
    echo "build      $b"

    # The gates. ctest-worktree.sh gates -L unit and -L serial; mpi-tests-worktree.sh drives the
    # MPI cases and the Python suite across four rank/partition layouts.
    c=$(sub $A --dependency="afterok:$b" -J "mp-ctest-$PR_NAME" \
                --export=ALL,TREE="$PORT_TREE",TAG="$PORT_TAG" \
                "$PORT_TREE/hpc/deucalion/sbatch/ctest-worktree.sh")
    echo "ctest      $c  (after $b)"

    m=$(sub_in "$PORT_TREE" $A --dependency="afterok:$c" -N2 \
                -J "mp-mpitest-$PR_NAME" \
                --export=ALL,EXPECT_TREE="$PORT_TREE" \
                "$PORT_TREE/hpc/deucalion/sbatch/mpi-tests-worktree.sh")
    echo "mpi tests  $m  (after $c)"
    deps="--dependency=afterok:$m"
fi

# ---------------------------------------------------------------- the cells

# One A/B job per (workload, nodes). Each is one allocation holding both arms.
ab_ids=()
# Exported into the environment and shipped with --export=ALL, rather than spelled into
# --export's comma list. A value containing a comma or an `=` silently truncates that list, and
# EXTRA_ENV is exactly such a value -- a cell that lost half its knobs would run the shipped
# arm while the label claimed the experimental one, which is a failure this project has already
# had once (an out-of-range knob falling back to the default).
submit_ab() {  # $1 label, $2 nodes, $3 env-csv
    local label="$1" nodes="$2" envcsv="$3" id
    export MAIN_VENV="$BASELINE_TREE/.venv" PORT_VENV="$PORT_TREE/.venv"
    export REPS RESULTS_ROOT
    export RESULTS_TAG="$label"
    export EXTRA_ENV="${envcsv//,/;}"
    export WORKLOAD="${CELL_WORKLOAD:-hubbard}"
    export CELL_SPEC="${CELL_SPEC_OVERRIDE:-}"
    export CPU_BIND="${CELL_CPU_BIND:-none}"
    export ALLOW_BOTH_PLACED="${CELL_ALLOW_BOTH:-1}"
    export RANKS_PER_NODE="${CELL_RANKS:-8}" PARTITIONS="${CELL_PARTS:-16}"
    # The cell directory ab.sh will create, spelled here the ONE way ab.sh spells it, and
    # recorded before the job exists. A cell that is cancelled, or whose afterok dependency is
    # never satisfied, writes nothing at all -- and pr_report.py, which only ever sees what is on
    # disk, would then produce a clean green report of fewer cells. The manifest is the only
    # place the intended cell list exists.
    local cell_dir="ab-${WORKLOAD}${RESULTS_TAG:+-${RESULTS_TAG}}-N${nodes}"
    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY  expect %s\n' "$cell_dir" >&2
    else
        printf '%s\n' "$cell_dir" >> "$RESULTS_ROOT/EXPECTED-CELLS"
    fi
    id=$(sub_in "$PORT_TREE" $A $deps -N"$nodes" --export=ALL \
            -J "mp-ab-$PR_NAME-$label-N$nodes" \
            "$PORT_TREE/hpc/deucalion/sbatch/ab.sh")
    ab_ids+=("$id")
    printf 'ab %-22s %s  N=%s  %s\n' "$label" "$id" "$nodes" "${envcsv:-}"
    # In dry mode print what the cell actually resolved to. The geometry, the bind and the
    # both-placed refusal travel as ab.sh's OWN variables rather than through EXTRA_ENV,
    # because each of them changes the label or a refusal -- and a highlight cell that
    # silently kept the default bind would measure the configuration it exists to avoid.
    [ "$DRY_RUN" = "1" ] && printf '     %s R=%s S=%s bind=%s both_placed=%s cells=%s knobs=%s\n' \
        "$WORKLOAD" "$RANKS_PER_NODE" "$PARTITIONS" "$CPU_BIND" "$ALLOW_BOTH_PLACED" \
        "${CELL_SPEC:-(default)}" "${EXTRA_ENV:-(none)}"
    return 0
}

# --- the common grid, identical in every PR -----------------------------------
# hubbard runs propagate ONLY: build_graph EXTENDS the graph, so its 29 Trotter steps retain 29
# layer-sets and exceed a 242 GiB node, while propagate releases each layer as it contracts and
# reaches 97M terms in 379 MiB. pauli is 1 step, so it runs all four operations.
for wl in $GRID; do
    for n in 1 2; do
        if [ "$wl" = "hubbard" ]; then
            CELL_WORKLOAD=$wl CELL_SPEC_OVERRIDE='fresh|hubbard and propagate' \
                submit_ab "grid-$wl" "$n" ""
        else
            CELL_WORKLOAD=$wl submit_ab "grid-$wl" "$n" ""
        fi
    done
done

# --- the PR's own highlight cells ---------------------------------------------
if [ -n "$EXTRA_CELLS" ]; then
    IFS=';' read -r -a _cells <<< "$EXTRA_CELLS"
    for spec in "${_cells[@]}"; do
        [ -z "$spec" ] && continue
        label="${spec%%|*}"; rest="${spec#*|}"
        envcsv="${rest%%|*}"; nodespec="${rest#*|}"
        # Pull the geometry and bind out of the env list: ab.sh reads them as its own
        # variables, not through EXTRA_ENV, because they change the label and the refusals.
        CELL_RANKS=8 CELL_PARTS=16 CELL_CPU_BIND=none CELL_ALLOW_BOTH=1
        CELL_WORKLOAD=hubbard CELL_SPEC_OVERRIDE='fresh|hubbard and propagate'
        passthru=""
        IFS=',' read -r -a _kvs <<< "$envcsv"
        for kv in "${_kvs[@]}"; do
            case "$kv" in
                RANKS_PER_NODE=*) CELL_RANKS="${kv#*=}" ;;
                PARTITIONS=*) CELL_PARTS="${kv#*=}" ;;
                CPU_BIND=*) CELL_CPU_BIND="${kv#*=}" ;;
                ALLOW_BOTH_PLACED=*) CELL_ALLOW_BOTH="${kv#*=}" ;;
                WORKLOAD=*)
                    CELL_WORKLOAD="${kv#*=}"
                    [ "$CELL_WORKLOAD" = "hubbard" ] || CELL_SPEC_OVERRIDE=""
                    ;;
                "") ;;
                # AN ARM-SELECTING KEY MUST FAIL CLOSED. Everything below used to fall into a
                # catch-all that exported the pair to both arms verbatim -- so `CPUBIND=cores`
                # or `RANKS_PERNODE=1`, one character off the names above, was exported, read by
                # nothing, and the cell ran the DEFAULT 8x16 bind=none geometry while its label
                # and its report claimed the experimental one. That is the shell form of
                # parse_positive_int capping at 1e6 and then .value_or(default).
                monoprop_PARTITIONS=* | monoprop_NUM_THREADS=* | MALLOC_ARENA_MAX=* \
                    | OMP_NUM_THREADS=*)
                    echo "refusing: cell '$label' sets ${kv%%=*}, which ab.sh derives from" \
                         "PARTITIONS and stamps into the layout label. Pass PARTITIONS=N" \
                         "instead, or the cell reports one world size and runs another." >&2
                    exit 2
                    ;;
                # The engine's own knobs, and ab.sh's workload-sizing variables. Anything else
                # is a typo until someone adds it here on purpose.
                monoprop_*=* | CUTOFF=* | LOWER_ATOL=* | OBS_TERMS=* | NUM_MODES=* \
                    | NUM_GENERATORS=* | MODEL_ARGS=* | REPS=*)
                    passthru="${passthru:+$passthru,}$kv"
                    ;;
                *=*)
                    echo "refusing: cell '$label' sets unknown key '${kv%%=*}'." \
                         "Cell variables are RANKS_PER_NODE, PARTITIONS, CPU_BIND," \
                         "ALLOW_BOTH_PLACED, WORKLOAD; pass-through keys are monoprop_*," \
                         "CUTOFF, LOWER_ATOL, OBS_TERMS, NUM_MODES, NUM_GENERATORS," \
                         "MODEL_ARGS, REPS." >&2
                    exit 2
                    ;;
                *)
                    echo "refusing: cell '$label' entry '$kv' is not NAME=value" >&2
                    exit 2
                    ;;
            esac
        done
        export CELL_RANKS CELL_PARTS CELL_CPU_BIND CELL_ALLOW_BOTH CELL_WORKLOAD CELL_SPEC_OVERRIDE
        for narg in $nodespec; do
            submit_ab "$label" "${narg#-N}" "$passthru"
        done
        unset CELL_RANKS CELL_PARTS CELL_CPU_BIND CELL_ALLOW_BOTH CELL_WORKLOAD CELL_SPEC_OVERRIDE
    done
fi

# ---------------------------------------------------------------- collate

# afterany, not afterok: a cell that fails its own refusals is a RESULT about this PR, and the
# report must say so rather than silently omit it. pr_report.py compares what is on disk against
# $RESULTS_ROOT/EXPECTED-CELLS (written by submit_ab above, at submit time) and exits non-zero if
# any cell is missing or void, so the chain still ends red -- for a cell that RAN and refused,
# and equally for one the scheduler cancelled, which leaves no directory to notice.
[ "${#ab_ids[@]}" -gt 0 ] || {
    echo "refusing: no A/B cells were submitted -- GRID is empty and no extra cells parsed." >&2
    echo "  A report job depending on nothing would run immediately over an empty tree." >&2
    exit 1; }
joined=$(IFS=:; echo "${ab_ids[*]}")
r=$(sub_in "$PORT_TREE" $A --dependency="afterany:$joined" \
        -J "mp-report-$PR_NAME" -p dev-x86 -N1 -t 0:20:00 \
        --wrap="'$BASELINE_TREE/.venv/bin/python' '$PORT_TREE/hpc/deucalion/tools/pr_report.py' \
                --pr '$PR_NAME' --ref '$BRANCH_REF' --out '$RESULTS_ROOT/PR-AB-$PR_NAME.md' \
                '$RESULTS_ROOT'")
echo "report     $r  (after all cells)"
echo
echo "results    $RESULTS_ROOT"
echo "report     $RESULTS_ROOT/PR-AB-$PR_NAME.md"
