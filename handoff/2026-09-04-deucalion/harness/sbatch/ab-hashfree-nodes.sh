#!/bin/bash -l
# N-arm interleaved A/B under srun: PR #317's ladder rungs L2b (NODES=1) and L3 (NODES>1).
#
# Same discipline as ab-hashfree-n1.sh -- N arms, every rep runs them all with the order rotated by
# one, paired ratios per (layout, model, group) row -- but every cell is an MPI launch at the
# NUMA-aligned 8x16 layout instead of one in-process run. L2b and L3 differ ONLY in --nodes: same
# arms, same rows, same R and P, so the L2b-to-L3 difference is the network.
#
#   cd worktrees/<port-tree> && \
#     ARMS="main=$PROJ/aaron/worktrees/base-296/.venv port=$PWD/.venv" \
#     sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" hpc/deucalion/sbatch/ab-hashfree-nodes.sh
#
#   # L3 on four nodes, or any non-default NODES/WALL/partition: run the script (no sbatch), and it
#   # submits itself with the matching flags, because #SBATCH lines cannot read NODES.
#   cd worktrees/<port-tree> && NODES=4 WALL=1:00:00 SLURM_PARTITION=normal-x86 \
#     ARMS="main=... port=..." bash hpc/deucalion/sbatch/ab-hashfree-nodes.sh
#
# SUBMIT FROM INSIDE THE WORKTREE. The script cd's to $SLURM_SUBMIT_DIR and sources
# hpc/deucalion/env.sh from there (as ab-hashfree-n1.sh does), so the submit directory must be a
# worktree carrying the hpc/deucalion symlink. Arm venvs are absolute and independent of it.
#
# Arms are >=2 venvs (installed _core.so's), given as ARMS: space-separated
# "name=venv[:VAR=VAL[,VAR=VAL...]]" tokens; the first arm is the reference. Arm names become the
# label's <arm> field (hpc/deucalion/tools/ab_pairs.py parses it), so they must be bare
# alphanumerics. A per-arm VAR=VAL applies only to that arm's srun -- it is prefixed to the task
# command with `env`, not exported here, so it cannot leak into a sibling arm's cell. Identity is
# the (md5, env) pair, so two arms may share a venv when their env differs.
#
# Environment (defaults in brackets):
#
#   ARMS             [main=$MAIN_VENV port=$PORT_VENV]  arm specs, first is the reference
#   MAIN_VENV        [base-296/.venv]      reference venv, when ARMS is unset
#   PORT_VENV        [one-round/.venv]     comparison venv, when ARMS is unset
#   BENCH_TREE       [bench-317]           benches/ + bench tools used for EVERY arm
#   NODES            [1]                   1 = L2b, >1 = L3. Must match the allocation.
#   WALL             [2:00:00]             --time for the self-submit path
#   SLURM_PARTITION  [dev-x86]             partition for the self-submit path (dev-x86 caps at 2)
#   ACCOUNT          [$MONOPROP_SLURM_ACCOUNT]  account for the self-submit path
#   RANKS_PER_NODE   [8]                   R: MPI ranks per node, one per NUMA domain
#   CORES_PER_RANK   [16]                  P: partitions and threads per rank
#   CORES_PER_NODE   [128]                 refuses unless R x P covers it exactly
#   REPS             [3]                   reps per row (3 sizes; >=10 for a claim, see §6)
#   ROWS             [hubbard pauli]       which of the L2 rows to run
#   RESULTS_TAG      []                    suffix on the results directory name
#   COMM_PROFILE     [0]                   1 exports monoprop_COMM_PROFILE to every arm alike
#   SMOKE            [0]                   1 = tiny rows, plumbing check only, not a measurement
#
# Rows are the L2a/L2b/L3 rows of benches/LADDER.md on BENCH_TREE, whose benches/ and bench tools
# serve every arm through PYTHONPATH, so the harness is one revision whatever the arms' trees carry.
#
#   hubbard : propagate, cutoff 10, lower_atol 3.38e-06  -- ~1.00B terms, 59.5 GiB/node at 8x16
#   pauli   : propagate, cutoff 14, lower_atol 8.9e-06   -- ~0.99B terms, 85.6 GiB/node at 8x16
#
# Memory: under srun there is no single process whose high-water mark describes the cell, and
# /usr/bin/time -v on the launcher measures srun itself, so peak memory comes from the bench JSON --
# `memhwm` (the ranks' VmHWM, summed by conftest.py) into PEAK-RSS.tsv's max_rss_kb column, where
# ab_pairs.py already reads it, and `memhwm_max` (worst rank) into WORST-RANK-RSS.tsv beside it.
# CELL-META.tsv records ranks_per_node, which is how ab_pairs.py knows to name the row
# "peak RSS (ranks summed)". The sum charges shared pages to every rank: it bounds a node for
# provisioning and is NOT comparable with a single process's peak (LADDER.md, L2b).
#
#SBATCH --job-name=mp-ab-hfn
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${NODES:=${SLURM_JOB_NUM_NODES:-1}}"
: "${WALL:=2:00:00}"
: "${SLURM_PARTITION:=dev-x86}"
: "${RANKS_PER_NODE:=8}"
: "${CORES_PER_RANK:=16}"
: "${CORES_PER_NODE:=128}"

[[ "$NODES" =~ ^[0-9]+$ ]] && (( NODES >= 1 )) || { echo "NODES must be a positive integer, got '$NODES'" >&2; exit 1; }

# The whole point of the 8x16 layout is that it covers the node: a shortfall leaves cores idle and
# an excess oversubscribes them, and either silently changes what the ratio means.
if (( RANKS_PER_NODE * CORES_PER_RANK != CORES_PER_NODE )); then
    echo "refusing: RANKS_PER_NODE=$RANKS_PER_NODE x CORES_PER_RANK=$CORES_PER_RANK != CORES_PER_NODE=$CORES_PER_NODE" >&2
    echo "  the layout must cover the node exactly (8x16=128 on x86); set CORES_PER_NODE for another machine" >&2
    exit 1
fi
if [ "$SLURM_PARTITION" = dev-x86 ] && (( NODES > 2 )); then
    echo "refusing: dev-x86 caps at 2 nodes; pass SLURM_PARTITION=normal-x86 for NODES=$NODES" >&2
    exit 1
fi

# #SBATCH directives are static, so NODES/WALL/partition can only reach Slurm as submit-time flags.
# Running the script outside an allocation submits it with them, and the environment (ARMS, REPS,
# SMOKE, ...) rides along on sbatch's default --export=ALL.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    ACCOUNT="${ACCOUNT:-${SBATCH_ACCOUNT:-${MONOPROP_SLURM_ACCOUNT:-}}}"
    [ -n "$ACCOUNT" ] || { echo "no account: export MONOPROP_SLURM_ACCOUNT (env.sh does) or ACCOUNT=<project>x" >&2; exit 1; }
    echo "submitting: nodes=$NODES time=$WALL partition=$SLURM_PARTITION account=$ACCOUNT chdir=$PWD"
    exec sbatch --nodes="$NODES" --time="$WALL" --partition="$SLURM_PARTITION" \
        --account="$ACCOUNT" --chdir="$PWD" "$0"
fi

# Slurm stages the script at submission, so the allocation is the fact and NODES is the claim.
if [ "$NODES" != "${SLURM_JOB_NUM_NODES:-$NODES}" ]; then
    echo "refusing: NODES=$NODES but this allocation has ${SLURM_JOB_NUM_NODES} nodes" >&2
    echo "  resubmit with matching values (or unset NODES to take the allocation's)" >&2
    exit 1
fi
if [ -n "${SLURM_CPUS_ON_NODE:-}" ] && [ "$SLURM_CPUS_ON_NODE" != "$CORES_PER_NODE" ]; then
    echo "!! SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE but CORES_PER_NODE=$CORES_PER_NODE; the layout may not cover the node"
fi

export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${MAIN_VENV:=$PROJ/aaron/worktrees/base-296/.venv}"
: "${PORT_VENV:=$PROJ/aaron/worktrees/one-round/.venv}"
: "${ARMS:=main=$MAIN_VENV port=$PORT_VENV}"
: "${BENCH_TREE:=$PROJ/aaron/worktrees/bench-317}"
: "${REPS:=3}"
: "${ROWS:=hubbard pauli}"
: "${RESULTS_TAG:=}"
: "${COMM_PROFILE:=0}"

RANKS=$((NODES * RANKS_PER_NODE))
if (( NODES > 1 )); then
    LAYOUT="C_${NODES}n_${RANKS_PER_NODE}x${CORES_PER_RANK}"   # L3
else
    LAYOUT="B_${RANKS_PER_NODE}x${CORES_PER_RANK}"             # L2b
fi

RESULTS="${MONOPROP_RUNS}/ab-hashfree-nodes${RESULTS_TAG:+-${RESULTS_TAG}}-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"
echo "results: $RESULTS"
echo "shape:   nodes=$NODES ranks=$RANKS (R=$RANKS_PER_NODE/node) P=$CORES_PER_RANK layout=$LAYOUT reps=$REPS"
echo "nodelist: $(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | tr '\n' ' ')"
export PYTHONPATH="$BENCH_TREE/packages/monoprop-bench-tools/src${PYTHONPATH:+:$PYTHONPATH}"

# Arm identity = md5 over the two binaries the interpreter actually loads (site-packages copies, not
# the editable source tree or the build dir): _core.so and, where most of the engine lives, libmonoprop.so.
md5_of() {
    "$1/bin/python" - <<'PY'
import hashlib, pathlib
from monoprop import _core
core = pathlib.Path(_core.__file__)
lib = core.parent / "lib64" / "libmonoprop.so"
h = hashlib.md5(core.read_bytes())
if lib.exists():
    h.update(lib.read_bytes())
print(h.hexdigest())
PY
}

# Parse ARMS: space-separated "name=venv[:VAR=VAL[,VAR=VAL...]]" tokens. The first token is the
# reference arm. ARM_ENV entries are comma-separated VAR=VAL pairs (or "") prefixed to that arm's
# srun task command in run_cell, so they never leak into a sibling arm's run.
declare -a ARM_NAME=() ARM_VENV=() ARM_ENV=() ARM_MD5=() ARM_REV=()
for tok in $ARMS; do
    name="${tok%%=*}"
    rest="${tok#*=}"
    venv="${rest%%:*}"
    envstr=""
    [ "$rest" != "$venv" ] && envstr="${rest#*:}"
    [[ "$name" =~ ^[A-Za-z0-9]+$ ]] || { echo "bad arm name '$name': letters/digits only, no _/=/: " >&2; exit 1; }
    ARM_NAME+=("$name"); ARM_VENV+=("$venv"); ARM_ENV+=("$envstr")
done
N_ARMS=${#ARM_NAME[@]}
(( N_ARMS >= 2 )) || { echo "need >= 2 arms in ARMS='$ARMS'" >&2; exit 1; }

declare -A _seen_names=() _seen_identity=()
for i in "${!ARM_NAME[@]}"; do
    name=${ARM_NAME[$i]} venv=${ARM_VENV[$i]}
    [ -z "${_seen_names[$name]:-}" ] || { echo "refusing: duplicate arm name '$name'" >&2; exit 1; }
    _seen_names[$name]=1
    [ -x "$venv/bin/python" ] || { echo "missing venv: $venv" >&2; exit 1; }
    # has_mpi is not optional here: without it every rank would build the whole problem alone.
    "$venv/bin/python" -c "import monoprop, pytest_benchmark; assert monoprop.has_mpi, 'built without MPI'" || exit 1
    ARM_MD5[$i]=$(md5_of "$venv")
    ARM_REV[$i]=$(git -C "$(dirname "$venv")" rev-parse --short HEAD)
    echo "arm $name: venv=$venv md5=${ARM_MD5[$i]} env=${ARM_ENV[$i]:-<none>}"
    identity="${ARM_MD5[$i]}|${ARM_ENV[$i]}"
    if [ -n "${_seen_identity[$identity]:-}" ]; then
        echo "refusing: arms '${_seen_identity[$identity]}' and '$name' load the same _core.so with the same env" >&2
        exit 1
    fi
    _seen_identity[$identity]=$name
done

{
    echo -e "key\tvalue"
    for i in "${!ARM_NAME[@]}"; do
        n=$((i + 1))
        echo -e "arm_${n}_name\t${ARM_NAME[$i]}"
        echo -e "arm_${n}_venv\t${ARM_VENV[$i]}"
        echo -e "arm_${n}_md5\t${ARM_MD5[$i]}"
        echo -e "arm_${n}_rev\t${ARM_REV[$i]}"
        echo -e "arm_${n}_env\t${ARM_ENV[$i]}"
    done
    echo -e "bench_tree\t$BENCH_TREE"; echo -e "bench_rev\t$(git -C "$BENCH_TREE" rev-parse --short HEAD)"
    echo -e "layout\t$LAYOUT"
    echo -e "rung\t$( ((NODES > 1)) && echo L3 || echo L2b )"
    echo -e "nodes\t$NODES"
    echo -e "ranks\t$RANKS"
    echo -e "ranks_per_node\t$RANKS_PER_NODE"
    echo -e "cores_per_rank\t$CORES_PER_RANK"
    echo -e "partitions_per_rank\t$CORES_PER_RANK"
    echo -e "reps\t$REPS"
    echo -e "smoke\t${SMOKE:-0}"
    echo -e "rss_source\tbench-json memhwm (ranks' VmHWM summed)"
    echo -e "partition\t${SLURM_JOB_PARTITION:-?}"
    echo -e "nodelist\t${SLURM_JOB_NODELIST:-?}"
    echo -e "job\t${SLURM_JOB_ID:-local}"
} > "$RESULTS/CELL-META.tsv"
echo -e "label\tmax_rss_kb\twall_s" > "$RESULTS/PEAK-RSS.tsv"
echo -e "label\tworst_rank_rss_kb\twall_s" > "$RESULTS/WORST-RANK-RSS.tsv"

# rows: name|selector|flags -- the layout is the job's, not the row's.
declare -A ROW_SPEC=(
    [hubbard]="test_model_propagate and hubbard|--hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06"
    [pauli]="test_model_propagate and pauli|--pauli-cutoff=14 --pauli-lower-atol=8.9e-06"
)
# SMOKE=1: tiny rows to check the plumbing (labels, srun shape, memhwm capture, summary). Same
# selectors and launch path, so it exercises everything but the size. NOT a measurement.
if [ "${SMOKE:-0}" = 1 ]; then
    ROW_SPEC[hubbard]="test_model_propagate and hubbard|--hubbard-cutoff=6 --hubbard-lower-atol=1e-3"
    ROW_SPEC[pauli]="test_model_propagate and pauli|--pauli-cutoff=6 --pauli-lower-atol=1e-2"
fi

declare -a ROW_LIST=()
for r in $ROWS; do
    [ -n "${ROW_SPEC[$r]:-}" ] || { echo "unknown row '$r'; known: ${!ROW_SPEC[*]}" >&2; exit 1; }
    ROW_LIST+=("$r|${ROW_SPEC[$r]}")
done

PY0="${ARM_VENV[0]}/bin/python"

# Peak memory and the recorded shape, from the cell's own bench JSON. Rank 0 writes it, so there is
# one file per cell however many ranks ran. memhwm is a dict of node id -> summed bytes; a grouped
# row has several entries and the MAXIMUM is the provisioning figure, never the sum (§5).
cell_mem() { # <label>.json -> "sum_kb max_kb ranks nodes partitions_env"
    "$PY0" - "$1" <<'PY'
import json, sys

try:
    with open(sys.argv[1]) as fh:
        data = json.load(fh)
except (OSError, ValueError):
    print("NA NA NA NA NA")
    raise SystemExit


def peak_kb(section):
    values = [v for v in (data.get(section) or {}).values() if isinstance(v, int)]
    return str(max(values) // 1024) if values else "NA"


meta = data.get("meta") or {}
print(
    peak_kb("memhwm"),
    peak_kb("memhwm_max"),
    meta.get("ranks", "NA"),
    meta.get("nodes", "NA"),
    meta.get("partitions_env", "NA"),
)
PY
}

run_cell() { # name venv envstr row rep
    local side=$1 venv=$2 envstr=$3 row=$4 rep=$5
    IFS='|' read -r model selector flags <<<"$row"
    local group=fresh
    case "$selector" in *gradient*|*energy*|*build_graph*) group=graph;; esac
    local label="N${NODES}_${LAYOUT}_${model}_${group}_${side}_r${rep}"
    local out="$RESULTS/${label}.log"
    echo "######## $label ########"

    # Exported, so srun's --export=ALL carries them to every rank. monoprop_PARTITIONS is
    # mandatory under MPI (conftest.py raises without it) and must equal the CPUs each rank sees.
    export monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="$RESULTS"
    export monoprop_PARTITIONS="$CORES_PER_RANK" monoprop_NUM_THREADS="$CORES_PER_RANK"
    export OMP_NUM_THREADS="$CORES_PER_RANK" MALLOC_ARENA_MAX="$CORES_PER_RANK"
    [ "$COMM_PROFILE" = 1 ] && export monoprop_COMM_PROFILE=1

    # Per-arm env goes on the task command, not in this shell: `env A=1 python ...` sets it in the
    # ranks and nowhere else, so the next arm's cell cannot inherit it.
    local -a task=()
    if [ -n "$envstr" ]; then
        local -a envpairs=()
        IFS=',' read -ra envpairs <<< "$envstr"
        task=(env "${envpairs[@]}")
    fi
    local -a flagwords=()
    read -ra flagwords <<< "$flags"
    task+=("$venv/bin/python" -m pytest "$BENCH_TREE/benches" -o filterwarnings=default
           -k "$selector" "${flagwords[@]}" --bench-rounds=1
           --benchmark-json="$RESULTS/time-${label}.json"
           -q -s -p no:cacheprovider)

    # Rank, core and binding flags are repeated here: allocation directives may not propagate to
    # the step, and an unbound step cost 1.45x at 8x16 (§4). -s keeps each rank's C++ stderr
    # (COMMPROF and friends) in the cell log instead of pytest's fd capture.
    local t0=$SECONDS
    srun --mpi=pmix --export=ALL \
        --nodes="$NODES" --ntasks="$RANKS" --ntasks-per-node="$RANKS_PER_NODE" \
        --cpus-per-task="$CORES_PER_RANK" --cpu-bind=cores --distribution=block:block \
        "${task[@]}" >"$out" 2>&1 \
        || echo "!! $label failed (rc=$?) -- see $out"
    local wall=$((SECONDS - t0))

    local sum_kb max_kb j_ranks j_nodes j_parts
    read -r sum_kb max_kb j_ranks j_nodes j_parts < <(cell_mem "$RESULTS/${label}.json")
    echo -e "${label}\t${sum_kb}\t${wall}" >> "$RESULTS/PEAK-RSS.tsv"
    echo -e "${label}\t${max_kb}\t${wall}" >> "$RESULTS/WORST-RANK-RSS.tsv"

    # The shape the ranks actually resolved, checked against the shape asked for. A mismatch means
    # the timing describes a different experiment (§4: all ranks must resolve the same S).
    if [ "$j_ranks" != "$RANKS" ] || [ "$j_nodes" != "$NODES" ] || [ "$j_parts" != "$CORES_PER_RANK" ]; then
        echo "!! $label: recorded shape ranks=$j_ranks nodes=$j_nodes partitions=$j_parts," \
             "expected ranks=$RANKS nodes=$NODES partitions=$CORES_PER_RANK"
    fi
    echo "   ${wall}s  peak(summed)=${sum_kb} kB  worst-rank=${max_kb} kB  ranks=$j_ranks partitions=$j_parts"
    grep -ao "COMMPROF[^[:cntrl:]]*" "$out" 2>/dev/null | sort -u | head -4 | sed 's/^/   /'
}

# arms_for_rep: rep -> space-separated arm indices, rotated by one arm per rep (rep 1: 0 1 2 ...,
# rep 2: 1 2 0 ..., rep 3: 2 0 1 ...). At N_ARMS=2 this is the per-rep flip.
arms_for_rep() {
    local rep=$1 off=$(( ($1 - 1) % N_ARMS )) i order=()
    for (( i = 0; i < N_ARMS; i++ )); do order+=( $(( (off + i) % N_ARMS )) ); done
    echo "${order[@]}"
}

echo
echo "=== $( ((NODES > 1)) && echo L3 || echo L2b ) at $LAYOUT ($REPS reps, ${#ROW_LIST[@]} rows, $N_ARMS arms) ==="
for rep in $(seq 1 "$REPS"); do
    for idx in $(arms_for_rep "$rep"); do
        for row in "${ROW_LIST[@]}"; do
            run_cell "${ARM_NAME[$idx]}" "${ARM_VENV[$idx]}" "${ARM_ENV[$idx]}" "$row" "$rep"
        done
    done
done

echo; echo "=== summary ==="
"$PY0" hpc/deucalion/tools/ab_pairs.py "$RESULTS" | tee "$RESULTS/AB-SUMMARY.md"
