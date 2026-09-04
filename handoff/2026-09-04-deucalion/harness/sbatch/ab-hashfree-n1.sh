#!/bin/bash -l
# N-arm interleaved A/B on one node, PR #317's ladder rows, for the hash-free engine phases.
#
#   cd worktrees/<port-tree> && MAIN_VENV=... PORT_VENV=... \
#     sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" hpc/deucalion/sbatch/ab-hashfree-n1.sh
#
#   # or, for 2+ arms with per-arm environment overrides:
#   cd worktrees/<port-tree> && \
#     ARMS="main=/path/venvA port=/path/venvB drop=/path/venvB:monoprop_DROP_SILENT_RECORDS=1" \
#     sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir="$PWD" hpc/deucalion/sbatch/ab-hashfree-n1.sh
#
# Arms are >=2 venvs (installed _core.so's), given as ARMS: space-separated
# "name=venv[:VAR=VAL[,VAR=VAL...]]" tokens; the first arm is the reference. When ARMS is unset it
# defaults to "main=$MAIN_VENV port=$PORT_VENV", so the old two-venv interface keeps working. Two
# arms may point at the same venv as long as their env differs -- identity is the (md5, env) pair,
# not the venv alone. Arm names become the label's <arm> field (hpc/deucalion/tools/ab_pairs.py
# parses it), so they must be bare alphanumerics, no "_"/"="/":" .
#
# Every rep runs all arms back to back, in an order that rotates by one arm per rep (rep 1: a b c,
# rep 2: b c a, rep 3: c a b, ... -- for two arms this reproduces the old per-rep flip), so a
# node-state swing hits every arm and cancels in the paired ratio (hpc/deucalion/tools/ab_pairs.py).
# Rows come from benches/LADDER.md on BENCH_TREE (PR #317), whose benches/ and bench tools are used
# for every arm via PYTHONPATH, so the harness is one revision whatever the arms' trees carry.
#
#   L1  : R=1, P=1, T=1  -- hubbard propagate, pauli propagate, random heisenberg gradient (REPS)
#   L2a : R=1, P=128     -- hubbard propagate at ~1B terms (REPS_L2)
#
# Peak RSS is the kernel's, from /usr/bin/time -v around each cell (PEAK-RSS.tsv); the engine's own
# ledger (opmembreak in <label>.json) is reported beside it as the secondary diagnostic.
#
#SBATCH --job-name=mp-ab-hf
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh

: "${MAIN_VENV:=$PROJ/aaron/worktrees/base-296/.venv}"
: "${PORT_VENV:=$PROJ/aaron/worktrees/hashfree-kernel/.venv}"
: "${ARMS:=main=$MAIN_VENV port=$PORT_VENV}"
: "${BENCH_TREE:=$PROJ/aaron/worktrees/bench-317}"
: "${REPS:=10}"
: "${REPS_L2:=3}"
: "${RESULTS_TAG:=}"
: "${L2_PARTITIONS:=128}"

RESULTS="${MONOPROP_RUNS}/ab-hashfree${RESULTS_TAG:+-${RESULTS_TAG}}-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"
echo "results: $RESULTS"
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
# reference arm. ARM_ENV entries are comma-separated VAR=VAL pairs (or "") applied only around that
# arm's pytest invocation in run_cell, so they never leak into a sibling arm's run.
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
    "$venv/bin/python" -c "import monoprop, pytest_benchmark; assert monoprop.has_mpi" || exit 1
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
    echo -e "node\t$(hostname)"; echo -e "job\t${SLURM_JOB_ID:-local}"
} > "$RESULTS/CELL-META.tsv"
echo -e "label\tmax_rss_kb\twall_s" > "$RESULTS/PEAK-RSS.tsv"

# rows: name|layout|partitions|selector|flags
L1_ROWS=(
    "hubbard|A_1x1|1|test_model_propagate and hubbard|--hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05"
    "pauli|A_1x1|1|test_model_propagate and pauli|--pauli-cutoff=12 --pauli-lower-atol=1.22e-04"
    "randheis|A_1x1|1|test_random_gradient and heisenberg|--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000"
)
L2_ROWS=(
    "hubbard|C_1x${L2_PARTITIONS}|${L2_PARTITIONS}|test_model_propagate and hubbard|--hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06"
)
# SMOKE=1: tiny rows on a login node to check the plumbing (labels, RSS capture, summary), no L2.
if [ "${SMOKE:-0}" = 1 ]; then
    L1_ROWS=(
        "randheis|A_1x1|1|test_random_gradient and heisenberg|--num-generators=20 --num-modes=32 --cutoff=4 --obs-terms=2000"
        "hubbard|A_1x1|1|test_model_propagate and hubbard|--hubbard-cutoff=4 --hubbard-lower-atol=1e-2 --hubbard-num-sites=8 --hubbard-observable-site=3"
    )
    L2_ROWS=()
fi

run_cell() { # name venv envstr row rep
    local side=$1 venv=$2 envstr=$3 row=$4 rep=$5
    IFS='|' read -r model layout parts selector flags <<<"$row"
    local group=fresh
    case "$selector" in *gradient*|*energy*|*build_graph*) group=graph;; esac
    local label="N1_${layout}_${model}_${group}_${side}_r${rep}"
    local out="$RESULTS/${label}.log"
    echo "######## $label ########"
    export monoprop_BENCH_LABEL="$label" monoprop_BENCH_RESULTS="$RESULTS"
    export monoprop_PARTITIONS="$parts" monoprop_NUM_THREADS="$parts" MALLOC_ARENA_MAX="$parts" OMP_NUM_THREADS="$parts"
    local -a cmd=()
    if [ -n "$envstr" ]; then
        local -a envpairs=()
        IFS=',' read -ra envpairs <<< "$envstr"
        cmd=(env "${envpairs[@]}")
    fi
    local -a flagwords=()
    read -ra flagwords <<< "$flags"
    cmd+=("$venv/bin/python" -m pytest "$BENCH_TREE/benches" -o filterwarnings=default
          -k "$selector" "${flagwords[@]}" --bench-rounds=1
          --benchmark-json="$RESULTS/time-${label}.json"
          -q -s -p no:cacheprovider)
    local t0=$SECONDS
    /usr/bin/time -v -o "$RESULTS/${label}.time" "${cmd[@]}" >"$out" 2>&1 \
        || echo "!! $label failed (rc=$?) -- see $out"
    local rss; rss=$(awk -F: '/Maximum resident set size/ {gsub(/ /,"",$2); print $2}' "$RESULTS/${label}.time")
    echo -e "${label}\t${rss:-NA}\t$((SECONDS - t0))" >> "$RESULTS/PEAK-RSS.tsv"
}

# arms_for_rep: rep -> space-separated arm indices, rotated by one arm per rep (rep 1: 0 1 2 ...,
# rep 2: 1 2 0 ..., rep 3: 2 0 1 ...). At N_ARMS=2 this is exactly the old per-rep flip.
arms_for_rep() {
    local rep=$1 off=$(( ($1 - 1) % N_ARMS )) i order=()
    for (( i = 0; i < N_ARMS; i++ )); do order+=( $(( (off + i) % N_ARMS )) ); done
    echo "${order[@]}"
}

echo "=== L1 ($REPS reps) ==="
for rep in $(seq 1 "$REPS"); do
    for idx in $(arms_for_rep "$rep"); do
        for row in "${L1_ROWS[@]}"; do
            run_cell "${ARM_NAME[$idx]}" "${ARM_VENV[$idx]}" "${ARM_ENV[$idx]}" "$row" "$rep"
        done
    done
done
echo "=== L2a ($REPS_L2 reps) ==="
for rep in $(seq 1 "$REPS_L2"); do
    for idx in $(arms_for_rep "$rep"); do
        for row in "${L2_ROWS[@]}"; do
            run_cell "${ARM_NAME[$idx]}" "${ARM_VENV[$idx]}" "${ARM_ENV[$idx]}" "$row" "$rep"
        done
    done
done

echo; echo "=== summary ==="
"${ARM_VENV[0]}/bin/python" hpc/deucalion/tools/ab_pairs.py "$RESULTS" | tee "$RESULTS/AB-SUMMARY.md"
