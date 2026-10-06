#!/bin/bash -l
# Strong and weak node-scaling ladders for the paper figures. ONE arm: the shipped default.
#
# WHY ONE ARM. These cells exist to show how monoprop scales, not to compare two transports, so
# `monoprop_ROUTING` is NOT SET anywhere here -- the measurement is the default configuration a
# user gets. The in-process gate below therefore requires the routing mode to read back as
# `default`; anything else means an env var leaked in from the submitting shell.
#
# WHY SIX TABLES. Strong and weak cannot share a family parameter: weak holds terms/node constant
# so terms/node IS its parameter, while strong holds the total constant and terms/node varies as
# S/N along every curve. Both families are instead built from ONE geometric term-count sequence,
# so a cell that appears in both is EXACTLY the same problem:
#
#   S0..S10 = 96.98M 184.1M 377.5M 781.7M 1.569G 3.105G 6.126G 12.25G 24.50G 49.01G 98.01G
#
#   weak curve j at N=2^i  uses total S_{j+i}   (constant terms/node, +-5% because the measured
#                                                sequence is x1.90-2.07, not exactly x2)
#   strong curve k         uses total S_k at every N
#
# Seven cells are shared between the two families and are measured ONCE. They double as free
# consistency checks: the same problem, the same node count, reached through two different ladder
# definitions in different allocations.
#
# S4 @ N=16 is also the check that an UNSET monoprop_ROUTING is identical to the explicit
# `linear` used by the earlier campaign's weak/16 cell -- if those two disagree, the default is
# not what was measured before and that is a finding, not a rounding error.
#
#   LADDER=weak_1569m ./scaleladder.sh              # print the rung table and submit lines
#   cd $PROJ/src/296-on-bench && LADDER=... REPS=1 sbatch ... scaleladder.sh
#
# LADDER must be exported into the job (`--export=ALL,LADDER=...`): it selects the rung table,
# the size knob and the results directory, and getting it wrong files a cell under another
# curve's name.
#
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

PROJ=/projects/EEHPC-DEV-2026D08-260
SELF="$PROJ/scratch/pr296bench/scaleladder.sh"
ACCOUNT=eehpc-dev-2026d08-260x
TREE="$PROJ/src/296-on-bench"
SO_MD5=${SO_MD5:-1157a5e2b421fb8bd18fc6d16fa39778}
TREE_SHA=${TREE_SHA:-d0a9e221}

# A new enum value has to reach EVERY validator. A previous campaign lost 12 cells to a value
# that reached the summariser and not the driver, so the list is written once, here, and the
# case below is generated from it.
LADDERS="weak_97m weak_377m weak_1569m strong_s4 strong_s6 strong_s8"
LADDER=${LADDER:-}
case " $LADDERS " in
    *" $LADDER "*) ;;
    *) echo "refusing: LADDER must be one of: $LADDERS -- got '${LADDER}'" >&2; exit 2 ;;
esac
MODE="${LADDER%%_*}"                       # weak|strong, for messages and the meta record

CUTOFF=10
NUM_SITES=60
OBSERVABLE_SITE=46
RANKS_PER_NODE=8
PARTITIONS=16
LAYOUT=B_8x16
CPUS_PER_TASK=$((128 / RANKS_PER_NODE))
CPU_BIND=cores
CELL_SPEC='fresh|hubbard and propagate'
EXPECT_PLACEMENT=${EXPECT_PLACEMENT:-both}
MEM_CEIL_GIB=242.0
RC_CONFIG=2; RC_ARM=3; RC_ARENA=4; RC_TERMS=5; RC_ARTIFACT=6; RC_ROUTING=7; RC_OOM=9

# nodes reps lower_atol expect_terms walltime partition
#
# Rungs are EXCLUDED where the problem does not fit: GiB/node = 4.71 + 0.0618 * Mterms/node,
# fitted on the two measured extremes and reproducing both (97M/node -> 10.7 GiB measured 10.7;
# 3104M/node -> 196.5 GiB measured 196.5), against a 198 GiB working ceiling. Excluding a rung
# shortens a curve; lowering the ceiling would risk an OOM that costs the whole cell.
#
# CALIB rows refuse to run until calibseq.sh measures the atol and its EXACT term count is pasted
# in -- the per-rep gate compares against that count, so a mistyped atol fails the cell instead
# of silently measuring a different problem.
#
# WALLTIME = reps x predicted seconds x 1.6 + 900 s for build, MPI init and pytest. Slurm bills
# elapsed, not the allowance, so erring wide costs idle minutes while a requeue costs the rung.
RUNGS_weak_97m="
1   10 1.25e-05      96981051   0:20:00 dev-x86
2   10 8.8e-06      184124520   0:20:00 dev-x86
4   10 5.9e-06      377482074   0:20:00 normal-x86
8   10 3.9e-06      781669404   0:20:00 normal-x86
16  10 2.6e-06     1569152761   0:20:00 normal-x86
32  10 1.73e-06    3104527573   0:25:00 normal-x86
64  10 1.14e-06    6125805627   0:25:00 normal-x86
"
RUNGS_weak_377m="
1    5 5.9e-06      377482074   0:25:00 dev-x86
2    5 3.9e-06      781669404   0:25:00 dev-x86
4    5 2.6e-06     1569152761   0:25:00 normal-x86
8    5 1.73e-06    3104527573   0:25:00 normal-x86
16   5 1.14e-06    6125805627   0:25:00 normal-x86
32   5 7.35e-07   12255330837   0:25:00 normal-x86
64   5 4.68e-07   24419998198   0:30:00 normal-x86
"
RUNGS_weak_1569m="
1    5 2.6e-06     1569152761   0:45:00 dev-x86
2    5 1.73e-06    3104527573   0:45:00 dev-x86
4    5 1.14e-06    6125805627   0:45:00 normal-x86
8    5 7.35e-07   12255330837   0:45:00 normal-x86
16   5 4.68e-07   24419998198   0:45:00 normal-x86
32   5 2.94e-07   48317129677   0:50:00 normal-x86
64   5 1.82e-07   94684031363   0:55:00 normal-x86
"
RUNGS_strong_s4="
1    5 2.6e-06     1569152761   0:45:00 dev-x86
2    5 2.6e-06     1569152761   0:30:00 dev-x86
4    5 2.6e-06     1569152761   0:25:00 normal-x86
8    5 2.6e-06     1569152761   0:20:00 normal-x86
16   5 2.6e-06     1569152761   0:20:00 normal-x86
32   5 2.6e-06     1569152761   0:20:00 normal-x86
64   5 2.6e-06     1569152761   0:20:00 normal-x86
"
RUNGS_strong_s6="
2    5 1.14e-06    6125805627   1:15:00 dev-x86
4    5 1.14e-06    6125805627   0:45:00 normal-x86
8    5 1.14e-06    6125805627   0:30:00 normal-x86
16   5 1.14e-06    6125805627   0:25:00 normal-x86
32   5 1.14e-06    6125805627   0:25:00 normal-x86
64   5 1.14e-06    6125805627   0:20:00 normal-x86
"
RUNGS_strong_s8="
8    5 4.68e-07   24419998198   1:15:00 normal-x86
16   5 4.68e-07   24419998198   0:45:00 normal-x86
32   5 4.68e-07   24419998198   0:35:00 normal-x86
64   5 4.68e-07   24419998198   0:30:00 normal-x86
"
eval "RUNGS=\$RUNGS_$LADDER"

rung_row() { awk -v n="$1" '$1==n {print; f=1} END {exit !f}' <<< "$RUNGS"; }

# ------------------------------------------------------------------ no allocation: print the plan
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "scaleladder LADDER=$LADDER ($MODE) -- hubbard cutoff=$CUTOFF sites=$NUM_SITES obs=$OBSERVABLE_SITE"
    echo "  tree $TREE  sha $TREE_SHA  so_md5 $SO_MD5"
    echo "  ONE arm: monoprop_ROUTING is NOT set; the in-process gate requires 'default'"
    echo "  layout $LAYOUT, MALLOC_ARENA_MAX UNSET, memory ceiling ${MEM_CEIL_GIB} GiB"
    echo
    printf '%5s %6s %5s %11s %13s %10s %9s %11s %s\n' \
        nodes P reps lower_atol terms M/node walltime partition submit
    while read -r n reps atol terms w part; do
        [ -z "${n:-}" ] && continue
        r=${REPS:-$reps}
        printf '%5s %6s %5s %11s %13s %10s %9s %11s %s\n' "$n" \
            "$((n * RANKS_PER_NODE * PARTITIONS))" "$r" "$atol" "$terms" \
            "$(awk -v t="$terms" -v n="$n" 'BEGIN{printf "%.0f", t/n/1e6}')" "$w" "$part" \
            "cd $TREE && sbatch -A $ACCOUNT -p $part -N $n -t $w --export=ALL,LADDER=$LADDER,REPS=$r --job-name=mp-sc-${LADDER}-N${n}-r${r} $SELF"
    done <<< "$RUNGS"
    cat <<NOTE

Submit from INSIDE $TREE: this script cds to SLURM_SUBMIT_DIR, so --chdir is ignored and the
signature of getting it wrong is a 3-second FAILED.

CALIB rows refuse until calibseq.sh measures the atol and its exact term count is pasted here.

Results land in \$PROJ/runs/scale-hubbard-296-$LADDER-r<reps>-N<nodes>/ -- collate from an
explicit list, never a glob: runs/ is shared and a glob has already swept another session's
campaign into a report.
NOTE
    exit 0
fi

# ------------------------------------------------------------------ inside the allocation
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit "$RC_CONFIG"
export MONOPROP_SRC="$PWD"          # before sourcing, or VENV resolves to src/monoprop
source hpc/deucalion/env.sh
case "$PWD" in
    "$TREE") : ;;
    *) echo "refusing: submitted from $PWD, not $TREE" >&2; exit "$RC_CONFIG" ;;
esac

unset MALLOC_ARENA_MAX
# A leaked routing knob would make this cell a different configuration than the one it claims.
unset monoprop_ROUTING monoprop_ROUTE_LINEAR_BITS

export PYTHONPATH="$PWD/benches${PYTHONPATH:+:$PYTHONPATH}"
export monoprop_NUM_THREADS="$PARTITIONS" monoprop_PARTITIONS="$PARTITIONS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NODES="${SLURM_JOB_NUM_NODES:-1}"
NTASKS=$((RANKS_PER_NODE * NODES))
row=$(rung_row "$NODES") || { echo "refusing: N=$NODES is not a rung of $LADDER" >&2; exit "$RC_CONFIG"; }
read -r _n RUNG_REPS LOWER_ATOL EXPECT_TERMS RUNG_WALL RUNG_PART <<< "$row"
REPS=${REPS:-$RUNG_REPS}

[ "$LOWER_ATOL" = CALIB ] && {
    echo "refusing: this rung's lower_atol is unmeasured. Run calibseq.sh and paste the atol AND" >&2
    echo "refusing: its exact term count into RUNGS_$LADDER." >&2; exit "$RC_CONFIG"; }
case "$EXPECT_TERMS" in ''|*[!0-9]*|0) echo "refusing: expect_terms='$EXPECT_TERMS' is not a positive integer" >&2; exit "$RC_CONFIG" ;; esac
case "$REPS" in ''|*[!0-9]*|0) echo "refusing: REPS='$REPS' is not a positive integer" >&2; exit "$RC_CONFIG" ;; esac
case "$EXPECT_PLACEMENT" in both|port-only|main-only|neither) ;;
    *) echo "refusing: EXPECT_PLACEMENT='$EXPECT_PLACEMENT' is not one of both/port-only/main-only/neither" >&2
       exit "$RC_CONFIG" ;;
esac
[ "$NODES" -gt 2 ] && [ "${SLURM_JOB_PARTITION:-}" = dev-x86 ] && {
    echo "refusing: N=$NODES on dev-x86 (MaxNodes=2)" >&2; exit "$RC_CONFIG"; }
case "$NTASKS" in
    1|2|4|8|16|32|64|128|256|512|1024|2048) ;;
    *) echo "refusing: R=$NTASKS is not a power of two; linear routing would clamp to d=0" >&2
       exit "$RC_CONFIG" ;;
esac

VENV_PATH="$TREE/.venv"
NAME="${CELL_SPEC%%|*}"
SELECT="${CELL_SPEC#*|}"

RESULTS="$MONOPROP_RUNS/scale-hubbard-296-${LADDER}-r${REPS}-N${NODES}"
if [ -e "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null)" ]; then
    echo "refusing: $RESULTS already holds results" >&2; exit "$RC_CONFIG"
fi
mkdir -p "$RESULTS" || exit "$RC_CONFIG"

PER_NODE_M=$(awk -v t="$EXPECT_TERMS" -v n="$NODES" 'BEGIN{printf "%.1f", t/n/1e6}')
echo "=== scaleladder LADDER=$LADDER  N=$NODES  partition=${SLURM_JOB_PARTITION:-?}  job=$SLURM_JOB_ID"
echo "=== reps      $REPS (table says $RUNG_REPS; the value ACTUALLY used is this one)"
[ "$REPS" = 1 ] && echo "=== PASS 1: plumbing and the memory gate. NO number from this pass is quotable."
echo "=== layout    $LAYOUT ranks=$NTASKS world=$((NTASKS * PARTITIONS)) cpus_per_task=$CPUS_PER_TASK"
echo "=== size      cutoff=$CUTOFF lower_atol=$LOWER_ATOL expect_terms=$EXPECT_TERMS (${PER_NODE_M}M/node)"
echo "=== arm       DEFAULT routing, monoprop_ROUTING unset -- ONE binary $TREE_SHA $SO_MD5"
echo "=== arena     shell sees MALLOC_ARENA_MAX='${MALLOC_ARENA_MAX-}' (expected empty)"
echo "=== results   $RESULTS"

[ -x "$VENV_PATH/bin/python" ] || { echo "refusing: no $VENV_PATH/bin/python" >&2; exit "$RC_CONFIG"; }
"$VENV_PATH/bin/python" -c 'import monoprop, monoprop_bench_tools.memory.cpu, pytest_benchmark
assert monoprop.has_mpi, "built without MPI"' || {
    echo "PRECHECK FAILED for $VENV_PATH -- not burning the allocation." >&2; exit "$RC_CONFIG"; }
echo "=== precheck  venv imports monoprop, monoprop_bench_tools and pytest_benchmark"

HARNESS_DIR="$(cd hpc/deucalion && pwd -P)" || exit "$RC_CONFIG"
HARNESS_SHA="$(git -C "$HARNESS_DIR" rev-parse --short HEAD 2>/dev/null)" || {
    echo "refusing: $HARNESS_DIR is not a git repository, so this cell is unprovenanced" >&2
    exit "$RC_CONFIG"; }
[ -z "$(git -C "$HARNESS_DIR" status --porcelain 2>/dev/null)" ] || HARNESS_SHA="${HARNESS_SHA}-dirty"

SIZE_ARGS=("--hubbard-cutoff=$CUTOFF" "--hubbard-lower-atol=$LOWER_ATOL"
           "--hubbard-num-sites=$NUM_SITES" "--hubbard-observable-site=$OBSERVABLE_SITE")

{
    printf 'campaign\tscaleladder LADDER=%s\n' "$LADDER"
    printf 'harness_sha\t%s\n'      "$HARNESS_SHA"
    printf 'harness_dir\t%s\n'      "$HARNESS_DIR"
    printf 'benches_sha\t%s\n'      "$(git -C "$PWD" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    printf 'cpu_bind\t%s\n'         "$CPU_BIND"
    printf 'distribution\tblock:block\n'
    printf 'cpus_per_task\t%s\n'    "$CPUS_PER_TASK"
    printf 'ranks_per_node\t%s\n'   "$RANKS_PER_NODE"
    printf 'partitions\t%s\n'       "$PARTITIONS"
    printf 'nodes\t%s\n'            "$NODES"
    printf 'ntasks\t%s\n'           "$NTASKS"
    printf 'world\t%s\n'            "$((NTASKS * PARTITIONS))"
    printf 'layout\t%s\n'           "$LAYOUT"
    printf 'workload\thubbard\n'
    printf 'size_args\t%s\n'        "${SIZE_ARGS[*]}"
    printf 'cells\t%s\n'            "$CELL_SPEC"
    printf 'reps\t%s\n'             "$REPS"
    printf 'expect_placement\t%s\n' "$EXPECT_PLACEMENT"
    printf 'venv\t%s\n'             "$VENV_PATH"
    printf 'malloc_arena_max\tUNSET\n'
    printf 'tree_sha\t%s\n'         "$TREE_SHA"
    printf 'so_md5\t%s\n'           "$SO_MD5"
    printf 'routing\tdefault\n'
    printf 'ladder\t%s\n'           "$LADDER"
    printf 'ladder_mode\t%s\n'      "$MODE"
    printf 'expect_terms\t%s\n'     "$EXPECT_TERMS"
    printf 'terms_per_node_m\t%s\n' "$PER_NODE_M"
} > "$RESULTS/CELL-META.tsv"

printf 'label\trep\tcell\tsum_kb\tworst_kb\tranks\n' > "$RESULTS/PEAK-RSS.tsv"
printf 'label\trep\tso_md5\tarena_seen\trouting_seen\tterms\trc\n' > "$RESULTS/CELL-CHECKS.tsv"

# ------------------------------------------------------------------ the run loop
for rep in $(seq 1 "$REPS"); do
    LABEL="N${NODES}_${LAYOUT}_hubbard_${NAME}_r${rep}"
    OUT="$RESULTS/${LABEL}.log"

    # Arm identity is the installed .so md5 and nothing else. Checked EVERY rep because rungs of
    # one curve are hours apart and a rebuild landing between them re-bases the whole curve.
    got_md5=$(md5sum "$VENV_PATH"/lib/python*/site-packages/monoprop/_core*.so 2>/dev/null | awk '{print $1}')
    echo "######## $LABEL  so_md5=$got_md5  R=$NTASKS S=$PARTITIONS world=$((NTASKS * PARTITIONS)) ########"
    if [ "$got_md5" != "$SO_MD5" ]; then
        echo "!!!! ARM IDENTITY CHANGED: expected $SO_MD5, found $got_md5" >&2
        echo "!!!! $VENV_PATH was rebuilt or reinstalled. Every point on this curve is re-based." >&2
        exit "$RC_ARM"
    fi

    export monoprop_BENCH_LABEL="$LABEL" monoprop_BENCH_RESULTS="$RESULTS"
    srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RANKS_PER_NODE" \
         --cpus-per-task="$CPUS_PER_TASK" \
         --cpu-bind="$CPU_BIND" --distribution=block:block \
         /usr/bin/time -v "$VENV_PATH/bin/python" -m pytest benches/bench_models.py \
         -o filterwarnings=default -k "$SELECT" -m slow "${SIZE_ARGS[@]}" \
         --bench-rounds=1 --benchmark-json="$RESULTS/time-${LABEL}.json" \
         -q -s -p no:cacheprovider > "$OUT" 2>&1
    rc=$?

    # Memory FIRST and unconditionally: a failed cell is exactly the one whose RSS matters.
    read -r rss_sum rss_worst rss_n < <(awk '
        /Maximum resident set size/ {kb=$NF; s+=kb; if (kb>m) m=kb; n++}
        END {print s+0, m+0, n+0}' "$OUT")
    node_gib=$(awk -v s="$rss_sum" -v n="$NODES" 'BEGIN{printf "%.1f", s/1048576/n}')
    pct=$(awk -v g="$node_gib" -v c="$MEM_CEIL_GIB" 'BEGIN{printf "%.1f", 100*g/c}')
    printf '#### MEMORY %s: node total %s GiB of %s (%s%%), worst rank %.1f GiB, %s of %s ranks reporting\n' \
        "$LABEL" "$node_gib" "$MEM_CEIL_GIB" "$pct" \
        "$(awk -v m="$rss_worst" 'BEGIN{printf "%.1f", m/1048576}')" "$rss_n" "$NTASKS"
    [ "$rss_n" -gt 0 ] && printf '%s\t%s\t%s\t%d\t%d\t%d\n' \
        "$LABEL" "$rep" "$NAME" "$rss_sum" "$rss_worst" "$rss_n" >> "$RESULTS/PEAK-RSS.tsv"
    [ "$rss_n" -lt "$NTASKS" ] && echo "!! only $rss_n of $NTASKS ranks reported RSS -- not a full max over ranks"

    # An OOM kill is a memory VERDICT, not broken plumbing, and must not read as one.
    # grep -a: a binary log makes plain grep print nothing, which looks like a clean job.
    if [ "$rc" -eq 137 ] || [ "$rc" -eq 9 ] \
       || grep -aqE 'oom.kill|Out of memory|Killed process|terminated by signal 9' "$OUT"; then
        echo "!!!! OOM: $LADDER N=$NODES DID NOT FIT at ${PER_NODE_M}M terms/node." >&2
        echo "!!!! node total ~$node_gib GiB of $MEM_CEIL_GIB (${pct}%). EXCLUDE this rung from" >&2
        echo "!!!! RUNGS_$LADDER and start the curve one rung later. Do not raise the ceiling." >&2
        grep -aE 'oom.kill|Out of memory|Killed process' "$OUT" | head -3 >&2
        exit "$RC_OOM"
    fi
    [ "$rc" -ne 0 ] && echo "!! $LABEL failed (rc=$rc) -- see $OUT"

    # The arena, the routing mode and the term count as the PYTHON PROCESS saw them. The shell's
    # unset is not evidence: a knob can arrive from the submitting environment, and a cell whose
    # routing is not the default is a different configuration than this curve claims.
    "$VENV_PATH/bin/python" - "$RESULTS/${LABEL}.json" "$EXPECT_TERMS" <<'PY'
import json, pathlib, sys
path, expect = pathlib.Path(sys.argv[1]), int(sys.argv[2])
if not path.exists():
    print("   !! no results json -- this rep is MISSING from the median, not equal"); sys.exit(0)
d = json.loads(path.read_text())
meta = d.get("meta", {})
arena = meta.get("malloc_arena_max")
routing = meta.get("monoprop_routing")
terms = sorted({v.get("terms") for v in d.get("opsize", {}).values()})
print("   arena seen by python:", arena, " threads:", meta.get("monoprop_threads"))
print("   routing seen by python:", routing, " bits:", meta.get("monoprop_route_linear_bits"))
print("   core md5 in-process :", meta.get("monoprop_core_md5"))
print("   placement:", meta.get("pinning", {}))
print("   terms    :", terms)
rc = 0
if arena != "default":
    print(f"   !!!! MALLOC_ARENA_MAX={arena} INSIDE THE PYTHON PROCESS, expected unset.")
    rc = 4
if routing is None:
    print("   !!!! this benches/conftest.py does not record monoprop_routing, so the")
    print("   !!!! configuration of this cell is UNPROVEN.")
    rc = 7
elif routing != "default":
    print(f"   !!!! ROUTING={routing} in-process, expected 'default'. A routing knob reached the")
    print("   !!!! ranks, so this cell is not the shipped configuration it claims to be.")
    rc = 7
if terms != [expect]:
    print(f"   !!!! TERM COUNT {terms}, expected [{expect}] -- wrong atol, or the model moved.")
    rc = 5
sys.exit(rc)
PY
    chk=$?
    read -r arena_seen routing_seen terms_seen < <("$VENV_PATH/bin/python" -c 'import json,sys
try:
    d = json.load(open(sys.argv[1])); m = d["meta"]
    t = sorted({v.get("terms") for v in d.get("opsize", {}).values()})
    print(m.get("malloc_arena_max"), m.get("monoprop_routing"), (t[0] if len(t) == 1 else t))
except Exception: print("NA NA NA")' "$RESULTS/${LABEL}.json" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$LABEL" "$rep" "$got_md5" "${arena_seen:-NA}" "${routing_seen:-NA}" \
        "${terms_seen:-NA}" "$chk" >> "$RESULTS/CELL-CHECKS.tsv"
    [ "$chk" -eq 4 ] && exit "$RC_ARENA"
    [ "$chk" -eq 5 ] && exit "$RC_TERMS"
    [ "$chk" -eq 7 ] && exit "$RC_ROUTING"

    # A rep whose artifacts vanished shrinks the median silently while the table still prints a
    # confident number. Named here, and recorded where the collator can see it.
    for want in "$RESULTS/time-${LABEL}.json" "$RESULTS/${LABEL}.json"; do
        [ -s "$want" ] || {
            echo "!! $LABEL produced no $(basename "$want") -- MISSING from the median, not equal"
            basename "$want" >> "$RESULTS/MISSING-ARTIFACTS"; }
    done
done

# ------------------------------------------------------------------ summary
echo
echo "=== identity, arena, routing and terms across reps ==="
cat "$RESULTS/CELL-CHECKS.tsv"
echo
echo "=== peak RSS, all reps ==="
cat "$RESULTS/PEAK-RSS.tsv"
echo
echo "=== cell summary (median over reps; ab_summary.py is NOT used -- it compares two arms) ==="
"$VENV_PATH/bin/python" - "$RESULTS" "$EXPECT_TERMS" "$NODES" <<'PY'
import json, pathlib, statistics, sys
d, expect, nodes = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
vals = []
for f in sorted(d.glob("time-*.json")):
    try:
        doc = json.loads(f.read_text())
    except (ValueError, OSError):
        continue
    for b in doc.get("benchmarks", []):
        if "propagate" in b.get("name", ""):
            vals.append(statistics.median(b["stats"]["data"]))
if not vals:
    print("   no timing data"); sys.exit(0)
med = statistics.median(vals)
lo, hi = min(vals), max(vals)
# Spread, not a p-value: these are unpaired points on a curve, so the honest summary is the
# median and how far the reps ranged, never a significance claim from one cell.
print(f"   reps={len(vals)}  median={med:.3f}s  min={lo:.3f}s  max={hi:.3f}s  "
      f"spread={100 * (hi - lo) / med:.1f}% of median")
print(f"   terms={expect}  nodes={nodes}  terms/node={expect / nodes / 1e6:.1f}M  "
      f"throughput={expect / med / nodes / 1e6:.3f} Mterms/s/node")
PY

if [ -s "$RESULTS/MISSING-ARTIFACTS" ]; then
    echo "!! $(wc -l < "$RESULTS/MISSING-ARTIFACTS") artifact(s) never appeared -- the summary" >&2
    echo "!! above is over FEWER reps than requested and does not say so." >&2
    exit "$RC_ARTIFACT"
fi
echo
echo "=== done: $RESULTS"
