#!/bin/bash -l
# Size PR #323's Bencher rungs for the CI runner (c7i.4xlarge: 8 physical cores, 32 GiB).
#
#   cd worktrees/bench-rungs2 && sbatch -A "$MONOPROP_SLURM_ACCOUNT" -t 1:45:00 \
#       hpc/deucalion/sbatch/bench-ci-calib.sh
#
# The workflow passes no size flags, so all three rungs run the option defaults and nobody
# has measured what they cost. This produces the terms / ~s / ~GiB columns for L1 (1x1),
# L2a (1x8) and L2b (4x2), and the `--obs-terms` value whose L2b node sum lands near 24 GiB.
#
# 8 CPUs, not 128: SMT is off here, so `taskset -c 0-7` is 8 physical cores and the same
# R*S world-slot count as the instance. Silicon differs (znver2 vs Sapphire Rapids), so the
# memory columns transfer and the time columns are indicative.
#
#SBATCH --job-name=mp-ci-calib
#SBATCH --partition=dev-x86
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

# --chdir is overridden by this on purpose; the *-worktree.sh scripts do the same.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export MONOPROP_SRC="$PWD"
source hpc/deucalion/env.sh || exit 1

# A default VENV is a different tree's _core.so with no bench tools in it.
case "$VENV" in "$PWD/.venv") : ;; *) echo "REFUSING: VENV=$VENV" >&2; exit 1 ;; esac

OUT="$PROJ/aaron/scratch/pr323-l2-calib"
mkdir -p "$OUT"
export monoprop_BENCH_RESULTS="$OUT"

CORES="0-7"
NCORES=8

echo "=== node ==="
echo "host        $(hostname)"
echo "cpus        $(getconf _NPROCESSORS_ONLN) online, AllocCPUS=${SLURM_CPUS_ON_NODE:-?}"
echo "mask        $CORES ($NCORES physical, SMT off)"
echo "commit      $(git rev-parse --short HEAD)"
echo

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    echo "=== build ==="
    bash hpc/deucalion/build.sh || { echo "BUILD FAILED -- refusing to measure" >&2; exit 1; }
fi

"$VENV/bin/python" - <<'PY' || exit 1
import sys
import monoprop
from monoprop import _core
import hashlib, pathlib
if not monoprop.has_mpi:
    sys.exit("monoprop built without MPI; every rung shares this binary")
p = pathlib.Path(_core.__file__)
print("has_mpi", monoprop.has_mpi, "MAX_NUM_MODES", monoprop.MAX_NUM_MODES)
print("_core.so", hashlib.md5(p.read_bytes()).hexdigest(), p)
PY

# ---------------------------------------------------------------------------
# Flag sets. L1 is LADDER.md's own L1 rows; the L2 group is one operator with
# three timed operations against it (build_graph publishes the graph energy and
# gradient evaluate), so its peak is the max over the three and not a sum.
# ---------------------------------------------------------------------------
L1_K='(test_model_propagate and (hubbard or pauli)) or (test_random_gradient and heisenberg)'
L1_FLAGS=(--hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05
          --pauli-cutoff=12 --pauli-lower-atol=1.22e-04
          --num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000)

L2_K='(test_random_build_graph or test_random_energy or test_random_gradient) and heisenberg'
L2_BASE=(--num-generators=1000 --num-modes=142 --cutoff=6)

# run_serial LABEL PARTITIONS ROUNDS -k EXPR -- flags...
run_serial() {
    local label="$1" parts="$2" rounds="$3" kexpr="$4"; shift 4
    echo "::: $label  1 rank x $parts partitions, rounds=$rounds"
    monoprop_BENCH_LABEL="$label" \
    monoprop_PARTITIONS="$parts" monoprop_NUM_THREADS="$parts" \
        taskset -c "$CORES" "$VENV/bin/python" -m pytest benches \
        -o filterwarnings=default -p no:cacheprovider \
        --benchmark-json="$OUT/time-$label.json" \
        -k "$kexpr" --bench-rounds "$rounds" "$@" \
        > "$OUT/log-$label.txt" 2>&1
    local rc=$?
    echo "    rc=$rc $(grep -aoE '[0-9]+ (passed|failed|error)[^,]*' "$OUT/log-$label.txt" | tr '\n' ' ')"
}

# run_mpi LABEL RANKS PARTITIONS ROUNDS -k EXPR -- flags...
# srun rather than mpiexec: PRRTE sizes itself from the Slurm allocation (128 cores) and
# would bind outside the taskset mask. Launcher affects placement, not terms or bytes.
run_mpi() {
    local label="$1" ranks="$2" parts="$3" rounds="$4" kexpr="$5"; shift 5
    echo "::: $label  $ranks ranks x $parts partitions, rounds=$rounds"
    monoprop_BENCH_LABEL="$label" \
    monoprop_PARTITIONS="$parts" monoprop_NUM_THREADS="$parts" \
        srun --mpi=pmix --ntasks="$ranks" --cpus-per-task="$parts" --cpu-bind=cores \
        "$VENV/bin/python" -m pytest benches \
        -o filterwarnings=default -p no:cacheprovider \
        --benchmark-json="$OUT/time-$label.json" \
        -k "$kexpr" --bench-rounds "$rounds" "$@" \
        > "$OUT/log-$label.txt" 2>&1
    local rc=$?
    echo "    rc=$rc $(grep -aoE '[0-9]+ (passed|failed|error)[^,]*' "$OUT/log-$label.txt" | tr '\n' ' ')"
}

# ---------------------------------------------------------------------------
# A. Shape and placement, at smoke sizes. Cheap, and it is the only step that
#    checks the launcher CI actually uses.
# ---------------------------------------------------------------------------
echo
echo "=== A. shape probe (smoke sizes) ==="
SMOKE=(--num-generators=8 --num-modes=8 --cutoff=6 --obs-terms=16)
run_serial probe-1x1 1 1 "$L2_K" "${SMOKE[@]}"
run_serial probe-1x8 8 1 "$L2_K" "${SMOKE[@]}"
run_mpi    probe-4x2 4 2 1 "$L2_K" "${SMOKE[@]}"

echo "::: probe-4x2-mpiexec  the CI launcher verbatim, inside the mask"
monoprop_BENCH_LABEL=probe-4x2-mpiexec \
monoprop_PARTITIONS=2 monoprop_NUM_THREADS=2 \
    taskset -c "$CORES" mpiexec -n 4 --map-by slot:PE=2 --bind-to core \
    -x monoprop_BENCH_LABEL -x monoprop_BENCH_RESULTS \
    -x monoprop_PARTITIONS -x monoprop_NUM_THREADS \
    "$VENV/bin/python" -m pytest benches \
    -o filterwarnings=default -p no:cacheprovider \
    --benchmark-json="$OUT/time-probe-4x2-mpiexec.json" \
    -k "$L2_K" --bench-rounds 1 "${SMOKE[@]}" \
    > "$OUT/log-probe-4x2-mpiexec.txt" 2>&1
echo "    rc=$?"

# ---------------------------------------------------------------------------
# B. L1 at its own sizes, 1x1, rounds 3.
# ---------------------------------------------------------------------------
echo
echo "=== B. L1 (1x1, rounds 3) ==="
run_serial L1 1 3 "$L1_K" "${L1_FLAGS[@]}"

# ---------------------------------------------------------------------------
# C. L2 sweep. 4x2 is the binding rung -- the per-rank cost is paid R times on
#    one node -- so the ceiling is read there and 1x8 is measured alongside.
# ---------------------------------------------------------------------------
echo
echo "=== C. L2 sweep (rounds 1) ==="
for obs in 1500000 2500000 3500000; do
    run_mpi    "L2b-obs$obs" 4 2 1 "$L2_K" "${L2_BASE[@]}" "--obs-terms=$obs"
    run_serial "L2a-obs$obs" 8 1   "$L2_K" "${L2_BASE[@]}" "--obs-terms=$obs"
done

# ---------------------------------------------------------------------------
# D. Summary. VmHWM, never a sampled peak.
# ---------------------------------------------------------------------------
echo
echo "=== D. summary ==="
"$VENV/bin/python" - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
GIB = 1024 ** 3

def rows(path):
    d = json.loads(path.read_text())
    meta, params = d.get("meta", {}), d.get("params", {})
    pin = meta.get("pinning", {})
    hdr = (f"{path.stem:<18} ranks={meta.get('ranks')} P={meta.get('partitions_env')} "
           f"obs={params.get('obs_terms')} "
           f"pinned={pin.get('single_cpu_threads_min')}/{pin.get('threads')} "
           f"mask={pin.get('affinity_cpus_min')}..{pin.get('affinity_cpus_max')}")
    print(hdr)
    terms = {k: v["terms"] for k, v in d.get("opsize", {}).items() if "::" not in k}
    print(f"{'':4}terms {terms}")
    for nid, sm in sorted(d.get("memhwm", {}).items()):
        mx = d.get("memhwm_max", {}).get(nid, 0)
        print(f"{'':4}{nid.split('::')[-1]:<38} sum {sm/GIB:7.2f} GiB   max {mx/GIB:7.2f} GiB")
    tj = out / f"time-{path.stem}.json"
    if tj.is_file():
        t = json.loads(tj.read_text())
        for b in t.get("benchmarks", []):
            print(f"{'':4}{b['name']:<38} {b['stats']['mean']:9.3f} s mean")
    print()

for p in sorted(out.glob("*.json")):
    if p.name.startswith("time-"):
        continue
    try:
        rows(p)
    except Exception as exc:  # noqa: BLE001
        print(f"{p.stem}: unreadable ({exc})")
PY

echo
echo "artifacts in $OUT"
exit 0
