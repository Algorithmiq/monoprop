#!/bin/bash -l
# Falsify (or fail to falsify) "the exchange count matrix is symmetric".
#
# The engine's resolve_recv runs an alltoall_counts to learn how much each peer will send it,
# and caches the answer per layer -- 8 B per world slot, 10.59 GiB at P=512. If what r sends me
# always equals what I send r, that collective and that cache are both unnecessary.
#
# The probe compares the alltoall's answer against the send counts on every resolve and writes
# counters to $MONOPROP_SYMCHECK.<pid>. A clean run must say "resolves>0, mismatched=0" --
# resolves=0 would mean the probe never saw cross-rank traffic, which is a vacuous pass, so the
# aggregator refuses it rather than printing a green line.
#
# The probe is NOT in the engine. Apply it, rebuild, and install the result into the venv, which
# is a separate step -- `cmake --build` leaves site-packages holding the previous binary:
#
#   cd $PROJ/src/mp-gws
#   git apply $PROJ/src/mp-gwsbench/hpc/deucalion/sbatch/symcheck-probe.patch
#   cmake --build build/editable/Release
#   SP=.venv/lib/python3.11/site-packages/monoprop
#   cp build/editable/Release/src/monoprop/bindings/_core*.so "$SP/"
#   cp build/editable/Release/libmonoprop.so "$SP/lib64/"
#   sbatch -A "$MONOPROP_SLURM_ACCOUNT" --chdir=$PWD -N2 \
#          $PROJ/src/mp-gwsbench/hpc/deucalion/sbatch/symcheck.sh
#
# Back up those two site-packages binaries first if the venv is a measured A/B arm, and restore
# them afterwards -- overwriting them silently redefines what that arm was.
#
#SBATCH --job-name=mp-symcheck
#SBATCH --partition=dev-x86
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:30:00
#SBATCH --output=/projects/EEHPC-DEV-2026D08-260/work-gws/%x-%j.out
#SBATCH --error=/projects/EEHPC-DEV-2026D08-260/work-gws/%x-%j.err

set -uo pipefail

PROJ=/projects/EEHPC-DEV-2026D08-260
GWS="$PROJ/src/mp-gws"
WORK="$PROJ/work-gws/symcheck-${SLURM_JOB_ID:-manual}"
mkdir -p "$WORK"

cd "$GWS"
export MONOPROP_SRC="$GWS"
source "$PROJ/src/mp-gwsbench/hpc/deucalion/env.sh"

NODES="${SLURM_JOB_NUM_NODES:-2}"

echo "=== build under test ==="
echo "branch : $(git rev-parse --abbrev-ref HEAD)   commit: $(git log --oneline -1)"
echo "dirty  : $(git status --porcelain | tr '\n' ' ')"
# The .so Python IMPORTS, not the one ninja wrote. This venv is an editable install that keeps
# its OWN COPY under site-packages: `cmake --build` updates build/editable/Release and leaves the
# imported binary untouched. A previous run of this job gated on the build-dir .so, passed, and
# then measured a binary from the day before -- three stages of "no probe files" that looked like
# "no cross-rank traffic". Ask the interpreter which file it loaded and check that one.
SO="$("$VENV/bin/python" -c 'from monoprop import _core; print(_core.__file__)')" || exit 1
echo "so     : $SO"
echo "so md5 : $(md5sum "$SO" | cut -d' ' -f1)"
"$VENV/bin/python" -c "import monoprop as m; assert m.has_mpi, 'built without MPI'" || exit 1
# The probe is compiled in unconditionally but inert without the env var. Prove the symbol is
# actually in the binary rather than trusting that the rebuild picked up the edit.
# grep -c, not grep -q: -q exits on the first match, SIGPIPEs `strings`, and under `pipefail`
# the pipeline then reports failure even though the string was found. That gate failed closed
# for a reason that had nothing to do with the binary.
PROBE_HITS=$(strings "$SO" | grep -c "mismatched_resolves")
if [ "$PROBE_HITS" -gt 0 ]; then
    echo "probe  : present in .so ($PROBE_HITS)"
else
    echo "probe  : ABSENT -- the .so predates the edit"
    exit 1
fi

rc_total=0

# ---------------------------------------------------------------------------
# One aggregation over a stage's per-rank probe files.
# ---------------------------------------------------------------------------
summarize() {
    local tag="$1"
    "$VENV/bin/python" - "$WORK" "$tag" <<'PY'
import pathlib, sys, re
work, tag = pathlib.Path(sys.argv[1]), sys.argv[2]
files = sorted(work.glob(f"sym-{tag}.*"))
tot = {"resolves": 0, "slots": 0, "mismatched_resolves": 0, "mismatched_slots": 0}
worst = []
for f in files:
    kv = dict(re.findall(r"(\w+)=(-?\d+)", f.read_text()))
    for k in tot:
        tot[k] += int(kv.get(k, 0))
    if int(kv.get("mismatched_slots", 0)):
        worst.append((f.name, kv))
print(f"  [{tag}] ranks reporting: {len(files)}")
print(f"  [{tag}] resolves={tot['resolves']:,}  slots compared={tot['slots']:,}")
print(f"  [{tag}] MISMATCHED resolves={tot['mismatched_resolves']:,} "
      f"slots={tot['mismatched_slots']:,}")
for name, kv in worst[:5]:
    print(f"    !! {name}: first_bad_slot={kv.get('first_bad_slot')} "
          f"send={kv.get('first_send')} recv={kv.get('first_recv')}")
if not files:
    print(f"  [{tag}] VERDICT: NO PROBE FILES -- inconclusive, not a pass")
    sys.exit(2)
if tot["resolves"] == 0:
    print(f"  [{tag}] VERDICT: VACUOUS -- no cross-rank resolve happened, nothing was tested")
    sys.exit(2)
if tot["mismatched_slots"]:
    print(f"  [{tag}] VERDICT: REFUTED -- the count matrix is NOT symmetric")
    sys.exit(1)
print(f"  [{tag}] VERDICT: survived -- every resolved recv count equalled its send count")
PY
}

# ---------------------------------------------------------------------------
# Stage 1: the Python MPI suite, three geometries. Broad path coverage.
# ---------------------------------------------------------------------------
echo
echo "############ stage 1: pytest --with-mpi ############"
# (ranks-per-node, partitions). ranks_per_node * S <= 128 or pinning silently disables.
for layout in "1 16" "8 16"; do
    read -r RPN S <<<"$layout"
    NTASKS=$((RPN * NODES))
    TAG="suite-${NTASKS}x${S}"
    echo
    echo "###### ranks=$NTASKS (${RPN}/node)  partitions=$S  world=$((NTASKS * S)) ######"
    export monoprop_NUM_THREADS="$S"
    export MONOPROP_SYMCHECK="$WORK/sym-${TAG}"
    srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RPN" \
         --cpus-per-task=$((128 / RPN)) \
         --cpu-bind=cores --distribution=block:block \
         "$VENV/bin/python" -m pytest tests --with-mpi -q 2>&1 | tail -15
    summarize "$TAG" || rc_total=$((rc_total + 1))
done

# ---------------------------------------------------------------------------
# Stage 2: a real operator. The suite's cases are small; this is the geometry and
# term count the memory claim is actually about, and it runs a GRADIENT, which is
# the round whose layout is now derived at scale 2.
# ---------------------------------------------------------------------------
echo
echo "############ stage 2: pauli c12, energy + gradient, world 256 ############"
# Driven from the BENCH worktree: the per-operation model benchmarks (test_model_energy,
# test_model_gradient) exist only there -- mp-gws's benches/bench_models.py has just test_model,
# so `-k "energy or gradient"` deselected everything and the stage measured nothing. The venv is
# still the gws one, so the binary under test is unchanged; only the test code comes from
# elsewhere. Verified below by printing which monoprop the interpreter actually imported.
cd "$PROJ/src/mp-gwsbench" || exit 1
"$VENV/bin/python" -c "import monoprop; print('stage 2 imports monoprop from:', monoprop.__file__)"
RPN=8
NTASKS=$((RPN * NODES))
TAG="models-${NTASKS}x16"
export monoprop_NUM_THREADS=16
export MONOPROP_SYMCHECK="$WORK/sym-${TAG}"
srun --mpi=pmix --ntasks="$NTASKS" --ntasks-per-node="$RPN" \
     --cpus-per-task=$((128 / RPN)) \
     --cpu-bind=cores --distribution=block:block \
     "$VENV/bin/python" -m pytest benches/bench_models.py -s -q \
     -k "pauli and (energy or gradient)" \
     --pauli-cutoff=12 --pauli-lower-atol=5e-05 \
     --bench-rounds=1 2>&1 | tail -20
summarize "$TAG" || rc_total=$((rc_total + 1))

echo
if [ "$rc_total" -eq 0 ]; then
    echo "########## SYMMETRY SURVIVED EVERY STAGE ##########"
else
    echo "########## $rc_total STAGE(S) REFUTED OR INCONCLUSIVE ##########"
fi
echo "probe files: $WORK"
exit "$rc_total"
