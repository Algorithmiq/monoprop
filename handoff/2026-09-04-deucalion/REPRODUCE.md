# Reproducing the monoprop memory/time campaign on another machine

Companion to `HANDOFF-2026-09-04.md`. That file says what was found; this one says how to rebuild the
arms, rerun the gates and the benchmark ladder, and start round 3, on a machine that is not Deucalion.
Absolute times will differ from the Deucalion numbers; the ratios between arms are what to compare.

## 0. What each step needs

| step | cores | memory | time (Deucalion, for scale) |
| --- | ---: | ---: | ---: |
| build one arm | 4–32 | 20 GiB | 15–40 min (LTO link of `_core` dominates) |
| serial ctest + pytest | 4 | 8 GiB | 5 min |
| 35-cell golden dump + diff | 4 | 8 GiB | 5–10 min, 1.5 GB JSON per arm |
| 9.26 M-term reproducer (VmHWM + ledger) | 1–128 | 1 GiB | 1–2 min per shape |
| ladder L1 (10 M terms, P=1) | 1 | 1 GiB | 26 s base / 43 s stack per cell |
| ladder L2a/L2b (1.0 B terms) | 128 | 60 GiB | 2.5–3 min per cell |

Only the L2 rows need a large node. Everything else, including round 3's codec and store tracks, runs
on a workstation.

## 1. Prerequisites

- GCC ≥ 14 (C++23; the build hard-fails below 14). Clang is untested on this campaign.
- CMake 3.28–4.0, Ninja, pkg-config.
- Boost ≥ 1.85 (Unordered, Test), msgpack-cxx (or let CPM fetch it), hwloc. Debian names are in
  `tools/packages/apt.txt`; add `apt-mpi.txt` for MPI (OpenMPI; Deucalion used 5.0.8 with PMIx 5).
- `uv` (manages Python 3.11–3.14 and the venv). `just` is optional but the `justfile` documents the
  supported recipes (`just build`, `just test`, `just test-mpi`).
- For MPI builds `mpi4py` must be compiled against the local MPI: `export UV_NO_BINARY_PACKAGE=mpi4py
  MPICC=$(command -v mpicc)`.
- Pin BLAS threads for every run: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
  (the harness raises `OMP_NUM_THREADS` per cell, see §6).

## 2. Get the code and lay out the arms

```bash
git clone https://github.com/Algorithmiq/monoprop.git && cd monoprop
git fetch origin perf/storage-levers perf/wire-zero-copy perf/one-round-exchange \
                 perf/linear-routing-on-wire-deucalion-base main
mkdir -p ../arms
git worktree add --detach ../arms/base-296 5ada3da3   # reference arm: local PR #296 base
git worktree add --detach ../arms/stack    a1c122a9   # the stack (one-round + storage), all gates green
git worktree add --detach ../arms/wire     91806a63   # zero-copy wire, off ce16f0f6, login gates only
git worktree add --detach ../arms/bench    origin/main # benches/ + bench tools (see note)
```

Arms are detached checkouts so nothing can edit them under a measurement. Identify an arm by its
commit and the md5 of the installed `_core.abi3.so`, never by `monoprop.__version__`.

Note on the bench tree: Deucalion used `worktrees/bench-317` at e3588ecb, a pre-merge PR #317 tip
that is not on GitHub. `origin/main` carries the merged ladder (`benches/LADDER.md`,
`benches/bench_models.py`, `benches/bench_random.py`, `packages/monoprop-bench-tools`) and differs from
e3588ecb only in `benches/conftest.py` (9 lines) and the bench tools' memory helper. Use `origin/main`
as the bench tree for every arm; it is one revision whatever the arms carry.

## 3. Build an arm

Run once per arm directory (`ARM=../arms/stack` etc.). This is `harness/build.sh` without the Deucalion
module loads:

```bash
cd "$ARM"
export monoprop_ENABLE_MPI=ON            # the env var is the switch; a CMake define alone skips mpi4py
export UV_NO_BINARY_PACKAGE=mpi4py MPICC=$(command -v mpicc)
uv sync --all-extras --group test --group bench -v \
    --reinstall-package monoprop --no-cache \
    --config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=ON" \
    --config-settings-package="monoprop:cmake.define.monoprop_MAX_NUM_MODES=1024" \
    --config-settings-package="monoprop:build-dir=build/{state}/{build_type}-$(basename "$ARM")" \
    --config-settings-package="monoprop:build.tool-args=-j8"
.venv/bin/python -c "import monoprop as m; print(m.has_mpi, m.MAX_NUM_MODES)"
md5sum .venv/lib/python3*/site-packages/monoprop/_core.abi3.so
```

- `--reinstall-package monoprop --no-cache` is mandatory: uv's cache key ignores the CMake defines and
  will serve a stale MPI-off wheel otherwise. `--group test` is mandatory or pytest is pruned.
- `monoprop_MAX_NUM_MODES=1024` matches Deucalion; it only widens the compiled storage tiers and does
  not change results. The default (250) also covers every ladder row.
- The `_core` link runs under LTO with `make -j$(nproc)`; on a shared login node cap it with
  `OMP_NUM_THREADS=4` (nproc honours it).
- Without MPI drop the three MPI lines; `has_mpi` will print `False` and the L2b row is unavailable.
- Deucalion md5s for cross-checking that you built the same source: base-296 `f991ce3a` (`_core`
  alone), stack a1c122a9 `8b3d724a` (dev venv) / `eccf7a15` (frozen venv, includes `libmonoprop.so` in
  the hash used by the A/B scripts). Different compilers give different md5s; the goldens are the
  cross-machine check.

## 4. Correctness gates (per arm, workstation)

```bash
cd "$ARM"
ctest --test-dir build/editable/Release-$(basename "$ARM")/cpp/tests -L serial   # read the final tally
monoprop_PARTITIONS=4 monoprop_NUM_THREADS=4 .venv/bin/python -m pytest tests -q  # expect 593 passed, 8 skipped
just test-mpi 4                                                                   # MPI suites, if built
```

`monoprop_PARTITIONS` is a pytest-only cap: exporting it into a ctest run fails 15 unrelated serial cases.
CTest's `--test-dir` must be the `cpp/tests` directory of the build; a wrong directory reports zero tests
and exits 0.

**Bit-identity (the gate that matters).** `golden.py` dumps 35 cells (7 msgpack fixtures in
`tests/data` × 5 cutoff/tolerance configs, propagate and graph modes, all terms with `repr` coefficients);
`golden_diff.py` compares two dumps to the ULP. Both scripts live on Deucalion under
`/projects/EEHPC-DEV-2026D08-260/aaron/scratch/hashfree/` (see §8 to fetch them; they are ~120 lines
each and depend only on the arm's venv and `tests/cases.py`).

```bash
export monoprop_PARTITIONS=1 monoprop_NUM_THREADS=4
../arms/base-296/.venv/bin/python golden.py ../arms/base-296 golden/base.json
../arms/stack/.venv/bin/python    golden.py ../arms/stack    golden/stack.json
python golden_diff.py golden/base.json golden/stack.json      # every maxULP column must be 0
```

Repeat with `monoprop_PARTITIONS=4` for a multi-partition arm-to-arm check (base at P=4 vs stack at
P=4). On Deucalion every landed commit was 0 ULP against base at P=1, 4 and 128 and across MPI layouts.

## 5. Memory on the 9.26 M-term reproducer

`w_repro.py 12 7 1e-6 <label> [--mpi]` (Hubbard, 12 Trotter steps, cutoff 7, `lower_atol` 1e-6;
also on Deucalion under `scratch/hashfree/wire-logs/`) builds the problem with
`monoprop_bench_tools.models.build_hubbard_problem`, propagates, and prints one `RESULT` line per rank
with `size`, an md5 of the evolved operator, the ledger (`total_bytes`, `gate_scratch_bytes`,
`d_gate_buffers_hwm_bytes`) and `VmHWM`. Run it with `PYTHONPATH=../arms/bench/packages/
monoprop-bench-tools/src` and the arm's venv python, under `/usr/bin/time -v`, at:

| shape | env | launch |
| --- | --- | --- |
| P=1 | `monoprop_PARTITIONS=1 monoprop_NUM_THREADS=1` | plain python |
| P=4 | `=4 / =4` | plain python |
| P=128 | `=128 / =128` (needs 128 cores; S must not exceed the CPUs visible) | plain python |
| 2×16 | `=16 / =16` | `mpiexec -n 2 --bind-to core --map-by ppr:2:node:pe=16 python w_repro.py 12 7 1e-6 r2x16 --mpi` |

Quote kernel `VmHWM`, not sampled RSS. Deucalion values for the stack at P=1: VmHWM ≈ 0.79× base;
`d_gate_buffers_hwm_bytes` is the transient the wire branch attacks (9.6 B/term at S=1, 16.7 at 2×16 on
the storage tip; W's per-buffer breakdown is `scratch/hashfree/wire-logs/breakdown-2.md`).

## 6. The ladder A/B (PR #317 rows)

Every cell is one pytest invocation of the bench tree's `benches/` with the arm's venv. Exact rows as run
on Deucalion (`harness/sbatch/ab-hashfree-n1.sh` and `ab-hashfree-nodes.sh`):

| row | layout | selector `-k` | flags | terms |
| --- | --- | --- | --- | ---: |
| L1 Hubbard | P=1 | `test_model_propagate and hubbard` | `--hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05` | ~10 M |
| L1 Pauli | P=1 | `test_model_propagate and pauli` | `--pauli-cutoff=12 --pauli-lower-atol=1.22e-04` | ~10 M |
| L1 graph | P=1 | `test_random_gradient and heisenberg` | `--num-generators=1000 --num-modes=142 --cutoff=6 --obs-terms=295000` | |
| L2a Hubbard | 1×128 | `test_model_propagate and hubbard` | `--hubbard-cutoff=10 --hubbard-lower-atol=3.38e-06` | 1.0016 B |
| L2b Hubbard | 8×16 | same as L2a | same as L2a | 1.0016 B |
| L2b Pauli | 8×16 | `test_model_propagate and pauli` | `--pauli-cutoff=14 --pauli-lower-atol=8.9e-06` | 0.99 B |

One in-process cell (L1, L2a), with `P` the partition count:

```bash
export PYTHONPATH=../arms/bench/packages/monoprop-bench-tools/src
export monoprop_PARTITIONS=$P monoprop_NUM_THREADS=$P OMP_NUM_THREADS=$P MALLOC_ARENA_MAX=$P
export monoprop_BENCH_LABEL="N1_A_1x1_hubbard_fresh_stack_r1" monoprop_BENCH_RESULTS="$RESULTS"
/usr/bin/time -v -o "$RESULTS/$monoprop_BENCH_LABEL.time" \
  ../arms/stack/.venv/bin/python -m pytest ../arms/bench/benches -o filterwarnings=default \
    -k "test_model_propagate and hubbard" --hubbard-cutoff=10 --hubbard-lower-atol=4.2e-05 \
    --bench-rounds=1 --benchmark-json="$RESULTS/time-$monoprop_BENCH_LABEL.json" -q -s -p no:cacheprovider
```

The label format `N<nodes>_<layout>_<model>_<group>_<arm>_r<rep>` is what `ab_pairs.py` parses; `group`
is `fresh` for propagate and `graph` for gradient rows. Time comes from the benchmark JSON, peak memory
from `Maximum resident set size` in the `.time` file (in-process) or from the bench JSON's `memhwm`
summed over ranks (MPI), and the ledger from the `opmembreak` block the bench writes.

One MPI cell (L2b, 8 ranks × 16 partitions on a 128-core node) replaces the python line with

```bash
export monoprop_PARTITIONS=16 monoprop_NUM_THREADS=16 OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=16
mpiexec -n 8 --map-by ppr:8:node:pe=16 --bind-to core \
  ../arms/stack/.venv/bin/python -m pytest ../arms/bench/benches ... (same arguments)
# Slurm equivalent used on Deucalion:
# srun --mpi=pmix --export=ALL --ntasks=8 --ntasks-per-node=8 --cpus-per-task=16 --cpu-bind=cores --distribution=block:block ...
```

Keep CPU binding (unbound cost 1.45× at 8×16) and make sure `S=16` does not exceed the CPUs each rank
can see; exceeding the affinity mask disables placement and costs 14–25×.

Measurement discipline (from `DEUCALION-MONOPROP.md` §6): run all arms in one session on an otherwise
idle node; per repetition run every arm back to back and rotate the order by one arm each repetition;
compute the ratio per repetition and report the paired median with the agreement count; ≥ 10 repetitions
for a claim (3 were used for the 1 B-term rows because of cost); separate results directory per job so
stale files never enter a glob; record commit, md5, environment and term count for every cell and check
the term counts are identical across arms. `harness/tools/ab_pairs.py <results-dir>` produces the
`AB-SUMMARY.md` tables from a directory laid out as above.

## 7. What to expect

Deucalion, x86 node (2 × 64-core zen2, 234 GiB), GCC 14.3, `-march=native`; stack a1c122a9 vs base
5ada3da3, ratios stack/base, so time > 1 is slower:

| row | base time | stack time | time | peak RSS | ledger | ledger B/term |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 Hubbard P=1 | 25.9 s | 42.7 s | 1.649 | 0.793 | 0.668 | |
| L1 Pauli P=1 | 20.6 s | 30.0 s | 1.458 | 0.821 | 0.677 | |
| L1 graph | 11.2 s | 10.7 s | 0.954 | 0.820 | 0.676 | |
| L2a 1×128 | 146 s | 192 s | 1.315 | 0.846 | 0.668 | |
| L2b 8×16 | 130 s | 176 s | 1.354 | 0.823 (58.5 → 48.2 GiB) | 0.668 | 52.3 → 34.9 |

The stack is compute-bound: the same slowdown at P=1, 1×128, 8×16 and 2 nodes, and callgrind shows
195 vs 160 instructions per visited row. A machine with a different cache hierarchy will move the time
ratio; the memory ratios follow from the data layout and should reproduce closely.

## 8. Files that are not on GitHub

Everything below is on Deucalion under `/projects/EEHPC-DEV-2026D08-260/aaron/`; a login node can serve
them without node hours. A bundle with all of it except the golden dumps is at
`scratch/hashfree/handoff-bundle.tar.gz` (see `HANDOFF-2026-09-04.md`):

```bash
scp deucalion:/projects/EEHPC-DEV-2026D08-260/aaron/scratch/hashfree/handoff-bundle.tar.gz .
```

Contents: `HANDOFF-2026-09-04.md`, `REPRODUCE.md`, `DEUCALION-MONOPROP.md`,
`scratch/hashfree/PLAN-rounds-2-3.md` (rounds 2 and 3 with every measurement), `harness/` (`env.sh`,
`build.sh`, `sbatch/ab-hashfree-*.sh`, `tools/ab_pairs.py`, the MPI test job scripts),
`scratch/hashfree/golden.py`, `golden_diff.py`, `wire-logs/w_repro.py`, every `*-logs/` directory
(PROGRESS, GATES, breakdown tables, callgrind notes) and the `runs/ab-hashfree-*/AB-SUMMARY.md` files.
The two golden references (`golden/one-round.json` 1.5 GB, `one-round-1p5.json` 752 MB) are not in the
bundle; regenerate them from the base arm with `golden.py` — base and stack are bit-identical, so a
base dump is the reference.

## 9. Starting round 3

```bash
git worktree add -b perf/syndrome-store ../syndrome a1c122a9
mkdir -p ../syndrome/cpp/monoprop/detail/syndrome
```

Tracks A (`SyndromeCode<NumModes, t>`: encode/decode/add, property tests, ≤ 20 ns encode) and B
(`SlicedBlockStore`: blocks, 8-mode bundles, residuals, checkpoint slices, `fold(G) -> (anti, sign)`,
sign mask checked against `Monomial` phases on 10⁶ random pairs) are pure C++ with Boost.Test cases
under `cpp/tests/`, and need only §1 and §3. Track C (the kernel behind `monoprop_ENGINE=syndrome`) is
gated on golden 0 ULP at P=1 (§4) and on the 9.26 M reproducer (§5): go if ≥ 1.5× faster than base
with VmHWM ≤ 0.85× the stack. Full design and rationale: `PLAN-rounds-2-3.md`, "Round 3".
