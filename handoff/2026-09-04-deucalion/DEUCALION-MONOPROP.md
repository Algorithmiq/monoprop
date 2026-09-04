# monoprop on Deucalion

Operational notes for building, running, and measuring monoprop on Deucalion x86. Numbers were
measured between 2026-08-11 and 2026-09-03. Read §6 before repeating an experiment. (Earlier editions carried
§8 "Experiments" and §10 "Settled findings"; they were lost in the 2026-09-01 cleanup and are not recoverable here —
the settled facts that survive are folded into §2–§7.)

## 1. Platform

| | x86 compute node | login node |
| --- | --- | --- |
| CPUs | 128, SMT off | 256, SMT on |
| NUMA | 8 × 16 cores | 2 domains |
| Memory | 234 GiB, ~223 GiB usable | 386 GB |
| `-march=native` | `znver2` | `znver2` |

| Partition | Max nodes | Max walltime | Internet |
| --- | ---: | ---: | --- |
| `dev-x86` | 2 | 4 h | yes |
| `normal-x86` | 64 | 2 d | no |
| `large-x86` | 128 | 3 d | untested |

- Use `-A <project>x`; the default account is ARM.
- CPU jobs are exclusive and billed for all 128 cores. Request a tight walltime; there is no
  checkpointing or overrun grace.
- Build on `dev-x86` or `ln04`; `normal-x86` has no outbound internet.
- `/projects` is the shared filesystem. Keep durable files there, not in `/tmp` or `$HOME`.
- `$HOME` is limited primarily by 20,000 inodes. Check `quota -s`, not `df`.
- `billing` reports core-hours; divide x86 totals by 128 for node-hours.
- Use one memory unit consistently. GB and GiB differ by 7%.

## 2. Job rules

- Start batch scripts with `#!/bin/bash -l`; never run `module purge` or re-source site module
  setup manually.
- Source `harness/env.sh` for both builds and runs. Compute nodes need its GCC runtime, and it
  provides CMake/CTest.
- System Python is 3.6. Use `$VENV/bin/python` or the managed Python under `caches/uv-python`.
- Slurm stages scripts at submission. Editing a pending script has no effect; resubmit it.
- Resolve paths from `cd "${SLURM_SUBMIT_DIR:-$PWD}"`; `${BASH_SOURCE[0]}` names Slurm's staged
  copy.
- Login nodes 1–3 cap the user slice at 150 tasks. Build on `ln04`. That cap is per *user*, not
  per session: `ln04` refuses forks past it too, so a `-j24` build dies with `c++: fatal error:
  cannot execute cc1plus` whenever a sibling agent is also building. Use `-j4`–`-j8`, or build only
  the targets you need. For small login-node runs use
  `monoprop_NUM_THREADS=4 OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=4`.
- `nproc` honors `OMP_NUM_THREADS`; use `os.sched_getaffinity(0)` or Slurm's `AllocCPUS` for the
  actual CPU grant.
- Treat empty `squeue` or `sacct` output as an error. Require a terminal state from `sacct`.
- Use `grep -a` on job output because files may contain NUL bytes.
- Store recovery commits under `refs/recover/*`; non-version tags break `uv sync`.
- A detached launch over `ssh ln04 'nohup … &'` dies with the ssh session; keep the connection open
  (or run the step inside `setsid` on ln04 itself) for anything longer than the session.

## 3. Build

```bash
ssh ln04
export MONOPROP_SRC=/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/<tree>
source /projects/EEHPC-DEV-2026D08-260/aaron/harness/env.sh
bash "$MONOPROP_SRC/hpc/deucalion/build.sh"
```

- Export `MONOPROP_SRC` before sourcing `env.sh`; `VENV` is derived from it.
- `build.sh` ends with `exec uv sync`, so invoke it as a script.
- MPI is opt-in. Set `monoprop_ENABLE_MPI=ON` in the environment; a CMake define alone does not
  install `mpi4py` in the isolated build environment.
- Rebuild with `--reinstall-package monoprop --no-cache`, and include `--group test` or
  `--all-groups` so pytest is not pruned.
- Toolchain: `foss/2025b` (GCC 14.3, OpenMPI 5.0.8, PMIx 5.0.8). Do not set `Boost_DIR`.
- `monoprop_MAX_NUM_MODES=1024` is a compile-time ceiling. `TermIndex` is always `uint32_t`; if one
  partition approaches 2³² terms, increase the partition or rank count.
- Use a distinct build directory per concurrent arm.
- nanobind links `_core` with LTO, and GCC's `-flto=auto` runs `make -j$(nproc)` = 256 on a
  login node, which blows the 150-task cap as `c++: fatal error: cannot execute`. `nproc` honors
  `OMP_NUM_THREADS`, so export `OMP_NUM_THREADS=4` for the install; `BUILD_JOBS` caps only Ninja.

Verify the installed build:

```bash
"$VENV/bin/python" -c "import monoprop as m; print(m.has_mpi, m.MAX_NUM_MODES)"
```

- Every `build.sh`/`uv sync` prunes `nanobind` from the venv and clobbers the CMake cache with
  scikit-build's deleted temp Python, so the fast ninja-loop fix (`uv pip install --python
  $VENV/bin/python nanobind "mpi4py>=4.1.0"`, then reconfigure) must be re-applied after *each*
  reinstall, not once per worktree.
- The `ln04` user cgroup is capped at 20 GiB across all of one user's processes. A golden run was
  OOM-killed while a sibling compile held 16 GiB; run golden dumps when the node is quiet or on
  `dev-x86`.

- The Release preset carries `-DNDEBUG`, so the ordinary gate never runs an `assert`. To test with
  asserts on, configure a separate build dir with `-DEXTRA_CXXFLAGS:STRING=-UNDEBUG` (the project flag
  lands after the module's `-DNDEBUG`, and GCC takes the last of a `-D`/`-U` pair; check
  `compile_commands.json`), build only `monoprop_unit_tests.x`, and run the serial CTest suite
  (`scratch/hashfree/one-round-logs/ir-assert.sh`).

- `EXTRA_CXXFLAGS` reaches only the `monoprop-objs` compile line, not the link line. Flags that GCC
  needs again at LTO link time (`-fprofile-generate`, sanitizers) must also go into
  `CMAKE_SHARED_LINKER_FLAGS`/`CMAKE_MODULE_LINKER_FLAGS`, or `_core` fails to link with
  `undefined symbol: __gcov_time_profiler_counter`.
- PGO (`-fprofile-generate` → S+Pauli training → `-fprofile-use`) was measured on 2026-09-03 (job
  1873677): bit-identical, ≤ 1.8 % faster on the redesign and 0.7–1.5 % *slower* on PR #296. Not worth a
  build option; recipe under `scratch/hashfree/pgo-logs/`.

Identify a benchmark arm by its commit and installed `_core.so` MD5. Editable version strings can
be stale, and a Ninja rebuild does not update the venv copies of `_core.so` and
`libmonoprop.so`; reinstall the package after rebuilding.

## 4. Layout and launch

The flat world is `R × S`: MPI ranks × in-process partitions.

| Layout | ranks/node | cores/rank | `monoprop_NUM_THREADS` |
| --- | ---: | ---: | ---: |
| whole node | 1 | 128 | 128 |
| NUMA-aligned, default | 8 | 16 | 16 |
| socket-aligned | 2 | 64 | 64 |

```bash
export monoprop_PARTITIONS=16 monoprop_NUM_THREADS=16
srun --mpi=pmix --cpu-bind=cores --distribution=block:block \
    --ntasks-per-node=8 --cpus-per-task=16 "$VENV/bin/python" script.py
```

- Launch `$VENV/bin/python` directly. `uv run` causes dependency resolution in every rank.
- Export `monoprop_PARTITIONS` for multi-rank jobs; the default is one partition per rank.
- All ranks must resolve the same `S`, and `S` must not exceed the CPUs visible to each rank.
  Exceeding the affinity mask disables placement and costs 13.8–24.6×.
- Repeat rank, core, and binding flags on `srun`; allocation directives may not propagate.
- Keep CPU binding unless measuring it explicitly. `--cpu-bind=none` cost 1.45× at 8×16.
- In MPI builds, `comm=None` means `MPI_COMM_WORLD`; use `MPI.COMM_SELF` for independent ranks.
- Results are bit-identical at fixed `(R,S)` and tolerance-equal across layouts, with identical
  term counts.

PR #296's GF(2)-linear routing is the shipping distributed path. Messages per rank are flat in
rank count; the scaling results below were measured on it.

## 5. Sizing and benchmarks

- Hubbard `propagate` at 8×16 fits
  `GiB/node = 3.39 + 0.0634 × Mterms/node` (68.1 B/term), within −2.9/+4.5 GiB over 38 runs.
  Other models have different process overheads.
- Unledgered memory scales mainly with world slots, and 78% of the gap measured was retained glibc
  heap. `MALLOC_MMAP_THRESHOLD_=131072` reduced peak RSS to 0.916× for ~5% more wall time.
- Quote kernel `VmHWM`, not sampled RSS. Peak-RSS windows nest correctly only after the 2026-09-01
  benchmark-tools fix; older `memhwm` values may omit construction.
- Size with the exact selector that will run. For grouped operations, use the maximum peak, not
  their sum. Run one model/picture per process because graph caches accumulate.
- Keep `--bench-rounds=1`; later rounds can overlap live propagators.
- The strong-scaling optimum follows terms per node, not core count. Reversal begins around
  50–100 Mterms/node. Weak efficiency at 64 nodes was 29% near 100M terms/node and 84% near
  1.5B terms/node.

Model controls:

- Hubbard and Pauli: size with `lower_atol`; default tolerances can saturate the cutoff.
- Keep Hubbard's `observable_site` at the same relative lattice position when changing sites.
- Random Heisenberg: size with `--obs-terms`.
- Random Schrödinger: size with `--num-generators`; `--obs-terms` does not size the evolved state,
  and each generator adds one gradient parameter. At 142 modes and cutoff 6, 5800 generators gave
  597M terms and 99.7 GiB for the full row.
- The former Schrödinger basis-construction blow-up is fixed. Current measurements at 142 modes
  include cutoff 7 at 467M terms/70.0 GiB and 180 modes at 45.6M terms/8.6 GiB.
- On the 8-core, 32 GiB CI runner, random Heisenberg graph operations fit at
  `--obs-terms=2500000` (168M terms, 23.71 GiB at 4×2); 3.5M does not fit.

See `benches/LADDER.md` for the portable benchmark procedure and calibrated rows.

## 6. Measuring

- Compare arms in one allocation, interleave them, and reverse order between repetitions.
- Compute paired ratios per repetition, then take the median. Report the paired median, agreement
  count, and statistical test.
- Use at least 10 repetitions for claims across multiple cells; three repetitions are for sizing.
- Use at least three sweep points before claiming a trend.
- Record the commit, installed `_core.so` MD5, exact environment, selector, shape, and term count.
- A benchmark arm must come from a tree nobody edits during the campaign: build each reference commit in
  its own detached worktree (`git worktree add --detach worktrees/ref-<hash> <hash>`) rather than pointing
  at a development worktree, whose venv is reinstalled whenever its agent rebuilds (an A/B was aborted on
  2026-09-03 because the reference venv's MD5 had changed underneath it).
- Use separate output directories to prevent stale results from entering globs.
- Hardware performance counters are unavailable (`perf_event_paranoid=3`). Use a standalone
  replay under callgrind when instruction-level evidence is needed.
- Under callgrind, `--dump-before=<fn>` splits the profile but callgrind defines its name table in
  part 1 only, so `callgrind_annotate` on part 2 attributes 100% to `(below main)`. Pass
  `--compress-strings=no --compress-pos=no`. Set `--LL` explicitly (one zen2 CCX is 16 MiB); the
  auto-detected 256 MiB is the whole node's L3 and no single-threaded run gets it.
- A profiling build needs only `libmonoprop.so` and `_core.abi3.so`; copying those two into a
  `.venv-dbg` skips the 45 unit-test TUs that dominate a full `uv sync`. RelWithDebInfo here is the
  Release flag set plus `-g3`, so codegen is unchanged.

## 7. Testing and worktrees

- Boost test discovery here registers `--run_test=<case>` with no suite prefix, so a
  `BOOST_AUTO_TEST_SUITE` wrapper makes every case in the file fail with
  `no test cases matching filter`. No test file in the repository uses one.
- CTest is rooted at the C++ test directory:
  `ctest --test-dir build/<tree>/cpp/tests -L serial`. Read the final case tally; a wrong directory
  reports zero tests and exits successfully.
- Run MPI CTests from a batch context with
  `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`. The OpenMPI 4 environment name does not
  work with OpenMPI 5.
- Cap login-node Python tests with `monoprop_PARTITIONS=4`. That cap is for pytest only: exporting
  `monoprop_PARTITIONS` into a CTest run fails 15 unrelated serial cases (pare-graph, dense-matrix and
  layer-range tests); unset it before `ctest`.
- The lint gate is `prek`, not Ruff alone:
  `uvx prek run --from-ref <base> --to-ref HEAD`. Put `XDG_CACHE_HOME` under `/projects`.
- New worktrees need the untracked harness link:
  `mkdir -p "$T/hpc" && ln -sfn "$PROJ/aaron/harness" "$T/hpc/deucalion"`.
- Pass an absolute path to `git worktree add`, and submit jobs from inside the worktree:
  `cd "$TREE" && sbatch ...`.
- Keep one agent per worktree and durable output under `$PROJ/aaron`.

