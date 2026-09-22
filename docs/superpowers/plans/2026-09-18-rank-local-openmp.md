# Rank-local OpenMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Read both documents completely; full-plan execution
> requires explicit approval. A separately authorized baseline-only calibration may precede campaign approval, as Task 1
> describes. Do not infer missing contracts. Checkboxes track future work.

**Goal:** Replace thread-owned partitions with one compact operator store per MPI rank and OpenMP worksharing, accepting
the replacement only after correctness, runtime, and peak-memory parity have been demonstrated.

**Architecture:** Keep the packed position-list rows, keyless hash index, inverted index, graph representation, and MPI
routing algorithms/settings. Setting the router's partition dimension to one can change historical physical-rank
ownership under splitmix routing; Task 10 makes that compatibility consequence explicit. Workers traverse frozen
structures or update preallocated disjoint ranges; the controlling thread owns structural publication and MPI. Prototype
fixed-graph replay first, then deterministic graph construction, and delete the old runtime last.

**Tech Stack:** C++23, OpenMP CXX, existing Boost and optional MPI, nanobind, Python 3.11+, uv/scikit-build-core,
Boost.Test, pytest. Monoprop's direct hwloc dependency is removed with the legacy partition runtime.

**Spec:** [Rank-local OpenMP design](../specs/2026-09-18-rank-local-openmp-design.md). Read both documents before
execution.

**Status:** Planning only. Source baseline: `perf/linear-routing-on-wire` at
`290112c8289ab8015eb9a2c7651ac409839c5a88`; published planning commit: `refactor-parallelization` at
`38eb13d80a95dd1ac60e762f14f68c9a5925f755`. This documentation-only interview revision records approved runtime,
compatibility, numerical and failure policies. OpenMP is mandatory; partition configuration is removed immediately;
thread budgets are launch-environment-only and captured at construction. The fixed32 capacity reduction is accepted.
The owner selected one AWS EC2 `c8a.metal-24xl` for implementation/testing and initial acceptance. Production targets
are dual-socket Sapphire Rapids and AMD Zen3/Zen4/Zen5 HPC nodes, commonly four NUMA domains/socket. Hardware SMT may
be on or off, but SMT workers are not recommended: always configure one worker per physical core. Passing the frozen
single-instance EC2 campaign is sufficient for initial acceptance; HPC performance remains separately unqualified.
Multi-node qualification is pending a proper HPC cluster/interconnect, not a cloud-network substitute. Record actual EC2
topology/software and agree representative workload sizes before freezing the campaign or collecting acceptance data.
The owner approved baseline-only, unscored sizing trials before the freeze, once separately authorized. Small/medium
profiles cover the full geometry sweep; large profiles cover only the full-node one- and two-ranks-per-domain cases.

No implementation, engine builds/tests or benchmarks are authorized by this revision. All future execution commands
below are planned, not evidence of passing gates. EXISTING labels describe current source; PROPOSED labels describe
future work. Full implementation and any publication of this revision require separate authorization.

## Global Constraints

- C++23; retain the GCC 14 / Clang 18 minimum compiler versions and Python 3.11 floor.
- MPI remains optional and OFF by default; retain working non-MPI wheels.
- OpenMP becomes a required implementation dependency for the replacement runtime; support Linux x86_64, Linux aarch64,
  and macOS builds covered by the current project.
- Each independent propagator instance owns one packed operator store per MPI rank; threads are never communicator
  ranks. Preserve independent copies, immutable graph-core sharing, retained coefficient/state snapshots and pared
  graphs.
- Remove monoprop's direct hwloc dependency and topology/affinity machinery; do not replace it with another
  library-owned hardware-discovery or pinning mechanism.
- Users choose P processes and T threads per process and configure resource allocation, binding and pinning; monoprop
  does not clamp T to physical cores or validate NUMA placement.
- SMT use is not recommended: allocate at most one monoprop worker per physical core, including on SMT-enabled hosts.
  Enforce this through user launch/binding settings and acceptance evidence, not a library hardware policy.
- Preserve packed operator rows, the keyless hash index, fixed uint32_t TermIndex, sparse state storage, and
  inverted-index cosine recomputation.
- Do not introduce boost::concurrent_flat_map, hidden shards, a dense duplicate store, a second persistent key store, a
  full per-thread operator, or full-vector coefficient double buffering.
- Preserve Majorana and Pauli bases, Heisenberg and Schrödinger pictures, cutoffs, graph build/replay/paring, gradients,
  partial contraction, copying, and initial-operator updates.
- MPI calls execute on the controlling thread outside library-created OpenMP worksharing regions; retain
  MPI_THREAD_SERIALIZED rather than changing the initialization contract.
- No exception escapes an OpenMP structured region; distributed operations fail coherently rather than leaving peers in
  collectives. Enforce invalidation after failed mutation; dependent functionals cannot reuse an invalid owner.
- No global omp_set_num_threads/omp_set_dynamic/omp_set_nested mutation; each worksharing region receives an explicit
  per-object thread budget.
- Use bitmap traversal or anticommutation pass for the existing operation; reserve prefix offsets for cumulative counts
  used to allocate output ranges.
- Do not change numerical tolerances, truncation rules, generator order, source ordering, or MPI routing to make a
  regression disappear.
- Do not declare success without target-hardware runtime and peak-memory measurements at the same rank count, CPU
  allocation, workload, and fixed 32-bit TermIndex as the baseline.
- No automatic regression allowance: the acceptance target is candidate/baseline <= 1.00 for the median runtime and
  median peak memory of every required workload cell. Noise triggers more measurement, not a silent waiver.
- Update AGENTS.md, README.md, and affected user/contributor docs in the same implementation changes that alter their
  documented behavior.
- No permanent legacy/new-backend selector. Temporary coexistence is confined to the development branch; remove it
  before accepting the replacement.

Additional execution rules:

- **No runtime work or implementation is authorized by the design interview.** Obtain separate authorization for a
  bounded baseline-only calibration/build phase; it may precede campaign approval. Candidate engine changes require
  implementation authorization and an owner-approved frozen campaign. Formal baseline collection also waits for that
  freeze. Simplification without regressions is the goal; reject the architecture if it cannot earn acceptance. The
  campaign bounds performance claims, not mathematical correctness across the supported API.
- P (called R in existing code/benchmark geometry) is the user-selected process count; T is the requested threads per
  process, normally set with `monoprop_NUM_THREADS`. Only an unset override selects the OpenMP runtime default. Users
  must allocate enough resources and configure rank/thread binding
  with OpenMP, `srun --cpu-bind`, and/or `mpiexec` options. Each rank's workers are assumed to fit its allocation within
  one NUMA domain. Documentation recommends against SMT workers: allocate one worker per physical core, including when
  hardware SMT is enabled. Do not create a topology service/NUMA scheduler, silently change P/T, or enforce that
  recommendation through library clamping or affinity changes.
- Do not touch pre-existing `.codegraph/` or `.direnv/`. Use an isolated implementation checkout and a separately built
  baseline. Follow the worktree/GitButler skills for version-control writes; no raw git write commands.
- New behavior follows RED → minimal implementation → GREEN → diff review. Run tests before production edits and record
  actual results. Preservation fixtures can already pass after an earlier task; do not manufacture a failure or remove
  completed plumbing. Each threading task also needs a kernel-specific worker-participation check, not only equal
  serial/threaded numerical output. For configuration/documentation use a failing integration check.
- Commit only when authorized, after the task gate. Use `<type>(<scope>): <gitmoji> <description>` and the required
  `Assisted-by: Pi:<actual-model>` trailer. Suggested boundaries are the numbered tasks, not every checkbox.
- Do not use a broad formatter over unrelated files. Public declarations receive a one-line summary; header
  declarations/members follow the repository's documentation convention.

## Execution map and gates

| Stage | Tasks | Gate before proceeding |
| --- | --- | --- |
| Baseline | 1 | Authorized baseline-only calibration, approved frozen inventory, fresh formal baseline evidence |
| Small prototype | 2–4 | OpenMP packaging works; one-store serial and threaded replay agree |
| Shared-store construction | 5–7 | Stable queries/IDs, safety, construction scaling and scratch measured |
| Remaining evaluation kernels | 8–9 | Replay, derivative, retained closures and reductions validated |
| Integration cutover | 10–11 | Both APIs and MPI×threads pass; benchmark parity at final geometry |
| Removal/release | 12–13 | Old runtime removed, final full matrix and parity rerun |

One writer per checkout. Kernel and storage tasks are ordered: do not have agents concurrently edit `Engine.h`,
`Evolution.cpp`, or `MonomialPropagator.inl`. Independent read-only reviews are safe. Do not batch past a failed gate.

Expected RED/GREEN checkpoints (record real results; these are planned conditions, not claims of execution):

| Task | RED before the task's change | GREEN required |
| --- | --- | --- |
| 1 | exactness/shape schema and new driver tests fail | complete baseline/validator evidence; no parity claim |
| 2 | missing Workshare/OpenMP exported dependency | helper and installed consumer pass |
| 3 | options overload/config/failure tests fail | serial behavior preserved; failures join/abort promptly |
| 4 | cosine/fused work-range participation remains serial; preservation tests may pass | exact coefficients and measured replay scaling |
| 5 | ordered range/window/self-stage tests fail | byte/order/ID equivalence and bounded scratch |
| 6 | new position helper/checked-boundary tests fail; Task 3 options already exist | position/hash probe and ordered serial publication |
| 7 | arithmetic-only capacity/new guard tests fail | uint32 count/ID guards; optional fill passes only if measured |
| 8 | endpoint work-range participation remains serial; preservation tests may pass | current stable derivatives and sparse transport preserved |
| 9 | named threaded reductions absent | fixed-block association approved and deterministic |
| 10 | one-store/default/compatibility tests fail | approved S=1 API/resource semantics |
| 11 | absent/incomplete evidence is not a pass | every runtime/peak cell <=1.00 after prescribed repetitions |
| 12 | static legacy/direct-dependency checks fail | runtime/topology removed LAST; parity rerun |
| 13 | clean wheel/Nix/consumer/docs checks expose missing integration | complete evidence and independent reviews |

## Baseline-drift checkpoint and handoff

Before any authorized implementation, record HEAD, status and these EXISTING anchors. Line numbers identify this
snapshot; symbol names are the durable anchors. If source differs semantically, stop, report the changed contract and
refresh/approve the plan before coding. Missing PROPOSED symbols are expected RED, not drift.

| EXISTING anchor | Must match before proceeding |
| --- | --- |
| `cpp/monoprop/detail/mpi/Routing.h:177–260,315–326` | boolean linear/splitmix, default linear, unsupported geometry throws |
| `cpp/monoprop/detail/evolution/layer_build/Scan.h:208–253` | `FusedScanResult<N>`, window, precomputed `gen_shift`, two self stages |
| `cpp/monoprop/detail/evolution/layer_build/Resolve.h:90–183` | `probe_incoming_queries(incoming,op,QueryForm)`, positions and hashes |
| `cpp/monoprop/detail/evolution/layer_build/Engine.h:486–565` | position/hash deferred publication and direct bounded self probe |
| `cpp/monoprop/detail/operator/OperatorIndex.h:282–288` | five-span `find_batch_positions`, optional hash output |
| `cpp/monoprop/TypeAliases.h:39`; `cpp/include/monoprop/Evolution.h:54–63` | uint32_t; derivative has `LayerAngle` and optional `CosRecordView` |
| `cpp/include/monoprop/MonomialPropagator.h:286–305` | virtual update/clone, nonvirtual apply helper |
| `justfile:75–122,169–171,371–397`; `pyproject.toml:162–168` | labels, centralized recipes, MPI env build override |
| `benches/bench_random.py:22–139`; `benches/conftest.py:248–274,411–493` | four operation families, shape guard, existing op/outer windows |
| `packages/monoprop-bench-tools/src/monoprop_bench_tools/memory/cpu.py:75–107,213–233` | nested peak preservation already implemented |

Handoff checklist: exact revision/binary and overlay hashes; owner approvals; last passed task; actual RED and GREEN
commands/results; remaining gate/capacity/placement blockers; fixed routing and workload manifest; raw artifact paths.
Never hand off only “tests pass”. No builds/tests/imports/installs/VCS writes are authorized by this document refresh.

### Remote Task 1 entry point

The baseline-only pilot belongs to Task 1. It is not a prerequisite that must run on the planning machine before handing
this plan to an implementer. Run it on the authorized execution host, currently c8a.metal-24xl. If the host differs,
clarify the hardware decision before treating its sizing results as applicable to the target campaign.
No remote access, provisioning or runtime execution is authorized merely by reading these documents.

Transfer the latest spec, plan and CONTEXT.md with the source checkout. The historical planning commit alone does not
include subsequent interview decisions. Verify document contents/digests on the execution host, including the 2-hour,
15-minute and 75% calibration limits. Documentation-only commits above the source anchor are not engine drift; inspect
semantic source/build/benchmark changes against the baseline-drift checkpoint rather than require HEAD to equal the old
planning commit. Keep the measured baseline source pinned to 290112c8289ab8015eb9a2c7651ac409839c5a88.

A bounded starter authorization may cover only host/source preflight, documented project-local setup, an isolated
baseline build, existing baseline API/benchmark sizing trials and their report. It does not authorize candidate engine
changes, formal acceptance samples, commits, pushes or a PR. Read both documents completely before starting; distinguish
EXISTING baseline behavior from PROPOSED replacement behavior rather than making the baseline conform to the new design.
Do not infer subagent/delegation permission from the plan's choice of execution skills.

Before expensive commands, state the observed host/allocation, source/doc identities, permitted scope, resource limits
and a durable artifact directory outside the source checkout. Preserve the source checkout and record any approved
measurement overlay separately. On interruption, retain commands/results, trial outcomes and remaining budget in a
checkpoint; resuming does not reset the pilot limits.

The pilot handoff contains absolute artifact/report paths and digests; host/toolchain/binary provenance; verified
placement; all tried configurations and outcomes; time/memory observations including export/validation; proposed exact
small/medium/large workload and geometry/routing cells; and the estimated time/cost of the formal campaign, including
prescribed extra repetitions. State blockers and unmeasured items explicitly. Stop for owner approval of that campaign
and budget, and authorization for the remaining work. Task 1 is not complete until its tools/tests and fresh post-freeze
formal baseline evidence also pass. No pilot result is a parity claim or an acceptance sample.

## File responsibilities

Paths in task lists are relative to the repository root.

| Area | Existing anchors | New files |
| --- | --- | --- |
| Runtime configuration | `detail/EnvConfig.h`, constructor/copy in `MonomialPropagator.inl`; delete `detail/partition/CpuTopology.*`, do not move it | `cpp/monoprop/detail/parallel/{Options.h,Workshare.h,ThreadBudget.h,ThreadBudget.cpp,CMakeLists.txt}` (budget validation only; no hardware discovery) |
| Replay | `Evolution.cpp`, `detail/evolution/CosineRecompute.h`, `MPFunctions.cpp` | no executor or new graph layer |
| Construction | `detail/evolution/layer_build/{Scan,Resolve,Engine,FusedApply}.h` | range helpers stay beside current kernels |
| Storage | `detail/operator/{OperatorIndex,MPOperator,InvertedIndex}.h` | no replacement map |
| MPI failure | `detail/mpi/{Comm,MPICompat,MPIUtils,Exchange}.h`, `MPICompat.cpp` | `cpp/monoprop/detail/mpi/OperationFailure.h` |
| Tests | existing suites listed per task | `cpp/tests/{openmp_workshare_tests,openmp_runtime_tests,openmp_kernel_tests,openmp_equivalence_tests}.cpp`, `tests/test_openmp_config.py`, `cpp/tests/mpi_failure_driver.cpp` |
| Benchmark evidence | `benches/conftest.py`, both benchmark-tool renderers | `tools/benchmark-rank-local-openmp.py`, `tools/rank-local-openmp-workloads.json`, `tests/test_rank_local_openmp_benchmark.py` |
| Documentation | `AGENTS.md`, `README.md`, docs listed in Task 13 | `docs/content/docs/openmp-migration.mdx` |

`detail/...` in the table means `cpp/monoprop/detail/...`. Task file lists below use full paths. New installed headers
must enter CMake FILE_SETs, not merely compiler include paths.

## Shared invariants: reviewer checklist

1. `OperatorIndex::find_batch_positions` (and dense `find_batch`) is read-only **only while rows, overflow storage, and
   hash slots are frozen**. `set()` is not a disjoint-row parallel setter: it touches a shared overflow map, even
   through `erase` for inline rows.
2. `MPOperator::inverted_index() const` and `InvertedIndex::row_parity_words() const` can mutate caches. Prepare them on
   the controlling thread before a team. Never retain a parity pointer across store growth.
3. Distinct inverted-index rows or graph records can share a packed word. Keep bitmap append and graph phase packing
   serial initially.
4. Leaders finish local resolution, remote resolution/insertion, responses, and matched marks **before** follower work.
   Deferred self misses are published after both passes. Physical insertion order is remote leader misses → remote
   follower misses → self leader misses → self follower misses.
5. XOR by one fixed generator is injective, and source terms are unique across owners. This is the reason query
   keys/targets are unique; the hash table does not deduplicate bulk input. Debug tests must verify this invariant.
   Generic duplicate input to a helper requiring distinct keys must not be silently accepted.
6. Query words, source IDs and captured values form parallel streams. Merge each owner/leader-or-follower stream in
   source-range order, not worker completion order.
7. Fused cosine scaling captures old source values before multiplying. Each rotation record owns its entire coefficient
   pair. No coefficient atomics.
8. A graph layer's `scaled_count` currently follows post-insertion finalization. Preserve it; the pre-gate source count
   is a different boundary.
9. Self derivative entries pair `k` with `k + count/2`, not `2*k` with `2*k+1`. Snapshot both required pre-update values
   before writing either endpoint.
10. Gate loops, reverse-gradient loops, repeated-parameter accumulation, graph append, store publication, and MPI remain
    ordered on the caller.
11. Empty local stores still follow the same distributed collective sequence. Launch-time thread budgets may differ by
    rank without changing ownership; each object's captured budget is immutable.
12. Internal numerical kernels fall back to serial inside an existing OpenMP region. The supported host enters through
    one controlling thread per rank, outside active host OpenMP teams, with no overlapping dependent-state or host MPI
    calls. The fallback does not extend that calling contract.
13. A post-mutation failure invalidates the affected owner and its dependent functionals. Check validity before later
    state access or copying; independent earlier copies remain valid. Pre-mutation validation errors do not poison.

## Build/test command library

These commands are to run **during implementation**, not evidence of passing tests now. Run from the relevant checkout.
Unset stale sanitizer/build-type and MPI override settings before Release commands; R explicitly uses
`monoprop_ENABLE_MPI=OFF` as well as its CMake OFF setting. A configuration change requires forced reinstall and
regenerated tests; `uv sync` alone does not reliably relink the C++ executable. For ordinary suite runs explicitly set
`monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1`; this is test-runner policy, not a library multi-rank default. Dedicated
kernel tests pass internal Options; propagator tests launch fresh processes with a selected environment budget on an
adequate allocation. Unset removed partition settings in candidate launches. OpenMP-fallback tests specifically unset
monoprop_NUM_THREADS.

### R: non-MPI Release

```bash
export monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1
monoprop_ENABLE_MPI=OFF uv sync --all-extras --group workspace-test --group bench \
  --reinstall-package monoprop --no-cache \
  --config-settings-package='monoprop:cmake.define.monoprop_ENABLE_MPI=OFF' -v
cmake --build --preset skbuild-release
OMP_NUM_THREADS=1 just test-cpp
OMP_NUM_THREADS=1 just test-py
```

For a targeted C++ gate after rebuilding:

```bash
ctest --test-dir build/editable/Release --label-exclude mpi --no-tests=error \
  --output-on-failure -R 'openmp_|operator_index|fused_|combined_recompute'
```

Use `ctest --test-dir build/editable/Release -N -V` to verify discovery. A regex matching zero tests is not a pass. New
Boost cases must be flat `BOOST_AUTO_TEST_CASE` names; CMake discovers `cpp/tests/*.cpp` automatically after
regeneration.

### M: MPI Release

```bash
export monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1
# Environment switch also adds mpi4py>=4.1.0 to the isolated build requirements.
monoprop_ENABLE_MPI=ON uv sync --all-extras --group workspace-test --group bench \
  --reinstall-package monoprop --no-cache \
  --config-settings-package='monoprop:cmake.define.monoprop_MPI_TEST_PROCS=1;2;3;4' -v
uv run --no-sync python -c 'import monoprop; assert monoprop.has_mpi, monoprop.__file__'
cmake --build --preset skbuild-release
OMP_NUM_THREADS=1 just test-cpp
# Every success matrix including three ranks uses explicit splitmix.
monoprop_ROUTING=splitmix OMP_NUM_THREADS=1 just test-cpp-mpi
monoprop_ROUTING=splitmix OMP_NUM_THREADS=1 just test-py-mpi '1;2;3;4'
# Rebuild/register a separate default-linear success matrix, excluding unsupported R=3.
monoprop_ENABLE_MPI=ON uv sync --all-extras --group workspace-test --group bench \
  --reinstall-package monoprop --no-cache \
  --config-settings-package='monoprop:cmake.define.monoprop_MPI_TEST_PROCS=1;2;4' -v
cmake --build --preset skbuild-release
monoprop_ROUTING=linear OMP_NUM_THREADS=1 just test-cpp-mpi
monoprop_ROUTING=linear OMP_NUM_THREADS=1 just test-py-mpi '1;2;4'
```

Run only on an allocation supporting those ranks; smaller local runs are smoke tests, not substitutes. Do not apply the
serial CTest fabric exclusions to real MPI/performance runs. `just bench-build-mpi` is the existing MPI benchmark build
recipe; `just test-mpi` is the centralized full test recipe. Verify executable mtimes after sync; use the explicit
`cmake --build` above before trusting C++ results (or update the recipe to guarantee relinking in its authorized task).
If direct CMake settings are used instead of the MPI environment switch, also supply
`--config-settings-package='monoprop:build.requires=mpi4py>=4.1.0'`. `--all-extras` supplies runtime extras, not this
isolated-build requirement. Use `--label-exclude mpi` for the non-MPI leg: the export probe is labeled `unit`, not
`serial`. Test linear R=3 rejection directly with `Router::for_modes`, not by treating a full-suite failed launch as
success. The `just bench-mpi` companion forwards extra arguments to **mpiexec**, not pytest.

There is no alternate index-width build. `cpp/monoprop/TypeAliases.h:39` fixes TermIndex to uint32_t;
`operator_index_tests.cpp:31–34` asserts it. Inspect cache and imported binary identity for every configuration.

### S: sanitizer profiles

```bash
export monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1
SKBUILD_CMAKE_BUILD_TYPE=AsanUbsan \
monoprop_ENABLE_MPI=OFF SKBUILD_CMAKE_DEFINE='monoprop_SANITIZER=asan-ubsan;monoprop_ENABLE_MPI=OFF' \
uv sync --group workspace-test --all-extras --reinstall-package monoprop --no-cache -v
cmake --build build/editable/AsanUbsan --target monoprop_unit_tests.x
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build/editable/AsanUbsan --output-on-failure
LD_PRELOAD="$(g++ -print-file-name=libasan.so):$(g++ -print-file-name=libstdc++.so.6)" \
ASAN_OPTIONS=detect_leaks=0 uv run --no-sync python -m pytest -s

SKBUILD_CMAKE_BUILD_TYPE=Tsan monoprop_ENABLE_MPI=OFF \
SKBUILD_CMAKE_DEFINE='monoprop_SANITIZER=tsan;monoprop_ENABLE_MPI=OFF' \
uv sync --group workspace-test --all-extras --reinstall-package monoprop --no-cache -v
cmake --build build/editable/Tsan --target monoprop_unit_tests.x
TSAN_OPTIONS=halt_on_error=1:history_size=4 \
  ctest --test-dir build/editable/Tsan --output-on-failure --no-tests=error -R 'openmp_'
```

Migrate `justfile::test-cpp-tsan` from `(partition_|shm_comm_)` to retained concurrency tests plus `openmp_`, with
`--no-tests=error`, before deletion; `.github/workflows/qa-analysis.yml` invokes that recipe. Direct selection above is
a focused gate, not a replacement for CI migration. Use existing recipes in workflows;
`tools/check-workflow-commands.py` (enforced by `prek.toml`) forbids duplicating ordinary workflow build/test commands.

ASan and TSan are separate Linux profiles. Qualify the compiler/OpenMP-runtime TSan combination using a known-racy and
known-safe small test first; unsupported runtime reports are a tooling limitation, not race-free evidence. No
stock-Python TSan gate, no blanket suppression of engine frames, and no sanitizer performance measurements.

---

### Task 1: Freeze baselines and repair measurement trustworthiness

**Files:**
- Modify: `benches/conftest.py` (exactness and shared shape preflight); preserve existing windows and timed operations
  in `benches/{bench_random,bench_models}.py`. No historical node renaming or restoration of removed pytest nodes.
- Modify: `packages/monoprop-bench-tools/src/monoprop_bench_tools/{report,bmf}.py`.
- Test: `packages/monoprop-bench-tools/tests/{test_memory,test_report,test_bmf}.py`.
- Create: `tools/benchmark-rank-local-openmp.py`, `tools/rank-local-openmp-workloads.json`,
  `tests/test_rank_local_openmp_benchmark.py`.
- Reference/migrate guidance: `benches/LADDER.md`; supplement with `tools/capture-baseline.py`.
- Modify: `AGENTS.md`, `README.md`, `docs/content/docs/benchmarks.mdx` where measurement commands/interpretations are
  documented.

**EXISTING:** The four pytest families in each benchmark file are `build_graph`, `propagate`, `energy`, `gradient`. All
have operation windows. `propagate` is graph-free, not a separate inplace node. Functional creation in energy/gradient
is outside their timed window. Nested peaks are preserved by `memory/cpu.py::_fold_open_windows`; keep its tests.

**PROPOSED interfaces:** One bounded driver and shared overlay used unchanged on both arms, preserving historical node
IDs. Add independent `opmemexact[nodeid]` and `memhwmexact[nodeid]` booleans (all ranks exact in that window); absent
means unknown. The separate construction worker records its own `construction_exact`. Each fresh process writes one
kind-specific artifact; `compare` joins a timed/construction pair into one observation as specified below. The new
paring-construction measurement below is deliberate additional measurement work requiring full-plan approval, not
instrumentation-only.

**Selected platform, not observed topology:** AWS's
[exact-size table](https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html#co_hardware) specifies c8a.metal-24xl as
AMD EPYC 9R45, 96 physical cores/96 vCPUs, one thread/core and 192 GiB. Its
[C8a description](https://aws.amazon.com/ec2/instance-types/c8a/) explicitly says no SMT. These are published facts, not
measurements. Socket/NUMA counts remain unverified. Do not label oversubscription as SMT coverage or assume this machine
has production's usual eight domains. Once execution is authorized, archive the following read-only observations from
inside the actual allocation; these commands have not been run here:

```bash
lscpu
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
numactl --hardware
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
```

Use these observations and per-rank placement evidence to select NUMA-contained geometries; a capacity-based 1x96 shape
is not automatically admissible. Missing diagnostic tools require another external observation method, not a new
library topology dependency. Record hardware SMT state, but configure only one worker per physical core. Do not add
SMT-sibling worker profiles; production-HPC and multi-node qualification remain separate and pending.

- [ ] Obtain separate authorization for baseline build/calibration work. On c8a.metal-24xl, record sockets/NUMA domains,
  physical cores, hardware SMT state, memory allocation and software stack before choosing geometries. Agree
  representative workloads/sizes and single-instance routing/rank/thread shapes. Initial acceptance is the EC2 campaign,
  including a two-ranks-per-domain case; all workers use distinct physical cores. Production-HPC performance and
  multi-node qualification on a proper interconnect remain pending, not prerequisites for initial EC2 acceptance.
  Production's common eight-domain layout is not a hard-coded EC2 topology. Historical profiles below are candidates,
  not an approved campaign.
  Confirm the bounded new paring-construction measurement in that campaign. Record baseline
  revision, allocation, compiler/flags, MPI version, index width, OpenMP runtime, allocator environment, imported
  extension path/hash, seed and complete workload configuration before collecting evidence.
- [ ] Before changing engine code, separately build the baseline using R/M. Preserve its environment and binary. Apply
  only the identical measurement-harness changes to both arms; record this overlay's diff/hash. Baseline numerical
  failures stop this task. After the freeze, any required baseline failure blocks collection, not permission to resize
  or remove the cell.
- [ ] Once authorized, run unscored baseline-only feasibility trials before freezing workload sizes. Use existing
  baseline APIs/benchmark entrypoints; add no fourth driver mode or general calibration framework. Check elapsed
  time and memory for construction, evaluation and numerical export/validation, including their temporary storage.
  The approved pilot budget is 2 hours total, excluding builds/setup, and 15 minutes per trial including all those
  phases. If insufficient, stop and report; do not silently extend either limit. Target at most 75% of observed usable
  allocation RAM across all ranks, including construction and validation/export temporary storage. Record the RAM basis
  and byte ceiling before trials; do not keep recomputing it from falling free memory. Exceeding this target requires
  owner review before larger trials. This is operator-side sizing, not a library memory clamp or an OOM guarantee.
  Record every attempted configuration, geometry/placement, baseline binary/overlay identity, elapsed time, memory
  observations, outcome and reason for retaining/rejecting a size. Keep raw trials in a separate calibration archive,
  without acceptance sample IDs. Capacity/time failures may inform pre-freeze sizing; numerical inconsistencies stop.
  Do not run/consult candidate performance to select the inventory. Estimate formal campaign time/cost from the pilot,
  then seek owner approval of sizes, coverage and campaign expenditure before the freeze and formal collection. Do not
  reduce required samples/coverage to fit an unapproved budget. Collect fresh formal baseline samples after the freeze.
  Never promote calibration runs, even with unchanged parameters. Approval of these limits is not execution permission.
- [ ] Add renderer tests independently varying operation and outer exactness: true/false/absent and mixed flags.
  `bmf.py::build_bmf` publishes `peak-memory` from `memhwm`, so gate it on `memhwmexact`; `operation-memory` comes from
  `opmemdelta`, so gate it on `opmemexact`. `report.py::build_report` must label each corresponding metric correctly.
  False means non-exact warning and no exact metric for that window; absent means legacy/unknown, never assumed true.
  RED: current renderers claim exactness without flags. GREEN: no cross-window certification; time/term fields survive.
- [ ] In `OpMemory.close`, collectively reduce `HighWaterMark.exact` across every rank into `opmemexact`; independently
  do the same in `record_memory` for `memhwmexact`. Retain peak/base/delta sum/max, outer memhwm/memhwm_max, opbytes,
  opmembreak, opsize and existing provenance. Tiny integration checks require complete fields for all four families. No
  rank-zero-only collective. Extend existing nested transient/fallback tests, not the reset implementation.
- [ ] Migrate `_require_shape` in the identical shared overlay. Add explicit manifest/CLI runtime declaration
  `--runtime-shape=partitions|openmp` to pytest and driver; this is PROPOSED measurement configuration, not a backend
  selector. Baseline declares partitions with `monoprop_PARTITIONS=T` and `monoprop_NUM_THREADS=T`; candidate declares
  openmp with `monoprop_NUM_THREADS=T` and removed partition settings unset. OMP_NUM_THREADS may also be T for the host
  runtime. Accept the documented candidate fallback with monoprop_NUM_THREADS unset and OpenMP resolving T; verify it
  separately in configuration tests. Require positive declared R/T and equal allocation.
  Record declared and observed shape separately; a renamed `--arm` label must not bypass validation. Before acceptance,
  diagnostic runs must verify actual baseline partition participation and candidate worker participation on
  representative large kernels, accounting for documented small/nested/runtime-limit paths. Reject missing or
  contradictory evidence. Tests: missing baseline declaration, invalid candidate thread override, leftover candidate
  partition setting in the measurement environment, wrong observed R/T, mislabeled arm, and valid explicit/fallback
  candidate declarations. This strict measurement preflight does not restore a removed production getenv reader.
  RED: current multi-rank guard rejects candidate configuration.
- [ ] Separate whole-construction measurement is intentional policy, not a repair for nesting. A fresh untimed worker,
  without pytest memory fixtures/session graphs, opens one `HighWaterMark`, constructs inputs/propagator/graph/callable
  as required, runs the operation and retains outputs until close. Record peak sum/max, floor, delta and exactness. Use
  existing model builders; no duplicate physics or general benchmark framework.
- [ ] After unscored calibration, freeze an owner-approved campaign inventory BEFORE formal baseline collection. Its
  JSON schema is
  `{"schema_version":1,"workloads_path":str,"workloads_sha256":str,"cells":[cell]}`. Each cell has unique slug `id`,
  `profile`, `node_id`, `operation`, `measurement_kind` (`pytest` or `driver`), `identity`, `config_digest` and
  `parameters_digest`. `identity` contains `ranks`, requested `threads`, `index_bits=32`, `cpu_allocation`,
  `compiler_flags`, resolved `config`, and `routing` (`mode`, derived `bits`, numeric `seed`). Expand the complete
  size-band-specific inventory below: small/medium profiles use the full geometry sweep, with the existing routing
  coverage requirements; selected large profiles use only full-node one- and two-ranks-per-domain geometries. Do not
  generate an all-sizes Cartesian product. Archive the inventory/digest with owner approval. This file, not later
  observations, is the authoritative expected-cell list. A full cell missing
  from BOTH arms must fail completeness. Reject duplicate/extra cell IDs and mismatched identities. Do not regenerate or
  shrink the inventory to make observations pass.
- [ ] `tools/rank-local-openmp-workloads.json` maps `profiles[profile][node_id]` to `operation`, `measurement_kind`,
  `basis`, `picture`, complete `config`, deterministic `parameters`, and `pare_threshold`. `tiny` is the small band and
  calibrated `reference` is medium; each band's `-pared` variant shares its geometry policy. Preserve the existing
  unpared four-family and threshold-1e-10 energy/gradient plus driver-construction coverage. Add `large`/`large-pared`
  entries for the owner-selected large workloads, restricted to the two full-node decompositions. Profile names do not
  freeze the historical reference sizes: record every calibrated override explicitly, with no ambiguity within a
  profile/node pair. Define
  config/parameter digests as SHA-256 of UTF-8
  `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)`. File digests are SHA-256 of exact file
  bytes. Resolve relative JSON paths against the containing file's directory. Do not put binary or placement-evidence
  hashes inside cross-arm `identity`: those legitimately differ, while allocation/routing/workload must match.
- [ ] Give the driver exactly three argparse modes: `observe` requires `--arm baseline|candidate`,
  `--runtime-shape partitions|openmp`, `--campaign FILE`, `--cell-id CELL`, `--sample-id SAMPLE`, `--placement FILE`,
  `--output DIR`, plus optional `--whole-process`; `validate` takes the same options except
  `--sample-id`/`--whole-process`; `compare --campaign FILE --manifest FILE --output FILE`. Resolve profile, historical
  node ID and complete workload from the campaign cell and digest-checked workload file; reject an observed
  geometry/routing/configuration that differs. The pytest overlay also takes `--runtime-shape`; never rewrite historical
  node IDs. Environment/launcher/allocation stay external; the driver installs nothing.
- [ ] Define the external placement JSON consumed by `--placement`: `schema_version=1`, `artifact_kind="placement"`,
  `arm`, `cell_id`, `binary_hash`, `identity`, `runtime_shape`, `declared`, and `observed`. `declared` contains `ranks`,
  `threads`, `cpu_allocation` and `settings` (the exact OpenMP/library environment). `observed` contains `ranks`,
  `threads_per_rank` and `cpu_allocation`, from Task 11's binary-linked representative large-kernel diagnostic. Preserve
  raw launcher/affinity evidence alongside this summary. Baseline shape is `partitions`; candidate shape is `openmp`;
  both must match the campaign R/T/allocation. A limited-team diagnostic cannot establish the requested full-team
  benchmark cell. Small-kernel serial fallback remains allowed and reported. Each artifact records `placement_path` and
  its exact `placement_sha256`; compare reads and validates that file, not just a caller boolean. Cross-arm
  placement-file hashes differ legitimately. Diagnostic provenance can be reused for matching samples of the same
  cell/arm/binary, never substituted for measured operation or construction artifacts.
- [ ] Normal `observe` writes `DIR/ARM/CELL/SAMPLE/timed.json`; `observe --whole-process` writes
  `DIR/ARM/CELL/SAMPLE/construction.json` in a SEPARATE fresh process with the same sample ID. Cell/sample IDs are
  path-safe slugs. Create files exclusively; reject overwrite. Each invocation generates a fresh unique `run_id` and
  records schema version, artifact kind, arm, cell/sample IDs, campaign digest, binary hash, profile/node/operation,
  identity, config/parameter digests and placement reference/hash. Keep pytest raw files in that sample's timed
  subdirectory so labels cannot overwrite another sample. Timed artifacts contain runtime, operation and outer-window
  fields only; construction artifacts contain construction-window fields only. Do not invent timing for the untimed
  worker. `validate` writes a separate uniquely named validation artifact under `DIR/ARM/CELL`, without sample ID.
- [ ] Define comparison input as
  `{"schema_version":1,"campaign_sha256":str,"cells":[{"id":str,"baseline":[sample],"candidate":[sample]}]}`. Each
  `sample` contains `sample_id`, `timed_path`, `construction_path`, `validation_path`. `compare` first verifies the
  campaign/workload digests and exact expected cell inventory. It then reads the three references and joins only if arm,
  cell, binary hash, profile/node/operation, identity, configuration/parameter digests, campaign digest and placement
  evidence agree; timed/construction sample IDs must also agree. Binary hashes need not match across arms. Require five
  distinct timed runs AND five distinct construction runs per arm/cell initially, ten after prescribed repetition.
  Reject missing/mismatched pairs, duplicate sample IDs within an arm/cell, reused measurement paths or run IDs,
  including reuse across cells/arms. A validation artifact may serve multiple samples only for exactly the same
  cell/arm/binary/configuration/parameters. Never count copied paths or five references to one run as five samples.
- [ ] The joined observation exposes `runtime_seconds`, operation `peak_sum_bytes`/`peak_max_bytes`,
  `construction_peak_sum_bytes`/`construction_peak_max_bytes`, and `op_exact`/`construction_exact`. Both exactness flags
  must be true for parity. Preserve `outer_peak_sum_bytes`/`outer_peak_max_bytes` and independent `outer_exact` as
  DIAGNOSTICS, not parity gates. Outer false/unknown must be labeled non-exact/unknown, not upgraded by op_exact; it
  does not invalidate otherwise exact operation/construction evidence. Missing outer flags in legacy raw files map to
  unknown; missing required operation/construction exactness fails acceptance. Include operation/construction
  floor/delta sum/max and persistent bytes as diagnostics, never substitutes for absolute peaks. Keep original raw
  artifact metric names: operation peaks come from opmempeak; outer peaks from memhwm/memhwm_max. Compare exactly FIVE
  per-cell medians: runtime, operation peak sum/max, construction peak sum/max. All five ratios must be <=1.00. Report
  outer ratios separately; an outer-only regression cannot fail this five-metric gate. Reject missing/non-finite
  required measurements rather than use zeros. Derive numerical success by reading validation products below; a
  caller-supplied boolean is insufficient. Exit 0 means all cells pass, 1 means unmet gates (including incomplete
  campaign/sample coverage), and 2 means malformed/incompatible evidence. Output `failed_cells`, missing coverage, five
  gate ratios and separately labeled diagnostic ratios.
- [ ] Implement `validate` in a **separate untimed/unmeasured process**, reconstructing the same
  model/configuration/operation with existing builders. Include the shared provenance above with
  `artifact_kind="validation"` and its own run ID; record `schema_version`, `node_id`, `operation` (one of the five
  operation rows below), `config_digest`, `parameters_digest`, `binary_hash`, `basis`, `picture`, finite `energy`, full
  `gradient` when relevant, `global_term_count`, `graph_layers`, and `term_files` paths to decoded term/coefficient
  files. Each sorted term file is JSON Lines with `{"key":[indices],"real":number,"imag":number}`; gradient arrays are
  in optimizer parameter order. `pare_functional_construct` and pared evaluation artifacts omit term files because the
  pared callable owns a subgraph rather than an independently exported operator; validate its scalar/full gradient
  against the baseline pared callable. Use deterministic input parameters; a graph benchmark replays at those parameters
  for validation. The required products are:

| Operation | Validation products |
| --- | --- |
| build_graph | energy, gradient, decoded evolved term/value map and graph layer count after replay |
| propagate | immediate energy and decoded evolved term/value map; independent graph-replay comparison on the same inputs where both paths use the same cutoff settings |
| energy | scalar result plus graph decoded map if unpared; pared scalar against same-threshold baseline |
| gradient | scalar/full gradient plus graph decoded map if unpared; pared scalar/full gradient otherwise |
| pare_functional_construct | evaluate the returned pared energy/gradient callable at the recorded parameters; compare to the baseline pared callable at the same threshold |

For random graph-free propagation, EXISTING `build_random_propagator(..., lower_atol=None)` is the current default;
record it explicitly. Fixed Hubbard/Pauli atols are below. Freeze separate energy/gradient profiles with
`pare_threshold=None` and `pare_threshold=1e-10`; the latter times evaluation, not construction. Pared profiles validate
the same returned scalar/full-gradient callable on both arms, not an unpared operator map. Do not compare different
truncation settings. Existing tiny/exact and finite-difference tests remain independent physics oracles; large-cell
evidence compares baseline/candidate of the identical operation, not an infeasible dense exact simulator.
- [ ] Implement exactly one NEW bounded measurement identity per picture/profile:
  `driver::pare_functional_construct[heisenberg]` and `driver::pare_functional_construct[schrodinger]`. In the workload
  manifest set `operation="pare_functional_construct"`, `measurement_kind="driver"`,
  `functional="expectation_value_and_gradient_functional"`, `pare_threshold=1e-10`, picture, full random config and
  parameter digest. Use the proposed `observe`/`validate` interface: `--cell-id` resolves this node/profile from the
  frozen campaign, not a new framework or pytest family. Workload key is `(profile,node_id)`; a campaign cell also fixes
  geometry/routing and gets a unique slug. Freeze tiny-pared, reference-pared and selected large-pared entries before
  formal baseline collection, respecting each band's geometry coverage.
  Example planned invocation after selecting the matching cell and binary-linked placement file (launcher/allocation
  supplied externally):

```bash
monoprop_NUM_THREADS="$T" OMP_NUM_THREADS="$T" \
uv run --no-sync python tools/benchmark-rank-local-openmp.py observe \
  --arm candidate --runtime-shape openmp --campaign "$CAMPAIGN" --cell-id "$CELL" \
  --sample-id s01 --placement "$PLACEMENT" --output "$RESULTS"
# Repeat in a separate fresh process with the same arguments plus --whole-process.
# Run validate separately; omit --sample-id and --whole-process.
```

  Unset removed partition settings before this candidate command; if launched at three ranks, explicitly export
  `monoprop_ROUTING=splitmix` to every rank. Use `--runtime-shape partitions` and both legacy T declarations for
  baseline. In normal driver observe, build random inputs first, then open an outer `HighWaterMark` before
  propagator/graph construction (the driver analogue of pytest `record_memory`). Construct graph fully before barrier,
  timer and operation `HighWaterMark(settle=False)`. Close operation then outer window, retaining graph/callable through
  both; publish outer and operation exactness independently. Time ONLY
  `built_graph.expectation_value_and_gradient_functional(1e-10)`, using existing barrier semantics (entry untimed,
  completion included); one call per fresh process, retain returned callable until the window closes. Record time as
  rank maximum and peaks/floors/deltas as rank sum/max with independent exactness. Do not evaluate inside timing. A
  separate `validate` process invokes the retained callable at deterministic parameters and records scalar/full gradient
  plus graph/config/binary provenance; compare the same pared callable on both arms. No fabricated term export from a
  private pared graph. `observe --whole-process` opens before random inputs/propagator/graph construction, builds the
  same callable and keeps graph/callable live until close; it is untimed and records construction peaks separately.
  Tests instrument event order and callable lifetime, forbid evaluation during timing, check exact CLI/manifest
  identity, threshold/config parity, missing outputs and numerical mismatch. RED: driver identities absent. GREEN:
  bounded driver and validation artifacts complete on both arms. This retains original paring-construction runtime/peak
  coverage.
- [ ] Export maps with the existing low-level `propagator._simulator.evolved_operator(parameters, 0.0)`; use an empty
  parameter vector after graph-free propagation and the repeated circuit parameters for graph replay. Sort raw encoded
  index tuples and store real/imaginary coefficient components in per-rank files. These exported coefficients are
  currently rounded to 1e-12; state that limitation and retain unrounded scalar/gradient checks. Stream-merge sorted
  rank files when comparing global maps instead of requiring historical rank ownership. For Heisenberg, the binding adds
  the replicated identity/core term on each rank: retain one copy after checking agreement, **do not sum it R times**.
  Other duplicated nonidentity terms fail unique ownership; Schrödinger identity follows ordinary owner semantics.
  Require equal global key sets, including zero-coefficient stored terms; compare values with the existing
  `test_utils::near` rule from `cpp/tests/TestUtilities.h.in`: `abs(a-b) <= 1e-9 + 1e-7*max(abs(a),abs(b))`,
  componentwise for complex values and gradients. Reject missing/non-finite outputs. Preserve all stricter existing
  unit-test tolerances; this benchmark comparison does not weaken them. If full validation export cannot fit, stop for
  an approved smaller equal-arm profile, not an unchecked `numerical_ok=true`.
- [ ] Link every performance observation to a successful validation artifact for the same binary/configuration/operation
  and compare baseline/candidate validation artifacts. Add tests with wrong energy, a changed gradient entry, one
  missing/extra term, duplicate ownership, replicated identity, non-finite data, stale binary hash and absent
  validation. A completed pytest benchmark with only type/length assertions is not numerical evidence.
- [ ] Test the comparator without running benchmarks:

```python
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_comparison_does_not_hide_one_regression(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/benchmark-rank-local-openmp.py"

    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def digest(value):
        return hashlib.sha256(encoded(value)).hexdigest()

    def save(name, value):
        path = tmp_path / name
        path.write_bytes(encoded(value))
        return str(path)

    config = {"gen_length": 4, "obs_terms": 16, "num_generators": 8, "num_modes": 8,
              "cutoff": 6, "seed": 0, "lower_atol": None}
    parameters = [0.0] * 8
    identity = {"ranks": 1, "threads": 2, "index_bits": 32, "cpu_allocation": "test-cpus",
                "compiler_flags": "test", "config": config,
                "routing": {"mode": "linear", "bits": 0, "seed": 0}}
    expected, entries = [], {}
    for name in ("build", "replay"):
        node = f"energy-fixture-{name}"
        entries[node] = {"operation": "energy", "measurement_kind": "pytest", "basis": "majorana",
                         "picture": "heisenberg", "config": config, "parameters": parameters,
                         "pare_threshold": None}
        expected.append({"id": name, "profile": "tiny", "node_id": node, "operation": "energy",
                         "measurement_kind": "pytest", "identity": identity,
                         "config_digest": digest(identity["config"]), "parameters_digest": digest(parameters)})
    workloads = {"schema_version": 1, "profiles": {"tiny": entries}}
    campaign = {"schema_version": 1, "workloads_path": save("workloads.json", workloads),
                "workloads_sha256": digest(workloads), "cells": expected}
    campaign_path = save("campaign.json", campaign)
    cells = []
    for target, candidate_time in zip(expected, (10.1, 5.0), strict=True):
        name = target["id"]
        cell = {"id": name}
        for arm, elapsed in (("baseline", 10.0), ("candidate", candidate_time)):
            shape = "partitions" if arm == "baseline" else "openmp"
            binary_hash = ("b" if arm == "baseline" else "c") * 64
            settings = {"monoprop_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"}
            if arm == "baseline":
                settings.update(monoprop_PARTITIONS="2")
            placement = {"schema_version": 1, "artifact_kind": "placement", "arm": arm,
                         "cell_id": name, "binary_hash": binary_hash, "identity": identity,
                         "runtime_shape": shape,
                         "declared": {"ranks": 1, "threads": 2, "cpu_allocation": "test-cpus",
                                      "settings": settings},
                         "observed": {"ranks": 1, "threads_per_rank": [2], "cpu_allocation": "test-cpus"}}
            placement_path = save(f"{name}-{arm}-placement.json", placement)
            common = {"schema_version": 1, "campaign_sha256": digest(campaign), "arm": arm,
                      "cell_id": name, "profile": target["profile"], "node_id": target["node_id"],
                      "operation": "energy", "binary_hash": binary_hash, "identity": identity,
                      "config_digest": target["config_digest"], "parameters_digest": target["parameters_digest"],
                      "placement_path": placement_path, "placement_sha256": digest(placement)}
            validation = {**common, "artifact_kind": "validation", "run_id": f"{name}-{arm}-validation",
                          "basis": "majorana", "picture": "heisenberg", "energy": 0.0,
                          "global_term_count": 0, "graph_layers": 0, "term_files": []}
            validation_path = save(f"{name}-{arm}-validation.json", validation)
            cell[arm] = []
            for i in range(5):
                sample_id = f"{name}-{arm}-{i}"
                timed = {**common, "artifact_kind": "timed", "sample_id": sample_id,
                         "run_id": f"{sample_id}-timed", "runtime_seconds": elapsed,
                         "peak_sum_bytes": 1000, "peak_max_bytes": 1000, "op_exact": True,
                         "outer_peak_sum_bytes": 1200, "outer_peak_max_bytes": 1200, "outer_exact": True,
                         "operation_floor_sum_bytes": 800, "operation_floor_max_bytes": 800,
                         "operation_delta_sum_bytes": 200, "operation_delta_max_bytes": 200}
                construction = {**common, "artifact_kind": "construction", "sample_id": sample_id,
                                "run_id": f"{sample_id}-construction", "construction_exact": True,
                                "construction_peak_sum_bytes": 1200, "construction_peak_max_bytes": 1200,
                                "construction_floor_sum_bytes": 800, "construction_floor_max_bytes": 800,
                                "construction_delta_sum_bytes": 400, "construction_delta_max_bytes": 400}
                cell[arm].append({"sample_id": sample_id,
                                  "timed_path": save(f"{sample_id}-timed.json", timed),
                                  "construction_path": save(f"{sample_id}-construction.json", construction),
                                  "validation_path": validation_path})
        cells.append(cell)
    manifest = save("manifest.json", {"schema_version": 1, "campaign_sha256": digest(campaign), "cells": cells})
    output = tmp_path / "comparison.json"
    result = subprocess.run([sys.executable, str(script), "compare", "--campaign", campaign_path,
                             "--manifest", manifest, "--output", str(output)], check=False)
    assert result.returncode == 1
    assert json.loads(output.read_text())["failed_cells"] == ["build"]
```

These two miniature inventory entries use empty-map energy fixtures, not real benchmark measurements. They follow the
same schema, reference loading and provenance checks as real artifacts; no synthetic-mode bypass is allowed. The
comparison test does not run physics, acquire placement, or establish actual hardware participation. Add tests for a
whole required cell deleted from both arms (including a large-profile cell), missing samples, mismatched
timed/construction pairs, reused measurement files/run IDs, raw calibration output offered as an acceptance artifact,
missing placement provenance (exit 2), each false/absent exactness flag, unequal
shape/configuration/index width, and sum-versus-max peaks. For only an outer-peak regression or unknown outer exactness,
report the diagnostic accurately while allowing the five-metric gate to pass. Operation/construction inexactness still
fails. The comparator reports that five more observations are needed on first ratio failure; it launches no jobs.
- [ ] Use the baseline-only trials and observed allocation to freeze owner-approved sizes for these coverage families in
  `tools/rank-local-openmp-workloads.json`: fixed Hubbard and Pauli models and random
  Heisenberg/Schrödinger × build_graph/propagate/energy/gradient. Energy/gradient have separate unpared and
  threshold-1e-10 profiles; do not change node IDs to distinguish the profile. Also require random Heisenberg and
  Schrödinger `pare_functional_construct` driver cells; no removed pytest node is revived. The following are EXISTING
  source-reference configurations, not the selected acceptance workloads:

```json
{
  "hubbard": {"num_sites": 60, "hopping": 1.0, "interaction": -2.0,
    "chemical_potential": 0.0, "trotter_dt": 0.2, "trotter_steps": 29,
    "observable_site": 46, "observable_spin": "up", "neel_start_spin": "down",
    "cutoff": 6, "lower_atol": 0.0001},
  "pauli": {"num_qubits": 127, "num_layers": 20, "observable_qubit": 62,
    "theta": 0.7853981633974483, "coupling": 0.7853981633974483,
    "cutoff": 8, "lower_atol": 0.0001},
  "random": {"gen_length": 4, "obs_terms": 10000, "num_generators": 100,
    "num_modes": 128, "cutoff": 6, "seed": 0, "lower_atol": null}
}
```

Add a tiny profile with random options `--num-generators 8 --num-modes 8 --cutoff 6 --obs-terms 16`; other random values
stay fixed. Both fixed models use their existing picture; random cases cover both pictures. Keep these complete
configurations in artifacts, not just a profile label.
- [ ] Collect node IDs with `uv run --no-sync python -m pytest benches --collect-only -q`; record exact IDs and run one
  node per process. The existing fixed-model guard skips retained graphs with more than two repeated steps
  (`MAX_GRAPH_STEPS=2`), so the 29-step Hubbard reference is intentionally a capacity decision: use
  `monoprop_BENCH_ALLOW_BIG_GRAPH=1` only on adequately provisioned hardware, or obtain approval for a fixed lower
  `--hubbard-trotter-steps` profile on both arms before formal baseline collection. A skipped cell is not a pass. Do not
  silently omit graph-heavy cells or bypass the guard on an unverified allocation.
- [ ] Select geometries from observed physical cores, not vCPU/SMT-sibling counts. For D allocated NUMA domains with
  C selected physical cores each, the small/medium sweep is `(R,T)=(1,1),(1,C),(D,1),(D,C)`, deduplicated. R=1 uses
  only one domain. Add `(2*D,floor(C/2))` with non-overlapping physical-core subsets when C>=2; record unused cores.
  Apply existing routing coverage requirements too. Selected large profiles use only `(D,C)` and `(2*D,floor(C/2))`
  from the full-node allocation, not the one-worker/partial-node sweep. Freeze the exact allowed cells per profile;
  do not reclassify profiles or drop cells after a regression. If domains/allocation are uneven, agree explicit balanced
  subsets before freezing rather than assume C=96/D. Each baseline/candidate pair has identical R/T/core allocation.
  All these cells are single-instance; network-HPC qualification stays pending. Old throughput uses R ranks x T
  partitions; retain an old single-store serial diagnostic but never compare only against forced-serial old code.

- [ ] Reuse `benches/LADDER.md` L1–L4 labels/provenance and term-count fields as historical calibration, never
  acceptance evidence from old runs. Where a full-node single rank crosses NUMA domains, do not copy that historical
  allocation: freeze a within-one-domain profile, identical on both arms, before formal baseline collection. Preserve
  historical nodes and fields while adding runtime-shape/observed-team metadata. Term counts must agree for the
  identical operation.
- [ ] Use `just capture-baseline` / `just diff-baseline` and `tools/capture-baseline.py --compare` only as supplementary
  small-fixture regression evidence. Its keys include rank, tolerance differs and it lacks operation
  gradients/provenance; it cannot replace the ownership-aware global-map comparator or independent
  exact/finite-difference tests.

**Gate:** Calibration archive and owner-approved size-band inventory/digest are retained; benchmark tools' unit tests
and baseline numerical suites pass. Fresh post-freeze baseline observations and placement evidence cover every required
cell; no calibration sample is counted. Historical benchmark keys remain unchanged. No parity claim yet.

### Task 2: Add the smallest OpenMP primitive and its build dependency

**Files:**
- Create: `cpp/monoprop/detail/parallel/{Options.h,Workshare.h,CMakeLists.txt}`.
- Modify: `CMakeLists.txt`, `cpp/monoprop/CMakeLists.txt`, `cpp/monoprop/detail/CMakeLists.txt`,
  `cmake/monopropConfig.cmake.in`.
- Create/test: `cpp/tests/openmp_workshare_tests.cpp`.
- Modify: `pyproject.toml`, `tools/install-deps.sh`, `tools/packages/{apt,brew}.txt`,
  `.github/actions/setup/action.yml`, `.devcontainer/Dockerfile`, `nix/{monoprop,devshell}.nix`, `flake.nix`,
  `.github/workflows/{test,nix,deploy}.yml` for required OpenMP across supported dependency routes.
- Extend: `cpp/tests/find_package_smoke/CMakeLists.txt`, `cpp/tests/link_export_probe/link_export_probe.cpp`; reuse
  `just test-find-package`, not a second consumer harness.
- Modify: `AGENTS.md`, `README.md`, `docs/content/docs/building.mdx` for required OpenMP.

**PROPOSED interfaces:** In namespace `monoprop::detail::parallel`:

```cpp
struct Options {
    int threads = 1; //!< Validated positive per-object budget; not a global runtime setting.
};

template <class Fn>
auto for_blocks(size_t count, Options options, Fn &&body) -> void;
// body(block_id), one invocation per logical block; returns only after all work joins.
// No MPI, no shared append, no escaping worker exception. Serial if nested or budget==1.
```

`Options.h` contains no OpenMP include; `Workshare.h` includes `<omp.h>`. No pool, scheduler, task queue, type-erased
jobs, or persistent worker-owned operator state.

- [ ] Add tests that require the new helper: every block visited exactly once; count 0/1/19; budget 1/2/3; more
  requested threads than blocks; nested invocation; original runtime max-thread setting unchanged; a worker throws
  `std::runtime_error` and caller catches it after all work has joined.
- [ ] Use this test pattern; each element has one writer:

```cpp
BOOST_AUTO_TEST_CASE(openmp_blocks_visit_once) {
    std::vector<int> visits(19);
    monoprop::detail::parallel::for_blocks(19, {.threads = 3}, [&](size_t b) {
        ++visits[b];
    });
    for (const auto n : visits) { BOOST_TEST(n == 1); }
}
```

For the exception test also record visited/completed blocks without asserting which ones finish after a failure. Never
use `BOOST_TEST` inside workers; store results and assert on the caller.
- [ ] Run the new test build: expect missing header/helper. Add required dependency integration:

```cmake
# Root discovery
find_package(OpenMP REQUIRED COMPONENTS CXX)
# Compilation and link/export requirements are separate: the shared library is
# assembled from TARGET_OBJECTS and does not inherit the object target's linkage.
target_link_libraries(monoprop-objs PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(monoprop PUBLIC OpenMP::OpenMP_CXX)
# Installed cmake/monopropConfig.cmake.in:
find_dependency(OpenMP REQUIRED COMPONENTS CXX)
```

Register `Options.h`/`Workshare.h` in the existing `headers` FILE_SET via the new subdirectory. Use imported target
flags, not hard-coded `-fopenmp` or `-lgomp`. There is no OpenMP-disabled build in this design; a one-thread build/run
is still OpenMP-enabled.
- [ ] Implement serial and parallel paths. Serial path calls `body(b)` in ascending order without allocating error
  slots. Parallel path requests `min(options.threads,count)` workers, uses
  `#pragma omp parallel for schedule(static) num_threads(team)`, catches into preallocated `std::exception_ptr` slots
  **indexed by actual worker ID**, joins, then rethrows the first populated slot. Allocate at most `team` errors, not
  one per row/block. Validate positive budget and loop-bound representability before the region. Correctness must not
  depend on actual team size equaling the request.
- [ ] Do not invoke throwing setup outside the worker's catch. Exceptions from allocation of error slots occur before a
  region and propagate normally. The helper cannot provide transactional rollback.
- [ ] Add `libomp` to the Homebrew manifest and the matching Clang OpenMP development package to the apt/setup route;
  GCC uses its selected libgomp. `.github/actions/setup/action.yml` and `.devcontainer/Dockerfile` consume the shared
  package manifests: update the source of truth rather than duplicate installs. Wire compiler-compatible OpenMP inputs
  into both Nix package and devshell, preserving Nix's explicit sandbox build-time mpi4py provisioning. Extend
  `flake.nix` package/MPI package/check/devshell checks and `.github/workflows/nix.yml` smoke validation. Verify OpenMP
  configure/link/run in both Nix packages and shell. Hwloc stays until Task 12; do not delete it early.
- [ ] Install `libomp` for macOS and Clang Linux jobs; verify keg-only Homebrew discovery with the actual toolchain.
  Keep GCC's selected libgomp when using GCC. Linux wheel dependency repair and macOS delocation must resolve the chosen
  runtime in a clean environment. Keep hwloc only while the legacy partition runtime still requires it; Task 12 deletes
  that dependency and its topology code rather than porting them. Audit Threads separately for remaining non-partition
  uses; OpenMP discovery/linking must not add a direct hwloc requirement.
- [ ] Run R and the new helper tests with `OMP_DYNAMIC=TRUE` and `FALSE`, plus `OMP_THREAD_LIMIT=1`. A limit of one must
  still visit every block. Add a test-only worker-ID observation proving more than one worker participates when
  hardware/runtime permits; report a skip when fewer than two are available, not a false success.

**Gate:** Helper tests and `just test-find-package` pass; extend its shared link-export probe source to instantiate an
OpenMP template and both basis APIs using installed targets only. RED: missing imported OpenMP dependency/header; GREEN:
clean consumer finds/links/runs with no manual flags or source includes. No engine behavior changes yet.

### Task 3: Plumb per-object budgets and distributed failure boundaries

**Files:**
- Create: `cpp/monoprop/detail/parallel/{ThreadBudget.h,ThreadBudget.cpp}`.
- Create: `cpp/monoprop/detail/mpi/OperationFailure.h`.
- Modify: `cpp/monoprop/detail/parallel/CMakeLists.txt`, `cpp/monoprop/detail/mpi/CMakeLists.txt`.
- Modify: `cpp/include/monoprop/{MonomialPropagator,Evolution,MPFunctions}.h`.
- Modify: `cpp/monoprop/detail/monomial_propagator/MonomialPropagator.inl`, `cpp/monoprop/{Evolution,MPFunctions}.cpp`.
- Modify: `cpp/monoprop/detail/evolution/{CosineRecompute,CosineRecomputeCallbacks}.h`,
  `cpp/monoprop/detail/evolution/layer_build/{Scan,Resolve,Engine,FusedApply}.h`.
- Modify: `cpp/monoprop/detail/mpi/MPICompat.cpp`; the final environment/configuration cleanup is in Task 10.
- Create/test: `cpp/tests/openmp_runtime_tests.cpp`, `cpp/tests/mpi_failure_driver.cpp`, `tests/test_openmp_config.py`;
  modify `cpp/tests/CMakeLists.txt` to exclude the failure-driver main from the unit-runner glob and build a separate
  executable.

**PROPOSED interfaces:**

```cpp
// ThreadBudget.h; include <optional>, <string_view> and Options.h.
namespace monoprop::detail::parallel {
// nullopt selects runtime_default; a present value must be decimal digits and in [1, INT_MAX].
// Throw std::invalid_argument on invalid input; no hardware query or global runtime mutation.
auto resolve_thread_budget(std::optional<std::string_view> configured, int runtime_default) -> Options;
// Read monoprop_NUM_THREADS once and obtain omp_get_max_threads(); called only during construction.
auto capture_thread_budget() -> Options;
}
// No new public constructor argument or binding keyword.
// Private object members: parallel::Options parallel_; bool invalid_ = false;
// Private helpers: auto require_valid_() const -> void; auto invalidate_() noexcept -> void;
// MonomialPropagator.h: declare monoprop::InvalidPropagatorError deriving std::runtime_error.
// EvalRequest: append detail::parallel::Options parallel = {};
```

All the following existing functions gain a trailing `parallel::Options options = {}` (fully qualify namespace in public
headers): `build_layer`, `fused_find_and_collect`, `probe_incoming_queries`, `resolve_incoming`, `apply_fused_contract`,
`scale_cos_lazy`, `accumulate_cos_lazy`, `scale_cos_mask`, `accumulate_cos_mask`, `evolve_step`, `evolve_operator`,
`state_operator_derivative_local`, and `build_cos_callbacks`. Private helpers receive/forward the same value. Keep
callback types `LayerCosScale`/`LayerCosAccumulate`/`LayerCosIndices` unchanged: closures capture options. EXISTING
derivative signature in `cpp/include/monoprop/Evolution.h:54–63` is the following signature (retain its existing export
annotation):

```cpp
auto state_operator_derivative_local(VecD&, VecD&, const MPGraphView&, size_t,
    LayerAngle, mpi::Comm, const detail::LayerCosAccumulate&,
    const detail::CosRecordView& = {}) -> double;
```
PROPOSED options follow `record`, not `angle` or `cos_acc`; preserve `LayerAngle{.gen_coeff=g,.param=theta}` and record
forwarding. `MPFunctions.cpp:44–172` records pre-layer values based on accumulated amplification AND vanishing-cosine
criteria; `Evolution.cpp:450–477` snapshots self endpoints, predivides, accumulates, restores and combines
pre-layer-unit sums. Keep record append/predivide/restore serial because duplicate indices are allowed. Do not replace
this stable protocol with a division-only reverse pass.

- [ ] Add compile/integration tests for old low-level numerical call sites (their default Options stay serial),
  prototype budgets 1/3 in separately launched processes, copied budget and retained-functional capture. Prove legacy
  children stay serial even when their process environment requests multiple threads.
- [ ] During coexistence, initialize `parallel_` to `{.threads=1}`. Only when the EXISTING constructor has explicit
  `partitions==1` AND `comm.kind==mpi::Comm::Kind::Mpi`, capture the environment budget before work. This also covers
  non-MPI builds, whose ordinary Comm has Kind::Mpi. Shm/Hybrid children never activate OpenMP; multi-partition facades
  keep their legacy path. Prototype tests use existing `partitions=1` and launch-time monoprop_NUM_THREADS; no new
  constructor keyword, runtime switch or test-only production setter is introduced. Task 10 removes the old arguments
  and makes capture unconditional for independent instances.
- [ ] Rebuild bindings through R/M without adding a thread keyword. Configuration already invokes binder generation
  and installation invokes dispatch generation. For manual regeneration of the default Release configuration only:

```bash
uv run --no-sync python tools/generate-binders.py --max-num-modes 1024 \
  --batch-size 8 --output-dir build/editable/Release/src/monoprop/bindings/generated \
  --binding-kind core
```

For a nondefault maximum use the actual `monoprop_MAX_NUM_MODES` cache value (current default 1024). Generated files
stay in the build tree and are not committed. Never hand-edit generated bindings, invent a mode-width policy, or modify
`tools/_binding_layout.py` for this task.
- [ ] Implement `resolve_thread_budget(configured,runtime_default)` as a pure parser: nullopt selects a positive runtime
  default; present input must be nonempty ASCII decimal digits, with the whole value in [1, INT_MAX]. Use checked
  parsing before narrowing; reject signs, whitespace, lists, junk, zero and overflow. Leading zeros are harmless. Do not
  retain the old arbitrary million-thread parser cap. Translate std::invalid_argument to PropagatorConfigError at the
  constructor boundary, then apply the coherent distributed failure policy if peers can be stranded. Tests cover nullopt
  with 1/3, nullopt with 0/-1 rejection, "8" with default 2 returning 8, and "1" with default 0 returning 1.
  Reject empty, "0", "-1", "abc", "2,3", " 3" and "+3". Test INT_MAX text acceptance and overflow rejection without
  creating teams.
- [ ] `capture_thread_budget()` reads getenv("monoprop_NUM_THREADS") once, distinguishes absent from present-empty,
  and calls the pure resolver with omp_get_max_threads(). No function-static budget cache or operation-time reload.
  During coexistence keep the old config parser solely for legacy partition selection; remove that cached budget path
  in Task 10. Copy resolved options unchanged. The supported application fixes ranks/threads at launch; no live
  configuration mechanism is provided or required.
- [ ] OpenMP supplies the fallback at any rank count, not an actual team size or hardware count. Do not independently
  parse OMP_NUM_THREADS or derive T from omp_get_num_procs, hardware_concurrency, affinity masks, places, hwloc, sysfs
  or NUMA. A request may exceed the allocation; resource suitability is the user's responsibility. Small-work/nested
  internal-helper fallback and OpenMP limits may reduce actual teams without changing the captured budget.
- [ ] In this task test parser/copy/callable capture and runtime-limited teams through the one-store prototype;
  final constructor surface and full default-selection subprocess tests belong to Task 10. Verify library calls leave
  OpenMP settings unchanged and issue no affinity-setting calls. A no-affinity-change test must disable OpenMP binding
  (`OMP_PROC_BIND=FALSE`), so it does not mistake the runtime's user-requested placement for library pinning. No
  hardware-topology query is needed by these tests; Task 11 placement checks are external benchmark-operator work.
- [ ] Forward options through ALL paths: graph-only build; graph-with-coefficients stored-mask closure; immediate fused
  apply; replay; retained functional capture and `EvalRequest`; `ev`/`ev_and_grad` → `prepare_evolved_operator` →
  evolution; derivative callback; partial contraction. `build_cos_callbacks` captures `[cache, sc, options]`. Keep
  parity pointers ephemeral. Do not alter the GIL policy or external callable lifetime contract.
- [ ] Implement a caller-side `mpi::operation_failed(comm, std::exception_ptr)` as a `[[noreturn]]` helper. For an
  ordinary real MPI communicator with size>1, report rank/error and call `MPI_Abort`; if MPI_Abort returns, terminate.
  For size one/non-MPI, rethrow the original exception. Keep transitional legacy communicator failures on their existing
  poison/abort paths until Task 12.
- [ ] Guard every newly throwable distributed phase, not merely the outermost function. Especially catch a worker
  failure in overlapped cosine work **inside the lifetime of an active exchange ticket** and abort before stack
  unwinding can wait on peers that never posted. Normal successful paths still explicitly wait. Do not rely on a ticket
  destructor to coordinate distributed failure; do not add a per-kernel allreduce to agree on allocation failure. Wrap
  constructor/build/replay/evaluation phases that can strand another rank, including retained-callable entry points.
  Single-rank failures preserve the original exception type. Add private `invalid_`, `invalidate_()` and
  `require_valid_()` to enforce non-reuse. The latter throws InvalidPropagatorError with a stable recreate-the-object
  diagnostic. On the caller, separate validation from the mutation phase using a local `mutation_started` flag; set it
  before the first mutable cache/state/workspace change. A catch after that boundary calls invalidate_() before the
  existing abort/rethrow path, including inside active-ticket lifetime. Successful calls never invalidate; pre-mutation
  validation errors preserve validity. No atomic flag, rollback framework or new persistent workspace is needed.
- [ ] Call require_valid_() before state-consuming/mutating public operations and dependent functional invocations.
  Retained closures capture the existing non-owning owner reference and check it before touching caches/snapshots;
  they invalidate that owner on their own post-mutation failure. Do not change C++ callable lifetime or Python owner
  retention. Validate a copy/clone source before any store/graph copy; cloning must not revive an invalid object.
  Already independent copies remain valid. Destruction is always allowed; immutable configuration diagnostics need
  not inspect invalid computational state. No production reset/setter clears invalid_.
- [ ] Add single-rank test-only failure injection at a real mutation boundary and during functional evaluation. Catch
  the original exception; then assert build/replay/update/functional creation, a previously retained functional and
  copy/clone reject the invalid owner. An independent earlier copy still evaluates. A validation-only failure leaves
  the original usable. RED: reuse currently reaches state; GREEN: entry guards reject before further work. Injection
  remains test-only and no production arbitrary-callback API is added.
- [ ] Validate already initialized MPI with `MPI_Query_thread`, not only monoprop-owned initialization. Reject a level
  below `MPI_THREAD_SERIALIZED` with an actionable diagnostic before distributed work. MPI worker calls are forbidden;
  low-level numerical kernels may serialize under a nested OpenMP call, but distributed public operations from an
  external OpenMP team are unsupported and must reject before communication. The host contract is one controlling
  caller per rank, no overlapping use of one propagator/dependent functionals and no overlapping host MPI calls.
  Do not add a process-wide locking service or nested-parallel execution machinery.
- [ ] Add a separate failure driver with command cases `worker-throw`, `before-exchange`, `active-ticket`,
  `insufficient-thread-level`. On rank zero inject the failure; peers execute the normal phase. Run from a subprocess
  with timeout 30s under two ranks. Expected outcome is prompt nonzero failure, **not** timeout or a successful exit. Do
  not put intentional MPI_Abort cases inside the ordinary suite executable. Test single-rank worker throws separately.

**Gate:** R/M compile and pass with kernels still serial. Focused gates include `combined_recompute_equivalence.cpp`
record restoration and `tests/test_deep_circuit_gradient.py` deep non-singular, vanishing-cosine and no-record cases,
before and after threading. Options reach every intended entry point; runtime global settings unchanged; failure tests
do not hang. Public/exported signature ABI change is recorded for a coordinated library/bindings rebuild.

### Task 4: Prototype cosine and fused updates without structural concurrency

**Files:**
- Modify: `cpp/monoprop/detail/evolution/CosineRecompute.h`, `cpp/monoprop/detail/evolution/layer_build/FusedApply.h`.
- Create/test: `cpp/tests/openmp_kernel_tests.cpp`.
- Extend: `cpp/tests/{combined_recompute_equivalence,fused_cos_sweep_tests}.cpp`.

**PROPOSED interfaces:** Consume Task 3 options. No further public interface. `for_blocks` handles region/exception
joining; caller chooses semantic blocks and the serial threshold.

- [ ] First add exact vector-equivalence tests using a stored mask spanning 4097 words; final mask word has only bit
  zero. Test budgets 1/2/3/4, zero words, 63/64/65-row tails and more workers than work.

```cpp
BOOST_AUTO_TEST_CASE(openmp_cos_mask_exact) {
    monoprop::CosMask mask;
    for (size_t w = 0; w < 4097; ++w) {
        const auto bits = w == 4096 ? uint64_t{1} : uint64_t{0x5555555555555555};
        mask.blocks.emplace_back(w * 64, bits);
        mask.total_count += static_cast<size_t>(std::popcount(bits));
    }
    monoprop::VecD s(4097 * 64, 1.25), p = s;
    monoprop::detail::scale_cos_mask(s.data(), mask, 0.625, {.threads = 1});
    monoprop::detail::scale_cos_mask(p.data(), mask, 0.625, {.threads = 3});
    BOOST_CHECK_EQUAL_COLLECTIONS(s.begin(), s.end(), p.begin(), p.end());
}
```

- [ ] Add lazy-mask tests in `combined_recompute_equivalence.cpp` with an `InvertedIndex<8>` exceeding three
  `kColumnBlockWords` blocks, mixed dense/sparse columns, odd/even Majorana, Pauli transformed fold, empty columns, zero
  generator, and partial `scaled_count`. Clear `row_parity_` before the **parallel-first** call to expose cold-cache
  races. Compare coefficients exactly against the existing serial oracle.
- [ ] Parallelize `scale_cos_lazy` over existing 1024-word fold blocks after caller-side parity preparation. Obtain
  scratch inside each worker invocation; use a fixed `std::array<uint64_t,kColumnBlockWords>` for no-allocation fold
  scratch or the existing TLS under the helper's catch. Never capture a reference to the caller's TLS buffer in the team
  lambda.
- [ ] Parallelize `scale_cos_mask` over disjoint mask-block ranges. Keep accumulate/reduction routines serial in this
  task. Keep `fold_to_cos_mask` and `fold_to_indices` serial to preserve diagnostics/ordering cheaply.
- [ ] In fused apply, size snapshot arrays on the caller, parallel gather insert-record snapshots, join, complete
  optional cosine phase, join, then workshare hit/insert/cross-half records. Each iteration performs the entire existing
  rotation formula; retain phase order. In debug tests record all updated row IDs and assert exactly one add-owner per
  destination before using this invariant.
- [ ] Keep first prototype thresholds conservative: at least two logical work blocks; a cosine block is 1024 words and a
  record block is 1024 records. Tiny cases run serial. Thresholds may be tuned only by Task 11 evidence, not semantic
  changes. Preservation fixtures above may already pass with Task 3's serial options plumbing. For RED/GREEN of actual
  threading, use test-build-only observations at each affected kernel work-range body: pre-size one worker-ID slot per
  logical range, have its sole owner record `omp_get_thread_num()`, and check distinct IDs after join on adequate
  hardware. Do not count a team created elsewhere as this kernel's participation. A runtime-limited skip is not scaling
  evidence. These observations must not become a production callback API or persistent per-term storage.
- [ ] Run R, M targeted fused/cosine tests, S, and repeated stress with runtime-limited teams. Measure serial versus
  threaded one-store replay before investing in construction. Archive timings/peak/worker scratch; this prototype cannot
  establish construction scalability.

**Gate:** Exact coefficient agreement and measured replay scaling on large cases; no coefficient-sized additional
persistent buffer. If replay is already bandwidth-bound with regressions, stop/profile rather than proceed on assumed
speedup.

### Task 5: Deterministic parallel bitmap traversal and query generation

**Files:**
- Modify: `cpp/monoprop/detail/evolution/layer_build/{Scan,Engine}.h`, `cpp/monoprop/algebra/AlgebraCommon.h`.
- Extend: `cpp/tests/{fused_cos_sweep_tests,sparse_query_tests,pauli_build_layer_tests}.cpp`,
  `cpp/tests/{exact_upper_atol_rescue,evolution_detail_tests}.cpp`.

**PROPOSED interfaces:** Add `CutoffEvaluator::parallel_safe() const noexcept -> bool`, initially true only for existing
typed `LengthCutoff`/`SupportCutoff` targets. Ordinary Python/C++ constructors take cutoff settings and basis-change
data, not arbitrary functions. Opaque predicates remain possible through protected C++ `cutoff_fn_` or low-level
helpers; keep their serial path. Current basis-change closures capture data by value, with no mutable capture found;
their initial serial path is conservative, not evidence of unsafety. Do not add a new callback API or claim a const
std::function alone proves safety.

EXISTING `Scan.h::fused_find_and_collect<N,A>` returns `FusedScanResult<N>` and takes `SlotWindow window`, `my_rank`,
router, precomputed `gen_shift`, then capture/scaling arguments. `Engine.h` derives the plan/window once from that
shift. PROPOSED: extract the following range helper in `Scan.h`; preserve these inputs and result type, prepare lazy
data once, and merge private results. Do not reintroduce rank-count-shaped query vectors. The range helper must never
call a lazy accessor itself:

```cpp
template <size_t N, Algebra A>
auto fused_find_and_collect_range(const MPOperator<N> &op,
    const Monomial<N> &gen, const CutoffEvaluator<N> &cutoff_eval,
    const CutoffContext &cut_st, const VecD &coeffs,
    std::optional<size_t> only_rotate_len_k, mpi::SlotWindow window, size_t my_rank,
    const routing::Router &router, size_t gen_shift, const InvertedIndex<N> &index,
    const uint64_t *row_parity, size_t wlo, size_t whi,
    bool capture_values, double *fused_scale_coeffs,
    double fused_scale_cos) -> FusedScanResult<N>;
```

Recomputing the small immutable generator context per range is acceptable initially; do not introduce a persistent
prepared-state cache. Tail masking uses the full index's final word, not each range's final word.

- [ ] Before changing traversal, add a serial/threaded result comparison helper checking each owner's leader/follower
  query words, source IDs, captured values, cosine indices, coefficients and both self stages (logical position spans,
  offsets, counts and phases). Include nonzero window bases, linear zero/nonzero shifts and splitmix dense windows.
  Cases must exercise dense and sparse pivots, odd Majorana correction even when selected fold columns are empty, Pauli
  J(G), no length cap, cap 0/positive, lower-atol equality, upper-atol rescue, cos=0 fallback, capture-values on/off,
  and no emitted queries.
- [ ] Extend the EXISTING stateless opaque C++ lambda fixture in `majorana_cutoff_tests.cpp:171–196`: evaluator
  classification must be non-parallel-safe and the kernel's test-only work-range observations must show its serial
  path. Do not invent a stateful Python callback or a callback-heavy performance cell for a nonexistent constructor
  surface. Preserve actual basis-change results; this first version leaves that traversal serial. Audit real cosine
  callbacks/parity-cache preparation separately. Serial fallback is not an exemption from approved campaign gates.
- [ ] On caller: validate arguments, prepare algebra/generator columns, obtain router once, initialize lazy
  index/parity, snapshot row/word count, check skip/tail conditions. Partition words into at most
  `min(options.threads,ceil(words/1024))` contiguous nonempty ranges. Range IDs increase with source word index and do
  not depend on actual worker count.
- [ ] Move `EvenParityNzWord` scratch, `CosineWordBuilder`, counters and the `emit` lambda **inside** each range
  invocation. Reuse `even_parity_scan_pass1(wlo,whi)` then immediately emit its results. Workers read/update only their
  assigned source rows. They may not resolve target coefficients or grow stores.
- [ ] If cutoff is not parallel-safe or work is small, call the original serial body without chunk metadata. Otherwise
  `for_blocks(range_count,options,...)` fills one `FusedScanResult<N>` per range. No shared push_back/critical section
  and no per-bitmap-block × rank vector table.
- [ ] Join, calculate checked total lengths, reserve once and merge reachable slots in ascending range ID using
  `mpi::WindowIndex{k}` or `.at_slot(flat_slot)`; never use a flat slot as a window index. All pieces share the exact
  `window` (base and count). Move the first useful buffer where safe and release drained buffers immediately.
  `leader_queries/src/val` and follower equivalents preserve each stream's source-order subsequence. Wire records have
  variable boundaries, not a fixed stride; use `QueryWire<N>` and preserve Plain/Fused semantics. Self wire stays empty.
- [ ] Merge `leader_self` and `follower_self` separately, aligned with the self slot's src/val streams. EXISTING
  `SelfQueryStage<N>` in `PartnerMerge.h:81–140` uses logical `size()`/`positions()`, not backing vector sizes. Use its
  existing `push(span,phase)` in source order initially: for each q below `piece.size()`, pass the span at
  `piece.pos_off[q]` of length `piece.k_of[q]` and `piece.phase_of[q]`. This appends to the logical position end and
  rebases offsets correctly without a new helper or accessing private counters. Check total queries/positions and
  reserve bounds first. Do not copy capacity tails. Assert self stage length equals self src length; values match only
  when capture is enabled. Keep cosine word seam handling and ascending disjoint cos_blocks in Engine.

```cpp
// PROPOSED merge loop shape; append concrete streams as specified above.
for (size_t k = 0; k < window.count; ++k) {
    const mpi::WindowIndex wi{k};
    const auto flat_slot = window.slot(wi);
    for (size_t range = 0; range < pieces.size(); ++range) {
        // pieces[range].leader_queries[wi] belongs to flat_slot, not flat slot k.
        // Append leader src/val in the same range order; handle followers separately.
        // If flat_slot==my_rank, merge self positions, never synthesize self wire.
    }
}
```

  Metadata bound is O(T*window.count), payload O(total emitted records), never per-block×R or per-thread full-store
  dense keys. At S=1, linear zero shift is all self; nonzero shift has exactly one reachable remote slot.
- [ ] Keep leader/follower resolution unchanged in this task. Compare graph data and row IDs at fixed rank count after
  full construction, not just energy.
- [ ] Measure simultaneous private+merged query capacities and worker TLS. If merge duplication fails peak parity, use a
  two-pass exact-fill variant **only for pure cutoffs**: count with no scaling/capture mutation, caller prefix/reserve,
  then fill and scale once. Count and fill must use identical pre-scale coefficients and predicates; do not scale in
  both passes. If this CPU/memory tradeoff still fails, stop. Do not replace compact storage to fix output buffering.

**Gate:** RED: new range helper absent; Task 3 options already exist. Exercise the new multi-range merge path in ordered
window/self regressions, not only the old serial path. GREEN: exact serial/threaded query words, self positions, IDs and
graph fields at fixed geometry. Ordered query/row/graph equivalence, no callback-thread contract change, no lazy-cache
races. Construction bitmap-traversal time and peak scratch are recorded separately from resolution.

### Task 6: Parallel frozen lookup while retaining serial insertion

**Files:**
- Modify: `cpp/monoprop/detail/evolution/layer_build/{Resolve,Engine,QueryWire}.h`.
- Extend: `cpp/tests/{operator_index_tests,sparse_resolve_tests,sparse_query_tests,mpi_fresh_insert_equivalence}.cpp`.

**PROPOSED interfaces:**

```cpp
// PROPOSED Resolve.h helper; per-query spans have equal lengths; store frozen until return.
template <size_t N>
auto probe_frozen_positions(const OperatorIndex<N> &store,
    std::span<const typename OperatorIndex<N>::PosT> pos_flat,
    std::span<const size_t> pos_off, std::span<const uint32_t> k_of,
    std::span<size_t> out, std::span<uint32_t> hash_out,
    parallel::Options options = {}) -> void;
```

EXISTING `OperatorIndex::find_batch_positions(pos_flat,pos_off,k_of,out,hash_out={}) const -> void`
(`OperatorIndex.h:282–288`) consumes position spans; use 256-query blocks, with absolute pos_off values into the shared
pos_flat and sliced query/output/hash spans. Retain batched lookup and cached hashes, not scalar find or dense keys.
Change incoming `miss_g` from `vector<TermIndex>` to `vector<size_t>` because it contains query ordinals, not term IDs.
Count that memory change explicitly.

- [ ] Add this preservation test to `sparse_resolve_tests.cpp`, using its EXISTING `make_op<N>` and `positions_of<N>`
  helpers/includes. Task 3 already added the trailing options argument, so this test may pass before Task 6. Its purpose
  is exact predicted/published IDs and nonzero-window addressing, not a fabricated RED. Use the missing
  `probe_frozen_positions` helper and malformed-boundary tests as this task's RED targets. Public Majorana indices are
  reversed when encoded; `QueryWire::push` needs raw ascending bit positions obtained with `positions_of`:

```cpp
BOOST_AUTO_TEST_CASE(openmp_incoming_miss_order) {
    using namespace monoprop;
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    const auto c = indices_to_bitset<8>({4, 5});
    for (const auto threads : {1, 3}) {
        auto op = make_op<8>(std::vector<Monomial<8>>{a});
        const mpi::SlotWindow window{.base = 4, .count = 3};
        mpi::WindowVec<VecZ> incoming(window);
        detail::QueryWire<8>::push(incoming.at_slot(4), positions_of<8>(b), +1);
        detail::QueryWire<8>::push(incoming.at_slot(6), positions_of<8>(a), -1);
        detail::QueryWire<8>::push(incoming.at_slot(6), positions_of<8>(c), +1);
        const auto pr = detail::probe_incoming_queries<8>(
            incoming, op, detail::QueryForm::Plain, {.threads = threads});
        BOOST_TEST(pr.sender_slot(0) == 4U);
        BOOST_TEST(pr.sender_slot(1) == 6U);
        const std::vector<size_t> offsets{0, 1, 1, 3}, ids{1, 0, 2}, misses{0, 2};
        BOOST_TEST(pr.base == 1U);
        BOOST_CHECK_EQUAL_COLLECTIONS(pr.goff.begin(), pr.goff.end(), offsets.begin(), offsets.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(pr.idx_of.begin(), pr.idx_of.end(), ids.begin(), ids.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(pr.miss_g.begin(), pr.miss_g.end(), misses.begin(), misses.end());
        detail::insert_incoming_misses(op, pr);
        BOOST_TEST(op.size() == 3U);
        BOOST_REQUIRE(op.store->find(a).has_value());
        BOOST_REQUIRE(op.store->find(b).has_value());
        BOOST_REQUIRE(op.store->find(c).has_value());
        BOOST_TEST(op.store->find(a).value() == 0U);
        BOOST_TEST(op.store->find(b).value() == 1U);
        BOOST_TEST(op.store->find(c).value() == 2U);
    }
}
```

Repeat at window base zero, empty/all-hit/all-miss inputs, >256 queries and short tails, fused-value codec, overflow
rows, and a prebuilt inverted index.
- [ ] Prepare safe decode on caller BEFORE workers. EXISTING `QueryWire::count_queries`/`read_query` assume valid input;
  do not use an unchecked walk as a safety check. In `Resolve.h::probe_incoming_queries`, walk each sender by window
  index, requiring off<buffer.size before `header_at`. Validate phase in [-1,1], k<=kMaxPositions and gw<=kPosBits.
  Require `h.bits == QueryWire<N>::header_bits_for(h.k)` before calculating payload size; reject noncanonical escaped
  counts below `kKEscape`. Otherwise `words_of_header` can undercount the actual header consumed by `read_positions`.
  Compute payload words using the validated header, add the Fused value word with checked arithmetic, and require the
  entire record to fit `buffer.size()-off`. Reject malformed/truncated streams; require final off==buffer.size. Add a
  pre-worker rejection fixture for `QueryWire<128>`: `VecZ{size_t{0x8037e}}`, one Plain word with phase +1, escaped k=6,
  gw=8 and zero payload bits. Its actual decode needs 68 bits but a canonical k=6 header implies 59; it must be rejected
  before any payload read. Keep valid escaped-count and Fused coverage. Checked-prefix query and decoded-position
  totals, plus goff, off_of, pos_off, k_of and sender_wi on caller. Check uint32 sender/count narrowing and vector
  max_size. Allocate all output arrays once; no worker vector growth.
- [ ] Decode disjoint records into their prepared `pos_flat` spans via `read_query(buf,form,off,span)`, preserving
  phase_of and Fused value offsets. Check returned next matches the prepared boundary. Extend EXISTING
  `QueryWire::read_positions` to check cumulative decoded size_t positions before narrowing to PosT: each must be
  strictly ascending and below 2*N. Checking only already-narrowed bytes could miss overflow. Headers already bound
  payload reads; invalid positions throw inside the worker catch before hashing/publication. No dense key
  reconstruction. Join and surface exceptions through Task 3's failure boundary, then `probe_frozen_positions`, join,
  normalize not-found and assign misses serially in ascending sender-window/query order. Retain hash_of for every query.
  Check ID/count limits before exposing predicted IDs. RED malformed-boundary/overflow tests must reject before worker
  decode; GREEN includes empty, multiword, escaped-popcount, Fused, overflow-row and partial-block cases without dense
  reconstruction.
- [ ] Keep `insert_incoming_misses` publication serial: grow → `set_positions(base+j,pr.positions_at(g))` →
  `bulk_insert_hashed(...pr.hash_of[g])` → `reindex_after_growth`. Preserve phase-3 scatter before insertion because it
  reads pre-insert coefficients. All ordinary set/overflow/hash/inverted/state-extension mutation stays single-writer.
  Existing paired-only `IncomingProbe::mono_at` for Schrödinger state scoring is a bounded per-record exception already
  in ContractSink; preserve it, never turn it into a dense batch/store. Keep that scoring path serial initially.
- [ ] Parallelize incoming scatter only after `responses` sizing and sink `prepare`. PROPOSED capability:
  `static constexpr bool parallel_resolve = true` in `GraphSink`/`ContractSink`; it certifies only independent
  `on_resolved` writes after prepare, not other sink methods. Missing/false capability means serial for generic sinks;
  use nested `if constexpr (requires { Sink::parallel_resolve; })` then test its value, avoiding absent-member lookup.
  For `ContractSink`, also require `!sink.schrodinger` initially; scoring remains serial. Keep shared
  `ContractSink::on_response_block` push_back serial. Matched leader marks are unique uint16 elements and bounded by
  pre-gate `combined_size`; join before followers read them.
- [ ] For self lookup, EXISTING `Engine.h::resolve_range_` uses 64-query batches of gathered offsets/counts into
  `SelfQueryStage` and returns cached hashes. PROPOSED: use bounded windows up to 4096 queries for parallel work; serial
  stable follower filtering → gather absolute position offsets/counts → parallel position probe → serial hit/deferred
  emission in original order. Leaders skip filtering. No self wire encode/decode, no full-store keys, and no store
  insertion between windows. Preserve deferred positions AND hashes, reuse bounded scratch and retain recording-sink
  compatibility.
- [ ] Add assertions/tests for mixed self hit/miss, remote hit/miss, and leader/follower ordering. Do not coalesce all
  insertion into a final phase. Keep remote-leader then remote-follower insertion and both-pass deferred self
  publication.

**Gate:** R/M lookup and fresh-insert tests pass with exact fixed-geometry IDs. Report
probe/prefix/publication/inverted-index times separately. If serial hash publication dominates, report an Amdahl limit
instead of inventing a concurrent table.

### Task 7: Harden rank-wide capacity and optionally parallelize append-only row fill

**Files:**
- Modify: `cpp/monoprop/detail/operator/{OperatorIndex,MPOperator}.h`.
- Modify only if row-fill optimization is justified: `cpp/monoprop/detail/evolution/layer_build/{Resolve,Engine}.h`.
- Extend:
  `cpp/tests/{operator_index_tests,inverted_index_tests,evolution_detail_tests,exchange_layout_precondition_tests}.cpp`,
  `cpp/tests/{graph_encoding_tests,ctor_validation_tests,majorana_cutoff_tests}.cpp`.

EXISTING `grow_rows_geometric` already checks append counts (`OperatorIndex.h:111–122,544–563`);
`operator_index_tests.cpp:68–99` checks ceiling/wrap refusal without huge allocation. Extend, do not replace, these
tests.

**PROPOSED required interface:** static `OperatorIndex::checked_append_end(size_t base,size_t count) -> size_t`
validates row-count/index limits without allocating. Keep capacity arithmetic distinct from valid row IDs.

**PROPOSED optional measured interface:**

```cpp
// PROPOSED OperatorIndex members, only if measured row fill justifies them.
[[nodiscard]] auto inline_width() const noexcept -> size_t;
auto set_new_inline_positions(size_t i, std::span<const PosT> positions) noexcept -> void;
auto reserve_index_for_size(size_t final_size) -> void;
```

- [ ] Add boundary tests before mutation: `(ceiling-1,1)` succeeds; `(ceiling-1,2)`, `(ceiling,1)` and arithmetic
  overflow fail; `(ceiling,0)` succeeds as a count boundary. No billion-row allocations. Extend the existing
  fixed-uint32 guard tests. Operator row IDs exclude the sentinel ceiling while some graph count fields can equal their
  representable max: keep those domains separate.
- [ ] Implement checked append addition as `base>ceiling || count>ceiling-base` before evaluating `base+count`. Check
  row-storage multiplication, vector max_size, table load-factor sizing and next-power-of-two rounding before
  allocations. Geometric spare capacity cannot wrap. Update diagnostics from per-partition to per-rank store at cutover.
- [ ] Use these checks before incoming predicted IDs and deferred insertion. Audit existing MPI checked counts without
  changing transport or count width. The owner accepts that consolidating old partitions reduces aggregate per-rank
  addressability. A required campaign cell above the new ceiling is still a **capacity acceptance blocker**, not
  permission to widen only the candidate, change ranks or introduce hidden shards. Do not infer a universal RAM limit
  from the illustrative 512-GiB/eight-domain machine. C8a has 192 GiB advertised RAM; per-rank budgets still
  require the observed allocation and actual memory demands, not a hard-coded equal split.
- [ ] Profile row fill. If it is not significant, leave row initialization and publication serial and mark the optional
  steps below skipped with evidence; the required capacity checks still ship.
- [ ] If justified, add tests proving simultaneous new inline writes do not modify old overflow entries.
  `set_new_inline_positions` requires already-grown new rows, each with one owner and position count<=inline width; it
  must contain no overflow lookup/erase, allocation, resize, table operation, or exception. Keep ordinary `set` serial.
- [ ] Do not introduce a dense KeyAt/Monomial append layer. Operate on EXISTING `IncomingProbe::positions_at(g)` and
  `deferred_pos_flat_` spans, preserving cached `hash_of[g]` / `deferred_self_misses[j].hash`. Check ranges before
  growth; input positions must outlive growth and never alias invalidated store views.
- [ ] Apply this order directly at `Resolve.h::insert_incoming_misses` and `Engine.h::insert_deferred_self_misses`:
  serial checked reserve/grow → parallel inline-position fill → join → serial `set_positions` for overflow rows → serial
  deferred `sink.emit_deferred` in k order (no incoming metadata emission) → serial `bulk_insert_hashed` using retained
  hashes → serial `reindex_after_growth`. No readers before publication; no shared overflow lookup/erase from workers.
  There is no production assign_row callback here to remove. Preserve the independent dense reference in
  `sparse_resolve_tests.cpp`; do not change unrelated `insert_absent_terms` callback contracts.
- [ ] Preserve streaming initialization during capacity changes: `for_each_paired_monomial` enumeration order,
  `paired_op_size` saturation guard, local reserve estimate and pre-worker rejection. At S=1 reserve/check against real
  rank share, not T. Keep `majorana_cutoff_tests.cpp` enumeration/count tests, `ctor_validation_tests.cpp` oversized
  constructor rejection and initial-basis partition-invariance coverage when migrating it to thread budgets.
- [ ] Compare all rows, keys, hash lookup, inverted bits/parity, sparse/dense state extension and `scaled_count` against
  serial append after every growth. Include boundaries 0/1/15/16/17/63/64/65/127/128/129 and appends starting at
  unaligned rows.

**Gate:** Capacity guards pass R/M. Optional fill must improve measured bottleneck without peak regression. Hash
insertion, overflow mutation, packed graph phases and inverted append remain serial.

### Task 8: Parallel replay endpoints and derivative pairs

**Files:**
- Modify: `cpp/monoprop/Evolution.cpp`, `cpp/include/monoprop/Evolution.h` only for signature consistency.
- Extend: `cpp/tests/{combined_recompute_equivalence,evolution_detail_tests}.cpp`,
  `cpp/tests/{mpi_distributed_layer_equivalence,flat_exchange_tests}.cpp`, `tests/test_deep_circuit_gradient.py`.

**PROPOSED interfaces:** Use Task 3 options in snapshot, pack, remote apply and self-pair helpers. If range flattening
needs a descriptor, define local `EndpointRange { size_t slot; size_t begin; size_t end; }` in `Evolution.cpp`; this is
proposed transient metadata, not a new graph field. Flatten **ranges within active slots**; merely parallelizing ranks
underutilizes sparse linear-routing peers.

- [ ] Write serial/threaded forward and derivative comparisons with multiple self pairs, nonadjacent row IDs,
  remote+self slots, empty peers, zero-angle, deep non-singular circuits, vanishing-cosine and no-record paths. Retain
  exact recorded restoration, including duplicate indices, indices outside the cosine set and cos rounding to one.
  Compare energy/gradient to current finite-difference/exact oracles. Assert even self endpoint count and unique output
  ownership.
- [ ] Record baseline PASS for existing mathematical/record fixtures where appropriate. Add test-build-only worker-ID
  observations inside the endpoint work-range bodies, one preallocated slot per range with one writer. Their multiworker
  requirement is RED while those bodies are serial; check after joining on an adequate allocation. Keep observations out
  of production APIs and distinguish runtime-limited skips from demonstrated participation.
- [ ] Pre-size snapshots/payloads on caller. Parallel gather and pack disjoint ranges, join before posting MPI. Preserve
  forward ordering: snapshot → pack → post → cosine → wait → remote apply → self apply. Do not reorder for overlap in
  the same patch.
- [ ] For self derivative workshare over `k in [0,pairs)`, keep endpoint `k+pairs` in the same iteration. Read both
  values required by the original formula before either overwrite. Never sort entries or independently workshare the two
  halves.
- [ ] Retain serial scalar accumulations in this task; if a loop both mutates arrays and produces a scalar, isolate its
  independent update portion only where the original formula permits it. Task 9 defines deterministic partials for the
  full loop.
- [ ] Keep layer loops and gradient reverse order unchanged. No worker may call `mpi::rank`, `router_for`, exchange
  begin/wait, or allreduce. Query rank/layout on caller.
- [ ] Cover worker exceptions after exchange posting using Task 3 active-ticket failure case. Buffers cannot resize or
  die while requests are outstanding. Preserve EXISTING replay `wire_bits` selection: plain MPI uses dense
  PeerPlan/nonzero pairwise legs, HybridComm derives its plan from all local partition rows. Do not introduce a new
  generator-derived replay plan. Run `flat_exchange_tests.cpp` dense, pairwise, dense-layout-on-pairwise and
  empty-layout cases after every transport edit.

- [ ] Keep `LayerAngle`, `CosRecordView` and `LayerCosIndices` intact. Serial forward record append, predivide/restore
  and snapshot-before-restore ordering are mandatory even if record ordinals differ: index duplication is legal.
  Preserve `A = cos_acc(...)*sec` and `-g*(sin*(A-ep.cos_terms)-ep.sin_terms)` in pre-layer units from Evolution.cpp;
  threading must not reintroduce division-only inversion or move cancellation after multiplication by sec.

**Gate:** RED endpoint participation checks; preservation tests need not fail first. GREEN includes the entire
deep-circuit gradient suite at budgets 1/2/3/4. Full replay and immediate-contraction equivalence across both
pictures/bases, gradients and MPI; useful active-slot worksharing even with few peers. No full-vector snapshot beyond
current algorithmic requirements.

### Task 9: Deterministic reductions without silently breaking serial dot contracts

**Files:**
- Modify: `cpp/include/monoprop/MPFunctions.h`, `cpp/monoprop/MPFunctions.cpp`, `cpp/monoprop/Evolution.cpp`,
  `cpp/monoprop/detail/evolution/CosineRecompute.h`.
- Extend: `cpp/tests/{mpfunctions,combined_recompute_equivalence,openmp_kernel_tests}.cpp`.

**PROPOSED interfaces:** Keep `inner_product(v,w)` and `EvalState::dot(op)` unchanged. Add explicitly named
`inner_product_threaded(v,w,parallel::Options)` and `EvalState::dot_threaded(op,parallel::Options) const`. They have
documented fixed-block association, not historical left-to-right association. `ev`/`ev_and_grad` use the new helpers
through `EvalRequest.parallel` under the approved numerical policy; implementation remains separately gated.

- [ ] Add cancellation-heavy finite vectors, zero/signed-zero, sparse gapped rows, very short arrays, >4096 rows,
  non-finite inputs, and repeated parameter indices. Check old serial dot tests unchanged; check new helper results
  bit-identical across thread budgets at fixed geometry and within existing oracle tolerances.
- [ ] Define fixed logical dot blocks of **4096 row positions**, independent of worker count or dense/sparse
  representation. Dense blocks sum in increasing row order; sparse blocks use lower_bound to locate entries in that same
  row range and sum in ascending row order. Fold partial doubles on caller in increasing block ID. Use the same
  partition even at budget one. Document non-finite/signed-zero limitations; do not strengthen sparse/dense guarantees
  beyond what tests and current finite-input contract support.
- [ ] `EvalState::sparse` currently does not enforce all documented ascending/unique conditions. Do not introduce unsafe
  parallel scatter. Keep `scatter_into` serial; for the new dot helper detect non-strictly-increasing rows and fall back
  to existing `dot` behavior. This preserves historical duplicate/unsorted-input behavior without adding a new public
  rejection in this refactor.
- [ ] For cosine accumulators use fixed 1024-word logical bitmap blocks; each block computes its scalar partial and
  performs its disjoint existing coefficient updates. Caller folds partials in block order. Stored masks use logical
  word ranges, not mask-vector ordinal ranges, when comparing to lazy recomputation. Leave old standalone serial
  accumulator oracles in tests; compare according to existing tolerances except binary-exact fixtures.
- [ ] For endpoint derivative contributions keep two partial doubles per fixed 1024-record block. A block owns full self
  pairs. Sum both fields in block order. Do not use OpenMP `reduction`, atomics, critical floating accumulation, or O(n)
  product materialization to recreate serial sum order.
- [ ] Keep `indices_above`, parameter mapping and export loops serial initially. If a later profile shows
  `indices_above` dominant, use count → caller offsets → disjoint exact fill, with logical row order, strict `>`,
  negative threshold admitting zeros, and NaN behavior unchanged. No new global threshold policy.
- [ ] Run `tests/test_deep_circuit_gradient.py` and `combined_recompute_equivalence.cpp` across budgets 1/2/3/4,
  including deep non-singular amplification, vanishing cosine, no-record and duplicate-record restoration cases.
- [ ] Run retained/direct evaluation, paring, partial contraction, repeated parameter mapping, both pictures/bases,
  fixed-rank deterministic repeats, and near-cutoff retained-key comparisons. Cross-rank-count sums need not be bitwise
  equal; don't widen tolerances to pass them.

**Gate:** Existing serial API guarantees remain intact; new numerical association is documented and deterministic by
team size. Global retained term keys must agree, including near cutoffs and stored zeros; matching energies/gradients
alone is insufficient. A key-set change blocks the gate. Stop and report, never widen tolerances or change truncation.

### Task 10: Cut over configuration and API semantics to one store

**Files:**
- Modify: `cpp/include/monoprop/MonomialPropagator.h`, `cpp/monoprop/detail/monomial_propagator/MonomialPropagator.inl`,
  `cpp/monoprop/detail/EnvConfig.h`, `cpp/monoprop/detail/parallel/{ThreadBudget.h,ThreadBudget.cpp}`.
- Modify: `src/monoprop/bindings/binder.h`, `src/monoprop/monomial_propagator.py` docstrings.
- Create/extend: `cpp/tests/openmp_equivalence_tests.cpp`, `tests/test_openmp_config.py`,
  `cpp/tests/{env_config_tests,routing_tests,mpi_distributed_layer_equivalence}.cpp`,
  `cpp/tests/{mpi_fresh_insert_equivalence,unit_tests,ctor_validation_tests,hybrid_comm_tests}.cpp`,
  `cpp/tests/link_export_probe/link_export_probe.cpp`, `tests/{test_monoprop_smoke,test_parameter_validation}.py`.
- Migrate now: `cpp/tests/{partition_equivalence_tests,partition_group_clone_tests}.cpp`; preserve useful coverage in
  the OpenMP equivalence suite and `cpp/tests/{simulator_copy_tests,update_initial_operator}.cpp` before retiring
  obsolete facade/factory assertions. Update `cpp/tests/README.md`; do not defer broken constructor users to Task 12.
- Modify: `AGENTS.md`, `README.md`, `docs/content/docs/features/parallelism.mdx`,
  `docs/content/docs/openmp-migration.mdx` (new).

**PROPOSED interfaces/policy:** Implement immediate removal, as approved in the design interview. No aliases,
deprecation window, public thread-count constructor argument or permanent backend selector.

- [ ] In fresh subprocesses test the final environment-only policy:

| Input | Result |
| --- | --- |
| `monoprop_NUM_THREADS=N`, valid decimal 1..INT_MAX | capture N exactly; no hardware clamp |
| valid override and a different OpenMP default | monoprop_NUM_THREADS wins |
| present empty/zero/negative/malformed/overflowing override | actionable PropagatorConfigError; no fallback |
| override unset, any MPI rank count | capture omp_get_max_threads() at construction |
| both thread environment variables unset | capture runtime default; no hardware-suitability promise |
| stale `monoprop_PARTITIONS` in application environment | no production reader, alias or effect |
| C++ partition/factory constructor arguments | removed; old source must migrate |
| low-level Python `partitions` or `num_threads` keyword | rejected as an unknown keyword |

Only absent monoprop_NUM_THREADS selects the OpenMP default; zero is not a public sentinel. Use Task 3's strict parser
rather than the old cached permissive one. Remove Settings::num_threads and its legacy accessor/parser once their
partition consumers are gone; remove resolve_partition_count_ declaration/definition in this task too, so its old parser
references do not survive as ill-formed dead template code. Retain independent routing settings/cache. Scan readers for
remaining partition-only knobs and remove them. At the inspected source the only partition-specific runtime variable is
monoprop_PARTITIONS; monoprop_ROUTING and monoprop_ROUTE_SEED remain MPI configuration. Do not add a removed-variable
reader for warning or rejection. Candidate benchmark preflight may reject a stale launch environment independently.

The supported application fixes ranks/threads at launch. Capture once per constructor, never reload during operations or
add a setter. Copies/functionals preserve their budget. OpenMP limits or small-work paths may use fewer actual workers;
this is not a budget change. There is no topology-based fallback or special multi-rank default of one thread.
- [ ] Test defaults in fresh processes with monoprop_NUM_THREADS unset: OMP_NUM_THREADS=1/3, list-valued 2,3 (use the
  runtime's outer-level default), and unset OMP_NUM_THREADS (compare against omp_get_max_threads, not assumed cores).
  Repeat at one and multiple ranks. Test override precedence, all invalid-input classes from Task 3 and limited teams.
  Invalid settings on one rank must fail coherently rather than strand peers. Add a bounded white-box test that changes
  the source setting after construction solely to prove existing objects/copies/closures do not reread it; this is not
  a supported hot-configuration API. Full-budget matrix tests use separate launches, not mid-calculation changes.
  Remove prototype partitions=1 arguments; public tests use environment configuration, internal kernels use Options.
- [ ] Apply the approved splitmix physical ownership change: `floor((hash % (R*S))/S)` becomes `hash % R`.
  Linear rank mapping remains unchanged; S=1 removes only the partition hash. Preserve boolean mode/seed,
  `linear_requested()`, `linear_bits_for(R)`, `make_router(R,1)`, generator `rank_shift` and `dest_from_shift` peer
  selection. No virtual shards. Exports/load/reduction order can change under splitmix; validate global term/value maps
  and resulting per-rank peaks at equal R/T/routing settings. Candidate ownership never depends on T.
- [ ] Extend `routing_tests.cpp`, `mpi_distributed_layer_equivalence.cpp` and `mpi_fresh_insert_equivalence.cpp`:
  default linear success at R=1/2/4; explicit splitmix success at R=1/2/3/4. Record `monoprop_ROUTING`,
  `monoprop_ROUTE_SEED` and resolved `linear_bits()` as derived metadata, not a requested bit count. Separately assert
  `Router::for_modes<8>(3,1,true)` throws `UnroutableGeometry` and splitmix construction succeeds. Keep fresh-process
  default tests distinct from splitmix matrix runs; never expect an automatic fallback. Assert unique global ownership,
  generator-peer agreement, zero-payload participation, fixed-candidate ordering across budgets and global numerical
  equivalence. Do not require old/new rank-local maps equal for splitmix.
- [ ] Migrate every constructor/helper/binding test using removed partition arguments in this same task. Use a source
  search over cpp/tests, tests and src; include the installed export probe, constructor-validation cases and Python
  parameter-validation/smoke tests. Old calls failing to compile or obsolete keywords still being accepted are
  meaningful RED conditions. Move useful numerical/copy/update/empty-rank coverage before retiring facade/factory-only
  cases; preserve ordinary-MPI cases currently housed in hybrid suites. Record each retired case's replacement in the
  Task 12 ledger. Remove unit_tests.cpp's monoprop_PARTITIONS override and use the serial launch budget instead.
  R/M must compile and pass here, not become green only after Task 12 deletes test files.
- [ ] Each independent propagator constructor creates one rank-local MPOperator and primary graph; comm_ represents
  only real MPI ranks. Remove facade branches, partitions/child_factory arguments, PartitionChildFactory and
  partition-specific error types such as MultiPartitionUnsupported. No diagnostic compatibility shims. Leave now-unused
  transport implementation files until Task 12. Independently copied propagators keep their own stores; retained
  coefficient/state snapshots, shared immutable graph cores and pared graphs remain legitimate existing objects.
- [ ] Preserve rank-local size/bytes accounting (do not allreduce twice), core term handling, copy independence,
  immutable graph-core sharing, virtual destructor, virtual `clone_`, overridden virtual `update_initial_operator`, and
  protected nonvirtual `apply_initial_operator_`. Do not add override to or make that helper virtual. Copy budget and
  ordinary MPI communicator. Raw accessors expose that instance's rank-local store/primary graph at every budget.
  Preserve Task 3 invalidation checks: invalid sources cannot be cloned; an already independent copy stays usable.
- [ ] Preserve `MonomialPropagator.inl:126–151,197–227` streamed paired-basis construction and local reservation:
  cardinality rejection remains before worker startup and uses real ranks at S=1, never T. Do not materialize
  `generate_paired_op` globally per rank. Migrate `partition_equivalence_tests.cpp` initial-basis invariance alongside
  enumeration/cardinality tests from `majorana_cutoff_tests.cpp` and `ctor_validation_tests.cpp`.
- [ ] Migrate numerical tests from partition counts to launch-time budgets 1/2/3/4 in fresh processes; lower-level
  kernel tests may pass explicit internal Options. Distinguish stored requests from actual team sizes under OpenMP
  limits. Library budget selection never clamps to affinity. Run parallel gates on a user/CI
  allocation supporting the requested teams and report limited-team skips honestly; pure budget tests need no extra
  CPUs. The default white-box harness selects a serial library budget (not CPU pinning); dedicated launches override
  it before construction. Test OpenMP-default selection in fresh processes with that library override absent.
- [ ] Test public Majorana/Pauli mathematical constructors unchanged. Remove binding-only partitions and do not add
  num_threads or any other public thread-count argument. Test copy, operator update, retained callbacks after growth,
  cutoff/basis-change update, paring, Schrödinger state extension and independent repeated evaluation.
- [ ] Regenerate bindings via the configured generator. Preserve 32-mode template dispatch. Document
  low-level/source/ABI breaks and unspecified coefficient-order changes; compare decoded term/value mappings rather than
  sorted coefficients alone.

**Gate:** R/M API/configuration tests pass; one independent propagator across R ranks owns R packed stores, regardless
of T. Copies and retained snapshots/graphs obey the qualified ownership rule. Removed configuration/constructor surfaces
stay absent; invalid objects cannot be reused. Record the approved immediate compatibility breaks in the release ledger.

### Task 11: Measure parity and apply only bounded optimizations

**Files:** Task 1 benchmark files and, only if profiling justifies, the kernels named below. Do not rename `bench_*`
functions or move benchmark files.

- [ ] Add metadata for `has_mpi`, requested/effective team size, OpenMP runtime,
  OMP_NUM_THREADS/PLACES/PROC_BIND/DYNAMIC/THREAD_LIMIT, any library override, routing mode/effective linear bits/seed,
  launcher command, rank/worker CPU masks, NUMA IDs, revision/dirty overlay and extension path/hash. Collect placement
  in separate diagnostic runs of the same binary/configuration with OpenMP runtime diagnostics
  (`OMP_DISPLAY_AFFINITY=TRUE`, `OMP_DISPLAY_ENV=VERBOSE`), launcher binding reports, and external OS/scheduler tools
  (on Linux, `lscpu -e=CPU,CORE,SOCKET,NODE` and `/proc/<pid>/task/*/status`). Observe actual active-worker masks/team
  size, not merely omp_get_max_threads; if a runtime cannot supply that evidence, use external sampling or test-only
  observation rather than adding a production topology API. Preserve old single-CPU-thread fields as historical data:
  OpenMP core places may contain both SMT siblings, but at most one worker may occupy each physical core in the
  recommended/acceptance configuration. A wider affinity mask is not evidence of extra worker capacity. Record distinct
  physical-core ownership from verified sibling topology across active ranks/workers, not only disjoint logical CPU IDs.
  No required hwloc dependency for the library or measurement helper.
- [ ] The benchmark operator, not monoprop construction, verifies that rank masks and actual worker places stay in one
  NUMA domain and within the reserved allocation, with no unintended overlap between ranks. Advise at least one MPI
  process per allocated NUMA domain, with disjoint CPU subsets when several processes share a domain. Use actual OS/site
  topology, not a constant four domains/socket; partial-node runs cover only their allocated domains.
  Do not require one logical CPU per worker mask. Record idle/small-kernel serial fallback versus representative
  large-kernel actual team size separately. Unverifiable placement blocks benchmark evidence, not library construction.
- [ ] Compare old partitions vs final new threads at identical R,T and routing settings, using Task 1's frozen
  size-band-specific inventory. Small/medium profiles cover the full sweep; large profiles only the two full-node
  decompositions. Do not add/remove required cells or reinterpret profile sizes after seeing candidate performance.
  Freeze the production routing mode in that matrix; additionally require the random reference profile under explicit
  splitmix at R=3 and R=4, plus default linear at R=4 so ownership-dependent imbalance is measured, not only
  unit-tested. Run five alternating **fresh-process** observations per arm/cell; external repetition only,
  `--bench-rounds=1`. Include first-call/cold cache and separately measured repeated warm calls without overlapping
  object construction.

Example candidate Open MPI invocation after selecting R (= P), K (ranks/domain), T, LABEL and absolute RESULTS.
Use K=1 or K=2 for the agreed decomposition profiles. Choose a compatible placement for routing-specific R=3/R=4 cells;
do not force ppr:1:numa onto an allocation with too few NUMA domains. All placements require observed disjoint cores.
This demonstrates one raw pytest run, not a complete parity observation. The Task 1 driver uses the same launcher
contract, unique per-sample raw directory, frozen campaign and paired construction/validation artifacts for acceptance.
Unset removed partition configuration and use the supported library override. The OpenMP fallback is tested separately:

```bash
unset monoprop_PARTITIONS
# This example chooses splitmix explicitly, including if R=3; use linear only in a separate R=1/2/4 cell.
export monoprop_ROUTING=splitmix
mkdir -p "$RESULTS"
monoprop_BENCH_LABEL="$LABEL" monoprop_BENCH_RESULTS="$RESULTS" \
monoprop_NUM_THREADS="$T" OMP_NUM_THREADS="$T" OMP_DYNAMIC=FALSE \
OMP_PLACES=cores OMP_PROC_BIND=close \
uv run --no-sync mpiexec -n "$R" --map-by "ppr:${K}:numa:PE=${T}" \
  --bind-to core --nooversubscribe --report-bindings \
  -x monoprop_BENCH_LABEL -x monoprop_BENCH_RESULTS -x monoprop_ROUTING \
  -x monoprop_NUM_THREADS -x OMP_NUM_THREADS -x OMP_DYNAMIC -x OMP_PLACES -x OMP_PROC_BIND \
  python -m pytest 'benches/bench_models.py::test_model_propagate[pauli]' \
  -o filterwarnings=default --bench-rounds=1 --runtime-shape=openmp \
  --benchmark-json="$RESULTS/time-$LABEL.json"
uv run --no-sync monoprop-bench-report "$RESULTS"
uv run --no-sync monoprop-bench-bmf "$RESULTS" "$LABEL"
```

This mapping syntax is Open MPI-specific. Verify scheduler/launcher support; do not escape the allocation. For the
**baseline only**, set `monoprop_PARTITIONS="$T"` and `monoprop_NUM_THREADS="$T"`, export both to ranks with
`-x monoprop_PARTITIONS -x monoprop_NUM_THREADS`, and replace `--runtime-shape=openmp` with
`--runtime-shape=partitions`. Keep OMP settings, routing, allocation and shared overlay identical. The candidate uses
monoprop_NUM_THREADS but never monoprop_PARTITIONS. The shared preflight is a required Task 1 change: the existing
guard would reject this candidate above one rank. Verify declared versus observed shape with binary-linked diagnostics;
labels alone cannot certify shape. After M build assert monoprop.has_mpi before invoking any launcher. Follow the
[physical-core launch recipes](../specs/2026-09-18-rank-local-openmp-design.md#physical-core-launch-recipes) for both
Open MPI and Slurm. Document srun's --threads-per-core=1, compatible core binding and OpenMP budgets/places, and the
required NUMA verification. Do not combine conflicting --hint options or claim that socket/block distribution implies
NUMA containment. Explicit-mask fallbacks require eligible full-CPU step allocation, not merely an exclusive job.
OMP_PLACES=cores is not a worker-count cap, and merely setting OMP_NUM_THREADS does not enlarge a one-core rank mask.
- [ ] Report runtime, exact operation peak/floor/delta, outer peak sum/max, separate whole-construction peak, persistent
  operator/graph bytes, query/snapshot/TLS capacities, term/graph sizes, and numerical checks. Rank peak sums are upper
  bounds on aggregate footprint, not node provisioning estimates; retain maximum rank peak as another statistic. Reject
  non-exact operation/construction windows as parity evidence; outer exactness controls diagnostic labeling only.
- [ ] Compare medians per cell using exactly the five Task 1 gates: runtime, operation peak sum/max and construction
  peak sum/max. Outer peak ratios are diagnostics only. All five required ratios must be <=1.00. If any fail, collect
  five additional observations per arm for those cells and compare all ten. Do not replace the strict gate with a
  geometric mean, a made-up 5% tolerance, or a changed problem size. A noisy failure means parity was not demonstrated,
  not proof of architectural slowdown; it still blocks acceptance. Do not drop the cell or quietly waive it.
- [ ] Profile phases before optimization: bitmap traversal, query merge, decode/probe, missing-ID assignment, row fill,
  hash publication/rehash, inverted-index maintenance, graph packing, MPI pack/wait, cosine, pairs, dot. Record miss
  fraction, overflow fraction and bytes, not just aggregate speed.
- [ ] Allowed bounded optimizations, in order of measured relevance:
  1. Serial thresholds and block sizes; reduce unnecessary tiny teams, keeping logical **reduction** partitions fixed
     for numerical determinism.
  2. Ordered query exact-fill from Task 5; bounded self-probe windows from Task 6; optional inline fill from Task 7.
  3. Incoming missing-ID prefix: parallel fixed-block miss counts, serial exclusive offsets, exact resize, parallel
     ascending fill. IDs remain sender/query order; scratch O(block count), not threads×queries.
  4. Serial index pre-reserve to reduce rehash (measure simultaneous old/new table capacity), without concurrent
     insertion.
  5. Packed graph phases by **whole-word ownership** only: serial validate/count/allocate; each worker constructs and
     assigns one uint64 word, including slot seams and zero padding. No endpoint-wise `|=`. Test slots 63/2/65 and
     nonbinary fallback; preserve derived D permutation instead of adding persistent D indices.
  6. Inverted-index append only if measured dominant: bounded windows of 64-row-aligned blocks; per-block/per-column
     integer counts, serial column offsets/allocations, disjoint sparse element writes and whole dense-word ownership,
     serial handling of an unaligned first word, column-owned promotion after fill. Scratch O(window_blocks×2N), never
     full-height bitmap per worker. Tests must compare dense/sparse promotion, parity and sorted posting order against
     serial after every append.
- [ ] Each optional optimization gets its own failing regression tests, focused review and full affected parity rerun.
  Incoming whole-batch streaming/windowing, concurrent hash publication, new transport, new executor, rank
  redistribution, full per-thread stores or a persistent key map require a new design; stop rather than improvise.

**Gate:** All required performance/memory cells pass with attached raw observations and numerical/placement evidence.
Missing required EC2 evidence, capacity failure, persistent serial-publication bottleneck or repeated ratio>1.00 blocks
removal/acceptance. Report it honestly; the design is not a promise of achievable parity.

### Task 12: Remove legacy partition runtime, topology machinery and direct hwloc dependency

**Files:**
- Delete: `cpp/monoprop/detail/partition/PartitionGroup.h`,
  `cpp/monoprop/detail/mpi/{ShmComm,HybridComm,PartitionBarrier}.h`.
- Delete `cpp/monoprop/detail/mpi/CpuRelax.h` only if no retained caller remains.
- Delete: `cpp/monoprop/detail/partition/CpuTopology.{h,cpp}`, `cpp/monoprop/detail/partition/CMakeLists.txt`,
  `cpp/tests/cpu_topology_tests.cpp`. Do not create replacement CpuTopology files under parallel/ or another path.
- Modify for dependency removal: `cpp/monoprop/CMakeLists.txt`, `cpp/tests/CMakeLists.txt`, `pyproject.toml`,
  `tools/install-deps.sh`, `tools/packages/{apt,brew}.txt`, `.github/actions/setup/action.yml`,
  `.devcontainer/Dockerfile`, `nix/{monoprop,devshell}.nix`, `flake.nix`, `justfile`,
  `.github/workflows/{test,qa-analysis,docpages,copilot-setup-steps,bench,nix,deploy}.yml`.
- Modify migration/build guidance in the same change: `AGENTS.md`, `README.md`,
  `docs/content/docs/{building,benchmarks,openmp-migration}.mdx`, `docs/content/docs/features/parallelism.mdx`.
- Modify: `cpp/monoprop/detail/CMakeLists.txt`, `cpp/monoprop/detail/{mpi,parallel}/CMakeLists.txt`,
  `cpp/include/monoprop/MonomialPropagator.h`, `cpp/monoprop/detail/monomial_propagator/MonomialPropagator.inl`,
  `cpp/monoprop/detail/mpi/{Comm,MPICompat,MPIUtils,Exchange,CheckedCount,Routing}.h`,
  `cpp/monoprop/detail/mpi/MPICompat.cpp`, `cpp/monoprop/MPFunctions.cpp`.
- Migrate/retire test files according to ledger below; update `cpp/tests/README.md` and architecture docs in the same
  change.

**PROPOSED interface:** Simplify `mpi::Comm` to the existing implicit MPI_Comm wrapper with `.mpi`; remove
Kind/Shm/Hybrid and associated pointers/constructors/dispatch. Keep non-MPI behavior, ordinary MPI, PeerPlan routing and
existing typed exchanges. Do not collapse it to a raw alias if that changes overload behavior unnecessarily.

- [ ] Start only after Task 11 parity under separately authorized implementation. The owner approved the immediate
  partition-related C++ extension/ABI break; confirm no runtime path constructs old groups before deleting definitions.
- [ ] Remove any remaining `partition_group_`, friendship and unused fan-out implementation. Confirm Task 10 already
  removed resolve_partition_count_, public partition/factory arguments and facade-only test users; remove remaining
  is_partition_facade/protected partition-only helpers. Keep `clone_` and virtual lifecycle/operator-update hooks. No
  compatibility facade that invokes old callbacks once with silently different semantics.
- [ ] Remove non-MPI-thread transport branches from utilities/tickets. Production router geometry is `(real_ranks,1)`.
  Preserve EXISTING boolean linear/splitmix default/rejection contract and generator shift/peer rules; no
  routing-default repair is needed. Keep ordinary MPI sparse query routing and sparse on-wire replay, sender order,
  count checks, communicator lifetime and serialized thread-level validation (including external initialization).
  `MPIUtils.h` routing agreement uses fixed S=1, never per-rank OpenMP budgets. Keep both MPI and non-MPI
  implementations.
- [ ] Delete the entire topology/affinity implementation and its legacy call sites, including automatic physical-core
  counting, L3 dealing, pinning and mandatory placement diagnostics. Do not move them into ThreadBudget or reimplement
  them with OpenMP-place unions, OS affinity, sysfs or another topology library. User-controlled OpenMP and launcher
  behavior replaces these policies, not another internal runtime.
- [ ] Remove `find_package(PkgConfig REQUIRED QUIET)`, `pkg_check_modules(HWLOC ...)`, the hwloc status message and both
  `PkgConfig::HWLOC` links in `cpp/monoprop/CMakeLists.txt`; remove the unit-runner's hwloc link in
  `cpp/tests/CMakeLists.txt`. PkgConfig is currently required there only for hwloc; do not require it on monoprop's
  behalf once that use is gone. Audit installed target/config output for residual direct hwloc requirements. Keep OpenMP
  target discovery/link/export intact; a transitive MPI/OpenMP-runtime dependency on hwloc is allowed and is not a
  monoprop dependency to remove.
- [ ] Remove hwloc installer implementation/configuration/help/output from `tools/install-deps.sh`, including
  `--skip-hwloc`; remove explicit hwloc/libhwloc-dev from `tools/packages/{apt,brew}.txt`, wheel hooks and any direct CI
  installs. Shared setup and devcontainer consume those manifests, so do not add duplicate replacement install blocks.
  Remove direct hwloc arguments/buildInputs in both Nix files and replace the Nix workflow's
  `pkg-config --modversion hwloc` smoke requirement with OpenMP configure/link/run validation via existing recipes. Keep
  required OpenMP across flake package/MPI package/devshell checks and the deploy wheel matrix. Remove documentation
  claiming hwloc/pkg-config is a direct monoprop prerequisite or that the library pins threads. Do not uninstall or
  strip a library needed by the selected MPI/OpenMP runtime, or remove another dependency's legitimate pkg-config use.
- [ ] Complete this migration ledger before deleting obsolete files. Constructor/facade-dependent cases must already
  have moved in Task 10; retain transport-only coverage until this task removes its implementation.

| Old coverage | Replacement/retirement |
| --- | --- |
| `partition_equivalence_tests.cpp`: energies, gradients, propagation, determinism, copies, Pauli, setters, partial contraction | `openmp_equivalence_tests.cpp`, budgets 1/2/3/4, both pictures, decoded term/value equivalence |
| facade raw-accessor errors | single-store raw-accessor consistency at every budget |
| factory throw/child construction | factory API removed; Task 3 validates joined worker exceptions without terminate |
| `partition_group_clone_tests.cpp`: derived children | retire child-existence assertions; retain derived clone/update-initial-operator independence in `simulator_copy_tests.cpp` and `update_initial_operator.cpp` |
| `shm_comm_tests.cpp` / `hybrid_comm_tests.cpp`: private barrier algorithms | retire only after ownership/source-order/empty-payload/count/error assertions map to MPI/OpenMP tests |
| `hybrid_comm_sparse_plan_on_the_plain_mpi_path` | move unchanged semantics to ordinary MPI tests; unknown and known receive counts |
| plain-MPI half of `hybrid_comm_known_recv_counts_are_masked_through_the_plan` | preserve masked non-peer counts and sparse peer payload assertions |
| `flat_exchange_tests.cpp` replay dense/pairwise/dense-layout/empty tests | retain and run under ordinary MPI after deletion |
| hybrid MPI energy and size case | retain as MPI×OpenMP matrix case in `mpi_distributed_layer_equivalence.cpp` |
| poisoned waiter release | Task 3 MPI failure-driver, joined exceptions and enforced invalid-owner/functionals tests |
| `cpu_topology_tests.cpp`: topology enumeration, mask classification, pinning/L3 dealing | retire these library-owned policy tests; Task 3/10 count/precedence/no-affinity-mutation tests cover the retained library contract, and Task 11 verifies placement externally |

Record individual retired case names and replacement names in the implementation acceptance artifact; do not merely
delete files until green.
- [ ] Audit references:

```bash
rg -n 'PartitionGroup|ShmComm|HybridComm|PartitionBarrier|CpuRelax|is_partition_facade|run_on_all' cpp src
rg -n 'partition_|partitions|monoprop_PARTITIONS' cpp src README.md AGENTS.md docs/content/docs
rg -n -i 'hwloc|CpuTopology|PkgConfig::HWLOC|libhwloc-dev' \
  cpp cmake CMakeLists.txt pyproject.toml tools .github .devcontainer nix flake.nix justfile \
  README.md AGENTS.md docs/content/docs
```

First command should have no production legacy-runtime matches; second must resolve only to mathematical/MPI
partition terminology or historical migration notes, never a retained constructor alias, getenv reader or fan-out.
The hwloc audit must find
no direct discovery/link/install requirement or CpuTopology implementation; migration notes about removal and
third-party/transitive dependencies are allowed. Inspect each match rather than deleting mathematical/MPI partition
terminology indiscriminately.
- [ ] Update `justfile::test-cpp-tsan` and `.github/workflows/qa-analysis.yml` selection before retiring files; require
  nonempty `openmp_` coverage. Keep `just test-cpp`'s unit-only export probe and `just test-find-package` consumer.
- [ ] Run R/M/S, C++ installed-consumer checks and full Python suite. Re-run Task 11 parity after deletion: removal can
  change layout, initialization and measurements. Preserve archived baseline; do not make candidate its own baseline.

**Gate:** No in-process communication runtime, library-owned worker threads, custom barriers, topology/affinity
subsystem, direct hwloc dependency or permanent legacy backend remain. Build/install/import a non-MPI candidate in an
environment without hwloc development headers/pkg-config metadata; inspect direct linkage instead of requiring the
absence of a runtime's own transitive hwloc dependency. One packed store per independent propagator per rank is verified
by construction, copy/functional tests and memory accounting, not merely a flag.

### Task 13: Final packaging, documentation and independent acceptance

**Files:**
- Modify as required: `.github/workflows/{test,bench,bench_bare_metal,deploy,nix,qa-analysis}.yml`, `pyproject.toml`,
  `tools/install-deps.sh`, `tools/packages/{apt,brew}.txt`, `.github/actions/setup/action.yml`, `justfile`,
  `nix/{monoprop,devshell}.nix`, `flake.nix`, `.devcontainer/Dockerfile`.
- Extend existing consumer/probe: `cpp/tests/find_package_smoke/CMakeLists.txt`,
  `cpp/tests/link_export_probe/link_export_probe.cpp`.
- Modify: `AGENTS.md`, `README.md`, `docs/content/docs/{building,benchmarks,openmp-migration}.mdx`,
  `docs/content/docs/features/parallelism.mdx`, `docs/content/docs/concepts/algorithm.mdx`, `cpp/tests/README.md`,
  public Python docstrings affected by ordering/runtime guidance.
- Test: existing wheel jobs; benchmark renderer tests; whole C++/Python suite.

- [ ] Validate the Nix package, explicit MPI package and devshell paths after dependency removal. Current
  `.github/workflows/nix.yml` owns these entrypoints; keep its OpenMP smoke gate consistent with the shared recipes.
  Planned commands (separate candidate environment; none were executed in this refresh):

```bash
nix flake check --print-build-logs
nix build --print-build-logs .#monoprop .#monoprop-mpi
nix run . -- -c 'import monoprop; print(monoprop.__version__)'
nix develop --command bash -euo pipefail -c '
  monoprop_ENABLE_MPI=OFF just build --reinstall-package monoprop --no-cache
  cmake --build --preset skbuild-release
  just test-find-package
'
```

  Require installed runtime resolution and the extended OpenMP/both-basis consumer, not a pkg-config hwloc probe.
  Archive MPI package import/smoke evidence too; the current default flake check alone covers only non-MPI packaging.
- [ ] Build/install/repair wheels through `.github/workflows/deploy.yml::build-wheels` and the `pyproject.toml`
  cibuildwheel hooks for the currently advertised Linux x86_64/aarch64 and macOS matrix. Validate import and a threaded
  propagation in a clean environment outside source checkout, MPI disabled. Inspect ELF/dylib dependencies and bundled
  runtime licensing; check coexistence with NumPy/other imported OpenMP users. Do not claim Windows or unsupported
  sanitizer/runtime combinations tested.
- [ ] Run `just test-find-package` against the installed package, extending its existing shared probe source rather than
  creating a duplicate harness; compile a template kernel using OpenMP and both Majorana/Pauli APIs. The consumer must
  not inherit source-tree include paths or manual linker flags.
- [ ] Document one packed store per independent propagator per rank, preserving copies, existing functional snapshots
  and pared graphs. Users choose P/T at launch and bind with OMP_PLACES+OMP_PROC_BIND, srun --cpu-bind and/or mpiexec.
  Normally use monoprop_NUM_THREADS=T; only unset selects the OpenMP runtime default. Present-invalid is an error.
  Capture once during construction, with no reload/setter. Describe no hardware clamp/topology discovery/NUMA
  validation, possible smaller OpenMP teams, and the within-one-domain allocation assumption. Show matching allocation,
  rank-binding and thread-place examples, including the one-core-rank-mask pitfall. Document one controlling host caller
  outside host OpenMP teams, no overlapping use of dependent state or host MPI calls, serialized MPI, callable lifetime,
  enforced failed-object invalidation, accepted capacity reduction and numerical contracts. Placement verification for
  benchmarks does not add a topology dependency. Advise at least one MPI process per allocated NUMA domain and identify
  actual topology externally; production's common four-domains/socket layout is not universal. Distinguish the selected
  c8a.metal-24xl initial acceptance from pending Intel/AMD HPC qualification. Explicitly recommend against SMT workers,
  even when hardware SMT is enabled, and demonstrate launcher plus OpenMP settings for one worker per physical core.
  Multi-node performance qualification remains pending a proper HPC cluster/interconnect; do not claim it from EC2
  same-instance MPI tests.
- [ ] State exact immediate breaks: partitions/child_factory arguments, PartitionChildFactory, partition-only helpers,
  errors and environment controls removed without aliases; no replacement public num_threads argument. Old automatic
  physical-core selection, multi-rank serial default, library pinning and direct hwloc requirement are gone. OpenMP is
  mandatory even without MPI. Raw order/splitmix physical ownership may differ; ABI requires rebuild. Requests are not
  hardware-clamped. Preserve high-level Python mathematical signatures. In prose/docstrings use `[Symbol][]` references,
  not hard-coded API URLs.
- [ ] Run `prek run --all-files` before any separately authorized push, plus relevant tests, `just gen-api` and
  `just build-docs`. Targeted hooks during development do not replace the all-files gate. Keep generated bindings/API
  artifacts consistent; no hand-edited dispatch.
- [ ] Obtain fresh-context read-only spec/correctness and performance-memory reviews. Reviewers receive both documents,
  implementation diff, retired-test ledger, raw baseline/candidate results, placement/capacity report, build logs and
  known tool limitations. Any concurrency, ordering, memory or unsupported-claim finding blocks completion until
  fixed/retested.
- [ ] Produce a final acceptance artifact containing: revision identities and overlay hashes; configuration matrix;
  actual executed commands/results; numerical cases; maximum observed store sizes; MPI failure tests; runtime/library
  identities; old/new line/file/runtime-component summary; per-cell time/peak medians and ratios; raw artifact paths;
  reviewer findings/fixes; residual risks. Include actual output/artifact paths when delegated, not filenames mentioned
  only in prompts.

**Final acceptance checklist:**

- [ ] Mathematical features, equal global retained term keys, copies/updates/closures and graph paring pass existing
  oracles; matching energies/gradients alone is insufficient.
- [ ] Environment-only immutable budgets, strict invalid-input errors, removed partition interfaces and enforced
  invalid-object/dependent-functional rejection pass; host calling assumptions and immediate breaks are documented.
- [ ] Thread-count-stable ordering and approved deterministic reductions pass; no shared append, lazy-cache, packed-word
  or MPI-worker race remains.
- [ ] Fixed uint32_t TermIndex and optional-MPI/non-MPI packaging pass.
- [ ] Rank-wide limits and verified one-worker-per-physical-core placement satisfy the single-instance campaign,
  including its two-ranks-per-domain case. HPC-family performance and proper-interconnect multi-node qualification are
  explicitly pending, not falsely passed or silently included in the initial acceptance claim.
- [ ] Every required runtime and exact peak-memory cell passes Task 11; unsupported or missing measurements are not
  marked passed.
- [ ] No legacy runtime, direct hwloc dependency or replacement topology/pinning service remains; no equivalent
  scheduler/concurrent store was introduced.
- [ ] Documentation, benchmark schema consumers and API generation agree. Operation/construction exactness gates the
  five parity metrics; outer-window exactness controls diagnostic labels only. Frozen campaign coverage, unique paired
  measurement artifacts and matching validation/placement provenance pass, including all four historical pytest families
  and new bounded paring-construction cells.
- [ ] Shared preflight proves declared/observed equal-allocation shape. Every successful R=3 launch uses explicit
  splitmix; linear rejection is tested separately. Current stable-gradient and streamed-initialization regressions pass.
- [ ] Required OpenMP works through shared package/setup, wheel/deploy, Nix package/devshell/check and
  installed-consumer routes; centralized TSan selection remains nonempty. `prek run --all-files` is complete before an
  authorized push.
- [ ] Owner receives evidence and integration choices; no unrequested commit/push/PR or merge is performed.

## Stop-and-report template

Use this instead of making an unapproved design change:

```text
Blocked task/gate:
Checkout/revision/configuration:
Command or workload cell:
Expected contract:
Observed failure or ratio (raw artifact path):
Smallest reproducer / phase profile:
Changes already made and worktree status:
Approved bounded options attempted:
Decision required from owner:
```

Examples requiring a stop: a full rank cannot fit fixed uint32_t TermIndex, serial publication prevents parity, query
buffering exceeds peak baseline, the benchmark operator cannot verify the target allocation/placement externally,
unsupported OpenMP packaging, failure injection hangs, numerical changes alter retained terms, or the owner declines a
compatibility break. None authorizes a concurrent map, extra shards, altered MPI geometry, weakened oracle, or a
performance waiver.
