# Rank-local OpenMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Read both documents completely; full-plan execution
> requires explicit approval. Task 0 now precedes all downstream work and ends with an owner architecture decision.
> After that gate, separately authorized baseline-only calibration may precede campaign approval, as Task 1 describes.
> Do not infer missing contracts. Checkboxes track future work.

**Goal:** First compare MPI-off index ownership/scaling with a mini-app. Reopen the one-store-per-rank choice using its
evidence; implement the conditional OpenMP replacement only after architecture approval and full-library validation.

**Conditional architecture (Tasks 1–13):** Keep packed position-list rows, keyless indexing, inverted indices, graphs
and MPI routing algorithms/settings. Setting the router's partition dimension to one can change historical physical-rank
ownership under splitmix routing; Task 10 makes that compatibility consequence explicit. Workers traverse frozen
structures or update preallocated disjoint ranges; the caller initially owns structural publication. A bounded Boost
concurrent-index trial may change hash publication, not row/graph ownership or canonical IDs. Final MPI calls belong to
MPI's initializing thread under FUNNELED. Prototype replay first, then construction, and delete the old runtime last.

**Tech Stack:** C++23, OpenMP CXX, existing Boost and optional MPI, nanobind, Python 3.11+, uv/scikit-build-core,
Boost.Test, pytest. Monoprop's direct hwloc dependency is removed with the legacy partition runtime.

**Spec:** [Rank-local OpenMP design](../specs/2026-09-18-rank-local-openmp-design.md). Read both documents before
execution.

**Status:** Planning only. Refreshed source baseline: current `origin/main` at
`90d57177c2b0cd93503f88dd931d2f8dd409aa23`, inspected in rebased `refactor-parallelization` at
`35f359f095fa3f1d628ca2fdc66c93fef6ea7564`. Only CONTEXT.md and these planning documents differ between those commits.
The original planning publication was `38eb13d80a95dd1ac60e762f14f68c9a5925f755`. This owner-requested current-main
refresh preserves the subsequent approved runtime, compatibility, numerical and failure policies and uncommitted edits.
The owner has now explicitly reopened the one-store architecture decision through Task 0. Tasks 1–13 retain the
previously approved one-store proposal as a CONDITIONAL path, not an automatic follow-up. Stop after the mini-app for
an owner decision; reconcile both documents before proceeding. The follow-up interview fixed full-machine MPI-off
library acceptance for small/medium and selected large profiles before seeing results. Task 0 is a staged index-phase
pipeline study, not an isolated hash-container contest or a replacement for those library gates.
On the retained one-store path, OpenMP is mandatory; partition configuration is removed immediately; thread budgets are
launch-environment-only and captured at construction. The fixed32 capacity reduction remains an explicit consequence,
not a restriction that predetermines Task 0's architectural conclusion.
The owner selected one AWS EC2 `c8a.metal-24xl` for implementation/testing and initial acceptance. Production targets
are dual-socket Sapphire Rapids and AMD Zen3/Zen4/Zen5 HPC nodes, commonly four NUMA domains/socket. Hardware SMT may
be on or off, but SMT workers are not recommended: always configure one worker per physical core. Passing the frozen
single-instance EC2 campaign is sufficient for initial acceptance; HPC performance remains separately unqualified.
Multi-node qualification is pending a proper HPC cluster/interconnect, not a cloud-network substitute. Record actual EC2
topology/software and agree representative workload sizes before freezing the campaign or collecting acceptance data.
The owner approved baseline-only, unscored sizing trials before the freeze, once separately authorized. Small/medium
profiles cover the full geometry sweep plus MPI-off full-machine operation. Selected large profiles cover the two
full-node MPI decompositions and MPI-off full-machine operation, not a new all-sizes Cartesian product.

No implementation, engine builds/tests or benchmarks are authorized by this revision. All future execution commands
below are planned, not evidence of passing gates. EXISTING labels describe the inspected baseline; PROPOSED labels
describe future work. Full implementation and any publication of this revision require separate authorization.

**Source-drift audit completed:** this refresh replaces the superseded
`290112c8289ab8015eb9a2c7651ac409839c5a88` pin with the main revision above, as requested by the owner. All 43 changed
paths were audited; affected contracts, source anchors, examples and preservation tests are updated below. The previous
d93d63f source-refresh blocker is resolved, not carried forward merely because HEAD contains planning commits. No
runtime gate is thereby passed or authorized. Both arms must use the refreshed pre-OpenMP engine as their common
starting point; samples from superseded engines cannot establish its baseline. Recheck subsequent source drift before
execution, and retain separate build/calibration authorization, campaign freeze and implementation gates.

## Global Constraints

The storage constraints describe the conditional one-store path in Tasks 1–13. Task 0 alone permits the explicit
per-thread-store experiment, not production shards. Full-machine MPI-off measurements are required both in Task 0
and in full-library acceptance, including cross-NUMA placement. Reconcile the storage design with the owner's
post-experiment decision without silently weakening that coverage or any numerical/performance gate.

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
- Preserve packed rows, keyless indexing through row references, fixed uint32_t TermIndex, sparse state storage, and
  inverted-index cosine recomputation. The Task 11 index A/B may replace the index, not rows.
- Do not introduce hidden operator shards, a dense duplicate store, a second persistent full-key store, per-thread
  operators, or full-vector coefficient double buffering. Select any concurrent index only after correctness/memory/time
  evidence; its container name is neither a rejection criterion nor proof of suitability.
- Required MPI-enabled cells keep each rank within one NUMA domain; MPI-enabled cross-NUMA L2a is diagnostic-only.
  Actual MPI-disabled full-machine operation is required for small/medium and selected large profiles, even across
  NUMA domains. Diagnostic results cannot replace, excuse or average away any required cell; correctness still applies.
- Preserve Majorana and Pauli bases, Heisenberg and Schrödinger pictures, cutoffs, graph build/replay/paring, gradients,
  partial contraction, copying, and initial-operator updates.
- Final MPI calls execute on the MPI-initializing thread, outside library-created OpenMP regions. Request/validate at
  least MPI_THREAD_FUNNELED, accepting higher provided levels without relaxing the caller rule. Retain
  MPI_THREAD_SERIALIZED during legacy Hybrid coexistence; lower the minimum only with its removal in Task 12.
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

- **No runtime work or implementation is authorized by this planning amendment.** Task 0's 2-hour and
  min(8 GiB, 25% RAM) study limits are approved, not permission to launch. Authorize implementation, host/allocation and
  setup/build separately. Task 1 retains its independent 2-hour/15-minute/75% baseline-only pilot. After Task 0's
  architecture gate, obtain separate authorization for that baseline calibration/build work before campaign approval.
  Candidate engine changes require
  implementation authorization and an owner-approved frozen campaign. Formal baseline collection also waits for that
  freeze. Simplification without regressions is the goal; reject the architecture if it cannot earn acceptance. The
  campaign bounds performance claims, not mathematical correctness across the supported API.
- P (called R in existing code/benchmark geometry) is the user-selected process count; T is the requested threads per
  process, normally set with `monoprop_NUM_THREADS`. Only an unset override selects the OpenMP runtime default. Users
  must allocate enough resources and configure rank/thread binding
  with OpenMP, `srun --cpu-bind`, and/or `mpiexec` options. Required MPI-enabled cells keep each rank within one NUMA
  domain; MPI-enabled cross-NUMA L2a stays diagnostic. Required MPI-off full-machine cells instead span the allocation
  in one process. Neither permits a parity waiver or calling a smaller NUMA-local run full-machine. Recommend one
  worker per physical core, including when hardware SMT is enabled. Do not create a topology service/NUMA scheduler,
  silently change P/T, or enforce that
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
| Architecture experiment | 0 | Done: evidence inconclusive for ownership; one store per rank is the working hypothesis (see Task 0 outcome) |
| Baseline (conditional) | 1 | Task 0 decision, authorized baseline-only calibration, frozen inventory, fresh formal baseline evidence |
| Small prototype | 2–4 | OpenMP packaging works; one-store serial and threaded replay agree |
| Shared-store construction | 5–7 | Stable queries/IDs, safety, construction scaling and scratch measured; post-Task-6 architecture go/no-go |
| Remaining evaluation kernels | 8–9 | Replay, derivative, retained closures and reductions validated |
| Integration cutover | 10–11 | Go/no-go passed before Task 10; parity at final geometry decides ownership; index A/B in Task 11 |
| Removal/release | 12–13 | Only after Task 11 confirms the single store; old runtime removed, final full matrix and parity rerun |

One writer per checkout. Kernel and storage tasks are ordered: do not have agents concurrently edit `Engine.h`,
`Evolution.cpp`, or `MonomialPropagator.inl`. Independent read-only reviews are safe. Do not batch past a failed gate.

Expected RED/GREEN checkpoints (record real results; these are planned conditions, not claims of execution):

| Task | RED before the task's change | GREEN required |
| --- | --- | --- |
| 0 | standalone target/self-tests and three variants absent | exact index results, honest scaling/memory evidence, owner decision; no library-parity claim |
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

Before authorized runtime work or implementation, record HEAD, status and these EXISTING anchors at source pin
90d57177c2b0cd93503f88dd931d2f8dd409aa23. Line numbers identify that snapshot; symbol names are durable anchors. The
current audit is complete. If later source differs semantically, report the changed contract and refresh/approve the
plan before coding. Missing PROPOSED symbols are expected RED, not drift.

The audit compared the superseded pin with current main across routing/MPI, construction, Python API/tests, build/CI,
version and docs tooling. Packed storage/QueryWire, fixed32, streamed paired initialization, stable-gradient formulas,
benchmark families/windows, dependency manifests and justfile recipes retain their earlier contracts. Main's routing
basis, communicator agreement/cache, request ownership and Python incremental-axis changes are now baseline behavior,
not optimizations to implement or credit to OpenMP. VERSION is 0.9.2a1; record source/binary hashes, not just that
label.

| EXISTING anchor | Must match before proceeding |
| --- | --- |
| `cpp/monoprop/detail/mpi/Routing.h:100–164,265–296` | full-rank basis draw, width-bound boolean router, default linear, Config and pure routes_pairwise predicate |
| `cpp/monoprop/detail/mpi/Pairwise.h:70–145`; `MPICompat.cpp:91–130` in the same directory | exact mode/partitions/seed agreement; communicator-attribute transport cache; request pre-sizing |
| `cpp/monoprop/detail/mpi/Comm.h:70–175`; `MPICompat.h:135–263` in the same directory | typed window iteration; sparse-plan validation; move-only PendingAlltoallv; WindowVec-only blocks; known-count width check |
| `cpp/monoprop/Evolution.cpp:102–117`; `cpp/monoprop/detail/mpi/Exchange.h:88–148` | rank-uniform routes_pairwise boolean, dense peer walk/nonempty legs, owned request vector |
| `cpp/monoprop/detail/mpi/HybridComm.h::derived_wire_plan_ / exchange_payload_` | fold every partition's send/receive peers; multi-peer fallback retains pairwise transport |
| `cpp/monoprop/detail/evolution/layer_build/Scan.h:208–249` | FusedScanResult has per-stream windows, no result.window; precomputed gen_shift and two self stages |
| `cpp/monoprop/detail/evolution/layer_build/Resolve.h:37–182` | sender_index plus window.slot, position/hash probing and ordered publication |
| `cpp/monoprop/detail/evolution/layer_build/Engine.h:348–372,470–574,600–612` | explicit plan/window constructor, position/hash deferred publication and bounded self probe |
| `cpp/monoprop/detail/operator/OperatorIndex.h:282–288` | five-span find_batch_positions, optional hash output |
| `cpp/monoprop/TypeAliases.h:39`; `cpp/include/monoprop/Evolution.h:54–63` | uint32_t; derivative has LayerAngle and optional CosRecordView |
| `cpp/monoprop/Evolution.cpp:437–464`; `cpp/monoprop/MPFunctions.cpp:44–172` | stable derivative snapshot/predivide/restore and record decisions |
| `cpp/include/monoprop/MonomialPropagator.h:288–306` | virtual update/clone, nonvirtual apply helper |
| `src/monoprop/monomial_propagator.py:231–270`; `tests/test_circuit.py:768–1008` | picture-correct incremental axis and post-call seed, both bases and shared-index gates |
| `justfile:75–122,169–171,371–397`; `pyproject.toml:162–168`; `cpp/tests/boost-test.cmake:1–9` | centralized recipes/MPI build override; CTest default 2;4, distinct from just's default 2 |
| `benches/bench_random.py:22–139`; `benches/conftest.py:248–274,411–493` | four operation families, shape guard, existing op/outer windows |
| `packages/monoprop-bench-tools/src/monoprop_bench_tools/memory/cpu.py:75–107,213–233` | nested peak preservation already implemented |

The updated CI action versions and docs package/lockfile are part of this source snapshot, not runtime-design changes.
Qiskit conversion also gained sparse-observable/single-Pauli and Pauli-product-rotation support: retain its existing
conversion tests and version-gated optional coverage; this OpenMP work neither rewrites conversion nor raises its floor.

Handoff checklist: exact revision/binary and overlay hashes; owner approvals; last passed task; actual RED and GREEN
commands/results; remaining gate/capacity/placement blockers; fixed routing and workload manifest; raw artifact paths.
Never hand off only “tests pass”. No builds/tests/imports/installs/VCS writes are authorized by this document refresh.

### Remote entry point: Task 0, then conditional Task 1

The initial handoff is Task 0's standalone mini-app, not a full monoprop build or the baseline pilot. Transfer both
planning documents and CONTEXT.md; verify the source pin, authorize the small implementation and execution within
Task 0's 2-hour/min(8 GiB, 25% RAM) limits, accounting for setup/build separately. Use c8a.metal-24xl unless
the owner approves another host. Return results and stop for the architecture decision. No Task 1 command below runs
until that decision and the corresponding document reconciliation.
No delegation, provisioning, commits or publication are implied by this handoff.

After that gate, the baseline-only pilot belongs to Task 1. It need not run on the planning machine before handing this
plan to an implementer. Run it on the authorized execution host, currently c8a.metal-24xl. If the host differs,
clarify the hardware decision before treating its sizing results as applicable to the target campaign.
No remote access, provisioning or runtime execution is authorized merely by reading these documents.

Transfer the latest spec, plan and CONTEXT.md with the source checkout. The historical planning commit alone does not
include subsequent interview decisions. Verify document contents/digests on the execution host, including the 2-hour,
15-minute and 75% calibration limits. Documentation-only commits above the source anchor are not engine drift; inspect
semantic source/build/benchmark changes against the baseline-drift checkpoint rather than require HEAD to equal a
planning commit. The refreshed source pin is 90d57177c2b0cd93503f88dd931d2f8dd409aa23. Build the baseline from that
engine, with separately recorded measurement overlays; do not auto-retarget to a later HEAD or reuse superseded-engine
measurements. The owner-requested current-main source refresh is complete; execution authorization is still required.

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
| Preliminary index study | `detail/operator/OperatorIndex.h`, `core/Monomial.h`; read-only reuse | `tools/index-miniapp/{CMakeLists.txt,main.cpp,README.md}`; no installed interface or engine changes |
| Runtime configuration | `detail/EnvConfig.h`, constructor/copy in `MonomialPropagator.inl`; delete `detail/partition/CpuTopology.*`, do not move it | `cpp/monoprop/detail/parallel/{Options.h,Workshare.h,ThreadBudget.h,ThreadBudget.cpp,CMakeLists.txt}` (budget validation only; no hardware discovery) |
| Replay | `Evolution.cpp`, `detail/evolution/CosineRecompute.h`, `MPFunctions.cpp` | no executor or new graph layer |
| Construction | `detail/evolution/layer_build/{Scan,Resolve,Engine,FusedApply}.h` | range helpers stay beside current kernels |
| Storage | `detail/operator/{OperatorIndex,MPOperator,InvertedIndex}.h` | bounded Boost index variant only; no duplicate full-key store |
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
10. Gate loops, reverse-gradient loops, repeated-parameter accumulation, graph append and publication phase boundaries
    remain caller-ordered. Only the bounded index trial may parallelize hash publication within a phase; MPI stays on
    its initializing thread in the final runtime.
11. Empty local stores still follow the same distributed collective sequence. Launch-time thread budgets may differ by
    rank without changing ownership; each object's captured budget is immutable.
12. Internal numerical kernels fall back to serial inside an existing OpenMP region. The supported host enters through
    one controlling caller per rank: MPI's initializing thread in MPI builds, outside active host OpenMP teams, with no
    overlapping dependent-state or host MPI calls. Higher provided support does not relax this rule. The fallback does
    not extend that calling contract.
13. A post-mutation failure invalidates the affected owner and its dependent functionals. Check validity before later
    state access or copying; independent earlier copies remain valid. Pre-mutation validation errors do not poison.
14. Keep the index swappable for Task 11's A/B trial:
    - Nothing outside `OperatorIndex` touches the hash table's internals (slot layout, `insert_slot_`, probe chains).
    - No caller depends on the table's iteration order. Clone and memory accounting go through the public API.
    - Publication goes through `bulk_insert_hashed`, whose parallel-options argument the packed index may ignore.
    - The duplicate-key behaviour of `insert_absent_terms` / `bulk_insert` is pinned by a test. Today the packed index
      indexes both rows; any change is an explicit, reviewed contract change.

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
The CTest default is `2;4`, but `just test-mpi`/`just test-py-mpi` default to `2` unless given ranks or the environment
setting. Supply the intended matrix explicitly; a default two-rank run is not coverage for four-rank multi-peer replay.
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

### Task 0: MPI-off index mini-app and architecture decision

**Question:** For one process using T physical cores, how do per-thread stores, a shared single-writer index and a
shared concurrent index scale under monoprop-like batches? The owner explicitly permits the answer to overturn the
one-store-per-rank choice. Compare the index-phase pipeline: ownership transfers and packed-row initialization,
not isolated containers. Owners initialize private rows in parallel; both shared variants initialize rows on the caller.
The concurrent index only parallelizes publication, so it need not fix a serial row-initialization bottleneck.
This is NOT a propagation benchmark or production cutover. Exact key/ID checks establish index-result agreement, not
retained-operator equivalence, which also requires coefficients.

**Size boundary and files:** Create only `tools/index-miniapp/{CMakeLists.txt,main.cpp,README.md}`. One translation unit
contains the generator, three small adapters, driver and `--self-test`; split no framework out of it. Reuse
`cpp/monoprop/detail/operator/OperatorIndex.h`, `cpp/monoprop/core/Monomial.h` and their header-only dependencies
without editing them. Do not build/link/import monoprop, add Python bindings, use MPI/ShmComm, copy the engine,
simulate gates, coefficients/inverted indices/graphs, capture library traces, or implement a scheduler/container
framework. Aim for a few hundred lines of C++, not a second library. If it needs a production storage extraction or
substantial additional machinery, stop and simplify or request a narrower experiment. It must not depend on Tasks 1–3
being implemented first.

**PROPOSED interface:** `index-miniapp --self-test`, or one variant/case per fresh process:

```bash
OMP_NUM_THREADS="$T" OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
  "$BUILD/index-miniapp" --variant shared-serial --mode grow --rows 65536 \
  --batch 4096 --batches 16 --miss-percent 10 --key-length 6 --modes 128 --seed 0
```

This is a proposed smoke-size invocation, not a frozen measurement size or an executed command. Supported variants are
`per-thread`, `shared-serial`, `shared-concurrent`; modes are `lookup` and `grow`. The first performance pass uses only
modes=128, key-length=6 and inline width=11. Width 1024 and spilled length-16 fixtures remain small correctness checks;
their performance studies are deferred, not automatic extra cells. Keep any supporting dispatch minimal, not a generic
container/type framework. Capture omp_get_max_threads once and record
the actual team. Optional `--profile` enables phase timers in a separate diagnostic run; normal runs time complete
batches only, leaving phase fields empty rather than reporting invented zeros. No library environment parser, global
OpenMP mutation or new public monoprop option. Reject malformed numbers, out-of-range percentages, impossible key/row
counts and checked-arithmetic overflow before allocation. Keep study sizes below the common one-store fixed32 ceiling;
report the architectural capacity difference without billion-row stress tests. A worker failure is caught inside its
region, all workers join, then the process reports failure; do not emit a success row.

- [ ] Obtain separate implementation, host/allocation and execution authorization. The approved Task 0 study limit is
  2 hours total for calibration and measurements, including process initialization, input generation, validation and
  phase profiles. Account for setup/build time and costs separately; they need separate authorization. Limit each fresh
  process's peak RSS, with setup/growth/checks, to min(8 GiB, 25% of observed usable allocation RAM). Record the
  RAM basis and fixed byte ceiling before trials; do not recompute it from falling free memory. Use external monitoring
  and
  stop on a limit; add no library resource clamp. Keep all attempts and consumed/remaining time across interruptions.
  If limits prevent completion, report incomplete evidence; do not extend them or omit losing cells.
  Record source/header and executable hashes, compiler/options, Boost/OpenMP versions, commands and placement. Task 0
  neither consumes nor extends Task 1's pilot; its 8-GiB cap does not restrict Task 1's calibrated workloads.
  Keep CSV/logs outside the checkout, separate from full-library calibration/acceptance. These limits do not authorize
  runtime measurements, setup or launch.
- [ ] Add the standalone self-test entry before implementing variants. RED is the absent target/adapter or incorrect
  expected result, not a broken production test. Use ordinary runtime checks that remain active in Release, not assert
  alone. The same driver executes tiny cases through all three variants at requested teams 1/2/3/4 on suitable hardware:

| Tiny fixture | Required result |
| --- | --- |
| Initial A={0,1}; query B={2,3}, A, C={4,5}; publish misses | Hit mask false/true/false; shared IDs 1/0/2; final three exact keys |
| Next batch C,B,A | All hits; shared IDs 2/1/0, including with the concurrent index |
| Repeated existing A,A | Both queries resolve correctly and preserve their output ordinals |
| Repeated absent B,B in a distinct-miss publication fixture | Reject the invalid fixture; do not silently create duplicate rows |
| Empty input/store, all hits, all misses; 15/16/17 and 63/64/65 query tails | Correct sentinels, no dropped work and correct growth |
| Two distinct keys with equal cached 32-bit hash | Exact comparison distinguishes them before and after table growth |
| Spilled rows and a reallocation of their backing store | Earlier lookups still work; no dangling key/query views |

  Across shards compare decoded keys and owner/local-ID pairs, not global numeric IDs. Also compare each variant with
  its own serial execution at fixed shard count, so scheduling cannot change IDs. Check every tiny result against a
  simple serial key oracle; keep that oracle out of measured processes. Test runtime-limited teams, joined worker
  exceptions and actual participation separately; fewer observed workers cannot be reported as T-core scaling.
- [ ] Add only this independent build configuration, enforcing the repository's compiler floors as well. It uses no
  root add_subdirectory, binding generation or monoprop runtime dependency; Task 2 still owns production OpenMP setup:

```cmake
cmake_minimum_required(VERSION 3.28...4.0)
project(index_miniapp LANGUAGES CXX)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "GCC 14 or newer is required")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "Clang 18 or newer is required")
endif()
find_package(Boost 1.85 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
add_executable(index-miniapp main.cpp)
target_compile_features(index-miniapp PRIVATE cxx_std_23)
target_include_directories(index-miniapp PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../cpp")
target_link_libraries(index-miniapp PRIVATE Boost::boost OpenMP::OpenMP_CXX Threads::Threads)
enable_testing()
add_test(NAME index-miniapp-self-test COMMAND index-miniapp --self-test)
```

  Keep monoprop_ENABLE_MPI undefined; reject a build where it is defined. Confirm no MPI library is linked. The owner
  waived compile/test verification against the Boost 1.85 floor after the first pass; building with the host's Boost
  (>= 1.85, as `find_package` requires) suffices. Do not silently raise the floor or substitute a different container.
- [ ] Generate identical deterministic global keys/query batches for every variant and T. Use an injective synthetic
  encoding of logical key ordinals into sorted positions: divide the 2*NumModes positions into key_length disjoint
  bands, encode ordinal digits in that radix, and apply a seed-derived cyclic offset within each band. Reject an
  ordinal that does not fit; each band contributes one position. This avoids an O(N) key-deduplication table or trace
  corpus. Stream initial rows and retain only one bounded query batch. For hits choose distinct earlier ordinals; for
  misses use fresh ordinals, unique within each publication batch. Use floor(batch*miss_percent/100) misses, report the
  actual count, and reject a requested distinct-hit count above the existing population. Mix order deterministically.
  Repeated hits are a separate self-test, not an accidentally shrinking working set. In lookup mode misses stay absent;
  in grow mode they become available to later batches. This structured synthetic family is NOT a claim about real
  workload ratios or ownership skew; report those limits rather than adding synthetic physics or a trace service.
- [ ] Implement exactly these ownership variants using the same OpenMP harness. T means one team including its primary
  thread; no multiple teams, per-NUMA shared stores, extra processes or fourth hybrid architecture in this task:

| Variant | Required implementation |
| --- | --- |
| per-thread | T disjoint current OperatorIndex stores; owner is monomial_hash<NumModes>(key)%T, as in R=1 routing; each owner is a single writer |
| shared-serial | One current OperatorIndex; parallel disjoint find_batch_positions output ranges; caller-ordered missing-ID assignment, grow_rows_geometric/set_positions and bulk_insert_hashed |
| shared-concurrent | One packed row backing and one boost::concurrent_flat_map index using compact row handles/cached hashes; ordered IDs and caller row initialization, then parallel index publication |

  Reuse public row access on OperatorIndex for the concurrent adapter; call grow_rows_geometric/set_positions but never
  populate its reference table or call reserve(), which would grow that unused table. Its constant empty 16-slot table
  is permitted ONLY as measured mini-app overhead, not a second populated index or a production design. OperatorIndex
  is noncopyable/nonmovable: keep stores in unique_ptr objects with stable addresses. Store row IDs, not raw row
  pointers or query spans; transparent hash/equality consult that same frozen backing store and compare monomial
  identity exactly. Retain cached hashes for publication. Never re-enter Boost from a visitor or
  assume bulk visitors arrive in query order. A bounded per-query visitor is sufficient; no bulk-API tuning project.
  Instantiate only the selected variant. Boost also changes table layout/probing, so its difference from the reference
  is not a pure measurement of locking overhead; report its T=1 overhead as well as parallel scaling.
- [ ] Replay only the index phase: prepare one query batch outside timing, then route/bucket → frozen probe → ordered
  missing-ID assignment → grow/fill packed rows → publish → return results in original query order → join. Hashing and
  ownership calculation belong inside timing, not input preparation; retain/reuse hashes across probe/publication.
  In lookup mode omit growth/publication, retaining hit/miss lookup work. Reference variants keep find_batch_positions
  and its internal 16-query prefetch pipeline. For shared lookup use balanced contiguous query spans over the actual
  team, including short spans/tails; use the same spans for Boost. Do not artificially cap participation at batch/256
  and mistake that chunking limit for map scalability. Small batches may have fewer active workers; record that.
  Preserve stable per-owner query subsequences for the sharded arm. That arm may grow/fill/publish different stores in
  parallel; shared rows/overflow remain caller-written even in the concurrent arm. Do not overlap reads/inserts or
  allocate IDs by scheduling. Use native table growth; time/report any reserve. No free final-capacity
  reservation for only one arm. Row initialization completes before concurrent map operations, and joins precede any
  row growth/relocation or dependent reads.
- [ ] Model owner bucketing with bounded flat arrays of ordinals/offsets and result scatter, not a copy of ShmComm.
  Parallel source chunks may use O(T*T) counts/offsets plus O(batch) records; use ordered prefixes, not one serial
  dispatcher doing all work or a T-fold copy of the keys. No producer-consumer queues or custom barriers. Include this
  overhead in complete-batch timing, and report index-only probe/publication timings separately. Label bucketing as a
  simplified shared-memory model; neither its cost nor its omission from kernel timing predicts legacy transport time.
- [ ] Check results outside timing without a second resident operator: every declared hit/miss, returned row's exact
  key, expected owner, final count and insertion order. A final streaming lookup of every expected key plus the exact
  row count checks the retained set without an O(N) shadow map. Checksums may prevent dead-code elimination and help
  compare runs, but are not the correctness oracle. A timed run with incorrect results is invalid, never fast evidence.
  Run separate ASan/UBSan checks and a qualified OpenMP race check on tiny cases; unsupported tooling stays a reported
  limitation. Do not use sanitizer timings for scaling or expand into the whole library's test matrix.
- [ ] Run the approved first pass: fixed GLOBAL initial rows and total queries across T, never N rows per thread.
  At modes=128/key-length=6/batch=4096/seed=0, use exactly three cases: frozen all-hit lookup, growth with 10% misses,
  and growth with 100% misses. Select one small and one larger memory-resident size before comparative timing, within
  the common study cap including growth/checking. Verify the accessed footprint exceeds the relevant last-level cache
  if claiming DRAM scaling; if the cap prevents that, state the limitation rather than enlarging the budget. Choose
  batches to exercise the footprint and record initial/final sizes. Sweep 1/2/4/... physical cores plus exact domain
  and allocation endpoints, deduplicated. Repeat within one domain and across the allocated machine; a one-domain host
  needs no duplicate series. Use the same core sets and external memory policy for all variants at a given T. The
  full-machine MPI-off series is required, even across NUMA domains; an MPI run cannot replace it.
- [ ] Report first-pass results before expanding scope. Defer wider-key, spilled-row performance, batch=64 and other
  hit/miss-distribution sensitivities until those results justify an owner-approved follow-up. Keep tiny correctness
  fixtures for collisions, both position widths, spilled rows and backing growth; deferring performance cases does not
  permit incorrect adapters. Qualify conclusions to measured inputs. Do not add a Cartesian product, weak-scaling
  campaign, containers or a trace subsystem; follow-up cannot silently extend the remaining time/RAM limits.
- [ ] Verify active worker masks against physical-core/NUMA topology using external tools and OpenMP diagnostics.
  Record natural first-touch: per-owner and caller-built rows can occupy memory differently under the same external
  policy. Count that difference as a real ownership/construction consequence; do not normalize away an architectural
  disadvantage. Separate locality effects from hash-table synchronization in the report. An explicitly matched
  interleaved-memory rerun may diagnose a result within the budget, but cannot replace the original comparison or apply
  only to one arm. No internal affinity setter, hwloc dependency or automatic geometry repair. Unknown placement makes
  scaling evidence incomplete.
- [ ] Emit a plain CSV row per process with full knobs, requested/actual team sizes, initial/final rows, hits/misses,
  complete-batch and optional phase seconds, queries/second, min/max shard rows, row/spill/index byte estimates where
  available, and validation outcome. Report setup separately. Measure complete batches without fine-grained timers,
  then collect --profile diagnostics in separate matching processes to avoid timer overhead bias. Label per-owner
  phase sums/maxima as nonadditive diagnostics when owners overlap; do not present their sum as elapsed batch time.
  Profiles also record active-worker counts from the actual probe/publication bodies, not merely team creation.
  Use three fresh-process observations per variant/cell with rotated order; retain every sample, median and range.
  Report speedup versus each arm's T=1 and time/memory ratios versus per-thread at the SAME T; do not select from
  throughput alone or hide losing cases.
  Capture whole-process peak RSS with an external tool in fresh processes, including setup/growth. Input generation
  and checking use bounded scratch, not a large reference map that masks index peaks. RSS includes allocator/runtime
  overhead; current memory_bytes/index_estimated_memory_bytes are estimates, not exact allocations or Task 1 windows.
  Archive compiler/runtime/placement metadata beside CSV rather than building a report schema/framework into the app.

**Planned standalone smoke commands, only after authorization:** BUILD is an absolute, isolated build directory; the
selected allocation must support the requested workers. No uv sync, monoprop imports or engine build is needed:

```bash
cmake -S tools/index-miniapp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --parallel
OMP_NUM_THREADS=1 ctest --test-dir "$BUILD" --output-on-failure --no-tests=error
for T in 1 2 3 4; do
  OMP_NUM_THREADS="$T" OMP_DYNAMIC=FALSE "$BUILD/index-miniapp" --self-test
done
```

**Gate and handoff:** Deliver the three-file mini-app, self-test evidence, raw CSV/peak-RSS/placement logs and a short
comparison with phase bottlenecks, uncertainty and omitted costs. Distinguish an ownership-model result from a measured
current-library baseline. Do not claim propagation, replay, gradients, MPI or full-library parity from this study.
If an arm is unavailable, placement is unverifiable, correctness fails or the budget ends, report incomplete evidence;
never declare the remaining arm the winner by default. Stop for the owner's decision to retain a shared-store path,
retain/reconsider sharding, or gather specified additional evidence. Record that decision and revise the spec and
Tasks 1–13 accordingly BEFORE proceeding. Preserve the already-approved required MPI-off full-machine library coverage
across all three size tiers; do not choose protected configurations after seeing which architecture wins. No fixed
mini-app speedup threshold replaces the owner decision or weakens the five full-library gates. Task 0 outputs are never
formal Task 1 samples, and its whole-process RSS cannot certify the library's operation/construction windows.

**Task 0 outcome (owner decision):** the evidence is inconclusive for the ownership axis. One packed store per rank
remains the working hypothesis for Tasks 1–13, and the per-thread (sharded) design stays available through the
existing partition runtime until Task 12. The ownership decision moves to Task 11's full-library comparison against the
old partitions, with an earlier diagnostic go/no-go after Task 6. The index axis — current packed index with serial
publication versus `boost::concurrent_flat_map` with parallel publication — stays open until Task 11's A/B trial.
Evidence (measured on the target host, not acceptance samples):
- The mini-app favoured neither design uniformly. Per-thread won growth with 100% new terms; the shared stores won
  lookups; the concurrent index lost 1.2–2.1× on one core.
- Instrumented `propagate` on the benchmark models: lookups hit 80–89% on Hubbard and Pauli, and 0% on the random
  circuits. The scan takes 66–95% of one-core time; lookups take 19–21% on the physics models and 2% on random.
- In-library Boost trial: +15–18% end-to-end on one core for Hubbard/Pauli at L1 size. Parallel publication gave no gain
  at per-gate batch sizes; the scan was still serial, so there is no full-machine result.

The results, overlays and budget ledger are archived outside the repository with the Task 0 report. The owner waived
the Boost 1.85 compile check, as stated above.

### Task 1: Freeze baselines and repair measurement trustworthiness

**Prerequisite:** Task 0 results have been reviewed by the owner and recorded (see the Task 0 outcome), and both
documents have been reconciled. The following remains the conditional one-store campaign, not an automatic continuation
of the mini-app. The formal baseline observations of the partition runtime, including MPI-off full-machine cells,
are also the sharded reference for the post-Task-6 go/no-go and Task 11's ownership decision. No extra measurement
framework is needed.

**Benchmark roadmap: use [benches/LADDER.md](../../../benches/LADDER.md), not a new suite.** N is nodes, C is usable
physical cores/node, R is ranks/node, and the ladder's P is old partitions/rank (not the generic process-count P used
elsewhere in this plan). Map old P to candidate T workers/rank. On the single target instance N=1:

| Rung/control | Candidate geometry | Classification |
| --- | --- | --- |
| L1 | R=1, T=1 | Required small/medium single-thread control |
| NUMA-local shared-memory control | R=1, T=C_dom | Required small/medium; not canonical full-node L2a |
| MPI-only control | R=D, T=1 | Required small/medium, with fewer active cores than full-node L2b |
| L2b | R=D, T=C_dom | Required one-rank/domain production shape |
| L2b variant | R=2D, T=floor(C_dom/2) | Required two-ranks/domain shape when C_dom>=2 |
| L2a (MPI-enabled) | R=1, T=C | Full-node diagnostic; cross-NUMA allocation expressly allowed |
| MPI-off full machine | R=1, T=C, has_mpi=false | Required small/medium and selected large profiles, including cross-NUMA |
| L3/L4 | N>1 | Pending proper-interconnect HPC qualification |

D is allocated NUMA domains, C_dom the selected cores/domain, C=D*C_dom. Record actual balanced subsets and unused cores
rather than infer topology. If D=1 makes MPI-enabled L2a identical to required L2b, deduplicate within that build mode
without downgrading required status. Never deduplicate MPI-off against MPI-enabled evidence, even at the same R/T.
Selected large profiles use both L2b decompositions AND actual MPI-off full-machine operation. Calibrate common profile
sizes using the MPI-off baseline too; a post-freeze capacity failure blocks acceptance, not permission to shrink a
workload or drop its cell. MPI-enabled diagnostic L2a retains the same flags even if it cannot fit. The NUMA-local
control cannot substitute for full-machine operation in either build mode.

The practical workflow is baseline-only ladder sizing → owner-approved frozen cells/flags → the same pytest nodes on
both arms at matched shapes → existing reports plus the five parity metrics and numerical evidence below. Prefer the
ladder's Hubbard/Pauli lower_atol, random-Heisenberg obs_terms and random-Schrödinger num_generators sizing controls;
record seed/structure and the resulting gradient length. These pre-freeze workload choices never relax oracle
comparison tolerances or permit post-regression retuning. Size graph rows using the largest build/energy/gradient peak,
including export/validation resources, within the already agreed pilot limits.

The ladder recommends grouped graph runs to save setup. Use that for pilot/diagnostic work where fixtures actually
share a graph; audit the fixtures rather than assume all models share build output. Formal per-operation observations
still use fresh processes and --bench-rounds=1 to avoid prior live graphs and fixture order affecting their peaks. Their
separate construction worker measures setup explicitly. Reuse the existing functions, builders and report files;
observe/validate/compare only supply the missing evidence joins/checks, not another benchmark framework.

Keep MPI-enabled cross-NUMA L2a in a separately labelled diagnostic inventory/output directory, outside the frozen
required-cell manifest supplied to compare. MPI-off full-machine cells belong IN that manifest, with unchanged
numerical checks and all five parity gates. Never reclassify a failed required cell as diagnostic or use L2a ratios
to waive it.
Numerical mismatches in diagnostic runs still require investigation. Historical ladder results are sizing examples,
not target-machine acceptance evidence. Update its shape/declaration guidance with the candidate cutover.

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

Use these observations to select NUMA-contained MPI acceptance and full-allocation MPI-off geometries. A 1x96 shape is
full-machine only with all 96 allocated physical cores. If an MPI-enabled run spans domains, it is diagnostic L2a;
in an actual MPI-disabled build, full-machine operation is required whether or not it crosses domains. Missing
diagnostic tools require another external observation method, not a library topology dependency. Record hardware SMT
state, but configure one worker per physical core. Do not add SMT-sibling profiles; HPC/multi-node qualification remains
separate and pending.

- [ ] Obtain separate authorization for baseline build/calibration work. On c8a.metal-24xl, record sockets/NUMA domains,
  physical cores, hardware SMT state, memory allocation and software stack before choosing geometries. Agree
  representative workloads/sizes and single-instance routing/rank/thread shapes. Initial acceptance is the EC2 campaign,
  including a two-ranks-per-domain case and actual MPI-off full-machine operation in all three selected size tiers;
  all workers use distinct physical cores. Production-HPC performance and
  multi-node qualification on a proper interconnect remain pending, not prerequisites for initial EC2 acceptance.
  Production's common eight-domain layout is not a hard-coded EC2 topology. Historical profiles below are candidates,
  not an approved campaign.
  Confirm the bounded new paring-construction measurement in that campaign. Record baseline
  revision, allocation, compiler/flags, has_mpi, MPI version where applicable, index width, OpenMP runtime, allocator
  environment, imported extension path/hash, seed and complete workload configuration before collecting evidence.
- [ ] Before changing engine code, separately build the baseline using R/M. Preserve its environment and binary. Apply
  only the identical measurement-harness changes to both arms; record this overlay's diff/hash. Use the refreshed
  main pin for the baseline and the candidate's starting engine, not the former routing-branch snapshot. Main's basis
  redraw, transport and parameter-axis changes must not be attributed to OpenMP. Baseline numerical failures stop this
  task. After the freeze, any required baseline failure blocks collection, not permission to resize or remove the cell.
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
  Never promote calibration runs, even with unchanged parameters. Include actual MPI-disabled baseline feasibility
  for small/medium and selected large profiles before freezing shared sizes. No candidate-influenced resizing or
  removal after freeze. Task 1 limits are independent of Task 0's 8-GiB study; approval is not execution permission.
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
- [ ] Include the build mode in shared preflight and evidence. Read monoprop.has_mpi from the imported extension,
  record it with the extension hash, and require it to equal identity.has_mpi for both arms. Actual MPI-off cells have
  has_mpi=false and R=1; a single rank of an MPI-enabled binary cannot substitute. The current conftest imports mpi4py
  whenever installed: gate that import on monoprop.has_mpi BEFORE touching mpi4py.MPI. For MPI-off, use comm=None and
  launch directly, with no MPI launcher/initialization or invented MPI thread-support/version values. Apply the same
  guard in the new driver. Use the identical overlay on both arms, without changing the installed Python API.
  Tests must cover mpi4py present/absent with has_mpi=false and a trap on attempted MPI import, unchanged MPI-enabled
  fixture behavior, requested build-mode mismatch, missing build-mode evidence and MPI-off declarations with R>1.
  Neither an arm label nor absence of mpiexec proves the build mode.
- [ ] Separate whole-construction measurement is intentional policy, not a repair for nesting. A fresh untimed worker,
  without pytest memory fixtures/session graphs, opens one `HighWaterMark`, constructs inputs/propagator/graph/callable
  as required, runs the operation and retains outputs until close. Record peak sum/max, floor, delta and exactness. Use
  existing model builders; no duplicate physics or general benchmark framework.
- [ ] After unscored calibration, freeze an owner-approved campaign inventory BEFORE formal baseline collection. Its
  JSON schema is
  `{"schema_version":1,"workloads_path":str,"workloads_sha256":str,"cells":[cell]}`. Each cell has unique slug `id`,
  `profile`, `node_id`, `operation`, `measurement_kind` (`pytest` or `driver`), `identity`, `config_digest` and
  `parameters_digest`. `identity` contains boolean `has_mpi`, `ranks`, requested `threads`, `index_bits=32`,
  `cpu_allocation`, `compiler_flags`, resolved `config`, and `routing` (`mode`, derived `bits`, numeric `seed`). Expand
  the size-band-specific inventory below: small/medium profiles use the full geometry sweep and existing routing
  requirements plus MPI-off full-machine operation. Selected large profiles use the two full-node MPI decompositions
  and MPI-off full-machine operation. Do not generate an all-sizes Cartesian product. Archive the inventory/digest
  with owner approval. This file, not later observations, is the authoritative expected-cell list. A full cell missing
  from BOTH arms must fail completeness. Reject duplicate/extra cell IDs and mismatched identities. Do not regenerate or
  shrink the inventory to make observations pass.
- [ ] `tools/rank-local-openmp-workloads.json` maps `profiles[profile][node_id]` to `operation`, `measurement_kind`,
  `basis`, `picture`, complete `config`, deterministic `parameters`, and `pare_threshold`. `tiny` is the small band and
  calibrated `reference` is medium; each band's `-pared` variant shares its geometry policy. Preserve the existing
  unpared four-family and threshold-1e-10 energy/gradient plus driver-construction coverage. Add `large`/`large-pared`
  entries for the owner-selected large workloads at both full-node MPI decompositions and MPI-off full-machine shape.
  Profile names do not freeze historical reference sizes: record every calibrated override, with no ambiguity within a
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
  `threads`, `cpu_allocation` and `settings` (the exact OpenMP/library environment). `observed` contains `has_mpi`,
  `ranks`, `threads_per_rank` and `cpu_allocation`, from Task 11's binary-linked large-kernel diagnostic. Preserve
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
  fields only; construction artifacts contain construction-window fields only. Every timed, construction and validation
  artifact records the imported binary's observed has_mpi as well as the expected identity; compare both to the cell and
  placement evidence. Do not invent timing for the untimed worker. `validate` writes a separate uniquely named
  validation artifact under `DIR/ARM/CELL`, without sample ID.
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
  per-cell medians: runtime, operation peak sum/max, construction peak sum/max. All five ratios must be <=1.00. At R=1
  with MPI disabled, sum and max coincide; retain all five fields/gates and the same numerical oracles. Report
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
  files. Each sorted term file is gzip-compressed JSON Lines (`.jsonl.gz`) with
  `{"key":[indices],"real":number,"imag":number}`; gradient arrays are in optimizer parameter order. Uncompressed,
  truncated or otherwise unreadable term files are malformed evidence. `pare_functional_construct` and pared evaluation
  artifacts omit term files because the pared callable owns a subgraph rather than an independently exported operator;
  validate its scalar/full gradient against the baseline pared callable. Use deterministic input parameters; a graph
  benchmark replays at those parameters for validation. The required products are:

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
  index tuples and store real/imaginary coefficient components in per-rank gzip files: level 1, with no timestamp or
  file name in the gzip header, so equal maps give equal bytes. The line format, sort order and rounding are unchanged
  by compression. The owner approved compression instead of adding storage. On the pilot's term files it measured
  4.4-16x smaller, and the draft campaign needs about 24 GB per arm instead of about 122 GB on the host's 128 GB volume.
  These exported coefficients are currently rounded to 1e-12; state that limitation and retain unrounded scalar/gradient
  checks. Stream-merge sorted rank files when comparing global maps instead of requiring historical rank ownership. For
  Heisenberg, the binding adds the replicated identity/core term on each rank: retain one copy after checking agreement,
  **do not sum it R times**. Other duplicated nonidentity terms fail unique ownership; Schrödinger identity follows
  ordinary owner semantics. Require equal global key sets, including zero-coefficient stored terms; compare values with
  the existing `test_utils::near` rule from `cpp/tests/TestUtilities.h.in`: `abs(a-b) <= 1e-9 +
  1e-7*max(abs(a),abs(b))`, componentwise for complex values and gradients. Reject missing/non-finite outputs. Preserve
  all stricter existing unit-test tolerances; this benchmark comparison does not weaken them. During pre-freeze
  calibration, export failures may require an approved smaller equal-arm profile. After freeze they block acceptance,
  not permission to resize or supply an unchecked `numerical_ok=true`.
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
    identity = {"has_mpi": False, "ranks": 1, "threads": 2, "index_bits": 32, "cpu_allocation": "test-cpus",
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
                         "observed": {"has_mpi": False, "ranks": 1, "threads_per_rank": [2],
                                      "cpu_allocation": "test-cpus"}}
            placement_path = save(f"{name}-{arm}-placement.json", placement)
            common = {"schema_version": 1, "campaign_sha256": digest(campaign), "arm": arm, "has_mpi": False,
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
whole required cell deleted from both arms (including a large MPI-off full-machine cell), missing samples, mismatched
timed/construction pairs, reused files/run IDs and calibration output offered as acceptance evidence. Also reject
missing/mismatched has_mpi between inventory, artifacts, placement and binaries; MPI-off declared at R>1; and a D=1
attempt to deduplicate MPI-off cells against MPI-enabled cells. Check missing placement provenance (exit 2),
each false/absent exactness flag, unequal
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
  silently omit graph-heavy cells or bypass the guard on an unverified allocation. Owner decision after the pilot:
  Hubbard graph cells (`build_graph`, `energy`, `gradient`) use 2 Trotter steps on smaller lattices, within the guard.
  Hubbard `propagate` keeps its 29 steps. The 29-step graph did not fit this host: the pilot killed it above the
  75% memory target, with at least 147 GiB resident. It may be checked separately on a larger machine; that check is
  not part of this campaign.
- [ ] Select required geometries from observed physical cores, not vCPU/SMT-sibling counts. For D allocated domains with
  C_dom cores each, keep `(R,T)=(1,1),(1,C_dom),(D,1),(D,C_dom)` for small/medium profiles. These R=1 controls use one
  domain. Add `(2*D,floor(C_dom/2))` with disjoint cores when C_dom>=2; record unused cores. Preserve MPI routing
  coverage. Add actual MPI-off `(1,C)` for every small/medium profile and the owner-selected large/large-pared profiles.
  Selected large acceptance uses `(D,C_dom)` and `(2*D,floor(C_dom/2))` with MPI, plus `(1,C)` without MPI. Freeze
  has_mpi and geometry per cell; deduplicate only within the same build mode and complete identity. In particular,
  D=1 never lets an MPI-enabled sample replace required MPI-off evidence. Freeze allowed cells per profile;
  do not reclassify profiles or drop cells after a regression. Uneven domains require approved balanced subsets, not
  assumed C_dom=96/D. Each pair has identical build mode, R/T/cores and inputs. Baseline uses T partitions per process;
  never compare only against forced-serial old code. Launch MPI-off directly after verifying not monoprop.has_mpi;
  its one process spans the allocation without an MPI launcher or fictitious NUMA containment.
- [ ] Run MPI-enabled canonical full-node L2a separately as a labelled diagnostic with the same model flags at R=1,T=C.
  It may cross NUMA domains; verify complete allocation and one worker/core rather than claiming containment. Keep its
  separately budgeted outputs outside required-cell artifacts, not in an extra driver mode. Do not substitute a smaller
  NUMA-local run or relabel the required MPI-off `(1,C)` cell as diagnostic. Record failed/unavailable diagnostics;
  no diagnostic ratio is a parity gate or waiver. Required MPI L2b stays NUMA-contained; L3/L4 remain pending. Preserve
  existing node IDs/term-count fields and add runtime-shape/build-mode/actual-team metadata. Counts alone cannot replace
  global term-map/value checks.
- [ ] Use `just capture-baseline` / `just diff-baseline` and `tools/capture-baseline.py --compare` only as supplementary
  small-fixture regression evidence. Its keys include rank, tolerance differs and it lacks operation
  gradients/provenance; it cannot replace the ownership-aware global-map comparator or independent
  exact/finite-difference tests.

**Frozen campaign (owner-approved 2026-09-25):** `tools/rank-local-openmp-campaign.json` (sha256
`ffde870c77a66c8ce37fb2cc07b39abef976e9d7938b7cea59f43cb407de068a`) lists 250 cells over
`tools/rank-local-openmp-workloads.json` (sha256 `508b65f1f0eb2171fc4fbd79feba2381054c6dcdd38a614600f5b957ab814416`).
Neither file may be regenerated or shrunk to make observations pass. The unscored pilot and its size decisions are
archived outside the repository with the Task 1 handoff report; no pilot output is a formal sample.
- The observed host has one NUMA domain (D=1, C=C_dom=96). Within each build mode, L1 and the MPI-only control
  coincide at 1x1, and the NUMA-local control, L2b and L2a coincide at 1x96. There is therefore no separate L2a
  diagnostic inventory. MPI-off 1x96 cells stay separate and required. The single-thread control runs in the MPI
  build.
- `tiny` and `reference` (plus their `-pared` variants) run at MPI 1x1, 1x96 and 2x48 and at MPI-off 1x96.
  - Each band has 16 unpared nodes. The `-pared` variants cover energy/gradient at threshold 1e-10 for all four
    families, plus `driver::pare_functional_construct` in both pictures.
  - Random reference `propagate` and `build_graph`, in both pictures, add 3x32 splitmix, 4x24 splitmix and
    4x24 linear cells.
- `large` and `large-pared` run at MPI 1x96 and 2x48 and at MPI-off 1x96. Their sizes (13-68M terms) are set by the
  cost of validation export, not by RAM.
- Owner decision on memory: the measured `VmHWM` shortfall is accepted as documented measurement noise for the
  five gates. With threads on many CPUs, a window can miss up to about 50 MiB of a real transient (see
  `docs/content/docs/benchmarks.mdx`, "Memory exactness"). The strict <=1.00 gates are not relaxed.
- The pilot killed one trial above the 75% memory target: the rejected 29-step Hubbard graph. The owner reviewed it;
  no cell in the campaign needs that graph.

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
- Modify: `cpp/monoprop/detail/mpi/MPICompat.{h,cpp}`; audit request/collective boundaries in `Exchange.h` and
  `Pairwise.h`.
  Final environment cleanup is Task 10, MPI-level cutover Task 12.
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
criteria; `Evolution.cpp:437–464` snapshots self endpoints, predivides, accumulates, restores and combines
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
  unwinding can wait on peers that never posted. This covers BOTH replay `Exchange.h::Ticket` and construction
  `MPICompat.h::PendingAlltoallv`: both are already move-only/self-draining; move assignment drains the old destination.
  Preserve unified request-vector ownership, moved-from posted=0 and pre-sized request storage. Normal successful paths
  still explicitly wait. Extend the active-ticket driver scenario to both handle types; source lifetime tests alone do
  not establish coherent distributed failure. Do not rely on a ticket
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
- [ ] Validate host-owned and library-owned MPI with MPI_Query_thread. During coexistence, keep the required/requested
  MPI_THREAD_SERIALIZED level: Hybrid still makes MPI calls from non-initializing workers. Task 12 lowers both checks
  to MPI_THREAD_FUNNELED only after removing that path. Do not reinitialize host-owned MPI or assume the provided level
  equals the requested one. Higher levels are valid; no speedup is presumed from requesting a lower minimum.
- [ ] Add an internal `mpi::require_initializing_thread()` guard in MPICompat, using MPI_Is_thread_main after MPI is
  initialized. Apply it before MPI-using constructor/operation/retained-functional/cleanup work on the new ordinary-MPI
  one-store path, including implicit communicator work before constructor bodies. Do not impose it on still-supported
  legacy Shm/Hybrid worker dispatch. MPI-off is a no-op; public operations still enter outside active host OpenMP teams.
  Keep `mpi::routes_pairwise` behind the caller guard on the ordinary-MPI path: first use may allreduce, warm use
  still calls MPI_Comm_get_attr, and a new/duplicated communicator must agree independently. Preserve the
  communicator-attribute
  cache and Hybrid's constructor agreement; do not replace them with per-gate agreement or a raw-handle global map.
  At Task 10 every new propagator uses this contract, even if provided support is SERIALIZED or MULTIPLE. Do not capture
  whichever thread first touched an object as a substitute for MPI's actual initializing thread.
- [ ] On wrong-thread entry, print an actionable initializing-thread diagnostic and fail fast locally without calling
  MPI_Abort, communicator queries, collectives or the normal operation_failed helper from that thread. This unsupported
  host-contract violation is distinct from valid-caller distributed failures. Test it under a launcher configured to
  terminate peers on a process's abnormal exit; do not promise peer cleanup under an arbitrary launcher policy. Add no
  process-wide lock, dispatch service or nested-execution machinery. Preserve no-overlap rules for dependent state and
  host MPI calls; an internal helper's nested fallback does not widen the public contract.
- [ ] Add a separate failure driver with `worker-throw`, `before-exchange`, `active-ticket`, `insufficient-thread-level`
  and `wrong-thread-entry`. Inject on rank zero while peers follow the normal phase. For wrong-thread-entry, invoke a
  real guarded entry from a host std::thread (outside OpenMP) after main-thread MPI initialization, at supplied
  SERIALIZED/MULTIPLE now; add FUNNELED in Task 12 after lowering the minimum. Never relax the caller rule.
  Use a 30s subprocess timeout under two ranks, with launcher abort-on-bad-exit enabled for local-fatal cases. Require
  prompt nonzero failure with the expected diagnostic, not timeout. Keep abort cases outside the ordinary unit runner.
  Test all required/provided-level decisions in a pure validator; real initialization tests record the actual provided
  value rather than assuming a request guarantees an insufficient level. Test single-rank worker throws separately.

**Gate:** R/M compile and pass with kernels still serial. Focused gates include `combined_recompute_equivalence.cpp`
record restoration and `tests/test_deep_circuit_gradient.py` deep non-singular, vanishing-cosine and no-record cases,
before and after threading. Preserve `tests/test_circuit.py` incremental-axis/seed behavior too; its existing passing
cases are baseline preservation evidence, not RED for a feature already shipped on main. Options reach every intended
entry point; runtime global settings unchanged; failure tests do not hang. Public/exported signature ABI change is
recorded for a coordinated library/bindings rebuild.

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
router, precomputed `gen_shift`, then capture/scaling arguments. `Engine.h::build_layer` derives the plan/window once
from that shift and passes BOTH explicitly to LayerBuildEngine after the sink argument. Keep that constructor contract
in direct-engine fixtures; it no longer derives geometry itself or supplies a default plan. FusedScanResult has no
separate window member: compare each populated stream's window(), with value streams empty when capture is disabled.
PROPOSED: extract the following range helper in Scan.h; preserve these inputs/result type, prepare lazy data once, and
merge private results. Do not reintroduce rank-count-shaped query vectors. The range helper must never call a lazy
accessor itself:

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
  `window.indices()` and `.at_slot(flat_slot)`; never use a flat slot as a window index. All pieces share the exact
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
for (const auto wi : window.indices()) {
    const auto flat_slot = window.slot(wi);
    for (size_t range = 0; range < pieces.size(); ++range) {
        // pieces[range].leader_queries[wi] belongs to flat_slot, not flat slot wi.value.
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
        BOOST_TEST(pr.window.slot(pr.sender_index(0)) == 4U);
        BOOST_TEST(pr.window.slot(pr.sender_index(1)) == 6U);
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
probe/prefix/publication/inverted-index times separately. If serial hash publication dominates, report its Amdahl limit
and carry the evidence into the Task 11 index A/B. Do not improvise a custom concurrent table here. Give
`bulk_insert_hashed` a trailing `parallel::Options` argument now (the packed index publishes serially and ignores it),
so the Task 11 variant needs no call-site changes (shared invariant 14).

**Architecture go/no-go (diagnostic, after Task 6 and before Task 10):** with the Task 1 harness, compare the
one-store prototype against the frozen partition baseline at matched geometries. The prototype has threaded replay
(Task 4), traversal (Task 5) and frozen lookup (Task 6). Cover L1 (R=1,T=1), the NUMA-local control and MPI-off full
machine for representative small/medium profiles. These are fresh diagnostic observations, never formal samples, and
they cannot waive a Task 11 cell. Report per-phase time and peak memory. If the one-store path trails clearly and no
bounded Task 11 optimization explains a route to parity, stop for the owner's decision before Task 10 removes the
partition API. This is the last cheap point to revert: only the additive one-store path would be dropped.

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
insertion, overflow mutation, packed graph phases and inverted append remain serial at this stage.

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
  remote+self slots, multiple remote peers, empty peers, zero-angle, deep non-singular circuits, vanishing-cosine and
  no-record paths. Retain
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
  die while requests are outstanding. Preserve EXISTING `mpi::routes_pairwise(comm)` selection and the boolean
  `post_flat_alltoallv(..., pairwise)` argument. Plain MPI walks a dense PeerPlan and posts only nonempty legs;
  `sparse_pairwise` sizes requests in its own pre-pass, with no active-leg estimate argument. Hybrid folds send AND
  receive summaries across every partition. Several peers or incompatible summaries retain every peer on the pairwise
  arm; that local fallback must NEVER switch transport to MPI_Alltoallv. Empty layouts post nothing on that same arm.
  Keep communicator-wide agreement/cache on the caller, not workers or a per-layer layout predicate. Do not introduce a
  new generator-derived replay plan. Run `flat_exchange_tests.cpp` dense, pairwise, dense-layout-on-pairwise and
  empty-layout cases after every transport edit, plus
  `hybrid_comm_derived_wire_plan_reads_every_partitions_row` and
  `hybrid_comm_derived_wire_plan_falls_back_to_every_peer` during coexistence. The latter needs at least four ranks.

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
  fixed-rank deterministic repeats, and near-cutoff retained-key comparisons. Include the incremental-axis and
  post-call-seed tests from `tests/test_circuit.py`: gradients must use the equivalent composed circuit's parameter
  order, including multi-monomial gates sharing one index. Cross-rank-count sums need not be bitwise
  equal; don't widen tolerances to pass them.

**Gate:** Existing serial API guarantees remain intact; new numerical association is documented and deterministic by
team size. Global retained term keys must agree, including near cutoffs and stored zeros; matching energies/gradients
alone is insufficient. A key-set change blocks the gate. Stop and report, never widen tolerances or change truncation.

### Task 10: Cut over configuration and API semantics to one store

**Prerequisite:** the post-Task-6 go/no-go passed, or the owner explicitly chose to continue. Reverting this task
later means reverting the API/configuration cutover; the partition runtime itself remains until Task 12.

**Files:**
- Modify: `cpp/include/monoprop/MonomialPropagator.h`, `cpp/monoprop/detail/monomial_propagator/MonomialPropagator.inl`,
  `cpp/monoprop/detail/EnvConfig.h`, `cpp/monoprop/detail/parallel/{ThreadBudget.h,ThreadBudget.cpp}`.
- Modify: `src/monoprop/bindings/binder.h`, `src/monoprop/monomial_propagator.py` docstrings.
- Create/extend: `cpp/tests/openmp_equivalence_tests.cpp`, `tests/test_openmp_config.py`,
  `cpp/tests/{env_config_tests,routing_tests,mpi_utils_tests,mpi_distributed_layer_equivalence}.cpp`,
  `cpp/tests/{mpi_fresh_insert_equivalence,unit_tests,ctor_validation_tests,hybrid_comm_tests}.cpp`,
  `cpp/tests/link_export_probe/link_export_probe.cpp`,
  `tests/{test_monoprop_smoke,test_parameter_validation,test_circuit}.py`.
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
  Linear rank mapping remains unchanged relative to the REFRESHED main baseline; S=1 removes only the partition hash.
  Preserve boolean mode/seed, `linear_requested()`, `make_router<NumModes>(R,1)`, router.linear_bits(), generator
  rank_shift and dest_from_shift peer selection. Keep the deterministic full-rank basis draw (up to 1024 attempts) and
  width bound log2(R)<=min(2*NumModes,64), where NumModes is the template width, not necessarily logical_num_modes_.
  The old transposed-plane/geometry-bit helper is gone; do not restore it. The pure routing::routes_pairwise predicate
  chooses transport for an agreed Config; its false result does not waive Router's invalid-geometry rejection.
  No virtual shards. Exports/load/reduction order can change under splitmix; validate global term/value maps
  and resulting per-rank peaks at equal R/T/routing settings. Candidate ownership never depends on T.
- [ ] Extend `routing_tests.cpp`, `mpi_distributed_layer_equivalence.cpp` and `mpi_fresh_insert_equivalence.cpp`:
  default linear success at R=1/2/4; explicit splitmix success at R=1/2/3/4. Record `monoprop_ROUTING`,
  `monoprop_ROUTE_SEED` and resolved `linear_bits()` as derived metadata, not a requested bit count. Separately assert
  `Router::for_modes<8>(3,1,true)` throws `UnroutableGeometry` and splitmix construction succeeds. Keep fresh-process
  default tests distinct from splitmix matrix runs; never expect an automatic fallback. Assert unique global ownership,
  generator-peer agreement, zero-payload participation, fixed-candidate ordering across budgets and global numerical
  equivalence. Do not require old/new rank-local maps equal for splitmix. Preserve current
  `routing_more_rank_bits_than_modes_is_refused`, `routing_basis_columns_are_independent_at_any_seed`,
  `routing_fibres_are_equal_sized`, `routing_dest_is_bit_identical_to_the_reference_bit_walk` and
  `routing_sparse_peer_plans_are_validated_before_use`. Run seed variations in separate processes because EnvConfig
  and the basis are cached; do not add environment reload to make tests convenient.
- [ ] Keep exact routing agreement on the caller: `routing::Config` carries raw mode, partitions and seed;
  `Pairwise.h::agree_routes_pairwise` reduces each value and its complement with MPI_MAX in one collective.
  `MPIUtils.h::check_routing_agreement` delegates to `mpi::routes_pairwise`, whose ordinary-MPI result is cached on
  that communicator with no attribute-copy callback. At cutover partitions=1 is fixed and budgets are not agreement
  inputs. Add ordinary-MPI tests in mpi_utils_tests.cpp: explicit matching Configs agree, differing valid mode/seed/part
  Configs throw routing::RoutingDisagreement on every rank, and cold/warm calls plus duplicated/freed/new communicators
  preserve independent agreement. Use explicit test Configs or fresh process environments, not production cache reset.
  Keep all ranks in the same cold-use sequence, including empty local stores. Do not reintroduce probabilistic digest
  agreement, per-gate allreduces or a cache keyed only by recycled MPI_Comm values.
- [ ] Preserve EnvConfig's separate route-seed parser: uint64 decimal via from_chars; unset/empty uses the default;
  signs, whitespace, trailing junk and overflow reject. Do not conflate it with the new present-empty thread error.
  Preserve COMMROUTE's ranks_per_coset/rank_cosets diagnostic and invalid-gate retry behavior. These describe reachable
  cosets, not observed idle ranks. Equal fibres over all possible monomials do not prove actual retained-term balance;
  per-rank load/peak measurements remain required.
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
- [ ] Preserve Python incremental graph construction exactly: build_graph(a); build_graph(b) denotes b+a in Heisenberg
  and a+b in Schrödinger. New Heisenberg blocks take low parameter indices and lift existing ones, without renumbering
  gate arrival IDs or splitting a multi-monomial gate's shared index. seed_parameters is on the POST-call axis; Python
  rotates it for the existing C++ build, then relabels the successful graph. C++ build_graph still takes its mapping
  as given. Keep this Python adaptation rather than shifting gates/parameters inside worker loops. Run these existing
  tests at budgets 1/2/3/4 in fresh launches, with no tolerance relaxation:
  `test_incremental_build_wires_the_picture_s_equivalent_circuit`,
  `test_incremental_heisenberg_moves_each_block_to_the_front_of_the_axis`,
  `test_incremental_reindex_keeps_a_multi_monomial_gate_on_one_index`,
  `test_incremental_build_wires_the_equivalent_circuit_for_pauli`, and
  `test_extend_seed_parameters_are_read_on_the_post_call_axis`. Compare decoded retained maps for seeded truncation
  as well as sizes/scalars; preserve failed-build retry and retained-functional invalidation coverage.
- [ ] Regenerate bindings via the configured generator. Preserve 32-mode template dispatch. Document
  low-level/source/ABI breaks and unspecified coefficient-order changes; compare decoded term/value mappings rather than
  sorted coefficients alone.

**Gate:** R/M API/configuration tests pass; one independent propagator across R ranks owns R packed stores, regardless
of T. Copies and retained snapshots/graphs obey the qualified ownership rule. Removed configuration/constructor surfaces
stay absent; invalid objects cannot be reused. Record the approved immediate compatibility breaks in the release ledger.

### Task 11: Measure parity and apply only bounded optimizations

**Files:** Task 1 benchmark files and, for bounded optimizations, the kernels below. The concurrent-index trial may
modify `cpp/monoprop/detail/operator/OperatorIndex.h`, its immediate MPOperator/Resolve/Engine callers and
`cpp/tests/{operator_index_tests,sparse_resolve_tests,simulator_copy_tests}.cpp`. Keep any adapter private and narrow;
no container-plugin framework. Do not rename `bench_*` functions or move benchmark files.

- [ ] Add metadata for `has_mpi`, MPI requested/provided thread support, requested/effective team size, OpenMP runtime,
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
- [ ] For required MPI-enabled cells, verify that rank masks and worker places stay within one NUMA domain and the
  reserved allocation, with no overlap. Required MPI-off full-machine cells and diagnostic MPI-enabled cross-NUMA L2a
  instead record the complete allocation and actual domain/core masks; do not falsely certify containment. In every
  case verify distinct physical-core use. These checks are measurement requirements, not library policy. For MPI
  deployments advise at least one process per allocated NUMA domain, with disjoint cores when several share a domain.
  Use actual topology, not a constant four domains/socket; partial-node runs cover only their allocated domains.
  Do not require one logical CPU per worker mask. Record idle/small-kernel serial fallback versus representative
  large-kernel actual team size separately. Unverifiable placement blocks benchmark evidence, not library construction.
- [ ] Compare old partitions vs final new threads at identical R,T and routing settings, using Task 1's frozen
  size-band-specific inventory and equal has_mpi. Small/medium cover the full sweep plus MPI-off full-machine operation;
  selected large profiles cover both full-node MPI decompositions and MPI-off full-machine operation. Do not add/remove
  required cells, change build modes or reinterpret profile sizes after seeing candidate performance.
  Freeze the production routing mode in that matrix; additionally require the random reference profile under explicit
  splitmix at R=3 and R=4, plus default linear at R=4 so ownership-dependent imbalance is measured, not only
  unit-tested. Run five alternating **fresh-process** observations per arm/cell; external repetition only,
  `--bench-rounds=1`. Include first-call/cold cache and separately measured repeated warm calls without overlapping
  object construction.
- [ ] Treat this comparison as the ownership decision recorded in Task 0: one store per rank against the sharded
  partition runtime, on the frozen campaign with all five gates. If a required cell still fails after the prescribed
  repetitions, the bounded optimizations and the index A/B below, keep the partition runtime and stop with a
  regression report. Task 12 does not start, and the owner decides whether to revise the design.

For required MPI-off full-machine observations, select the R (MPI-disabled) build on both arms and verify with
`uv run --no-sync python -c 'import monoprop, sys; sys.exit(int(monoprop.has_mpi))'` (exit 0 required).
Use Task 1's direct driver invocation without mpiexec. Set T=C with matching physical-core allocation and OpenMP places;
the baseline still declares both
legacy T variables. The shared overlay must avoid importing mpi4py.MPI even if it is installed. Record has_mpi=false
from the imported binary in timed/construction/validation/placement evidence, not just the requested build setting.
The usual five fresh observation pairs, numerical oracles and all five ratios apply, including selected large profiles.

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
  4. Serial index pre-reserve to reduce rehash (measure simultaneous old/new table capacity); the separate bounded
     concurrent-index trial below is also allowed, not assumed to win.
  5. Packed graph phases by **whole-word ownership** only: serial validate/count/allocate; each worker constructs and
     assigns one uint64 word, including slot seams and zero padding. No endpoint-wise `|=`. Test slots 63/2/65 and
     nonbinary fallback; preserve derived D permutation instead of adding persistent D indices.
  6. Inverted-index append only if measured dominant: bounded windows of 64-row-aligned blocks; per-block/per-column
     integer counts, serial column offsets/allocations, disjoint sparse element writes and whole dense-word ownership,
     serial handling of an unaligned first word, column-owned promotion after fill. Scratch O(window_blocks×2N), never
     full-height bitmap per worker. Tests must compare dense/sparse promotion, parity and sorted posting order against
     serial after every append.
- [ ] Each optional optimization gets its own failing regression tests, focused review and full affected parity rerun.
  Beyond the bounded index trial below, whole-batch streaming/windowing, a custom concurrent table, new transport,
  executor, rank redistribution, full per-thread stores or a persistent duplicate full-key map require a new design.

#### Index A/B: shared-serial versus shared-concurrent

Compare two build variants of the same one-store engine, with everything else identical:
- **shared-serial:** the current packed `OperatorIndex`, parallel frozen probes and serial publication.
- **shared-concurrent:** the same packed rows indexed by `boost::concurrent_flat_map` over `{row ID, cached hash}`
  handles, with bulk lookup and parallel publication of preassigned IDs.

The variants differ only inside `OperatorIndex` and at its publication call sites; shared invariant 14 keeps them
local. This is a development-only A/B between an existing container and the reference, not a container search or a
permanent backend selector. It needs authorized implementation/measurement budget. Task 0's in-library trial is the
starting point (archived overlay patch); its timings are not acceptance evidence.

- [ ] Prerequisite: Tasks 5–6 are complete, so traversal and lookup are parallel. With a serial scan (66–95% of one-core
  time in Task 0) the variants cannot be told apart at full machine.
- [ ] Build the variants from one development-branch compile-time option; never keep two resident indices. Run the full
  C++/Python suites on both during development. Task 0 found bitwise-identical term maps and energies, and one
  duplicate-key fixture that depends on the contract pinned by invariant 14.
- [ ] Boost: the owner waived compile/test verification against the 1.85 floor. Bulk visitation exists at 1.85.
  Container statistics (`BOOST_UNORDERED_ENABLE_STATS`) need Boost 1.86 or newer, are for diagnostic builds only, and
  serialize on one lock: never time them.
- [ ] Use `boost::concurrent_flat_map` as an index into the packed rows, with compact stored row references and
  heterogeneous position-query lookup. Full monomials, query-buffer spans and a second persistent reference index are
  forbidden as stored duplicate keys. Hash/equality must represent monomial identity, not compare row IDs as if they
  were term keys. Retain exact collision checks and cached query hashes.
- [ ] Preserve the batched lookup/result contract.
  - Use bulk visitation in chunks of at most 16 queries. The visitor receives only matched elements, in no documented
    order, so it only records `(cached hash, row ID)` pairs.
  - After the call and outside Boost's locks, match them to queries: a query alone with that hash in its chunk is the
    match; several such queries (duplicates or a collision) are separated by exact row comparison. Misses produce no
    call.
  - Never use side-effecting hash/equality, re-enter the container from a visitor, or assume callback order. A
    per-query visitor is an acceptable fallback.
  - Canonical row order comes from row IDs, not bucket iteration.
- [ ] Complete frozen probing and deterministic missing-ID assignment before any publication. Caller checks fixed32 and
  allocation limits, reserves rows/table, initializes packed rows and overflow entries, then workers publish only the
  index entries for preassigned IDs. No row growth, coefficient resize, overflow mutation, inverted-index mutation,
  graph packing or external-key relocation while map workers run. Indexed row contents remain immutable until the
  index is cleared/rebuilt. Join before dependent reads, structural maintenance or freeing scratch. Preserve remote
  leader → remote follower → deferred self leader → deferred self follower publication phase order and matched timing.
  Callbacks supplied by callers of `bulk_insert` (construction) are not assumed thread-safe: publish those serially.
- [ ] Keep one index per independent packed store. Copy/move/clone must bind row-aware hash/equality state to the
  destination's rows, never a destroyed source. Container synchronization does not make external-row lifetimes safe.
  Keep exceptions inside Workshare, join, then use existing caller invalidation and distributed-failure boundaries,
  including during active-ticket lifetime. No worker MPI or scheduling-based ID counter.
- [ ] Extend tests before adoption: deliberate hash collisions with distinct keys; repeated lookup queries; mixed
  hit/miss batches and exact row IDs at budgets 1/2/3/4; inline and spilled rows; reserve/growth boundaries; the pinned
  duplicate-key contract; clone lookup after source mutation/destruction; allocation/publication failure invalidation;
  and end-to-end term/graph equivalence at the same geometry. Add a worker-participation test on the variant path. Run
  qualified race checks without blanket suppressions.
- [ ] Measure both variants with the Task 1 harness on identical frozen inputs, allocation and placement: fresh-process
  pairs, alternating order, all five metrics, plus per-phase lookup/publication/reindex times and peaks. Cover at least:
  - L1 (R=1,T=1) cells: Task 0 measured shared-concurrent 15–18% slower on Hubbard/Pauli L1;
  - MPI-off full machine;
  - the all-new-terms random profiles, where serial publication is the concern;
  - the physics models (80–89% hits).

  No new sizes, higher memory allowance or acceptance waiver. Microbenchmark gains do not override end-to-end
  regressions. Obtain approval for any spend beyond the approved budget.
- [ ] Decide from the required cells, not from index-phase speedups. Record the choice, remove the losing variant and
  the build option before final acceptance, and rerun the affected gates on the final binary. If neither variant can
  pass, stop: the A/B is permission to compare an existing container, not to rescue the architecture at any cost.

**Gate:** All required performance/memory cells pass with attached raw observations and numerical/placement evidence.
The ownership decision and the index A/B outcome are recorded, and only the chosen variant remains.
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
  `cpp/monoprop/detail/mpi/{Comm,MPICompat,MPIUtils,Exchange,Pairwise,CheckedCount,Routing}.h`,
  `cpp/monoprop/detail/mpi/MPICompat.cpp`, `cpp/monoprop/MPFunctions.cpp`.
- Migrate/retire test files according to ledger below; update `cpp/tests/README.md` and architecture docs in the same
  change.

**PROPOSED interface:** Simplify `mpi::Comm` to the existing implicit MPI_Comm wrapper with `.mpi`; remove
Kind/Shm/Hybrid and associated pointers/constructors/dispatch. Keep non-MPI behavior, ordinary MPI, PeerPlan routing and
existing typed exchanges. Do not collapse it to a raw alias if that changes overload behavior unnecessarily.

- [ ] Start only after Task 11 parity, and only if Task 11's ownership decision confirmed the single store, under
  separately authorized implementation. This is the irreversible step: afterwards, returning to sharding means
  restoring the partition runtime, transport, topology code and hwloc dependency. The owner approved the immediate
  partition-related C++ extension/ABI break; confirm no runtime path constructs old groups before deleting definitions.
- [ ] Remove any remaining `partition_group_`, friendship and unused fan-out implementation. Confirm Task 10 already
  removed resolve_partition_count_, public partition/factory arguments and facade-only test users; remove remaining
  is_partition_facade/protected partition-only helpers. Keep `clone_` and virtual lifecycle/operator-update hooks. No
  compatibility facade that invokes old callbacks once with silently different semantics.
- [ ] Remove non-MPI-thread transport branches from utilities/tickets. Production router geometry is `(real_ranks,1)`.
  Preserve EXISTING boolean linear/splitmix default/rejection contract and generator shift/peer rules; no
  routing-default repair is needed. Keep ordinary MPI sparse query routing and sparse on-wire replay, sender order,
  count checks, communicator lifetime and initializing-thread guards (including external initialization).
  `MPIUtils.h::check_routing_agreement` still delegates to mpi::routes_pairwise; retain exact Config agreement,
  communicator attributes/no-copy lifetime semantics and fixed S=1, never per-rank OpenMP budgets. Preserve
  require_routable validation, the flat known_recv_counts width check, and WindowVec-only begin_alltoallv/wait_into.
  Keep cpp/tests/SlotBlocks.h as the test-only whole-world adapter where migrated tests use it; do not restore
  production vector-of-vectors overloads. Ticket/PendingAlltoallv move/drain semantics survive deletion. Keep both MPI
  and non-MPI
  implementations.
- [ ] With the last Hybrid caller removed, change MPICompat's requested and required minimum to MPI_THREAD_FUNNELED.
  Preserve MPI_Query_thread validation for externally initialized MPI and Task 3's caller guard even at higher levels.
  Test library-owned initialization; external FUNNELED, SERIALIZED and MULTIPLE; rejection of actual SINGLE;
  wrong-thread entry at higher levels; and valid caller execution with workers. Record requested/provided support
  and distinguish unavailable runtime scenarios from exercised cases. In Python subprocesses, configure mpi4py before
  importing MPI/monoprop; its default MULTIPLE is accepted, not downgraded. Audit finalization/cleanup and failure paths
  as well as hot kernels: no worker MPI calls, including MPI_Abort. Update diagnostics/docs/tests in the same task, then
  rerun MPI correctness/failure tests and affected parity cells on the final binary. Changing only Init_thread's
  constant does not complete this gate.
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
| plain-MPI half of `hybrid_comm_known_recv_counts_are_masked_through_the_plan` | preserve masked non-peer counts and sparse peer payload assertions; add short known-count rejection before posting |
| `hybrid_comm_unroutable_peer_plan_is_refused` | move ordinary-MPI invalid-shift/count-entry checks; retain routing_sparse_peer_plans_are_validated_before_use |
| `hybrid_comm_pending_alltoallv_owns_its_requests` | move ordinary-MPI move/noncopyable/destructor-drain coverage unchanged; extend active-destination move assignment with live buffers |
| `hybrid_comm_derived_wire_plan_reads_every_partitions_row` / `hybrid_comm_derived_wire_plan_falls_back_to_every_peer` | retire only partition-row folding; preserve empty, one-peer and multi-peer replay under ordinary MPI, including four-rank fixtures |
| `flat_exchange_tests.cpp` replay dense/pairwise/dense-layout/empty tests | retain and run under ordinary MPI after deletion; dense gate test is flat_exchange_dense_gate_always_takes_the_collective |
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
  `docs/content/docs/features/{parallelism,evaluation}.mdx`, `docs/content/docs/concepts/algorithm.mdx`,
  `cpp/tests/README.md`,
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
  validation and possible smaller OpenMP teams. Distinguish NUMA-contained required MPI cells, MPI-enabled cross-NUMA
  L2a diagnostics, and required MPI-off full-machine operation spanning domains. Show matching allocation, rank-binding
  and thread-place examples, including the one-core-rank-mask pitfall and direct MPI-off execution. Document one
  controlling host caller on MPI's initializing thread outside host OpenMP teams,
  no overlapping use of dependent state or host MPI calls, FUNNELED-or-higher support, caller checks, callable lifetime,
  enforced failed-object invalidation, accepted capacity reduction and numerical contracts. Placement verification for
  benchmarks does not add a topology dependency. For MPI deployments advise at least one process per allocated NUMA
  domain; do not tell MPI-off users to add unavailable MPI ranks to avoid regressions. Identify topology externally;
  production's common four-domains/socket layout is not universal. Distinguish the selected
  c8a.metal-24xl initial acceptance from pending Intel/AMD HPC qualification. Explicitly recommend against SMT workers,
  even when hardware SMT is enabled, and demonstrate launcher plus OpenMP settings for one worker per physical core.
  Multi-node performance qualification remains pending a proper HPC cluster/interconnect; do not claim it from EC2
  same-instance MPI tests.
- [ ] State exact immediate breaks: partitions/child_factory arguments, PartitionChildFactory, partition-only helpers,
  errors and environment controls removed without aliases; no replacement public num_threads argument. Old automatic
  physical-core selection, multi-rank serial default, library pinning and direct hwloc requirement are gone. OpenMP is
  mandatory even without MPI. Raw order/splitmix physical ownership may differ; ABI requires rebuild. Requests are not
  hardware-clamped. Record the stronger initializing-thread MPI caller rule as a compatibility break, even at supplied
  SERIALIZED/MULTIPLE. Preserve Python mathematical signatures and current incremental-axis/post-call seed semantics
  in the evaluation guide/docstrings. Keep Qiskit conversion and its optional-version tests unchanged; do not advertise
  those already-shipped additions as OpenMP work. Explain routing geometry and COMMROUTE cosets without claiming a
  balanced basis guarantees a balanced retained operator. Use `[Symbol][]` references in prose/docstrings,
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

- [ ] Task 0 evidence and the owner's architecture decision are archived; this conditional implementation and its
  MPI-off acceptance coverage match the reconciled spec/plan, not a superseded assumption of one store.
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
- [ ] Every required runtime and exact peak-memory cell passes Task 11, including actual MPI-off full-machine
  small/medium and selected large profiles. All five metrics and numerical checks apply; a NUMA-local win cannot
  excuse a full-machine regression. Unsupported or missing measurements are not marked passed.
- [ ] No legacy runtime, direct hwloc dependency, replacement topology/pinning service or scheduler remains. A selected
  Boost concurrent index passed its bounded tests and affected parity gates, with no duplicate operator/full-key store.
  MPI requests/requires FUNNELED and checks the initializing caller even at higher provided levels.
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
compatibility break. Task 0 permits the three standalone index variants and reopens the architecture with the owner;
Task 11 permits only the retained candidate's bounded integration trial. Neither silently authorizes production shards,
an unbounded/custom concurrent store, altered acceptance geometry, a weakened oracle or a waiver.
