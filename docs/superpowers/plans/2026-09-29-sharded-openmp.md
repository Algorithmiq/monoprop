# Sharded OpenMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for authorized inline execution, or
> superpowers:subagent-driven-development only when the operator explicitly authorizes delegation. Read this plan and
> its spec completely. Steps use checkboxes. A task approval does not authorize the next task, measurements or
> publication.

**Goal:** Replace the custom partition runtime with coarse-grained OpenMP execution while retaining sharded storage,
owner-local mutation and the frozen numerical/runtime/memory acceptance contracts.

**Architecture:** One rank-level propagator owns T internal shard states, one per OpenMP worker. A single team spans
each operation; explicit local phases and published buffers replace child-propagator dispatch and in-process
communicators. MPI's initializing thread is the OpenMP primary and owns physical-rank MPI calls inside that team.

**Tech Stack:** C++23, required OpenMP CXX, existing Boost and optional MPI, nanobind, Python 3.11+,
uv/scikit-build-core, Boost.Test and pytest. No replacement affinity/topology dependency.

**Spec:** [Sharded OpenMP execution design](../specs/2026-09-29-sharded-openmp-design.md).

## Status and authority

Planning revision at `a6a995a4352bc5a9494acc4c13df33aa61ef84ef`, dated 2026-09-29. The owner approved writing the
replacement documents, not executing this plan. No implementation, build, experiment, delegation, commit or publication
is authorized by these checkboxes. Review subsequent source differences before starting; never reset to a historical
pin.

Progress: S0 was separately authorized, executed and accepted by the owner on 2026-09-29 (outcome under Task S0).
S1 was separately authorized and implemented on 2026-09-29 (outcome under Task S1); owner acceptance is not recorded
here. S2 has not started and needs its own authorization.

This replaces the abandoned one-store Tasks 7–13 in the
[historical plan](2026-09-18-rank-local-openmp.md). Its Tasks 0–6 remain historical evidence, not an unexecuted queue.
Its Task 1 frozen inventory, artifacts and measurement protocol remain authoritative where imported below. New task IDs
are **S0–S8** to prevent confusion with that experiment. Task 6 is not the performance baseline.

## Global constraints

- C++23; GCC 14 / Clang 18 minimum; Python 3.11 minimum.
- OpenMP CXX is required, including MPI-off builds and installed consumers. MPI remains optional and OFF by default.
- Retain Linux x86_64, Linux aarch64 and macOS packaging coverage; report untested platforms as pending.
- Keep packed position-list rows, keyless indexing, sparse state, inverted indices and packed graph representation.
- Keep `TermIndex = std::uint32_t`, with its reserved sentinel, per shard. Retain checked count/allocation arithmetic.
- Preserve both bases and pictures, cutoffs, basis changes, propagation, graphs, gradients, paring, copies and updates.
- Remove the custom worker pool, in-process communicator abstractions, custom barriers and direct hwloc dependency.
- Allocation, placement and pinning belong to the launcher/OpenMP configuration. Recommend one worker per physical core.
- Numerical preservation and the frozen runtime/memory gates govern this refactor; neither is waived by a faster kernel.
- Fixed uniform P processes × T threads; T is captured at construction and is exactly the internal shard count.
- No T-consistency collectives/checks, reduced-team handling, resharding, hardware clamp or independent shard-count
  knob.
- No nested kernel parallelism initially. Preserve ascending-shard combination and existing MPI reduction association.
- No arithmetic reordering, new floating-point reductions or FMA-policy change during this refactor. These options
  remain open after landing, individually or together, under the explicitly separate follow-up at the end of this plan.
- Preserve immutable graph sharing and independent store copies. Raw root accessors work only at T=1.
- C++ naming/style/Doxygen, Python docstrings, generated-file restrictions and API links follow `AGENTS.md`.
- Use traversal for query-producing bitmap work, and prefix-sum for cumulative offsets. Do not call both a scan.
- Build/artifact isolation is mandatory; do not alter baseline environments, archived evidence or unrelated checkout
  data.
- Before an authorized push, run `prek run --all-files` and relevant tests, and fix every finding the change
  introduces. Standing owner waiver (recorded at S0 acceptance, 2026-09-29), valid for the whole execution of this
  plan: ignore the pre-existing `ruff check` findings in `docs/notebooks/chemistry/chemistry.ipynb`,
  `docs/notebooks/fermi_hubbard/fermi_hubbard.ipynb` and `docs/notebooks/kicked_ising/kicked_ising.ipynb` (49 at
  S0). Do not edit those notebooks to silence them. The waiver covers no other file, hook or finding; report any new
  finding in those notebooks, or any change in their count, rather than treating it as waived.

## Evidence pins and scope

The preserved production baseline is `90d57177c2b0cd93503f88dd931d2f8dd409aa23`. Baseline tools were subsequently
synchronized by overlay without rebuilding its binaries. At this planning checkpoint the local frozen inputs match:

| File | SHA-256 |
| --- | --- |
| `tools/rank-local-openmp-campaign.json` | `ffde870c77a66c8ce37fb2cc07b39abef976e9d7938b7cea59f43cb407de068a` |
| `tools/rank-local-openmp-workloads.json` | `508b65f1f0eb2171fc4fbd79feba2381054c6dcdd38a614600f5b957ab814416` |
| `tools/benchmark-rank-local-openmp.py` | `2207529fb45454990dfba3ad0e1cf944ba1ae1d15385489aef4b16a365426697` |
| `CONTEXT.md` | `1eaf2d92f23f3d0f349e281363e846f7baa5235415f20117abcad40eb5249d66` |

The recorded remote baseline index is `/home/ubuntu/task1-artifacts/formal/baseline-index.json`, hash
`82479d6ff83deb3113306ac1caa1720d25f15a400cbc89b5693f85ffad47fff5`. Its identity and raw artifacts have not been
independently reverified during this redesign. Reverify them during separately authorized preflight; do not reconstruct
or overwrite the formal baseline. Preserve the whole `/home/ubuntu/task1-artifacts/` tree and later task reports.

The 250-cell inventory and five independent <=1.00 median gates remain: runtime, operation peak sum/max, construction
peak sum/max. Initial observations use five distinct fresh-process pairs; failing cells get the specified additional
five. Numerical validation is separate and includes global retained operators. No workload resizing, cell dropping,
geometric-mean masking, memory allowance or substitution of MPI-enabled P=1 for actual MPI-off operation.

Task 6's full-machine failure motivates two early architecture checkpoints, S4 and S5. Neither is a formal campaign or
permission to spend the old 180-minute budget. Each experiment requires its own written scope, elapsed/per-trial/RAM
limits, allocation/provenance plan and owner approval. Budgets never roll over. Keep diagnostics and formal results
separate. A checkpoint failure stops the plan for an owner decision; it does not silently move optimization into S6.

## Module and source map

New private names below are proposed interfaces, not claims about existing source. Install private headers reachable
from public templates using the existing CMake header/export mechanism. Avoid a broad source reorganization.

| Path | Responsibility |
| --- | --- |
| New `cpp/monoprop/detail/sharded/Team.h` | Operation-scoped team and checkpoint-stable exception publication; no queue, threads or transport ownership |
| New `cpp/monoprop/detail/sharded/State.h` | Stable shard addresses and owner-initialized/copyable mathematical state |
| New `cpp/monoprop/detail/sharded/Construction.h` | Rank-level gate/pass orchestration over shard-local construction phases |
| New `cpp/monoprop/detail/sharded/Evaluation.h` and `Evaluation.cpp` | Replay, reverse pass and ordered rank-local result combination |
| New `cpp/monoprop/detail/sharded/Exchange.h` and `Exchange.cpp` | Physical-MPI staging/request ownership only; no in-process collective facade |
| New `cpp/monoprop/detail/sharded/CMakeLists.txt` | Source/header registration, including exports needed by installed template consumers |
| `cpp/monoprop/detail/evolution/layer_build/{Engine,Resolve,QueryWire,Scan,FusedApply}.h` | Preserve owner-local algorithms; separate pure phases from communication; read-only input views |
| `cpp/monoprop/{Evolution,MPFunctions}.cpp` | Extract reusable local replay/derivative operations and scratch without changing formulas |
| `cpp/include/monoprop/MonomialPropagator.h`, `cpp/monoprop/detail/monomial_propagator/MonomialPropagator.inl` | Rank-level state, public operations, lifecycle, aggregates and exports |
| `cpp/monoprop/detail/mpi/{Comm,Exchange,MPICompat,MPIUtils,OperationFailure,Pairwise,Routing}.h` | Real-MPI ownership, routing cache and failure contracts; remove fake communicator dispatch at cutover |
| `cpp/monoprop/detail/mpi/MPICompat.cpp`, `cpp/monoprop/detail/EnvConfig.h` | Initialization/thread support and removal of partition configuration |
| `cpp/tests/PropagatorTestAccess.h` | Private shard inspection, actual-phase observers and failure injection |
| New `cpp/tests/sharded_{team,state,construction,evaluation,exchange}_tests.cpp` | Focused flat Boost tests for new seams |
| New `tests/test_sharded_openmp.py` | Process-isolated launch geometry, full API and exporter integration |
| `cpp/tests/CMakeLists.txt`, `justfile`, `.github/workflows/test.yml` | Separate T=1 internals and fixed-T sharded test launches |

Existing high-risk seams to inspect before editing:

- `Engine.h::run_exchange`, `resolve_self_queries`, `insert_deferred_self_misses`, `finish`, `build_layer`.
- `GraphSink` / `ContractSink`, `IncomingProbe`, `SelfQueryStage`, `SlotWindow` and the Task 6 checked decoder.
- `Evolution.cpp::CrossRankExchangeHandle`, `begin_flat_exchange`, `wait_flat_exchange`, the two exchange finish paths.
- `MPFunctions.cpp::prepare_evolved_operator`, `ev`, `ev_and_grad` and `eval_scratch`.
- `MonomialPropagator.inl::run_operation_`, constructors/copy, `evolved_operator_terms`, functionals and updates.
- `PartitionGroup.h`, `ShmComm.h`, `HybridComm.h`, `PartitionBarrier.h`: algorithm/order reference, not a runtime to
  wrap.

## Shared correctness contracts

1. A store and its overflow map have one mutation owner at a time. Reserving capacity does not make concurrent
   `set_positions` safe. Different stores may mutate concurrently; lazy caches belong to their owner too.
2. At fixed P/T preserve owner mapping, source/query order, per-shard row IDs and graph records. Keep different local
   shards distinct from self; combine local and remote sender blocks in ascending flat-owner order.
3. Publication order is cross-owner leaders, cross-owner followers, deferred self leaders, deferred self followers.
   Match followers against the completed leader pass. Keep the source boundary distinct from final `scaled_count`.
4. XOR with a fixed generator gives distinct partners for distinct sources. Preserve the existing duplicate-key behavior
   rather than silently deduplicating; see invariant 14 in the historical plan and `operator_index_tests.cpp`.
5. Queries remain compact. Keep absolute position offsets, cached hashes, checked wire boundaries/position narrowing and
   shard-local ID/sentinel limits. Query ordinals remain `size_t`. Do not restore a rank-wide single-store capacity
   limit.
6. Send and endpoint-snapshot buffers are immutable through consumption. MPI request storage and buffer addresses do not
   move after posting. Requests outlive region exit; failure handling precedes any destructor that could wait on a peer.
7. All supported workers reach the same checkpoints. Failure decisions are stable for a checkpoint generation; a fast
   worker's next-phase failure cannot make a slower worker branch out of the previous phase. Empty shards participate.
8. Preserve self-pair layout (`k`, `k+count/2`), signs, cosine masks/tails, `LayerAngle`, record/restore order and
   derivative `A=cos_acc(...)*sec`, `-g*(sin*(A-ep.cos_terms)-ep.sin_terms)`. Keep sparse/duplicate dot and scatter
   semantics.
9. No new arithmetic association in S0–S8. Keep `kVanishingCos=0x1p-20`, `kRecordSpreadBits=53.0`, the contraction
   workaround and existing tolerances. Preserve Python incremental axes and post-call seeds in both pictures/bases.
10. Validate full retained maps, including zeros/near-cutoff rows. Replicated Heisenberg identity agrees and is included
    once. Duplicate nonidentity ownership or nonfinite values fail. General map comparison remains
    `abs(a-b) <= 1e-9 + 1e-7*max(abs(a),abs(b))`; stricter existing tests and unrounded scalar/gradient checks remain.

## Verification commands and task handoffs

During authorized execution, use fresh R (MPI OFF) and M (MPI ON) environments/builds, outside baseline directories.
The following is a build/test recipe, not an instruction to execute while reviewing the plan. Set absolute `ARTIFACTS`
to the approved task directory. A stable installed Ninja avoids the expired build-isolation Ninja path in old caches.

```bash
: "${ARTIFACTS:?Set the approved absolute task artifact directory}"
export R_ENV="$ARTIFACTS/venv-r" R_BUILD="$ARTIFACTS/build-r"
export monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
monoprop_ENABLE_MPI=OFF UV_PROJECT_ENVIRONMENT="$R_ENV" \
  uv sync --all-extras --group workspace-test --reinstall-package monoprop --no-cache -v \
  --config-settings-package="monoprop:build-dir=$R_BUILD" \
  --config-settings-package="monoprop:cmake.define.CMAKE_MAKE_PROGRAM=$(command -v ninja)"
cmake --build "$R_BUILD" --target monoprop_unit_tests.x
ctest --test-dir "$R_BUILD" --show-only
ctest --test-dir "$R_BUILD" --output-on-failure --no-tests=error
UV_PROJECT_ENVIRONMENT="$R_ENV" uv run --no-sync python -c \
  'import monoprop, sys; sys.exit(int(monoprop.has_mpi))'
UV_PROJECT_ENVIRONMENT="$R_ENV" uv run --no-sync pytest
```

For M, use distinct `M_ENV`/`M_BUILD` and `monoprop_ENABLE_MPI=ON` in the build environment; verify `has_mpi` is true.
Use the documented `just test-mpi` routes, adapted to the isolated tree when necessary. MPI-off tooling must not import
`mpi4py.MPI`. Keep `--no-sync` when running an existing build. Verify executable/extension/library paths, hashes,
mtimes, actual linkage and regenerated nonempty Boost discovery; `uv sync` alone does not prove the C++ binary was
relinked.
New sources/tests must be registered/reconfigured before claiming a missing-test RED or a passing run.

Once sharded test registration exists, launch each supported T in a separate process; do not change T inside a test:

```bash
monoprop_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
  ctest --test-dir "$R_BUILD" --output-on-failure --no-tests=error -R '^sharded_'
monoprop_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE \
  UV_PROJECT_ENVIRONMENT="$R_ENV" uv run --no-sync pytest tests/test_sharded_openmp.py -v
```

These commands require an allocation suitable for four workers. Ordinary low-level tests stay at T=1. A helper-level
limited-team test may remain for `for_blocks`, but it does not define a supported sharded-operation launch.
Run ASan/UBSan separately from TSan. Qualify Clang/libomp/Archer with known-safe/racy probes; unsupported libgomp TSan
is not race-free evidence. Observe actual workers inside real phases; assert only after joining, not from worker bodies.

Every task ends with a written handoff outside the checkout: source/build identity, changed interfaces, actual RED/GREEN
commands and outcomes, preservation tests, request/memory/ordering review, pending platforms and the next authorization
boundary. A passing old fixture is preservation evidence, not an invented RED. Commits require separate authority and
repository message/trailer/signing policy; no task here contains an automatic git-write step.

## Task S0: A fixed-team, phase-safe execution primitive

**Deliverable:** independently tested operation-scoped team/checkpoint code, with no propagator or transport migration.

**Files:** create `detail/sharded/Team.h`, its `CMakeLists.txt` and `cpp/tests/sharded_team_tests.cpp`; register through
`cpp/monoprop/CMakeLists.txt` and `cpp/tests/CMakeLists.txt`; clarify `detail/parallel/Options.h` documentation.
Paths under `detail/` here and below are under `cpp/monoprop/`.

**Interfaces:** in `monoprop::detail::sharded`, introduce `TeamFailure` with a constructor taking `size_t threads`,
`record(size_t shard, std::exception_ptr error) noexcept -> void`, `checkpoint() noexcept -> bool`, and
`first_error() const noexcept -> std::exception_ptr`. One error slot per worker; `first_error` selects the lowest
failing shard after joining. Add these templates:

```cpp
template <class Fn>
auto run_team(parallel::Options options, Fn &&body) -> std::exception_ptr;

template <class Fn>
auto phase(TeamFailure &failure, size_t shard, Fn &&body) noexcept -> bool;
```

`run_team` constructs error storage before entering `omp parallel num_threads(options.threads)`, calls the noexcept
body with `(size_t shard, TeamFailure&)`, then returns the error without rethrowing. `phase` catches its local body,
records its own slot and returns the collective checkpoint decision. Every phase result is checked; false ends the
common sequence without invoking later phase bodies. No MPI, configuration check or topology query.

**Outcome:** implemented and **accepted by the owner on 2026-09-29**. Start: `f31174c`. Handoff with the
contract, the barrier/generation/join argument, commands, toolchains and full evidence:
`/home/ubuntu/s0-artifacts/HANDOFF.md` (outside the checkout).

- Interface as specified, plus a `TeamBody` concept that requires a noexcept orchestration body; `run_team` rejects a
  nonpositive budget before the region. The checkpoint uses `omp masked` (the non-deprecated `master`).
- RED: the registered test failed to compile only on the missing `monoprop/detail/sharded/Team.h`. GREEN: 17 flat
  `sharded_team_*` cases in fresh `sharded_team_env_t{1,2,4}` and `_t4_passive` launches observed 1/2/4 distinct
  workers; GCC/libgomp and Clang 18/libomp R (413/413), GCC ASan/UBSan (413/413), GCC M with Open MPI 5.0.10
  (459/459, compatibility only), R pytest (767 passed, 25 skipped). `for_blocks` preservation passed under dynamic and
  thread-limited teams.
- 13 `Team.h` mutants: 12 fail specific assertions (an asynchronously changing stop flag fails the generation case at
  T>=2 only); a failing worker that skips the checkpoint shows only as a bounded 120 s timeout.
- Clang/libomp/Archer TSan qualified with known-safe/racy probes; the real tests are silent, and the missing-leading-
  barrier mutant is reported in `TeamFailure::record`. The installed `find_package_smoke` consumer instantiates
  `Team.h` through `monoprop::monoprop` for the R, M and Clang packages.
- Pending: macOS, Linux aarch64, wheels, Nix and minimum-version compiler routes.

- [x] Add a real-team test and run it before implementation; the absent helper is the initial compile RED:

```cpp
namespace parallel = monoprop::detail::parallel;
namespace sharded = monoprop::detail::sharded;

BOOST_AUTO_TEST_CASE(sharded_team_visits_each_owner) {
    const auto options = parallel::capture_thread_budget();
    auto visits = std::vector<int>(static_cast<size_t>(options.threads), 0);
    const auto error = sharded::run_team(options, [&](size_t shard, sharded::TeamFailure &failure) noexcept {
        sharded::phase(failure, shard, [&] { ++visits[shard]; });
    });
    BOOST_CHECK(!error);
    for (const auto count : visits) {
        BOOST_CHECK_EQUAL(count, 1);
    }
}
```

- [x] Implement a checkpoint-stable decision, not a single asynchronously changing stop flag. A valid initial shape
  has owner-written `errors_` and primary-written `proceed_`:

```cpp
auto TeamFailure::checkpoint() noexcept -> bool {
#pragma omp barrier
#pragma omp master
    {
        proceed_ = std::none_of(errors_.begin(), errors_.end(), [](const auto &error) {
            return static_cast<bool>(error);
        });
    }
#pragma omp barrier
    return proceed_;
}
```

  The next decision cannot be written until every worker reaches the next leading barrier. Integrate these boundaries
  with handoff needs instead of layering an unrelated failure protocol on top. Add Qt-style declarations/member docs.
  Clarify `Options` as a requested budget: the reduced-team guarantee belongs to `for_blocks`, not every consumer.
  The sharded team requires exactly T; this documentation change must not add actual-team checks or weaken the generic
  helper's existing behavior.
- [x] Test T=1/2/4, empty work, nonprimary failure, simultaneous failures, primary-only work, skipped later mutation and
  delayed-worker checkpoint stability. Add a mutation check that removes synchronization/actual work and makes the
  corresponding tests fail; use bounded test timeouts rather than treating a hang as success.
- [x] Run the focused R tests with `-R '^sharded_team_'`, qualified sanitizer probes and installed-header compilation.
  Keep whole-body callbacks noexcept; every potentially throwing operation belongs inside a protected phase.
- [x] Hand off the helper and barrier/lifetime argument. Do not add MPI or start an experiment in this task.

## Task S1: Owner-initialized shard state and copy semantics

**Deliverable:** internal state can be seeded, copied, counted and destroyed without a pool or child propagators.

**Files:** create `detail/sharded/State.h`, `cpp/tests/sharded_state_tests.cpp`; adapt initialization/copy extraction in
`MonomialPropagator.inl`, `detail/operator/MPOperator.h` and private `PropagatorTestAccess.h` where needed. Do not
switch the public runtime yet.

**Interfaces:** `ShardState<NumModes>` contains the existing `MPOperator<NumModes>`, `MPGraph` and `MatchedEpochSet`
state, not another `MonomialPropagator`. Use stable `std::unique_ptr` addresses. Define
`Shards<NumModes> = std::vector<std::unique_ptr<ShardState<NumModes>>>` and a construction helper:

```cpp
template <size_t NumModes, class Initialize>
auto make_shards(parallel::Options options, Initialize &&initialize) -> Shards<NumModes>;
```

The initializer consumes `size_t shard` and returns `std::unique_ptr<ShardState<NumModes>>`; it contains only
owner-local initialization. Caller-side configuration/router/MPI setup precedes it. New state and source copies
preserve captured T.

**Outcome:** implemented on 2026-09-29; **owner acceptance pending**. Start: `4cdffac`. Handoff with the
interfaces, seeding/ownership/copy/cleanup arguments, commands, toolchains, memory model and full evidence:
`/home/ubuntu/s1-artifacts/HANDOFF.md` (outside the checkout).

- `detail/sharded/State.h`: `ShardState` (the existing `MPOperator`, `MPGraph` and `MatchedEpochSet`), `Shards`,
  `make_shards`, `seed_shards`, `copy_shards`, and per-shard `total_size`/`operator_memory_usage`/
  `graph_memory_usage`. Caller-side preparation (`validate_initial_operator`, which returns the identity core term,
  `paired_basis_bounds`, `packed_inline_width`, `OperatorSeed`) is separate from the owner-local `seed_operator`. The
  legacy single-store constructor now seeds through the same `seed_operator`; `MPOperator::initialize_caches` holds
  the extracted Heisenberg-sparse/Schrödinger-dense warm-up. A defaulted test-only `NoShardObserver` is called
  inside each owner's seed and copy body. Arguments are checked before the team, including `threads < 1` before any
  container size; there is no actual-team or cross-rank check.
- `MPOperator()` was `noexcept` although its store initializer allocates. RED: a real injected allocation failure
  hit `std::terminate`; removing only `noexcept` made the sweeps pass.
- RED: the registered test failed to compile only on the missing `State.h`. GREEN: 19 flat `sharded_state_*` cases
  in fresh `sharded_state_env_t{1,2,4}` launches (16/19/19 cases). They check a field-by-field partition oracle at
  (1,T), plus an independent test-side reference, because the oracle now shares `seed_operator`. They also check flat
  ownership at P=1/2/4 linear and P=3 splitmix, identity roles, empty owners, cache selection and pending entries,
  exact and independent copies with shared layer cores, the captured budget and the real owner guard (preservation).
  Failure cases cover primary, nonprimary and concurrent owners. Allocation-failure sweeps fail every site of the
  seed, copy and legacy-constructor paths, with no leak and the source left intact. The test-only
  `cpp/tests/AllocationProbe.cpp` replaces global new/delete in the unit-test executable and is compiled out under
  Clang sanitizers and GCC TSan.
- 11 `State.h`/`MPOperator.h` mutants (lost ownership, caller-side copy, lost copy mapping, wrong flat owner,
  identity per shard, global reserve, swallowed errors, leaked siblings, store theft, Heisenberg densification,
  restored `noexcept`) are all detected, with no hangs. A materialize-the-basis-per-owner mutant would be
  state-identical and is guarded by review only.
- Results:
  - GCC/libgomp R: 435/435 and pytest 767 passed, 25 skipped.
  - Clang 18/libomp R: 435/435.
  - GCC ASan/UBSan: 435/435.
  - GCC M with Open MPI 5.0.10: 465/465 + 16/16 and pytest 821 passed, 32 skipped (compatibility only).
  - Clang/libomp/Archer TSan, qualified with known-safe/racy probes: silent on the real cases, and a publication-race
    mutant is reported.
  - The installed `find_package_smoke` consumer compiles `State.h` alone (chain g) through `monoprop::monoprop` for
    the R, M and Clang packages.
- Pending: macOS, Linux aarch64, wheels, Nix, minimum-version compiler routes and actual P>1 sharded runs (S5).
  Public runtime integration is S4 work.

- [x] Add fixtures for each basis/picture, empty owners, replicated identity, source mutation after copy, invalid-source
  rejection and constructor failure. Verify the union of shard keys matches the existing partition oracle at fixed P/T.
  For each owner, the test checks `router.dest<NumModes>(key)` against its flat owner ID.
- [x] Implement first-touch with only the pointer vector allocated on the caller; the protected owner body allocates the
  substantial state. The helper's central pattern is:

```cpp
auto shards = Shards<NumModes>(static_cast<size_t>(options.threads));
const auto error = run_team(options, [&](size_t shard, TeamFailure &failure) noexcept {
    phase(failure, shard, [&] { shards[shard] = initialize(shard); });
});
if (error) {
    std::rethrow_exception(error);
}
return shards;
```

  Caller distributed guards must still abort peers on failed distributed initialization. No requests may be owned by
  this helper when it rethrows. Audit allocating `noexcept` constructors, including the current default `MPOperator`
  constructor, before claiming catchable allocation-failure coverage.
- [x] Extract seeding rather than copying full child constructors. Preserve hash ownership, streamed Schrödinger paired
  basis generation, per-shard reserve and lazy-cache ownership. Reuse existing deep-store-copy and
  immutable-core-sharing semantics. Preserve rank-level virtual clone/update hooks; do not recreate
  `PartitionChildFactory` privately.
- [x] Run `-R '^sharded_state_'` at supported separate launches; prove stores are distinct, owner allocation occurs in
  the owner body, copied budget is unchanged and earlier independent copies survive invalidation of another instance.
- [x] Hand off memory accounting for state, graph sharing and retained scratch. No global operator mirror or benchmark.

## Task S2: Direct-buffer construction and graph-free propagation

**Deliverable:** a complete P=1, T-shard construction/propagation engine with testable local phase boundaries.

**Files:** create `detail/sharded/Construction.h`, `cpp/tests/sharded_construction_tests.cpp`; extract phases from
`detail/evolution/layer_build/Engine.h`; adapt `Resolve.h`, `QueryWire.h`, `Scan.h` and `FusedApply.h` only as needed.
Retain `sparse_query_tests.cpp`, `sparse_resolve_tests.cpp`, `operator_index_tests.cpp`, `majorana_cutoff_tests.cpp`,
`pauli_build_layer_tests.cpp` and fused tests as preservation coverage.

**Interfaces:** retain `LayerBuildEngine`, `GraphSink` and `ContractSink`, but move blocking transport out of the owner
engine. Existing legacy calls may use an adapter while the old runtime is retained. Introduce these owner-local methods
on `LayerBuildEngine<NumModes, Sink, Observer>`; `Response` is `typename Sink::Response`:

```cpp
auto prepare_exchange(bool leaders,
                      mpi::WindowVec<VecZ> &&queries,
                      mpi::WindowVec<std::vector<size_t>> &&sources,
                      mpi::WindowVec<std::vector<double>> &&values,
                      SelfQueryStage<NumModes> &&self_stage) -> const mpi::WindowVec<VecZ> &;
auto resolve_published(const mpi::WindowVec<std::span<const size_t>> &incoming, bool leaders)
    -> mpi::WindowVec<std::vector<Response>>;
auto consume_published(const mpi::WindowVec<std::span<const Response>> &responses) -> void;
```

`prepare_exchange` performs the existing stable follower filtering/self resolution and returns the sink's outgoing
buffer. `resolve_published` performs checked decode/probe/scatter/publication within one shard. `consume_published`
preserves response order and leader marks. Existing `finish(CosMask&&, CosMask*)` remains the finalization seam.
These methods perform no MPI, barrier or nested worksharing; `SlotWindow` stays a data-layout type, not a communicator.

- [ ] Add differential fixtures with same-shard, other-local-shard and empty destinations; mixed hits/misses; all-hit,
  all-miss, fresh growth/reindex; Plain/Fused forms; both sinks; and fixed-geometry row/graph/query/response equality.
  A genuine RED must show missing sharded behavior/owner participation, not merely rename passing serial assertions.
- [ ] Split `run_exchange` at the above boundaries and adapt incoming containers to immutable spans without duplicating
  whole local query buffers. `QueryWire::WireView` already exists; preserve its checks and generalize the remaining
  owning-container callers. Use `window.indices()`/`window.slot(wi)`/`at_slot(flat)` correctly.
- [ ] Implement leader then follower orchestration with published views. The pass order is executable structure:

```text
owner traversal -> publish outgoing -> resolve destination-owned queries -> publish responses
-> consume responses -> complete leader marks -> repeat for followers
-> insert deferred self misses -> finalize graph/fused updates
```

  Each arrow is a lifetime/synchronization boundary, not a new local collective API. Fixed source order must match
  `ShmComm`'s historical order. Pass serial `parallel::Options{}` to reused within-shard kernels.
- [ ] Keep same-shard position stages unencoded, bounded self lookup and deferred hashes. Preserve scatter before
  insertion, graph versus fused response types, Schrödinger scoring, cosine tails/pivots and paired cutoff exceptions.
  Retain the malformed `QueryWire<128>` Plain fixture `VecZ{size_t{0x8037e}}` and checked overflow tests.
- [ ] Test throw sites at traversal, decode, publication and finalization. Observers record actual shard owners in those
  bodies; no empty-region proxy. Test opaque callback/basis-change supported paths without assuming thread safety.
- [ ] Run `-R '^(sharded_construction_|sparse_|operator_index_|fused_|pauli_)'` plus relevant cutoff tests, verifying
  the selection is nonempty and broad enough. Hand off exact fixed-geometry/global-map evidence and buffer live ranges.

## Task S3: Snapshot-safe replay, energy, gradients and paring

**Deliverable:** complete P=1 sharded evaluation, including reverse derivatives and retained/pared functionals, without
cross-shard live-coefficient reads or per-kernel OpenMP regions.

**Files:** create `detail/sharded/Evaluation.h`, `Evaluation.cpp`, `cpp/tests/sharded_evaluation_tests.cpp`; extract
local functions from `Evolution.cpp`, `MPFunctions.cpp`, `detail/evolution/CosineRecompute.h` and callback plumbing.
Integrate graph views from `detail/pare/PareGraph.cpp`; preserve `combined_recompute_equivalence.cpp`,
`pare_graph_tests.cpp`, `evolution_detail_tests.cpp` and `tests/test_infinite_cutoff.py`.

**Interfaces:** use one `EvalRequest` and `detail::CosCallbacks` per shard with this rank-level detail interface:

```cpp
auto ev_sharded(std::span<const EvalRequest> requests,
                std::span<const detail::CosCallbacks> callbacks,
                parallel::Options options, mpi::Comm comm) -> double;
auto ev_and_grad_sharded(std::span<const EvalRequest> requests,
                         std::span<const detail::CosCallbacks> callbacks,
                         parallel::Options options, mpi::Comm comm) -> std::pair<double, VecD>;
```

These own one team for the entire call. Initially the physical communicator has size one; S5 adds remote legs without
changing callers. Requests/callbacks are prebuilt views with T entries and no shard-count setter. Keep existing exported
single-store functions as low-level compatibility/reference paths where appropriate; new detail functions need exports
if public template instantiation calls them across the shared-library boundary.

- [ ] Add exact small shard-pair fixtures where one owner overwrites a coefficient before another consumes its old
  endpoint; only a published snapshot can pass. Cover empty owners, identity once, sparse/duplicate dot/scatter and
  repeated parameter indices, then finite-difference/reference gradients and pared/unpared functionals.
- [ ] Extract pure owner-local operations from the existing replay/derivative begin/apply/finish functions. Publish only
  needed endpoint values, not a full new operator mirror. Preserve this ordering:

```text
snapshot/pack all required endpoints -> publish -> independent self/local work
-> consume peer snapshots -> cosine/record/restore phases in their original order
-> local dot/gradient contributions -> ascending-shard fold -> scalar/vector result
```

  Keep each self pair together; no adjacent-pair reinterpretation. Forward and derivative scratch must stay valid
  across phases. Reuse legitimate mathematical snapshots; account for retained TLS versus per-instance scratch.
- [ ] Implement ordered local combination. For every result position, the reference fold is:

```cpp
auto total = 0.0;
for (size_t shard = 0; shard < contributions.size(); ++shard) {
    total += contributions[shard];
}
```

  Add core/identity only once using existing semantics. Do not parallelize this fold with a new floating-point
  association, change `std::fma` policy or remove `[[gnu::noinline]] apply_fused_record_range` without neutrality proof.
- [ ] Verify record thresholds, vanishing cosine, restoration after derivative calls, signs in both pictures/bases,
  basis changes, incremental graph axes, repeated evaluation, copies, updates and functional owner lifetimes.
- [ ] Run `-R '^(sharded_evaluation_|pare_|combined_|evolution_)'`, the named Python numerical suites and qualified
  sanitizers. Compare retained operators as well as energy/gradient; performance work is not part of this task.

## Task S4: Integrated MPI-off prototype and first architecture checkpoint

**Deliverable:** public mathematical operations use the new P=1 sharded engine in an isolated development build, with
credible end-to-end numerical/memory/runtime evidence or an explicit stop decision.

**Files:** `MonomialPropagator.h`, `MonomialPropagator.inl`, `detail/sharded/{State,Construction,Evaluation}.h`,
`src/monoprop/bindings/binder.h`, `cpp/monoprop/CMakeLists.txt`, `cpp/tests/PropagatorTestAccess.h`;
create `tests/test_sharded_openmp.py`; adapt `tools/benchmark-rank-local-openmp.py`,
`tests/test_rank_local_openmp_benchmark.py` and
`packages/monoprop-bench-tools/src/monoprop_bench_tools/{models,preflight,bmf}.py` for candidate construction.

**Interfaces:** connect the S1 state, S2 engine and S3 evaluation to existing root build/propagate/functional/update
entry points. A temporary compile-time `monoprop_SHARDED_OPENMP_PROTOTYPE` option selects the integrated candidate;
normal builds retain the legacy path until cutover. This is development instrumentation, not a public runtime setting
or permanent backend selector. Old constructor controls may remain in the legacy build during coexistence, but cannot
select a different shard count in the candidate. Explicit nondefault legacy controls must not silently be honored there.

- [ ] Add separate-process T=1/2/4 Python cases through the real public API. One concrete core fixture is:

```python
from pathlib import Path

import pytest

from monoprop import MajoranaPropagator
from tests.cases import load_problem


def test_sharded_propagation_matches_graph() -> None:
    problem = load_problem(Path(__file__).parent / "data" / "random_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    graph_sim, direct_sim = [
        MajoranaPropagator(
            problem.operator, problem.monomial_circuit.initial_state, cutoff=2 * problem.n_modes
        )
        for _ in range(2)
    ]
    graph_sim.build_graph(circuit)
    direct_sim.propagate(circuit)
    assert graph_sim.expectation_value(circuit) == pytest.approx(
        direct_sim.expectation_value(), rel=0, abs=1e-12
    )
```

  This is preservation coverage; sharded ownership/participation needs its own genuine RED. Also compare decoded full
  operators, unrounded energies and every gradient component. Across T, use subprocess output maps rather than changing
  configuration mid-process.
- [ ] Integrate rank-level state and validity, count/memory aggregation and direct shard iteration in
  `evolved_operator_terms()`. Pair each shard index with its own evolved vector; do not use root `indexing()` at T>1.
  Include retained zeros, identity de-duplication and count/export failure tests before timing any result.
- [ ] Record a migration ledger for all old facade/one-store tests: kept at T=1, rewritten shard-aware, or retired with
  equivalent coverage. Prove actual participation inside construction, publication, replay and derivative phases.
  Audit trial selection from binary/build provenance, not `--runtime-shape openmp` or environment labels alone.
- [ ] Run integrated R correctness, installed-consumer smoke and M-with-P=1 correctness. Public P>1 candidate support is
  not claimed before S5. Update the candidate builder to use the new root constructor instead of `partitions=1`;
  retain the legacy partition builder for the baseline. Keep `observe`, `validate`, `compare` and the
  allocation-before-import fix. Record/synchronize any approved tool overlay without rebuilding baseline binaries or
  rewriting formal records. If an adapter/schema change cannot consume the frozen evidence honestly, stop for review.
- [ ] **Stop for experiment approval.** Propose matched baseline/candidate single-thread controls and actual MPI-off
  full-machine reference cells spanning build_graph, propagate, energy and gradients, both bases/pictures. Use frozen
  builders/parameters and operation/construction memory windows. Specify repetitions, task/per-trial/RAM caps and
  first-touch/placement evidence before running; do not reuse Task 6's allowance.
- [ ] If authorized, run through the existing driver against preserved partitions, not just candidate T=1. Report every
  runtime and memory cell plus full numerical validation. Single-kernel speedup is not the checkpoint. Archive results
  outside the checkout; retain natural first-touch effects as architectural evidence.
- [ ] Obtain the owner's written proceed/rework/stop decision before S5. Failed parity needs a bounded proposed remedy,
  not an automatic move to runtime deletion. The later frozen campaign is still mandatory after an early pass.

## Task S5: Physical-MPI exchange and the MPI+OpenMP checkpoint

**Deliverable:** supported P×T candidate with primary-only MPI, owner-parallel payload work and failure-safe requests.

**Files:** create `detail/sharded/Exchange.h`, `Exchange.cpp`, `cpp/tests/sharded_exchange_tests.cpp`; extend S2/S3
orchestration; adapt `detail/mpi/{Pairwise,Routing,MPIUtils,OperationFailure}.h`, `MPICompat.cpp`,
`cpp/tests/mpi_failure_driver.cpp`, `run_mpi_failure_scenario.cmake`, `mpi_fresh_insert_equivalence.cpp` and
`mpi_distributed_layer_equivalence.cpp`.

**Interfaces:** a move-only physical exchange owns real-MPI request objects and stable staging buffers outside the team.
Expose physical post/wait operations to the primary and disjoint payload views to owners, not fake collective methods
on a shard. Reuse `begin_flat_exchange` / `wait_flat_exchange` and existing pending-request semantics when extracting
this code. The S2 owner-phase and S3 rank-evaluation signatures remain unchanged.

- [ ] Add two-process tests with T=1/2/4 for mixed local/remote destinations and canonical sender ordering. Check that
  local sender blocks do not precede lower-numbered remote senders. Verify response source/query alignment, empty
  messages, nonzero slot windows and exact pre-update endpoint values.
- [ ] Define one staging layout with checked counts and prefix-sum offsets for `(source shard, destination shard)`
  segments per physical peer. Workers write disjoint payload slices; the primary exchanges metadata/posts/waits and
  publishes received extents; destination owners decode and mutate. Never serialize all payload processing on shard 0.
- [ ] Preserve construction's linear peer window and replay's actual multi-peer summaries. Warm/use communicator-agreed
  `routes_pairwise` on the caller/primary, including MPI attribute accesses. Keep mode/seed agreement and remove the
  partition-count agreement field rather than replacing it with T agreement. No geometry-repair fallback.
- [ ] Implement request lifetime explicitly. The owner of `pending` and every referenced buffer surrounds the team:

```text
allocate operation frame and request owners
run_team: protected owner/primary phases, post -> independent work -> wait -> dependent work
inspect returned error while request owners are still alive
if error: invalidate after mutation; multi-rank abort, otherwise rethrow
only then permit normal request/buffer destruction
```

  A primary post may fail after some requests became live; those requests still belong to the outer frame. Do not
  return through a draining local destructor before reaching distributed failure handling.
- [ ] Adapt failure scenarios for worker throw, before-exchange throw, active-ticket throw, malformed receive and
  wrong-thread entry. Supervised two-process runs must finish in failure before the 30-second timeout; timeout fails the
  test. Single-process rethrow/invalidation and independent-copy survival must also pass. Test actual MPI support, not
  merely the requested level. Keep SERIALIZED until S8 removes executable legacy workers.
- [ ] Run linear supported geometries and splitmix with three processes, numerical/full-map comparisons and qualified
  MPI sanitizer runs. Assert MPI occurs only on the initializing primary; no arbitrary `omp single` winner may post.
- [ ] **Stop for a separate bounded measurement approval.** Propose matched MPI-only, MPI+OpenMP and single-thread
  controls plus relevant full-machine cells. Use the same partition baseline and all numerical/memory surfaces. Obtain
  an owner proceed/rework/stop decision before API cutover; do not claim multi-node qualification from ordinary EC2
  links.

## Task S6: Final public surface, exports and test migration

**Deliverable:** candidate has the complete intended interface and coverage; no test depends on a one-store facade
at T>1.

**Files:** `MonomialPropagator.h`, `MonomialPropagator.inl`, `src/monoprop/bindings/binder.h`,
`src/monoprop/monomial_propagator.py`, `detail/EnvConfig.h`, `cpp/tests/{unit_tests,partition_equivalence_tests,
openmp_runtime_tests,fused_cos_sweep_tests,mpi_fresh_insert_equivalence,mpi_failure_driver}.cpp`,
`cpp/tests/PropagatorTestAccess.h`, `cpp/tests/link_export_probe/link_export_probe.cpp`, `cpp/tests/CMakeLists.txt`,
`justfile`, `.github/workflows/test.yml`, `tests/test_openmp_config.py`; documentation below.

**Interfaces:** no `partitions`, `child_factory`, `PartitionChildFactory` or `monoprop_PARTITIONS` reader in the
candidate.
No replacement constructor/setter. Single-shard raw accessor rejection uses launch-time T, not a user shard selector.
Mathematical operations, virtual clone/update hooks, aggregate reporting and decoded exports retain their contracts.

- [ ] Add compile/API tests proving old explicit partition/factory signatures are absent from the candidate, Python
  construction has no new thread/shard argument, and budget parsing/capture/copy use the existing environment contract.
- [ ] Implement single-shard access with an explicit T>1 exception and a T=1 remedy. Do not fabricate raw merged
  storage. Update the old `MultiPartitionUnsupported` diagnostic/type as part of the documented C++ break.
- [ ] Migrate the audit's real call sites and tests, preserving coverage. Export through shard indices/coefficient
  vectors; fixed-geometry graph-layout tests inspect shards privately. Cross-geometry tests compare full retained maps.
  Retain actual-participation, capacity, malformed-input, fresh-insertion, copy/clone/update and failure-lifetime tests.
- [ ] Remove the test main's `monoprop_PARTITIONS=off` assignment; ordinary C++/Python tests launch at T=1. Register
  dedicated fixed-T launches without `ScopedBudget` mutation in integration tests. Do not use `OMP_THREAD_LIMIT<T` to
  test the unsupported sharded configuration, or make all tests serial to avoid migration.
- [ ] Update `README.md` when relevant, `docs/content/docs/features/parallelism.mdx`,
  `docs/content/docs/building.mdx`, `cpp/tests/README.md`, `benches/LADDER.md` and `CONTEXT.md` if terminology needs
  clarification. Explain required OpenMP, launch-before-import configuration, fixed geometry, single-shard accessors,
  removed extension surface, NUMA/physical-core allocation and new test commands. Update docstrings/API links.
- [ ] Regenerate bindings/dispatch through the supported build, never by editing generated files. Run full R/M C++ and
  Python suites and installed consumers; rebuild all ABI-coupled artifacts together. Close every migration-ledger row.
- [ ] Hand off the source/API coverage audit and pending platforms. Runtime deletion and formal measurements require
  their following gates; this task alone does not establish parity or publication readiness.

## Task S7: Frozen full-library parity before removal

**Deliverable:** a complete, machine-readable acceptance report for the integrated candidate, or a blocking failure.

**Files:** ordinarily no production changes. Use `tools/benchmark-rank-local-openmp.py`, the two frozen JSON inputs,
`tests/test_rank_local_openmp_benchmark.py` and existing bench-tools. Put campaign artifacts outside the checkout.

- [ ] Obtain campaign approval with a fresh budget. Reverify frozen file/index identities, baseline binary/environment
  preservation and the actual current target allocation. c8a.metal-24xl previously had one NUMA domain and 96 physical
  cores; verify rather than infer. No SMT-worker campaign or automatic provisioning.
- [ ] Before timings, complete the candidate's compiler/build correctness matrix, full retained-map export checks and
  actual-worker/placement evidence. Assert actual R/M build identity. Both arms use the fixed global workloads and
  original timing boundaries; candidate selection needs more than its requested thread count.
- [ ] Follow historical Task 1's existing `observe`, `validate`, `compare` schema and distinct timed/construction
  processes. Collect five fresh pairs per required cell, then the specified five more for failing cells. Keep all
  samples and numerical validation. Frozen baseline data cannot be overwritten/recollected to improve a comparison;
  any additional prescribed observations are separately identified records under the approved protocol.
- [ ] Check all five per-cell median ratios <=1.00. Reject missing cells in both arms, duplicated/reused runs, extra or
  mismatched sample identities, mismatched provenance, absent validation and inexact required memory windows. Report
  operation, construction and diagnostic outer peaks separately; historical residual peak growth grants no allowance.
- [ ] If a gate fails, profile only with separately authorized time/RAM. The permitted refactor-preserving remedies are
  ownership/buffer reuse, redundant copies, packing/layout overhead, barrier placement, cache locality and phase
  balance.
  Keep payload/source order and numerical policy fixed. No hidden backend, new concurrent-index study, smaller workload,
  restored one-store plan or unapproved tuning campaign.
- [ ] Stop on unresolved regressions. A pass permits an owner decision about S8, not an automatic deletion/commit/push.
  Record any platform/HPC qualification still pending separately from the target's frozen acceptance scope.

## Task S8: Remove legacy machinery, qualify the final binary and close out

**Deliverable:** one sharded OpenMP runtime, correct FUNNELED support and verified final packaging without direct hwloc.

**Files:** remove `detail/partition/PartitionGroup.h`, `CpuTopology.h`, `CpuTopology.cpp` and obsolete partition CMake
registration; remove `detail/mpi/{ShmComm,HybridComm,PartitionBarrier}.h` and runtime-specific tests after their
coverage has migrated. Update `detail/mpi/{Comm,MPICompat,MPIUtils,Exchange}.h`, `MPICompat.cpp`, parent CMake files,
`cmake/monopropConfig.cmake.in`, `nix/monoprop.nix`, `nix/devshell.nix`, `.github/workflows/nix.yml`,
`pyproject.toml`, installed smoke tests and docs as applicable. Audit other tracked dependency references before
editing; do not touch unrelated caches.

- [ ] Add/activate host-FUNNELED acceptance and insufficient-actual-support tests before lowering initialization/support
  requirements. Test initializing-thread ownership inside the library team and wrong-host-thread rejection. Ensure no
  old worker path remains executable before claiming FUNNELED.
- [ ] Remove facade dispatch, custom barriers/queues/thread pinning, local collective tags and the temporary development
  selector. Keep real-MPI communication and source-order data types. Search confirms no production references remain:

```bash
rg -n 'PartitionGroup|ShmComm|HybridComm|PartitionBarrier|PartitionChildFactory|monoprop_PARTITIONS' \
  cpp src CMakeLists.txt cmake pyproject.toml justfile .github
rg -n 'hwloc|pkg.?config|SHARDED_OPENMP_PROTOTYPE' \
  CMakeLists.txt cpp cmake pyproject.toml .github
```

  Explain legitimate negative fixtures/history; production remnants fail the audit. Delete direct hwloc discovery,
  headers, link/export flags and native/wheel/Nix build inputs. An MPI/OpenMP runtime's transitive hwloc is allowed.
- [ ] Preserve PUBLIC `OpenMP::OpenMP_CXX` on object and shared targets plus installed
  `find_dependency(OpenMP REQUIRED COMPONENTS CXX)`. Run `just test-find-package` against each installed R/M package,
  link-export probes, wheel-repair imports and relevant Nix/platform builds. Record unavailable qualification as
  pending.
- [ ] Rebuild the final candidate, rerun full R/M C++/Python tests, installed consumers, failure drivers, stress and
  qualified sanitizers. Check ordinary aggregate/export semantics and raw-accessor rejection again on the final binary.
- [ ] Obtain any required final-measurement budget and rerun affected parity on the final binary. If removal changes
  measured code/configuration, all affected frozen cells need new candidate evidence; do not relabel pre-removal hashes
  as final. Retain the baseline and original formal evidence.
- [ ] Self-review the diff against the spec and record unresolved numerical/concurrency/MPI concerns. Independent review
  requires separate authorization to delegate; do not silently spawn reviewers. Before any authorized push, run
  `prek run --all-files` and relevant tests and fix findings; the only waiver is the standing notebook Ruff waiver in
  the global constraints.
- [ ] Present final acceptance evidence and residual platform/HPC risks to the owner. Only separately authorized
  integration/publication closes the refactor. Do not start the following numerical experiments automatically.

## Explicit post-landing follow-up: numerical performance policy

**Deferred boundary:** after this refactor has landed, with separate scope and approval. This is not Task S9 and is not
a prerequisite for S0–S8. The superseded one-store Task 9 is not revived.

The owner explicitly wants to leave open performance improvements involving any of:

1. Arithmetic operation ordering within local kernels or derivative/replay calculations.
2. The order or method used to combine shard contributions.
3. Floating-point reduction schemes, including alternatives to the initial ascending-shard fold.
4. FMA policy, including explicit FMA and compiler contraction choices.

Investigations may change these **individually or in combination**. A combination may be valuable even when one of its
components is not independently faster. The no-reordering/no-new-reduction/no-new-FMA rules above isolate the refactor;
they do not permanently prohibit these later changes or require every future implementation to be bit-identical to it.

For a future approved study, retain the landed refactor as the control, state the numerical/reproducibility contract,
choose matched end-to-end runtime/memory/numerical cases and bound the budget before running. Validate retained keys and
coefficients, energies, full gradients, near-cutoff/cancellation behavior, record/restoration and compiler/FMA behavior.
Record component and combined variants where useful, without demanding each component individually win. Do not silently
weaken tolerances or change accepted semantics to declare a speedup. Adoption requires explicit review of the measured
benefit and numerical policy, with updated tests/docs; keeping the option open grants no automatic execution authority.
