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
here. S2 was separately authorized and implemented on 2026-09-30 (outcome under Task S2); owner acceptance is not
recorded here. S3 was separately authorized and implemented on 2026-09-30 (outcome under Task S3); owner acceptance
is not recorded here. S4 was separately authorized, implemented on 2026-10-01 and accepted by the owner on 2026-10-01
(outcome under Task S4). Its architecture checkpoint ran on 2026-10-01; the owner decided **rework**, and a first
profiling-driven rework was implemented the same day. The checkpoint re-run on the reworked code (2026-10-01) still
failed; after profiling the owner decided a second **rework** (C, D, E), implemented on 2026-10-02. Its checkpoint
re-run (2026-10-02) left small deviations, which the owner accepted on 2026-10-02 as reasonable, to be re-evaluated at
a later stage (see S4's outcome). That is the owner's **proceed** decision for the S4 checkpoint. S5 was separately
authorized and implemented on 2026-10-02 (outcome under Task S5); its MPI+OpenMP checkpoint (tier A) ran the same day,
and the owner accepted S5 as done on 2026-10-02, with the remaining deviations and the optimization opportunities
recorded under S5 for later. S6 was separately authorized, implemented and accepted by the owner on 2026-10-03
(outcome under Task S6). Bounded diagnostics and one optimization followed before S7 (recorded after S6's checklist):
a re-check of the deviating checkpoint cells, B-replay (owner-parallel replay layout, landed) and B-memory (staging
change rejected). The owner decided to run S7 with the remaining MPI-off memory cells recorded as known deviations.
S7 was separately authorized and ran on 2026-10-05/06 (outcome under Task S7): evidence complete, strict parity not
demonstrated (102 of 250 cells pass all five gates). Four owner-authorized remedy rounds followed on 2026-10-06/07
(recorded after S7's checklist); their kept source changes landed without a new campaign. The owner decided on
2026-10-07 to keep glibc malloc: no allocator change. A further owner-authorized round (A, 2026-10-07, recorded after
the remedy rounds) fixed the two largest remaining failure classes and a regression of the remedy rounds. S7 has not
been re-run on the remedied source. On 2026-10-07 the owner waived the S7-pass prerequisite for S8: S8 may proceed
without a passing pre-removal campaign, and one parity campaign on S8's final binary replaces an S7 re-run. The final
acceptance gates are unchanged. S8 was separately authorized on 2026-10-07; its cutover is implemented and locally
verified (outcome under Task S8). Its optimization points were taken up in owner-approved rounds on 2026-10-07/08 and
the kept changes verified; the owner had the work committed and pushed on 2026-10-08. S8's final parity campaign
awaits separate authorization.

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

  The next decision cannot be written until every worker reaches the next leading barrier. (Superseded by the second
  S4 rework, fix C: one barrier per checkpoint with a generation-stamped first failure; see `Team.h`.) Integrate these boundaries
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

**Outcome:** implemented on 2026-09-30; **owner acceptance pending**. Start: `a56daac`. Handoff with the
interfaces, phase sequence, buffer-lifetime table, ordering/numerical/failure arguments, memory accounting, commands,
toolchains and full evidence: `/home/ubuntu/s2-artifacts/HANDOFF.md` (outside the checkout).

- `detail/sharded/Construction.h`: `build_graph` and `propagate` over P=1, T shards in one `run_team` for the whole
  gate loop, with `ConstructionContext`, `GraphCircuit`/`PropagationCircuit`, the test-only
  `NoConstructionObserver` and `gather_published`. Argument errors throw before the team. Phase failures are returned
  after the join as `ConstructionOutcome{error, mutation_started}`, never rethrown. `mutation_started` is false only
  if the pre-mutation frame phase failed; there is no rollback.
- Per gate: traversal and leader prepare, then leader resolution, then leader consume and follower prepare, then
  follower resolution, then follower consume and finish/apply. That is five checkpoints for an exchanging gate; an
  identity gate skips the exchange, and T=1 keeps same-shard resolution only. Destinations read other shards'
  published blocks as spans in ascending flat-source order. Opaque cutoffs and basis-change closures traverse every
  shard on the primary, in an exclusive phase that every worker enters. Within-shard kernels get serial `Options{}`.
- Engine: `prepare_exchange`, `resolve_published`, `consume_published`, the extracted `scan_gate` and
  `stamp_layer_metadata`. The communicator left the engine: the legacy transport is the `run_exchange(comm, plan, …)`
  adapter over the same phases, with unchanged request lifetimes and guards. The resolver, the response path and the
  sink callbacks take read-only views (`mpi::views_of`). `MPOperator::current_picture` and
  `extend_from_current_picture` were extracted, and the propagator delegates to them.
- `QueryWire::Writer::put`/`flush` were `noexcept` but call `push_back`. RED: an injected allocation failure while
  encoding a query hit `std::terminate`; removing only `noexcept` made the sweeps pass. This also fixes the legacy
  path.
- RED: the registered test failed to compile only on the missing `Construction.h`. GREEN: 17 flat
  `sharded_construction_*` cases in fresh `sharded_construction_env_t{1,2,4}` launches (15/17/17 cases). At (1,T),
  every shard's rows, IDs, coefficients, caches and graph layers are bit-identical to the legacy partition child, and
  so are the query/source/value/answer streams of every pass. Because both paths share the phases, the cases also
  check an independent insertion-order reference, an independent coefficient-map propagator, the frozen
  `random_exact` energy and the cross-T global retained maps. They also cover participation, empty and identity
  inputs, the span seam (nonzero windows, empty senders, 255/256/257 queries, narrow and wide positions, the
  malformed `QueryWire<128>` fixture), self streams beyond 4096, throws in every phase, kernel and publication
  failures, and allocation sweeps.
- 11 mutants were all detected, with no hangs. The shared-engine order and follower-filter mutants pass the legacy
  comparison and are caught only by the independent references. A race mutant is reported by TSan/Archer.
- Results:
  - GCC/libgomp R: 455/455, the fixed-T launches 10/10, pytest 767 passed, 25 skipped.
  - Clang 18/libomp R: 455/455.
  - GCC ASan/UBSan: 455/455.
  - GCC M with Open MPI 5.0.10: 485/485 + 16/16 (every `mpi_failure_*` scenario), pytest 821 passed, 32 skipped
    (legacy compatibility only).
  - Clang/libomp/Archer TSan, qualified with known-safe/racy probes: silent on `sharded_*` at T=1/2/4 and on
    `openmp_*`.
  - The installed `find_package_smoke` consumer compiles `Construction.h` alone (chain h) through
    `monoprop::monoprop` for the R, M and Clang packages.
- Not in S2: coefficient-informed `build_graph`, replay, energy, gradients and paring (S3), rank-level integration
  (S4) and P>1 (S5). Allocation injection is unavailable under Clang sanitizers and GCC TSan. Pending: macOS, Linux
  aarch64, wheels, Nix and minimum-version compiler routes.

- [x] Add differential fixtures with same-shard, other-local-shard and empty destinations; mixed hits/misses; all-hit,
  all-miss, fresh growth/reindex; Plain/Fused forms; both sinks; and fixed-geometry row/graph/query/response equality.
  A genuine RED must show missing sharded behavior/owner participation, not merely rename passing serial assertions.
- [x] Split `run_exchange` at the above boundaries and adapt incoming containers to immutable spans without duplicating
  whole local query buffers. `QueryWire::WireView` already exists; preserve its checks and generalize the remaining
  owning-container callers. Use `window.indices()`/`window.slot(wi)`/`at_slot(flat)` correctly.
- [x] Implement leader then follower orchestration with published views. The pass order is executable structure:

```text
owner traversal -> publish outgoing -> resolve destination-owned queries -> publish responses
-> consume responses -> complete leader marks -> repeat for followers
-> insert deferred self misses -> finalize graph/fused updates
```

  Each arrow is a lifetime/synchronization boundary, not a new local collective API. Fixed source order must match
  `ShmComm`'s historical order. Pass serial `parallel::Options{}` to reused within-shard kernels.
- [x] Keep same-shard position stages unencoded, bounded self lookup and deferred hashes. Preserve scatter before
  insertion, graph versus fused response types, Schrödinger scoring, cosine tails/pivots and paired cutoff exceptions.
  Retain the malformed `QueryWire<128>` Plain fixture `VecZ{size_t{0x8037e}}` and checked overflow tests.
- [x] Test throw sites at traversal, decode, publication and finalization. Observers record actual shard owners in those
  bodies; no empty-region proxy. Test opaque callback/basis-change supported paths without assuming thread safety.
- [x] Run `-R '^(sharded_construction_|sparse_|operator_index_|fused_|pauli_)'` plus relevant cutoff tests, verifying
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

**Outcome:** implemented on 2026-09-30; **owner acceptance pending**. Start: `0c2503b`. Handoff with the
interfaces, phase and buffer-lifetime tables, ordering/numerical/callback/failure arguments, memory accounting,
commands, toolchains and full evidence: `/home/ubuntu/s3-artifacts/HANDOFF.md` (outside the checkout).

- `detail/sharded/Evaluation.h`/`.cpp`: `ev_sharded` and `ev_and_grad_sharded` (one team for forward and reverse;
  the communicator is touched only on the caller, and a phase failure reaches `mpi::operation_failed` before the
  reduction), the S4 seam `evaluate_shards` (per-shard terms and gradients plus the error, returned after the join),
  `combine_contributions`/`combine_gradients` (ascending-shard fold), `replay_shards` (partial contraction),
  exported owner-local forward phases with `replay_forward_in_team`, and `prepare_retained`/`RetainedEvaluation`
  for retained and pared functionals. `Construction.h` gains coefficient-informed `build_graph_informed` (seed
  replay and per-gate new-layer replay in the same team). `detail::pare_graph_owner` takes the flat owner;
  `detail::make_cos_callbacks` and `full_cos_mask` moved to `CosineRecompute.h`; `CosCallbacks::owner_parallel`
  marks library-built callbacks (anything else runs on the primary in an exclusive phase).
- Each replay step publishes pre-cosine endpoint snapshots into a double buffer: one checkpoint per step, partners
  never read live coefficients. The kernels were extracted into the library-internal
  `detail/evolution/LayerReplay.h`, shared with the rewired legacy `Evolution.cpp`/`MPFunctions.cpp`.
- Contraction fidelity: the inlined legacy derivative fuses `cos_acc(...)*sec` into `A - ep.cos_terms`. A first
  out-of-line derivative changed legacy gradient bits versus HEAD in record-bearing cases; the final form keeps the
  legacy code inline and gives the sharded path the raw accumulation. A cross-build public-API probe shows legacy
  results bit-identical to HEAD (GCC, Clang, MPI) and the sharded results equal legacy bitwise.
- RED: the registered test failed to compile only on the missing `Evaluation.h`. GREEN: 20 flat
  `sharded_evaluation_*` cases in fresh `sharded_evaluation_env_t{1,2,4}` launches: bitwise equality with the legacy
  partitions (per-shard evolved operators and terms, energies, every gradient component, pared functionals, partial
  contraction in both pictures, informed builds); independent map-propagator, frozen-energy, closed-form and
  finite-difference references including the `test_deep_circuit_gradient.py` cases; layouts, identity, records,
  lifetimes, participation, opaque callbacks and failures including allocation sweeps.
- 14 mutants (missing snapshots, wrong publication buffer, self pairing, records, restore, contraction, fold order,
  identity per shard, empty pruned mask, opaque on owners, informed replay, swallowed error) are all detected, with
  no hangs; a single-buffer race mutant is reported by TSan/Archer.
- Results:
  - GCC/libgomp R: 478/478 and pytest 767 passed, 25 skipped.
  - Clang 18/libomp R: 478/478.
  - GCC ASan/UBSan: 478/478, no reports.
  - GCC M with Open MPI 5.0.10: 508/508 + 16/16 (every `mpi_failure_*` scenario) and pytest 821 passed, 32 skipped
    (legacy compatibility only).
  - Clang/libomp/Archer TSan, qualified with known-safe/racy probes: silent on `sharded_*` at T=1/2/4 and on
    `openmp_*`.
  - The installed `find_package_smoke` consumer (chain i) calls both exported evaluators through
    `monoprop::monoprop` for the R, M and Clang packages.
- Also reformatted three S2 files that were not clang-format clean at `0c2503b` (whitespace only).
- Pending: macOS, Linux aarch64, wheels, Nix, minimum-version compiler routes, the Python ASan leg and P>1 sharded
  runs (S5). Rank-level wiring of these seams into `MonomialPropagator` is S4 work.

- [x] Add exact small shard-pair fixtures where one owner overwrites a coefficient before another consumes its old
  endpoint; only a published snapshot can pass. Cover empty owners, identity once, sparse/duplicate dot/scatter and
  repeated parameter indices, then finite-difference/reference gradients and pared/unpared functionals.
- [x] Extract pure owner-local operations from the existing replay/derivative begin/apply/finish functions. Publish only
  needed endpoint values, not a full new operator mirror. Preserve this ordering:

```text
snapshot/pack all required endpoints -> publish -> independent self/local work
-> consume peer snapshots -> cosine/record/restore phases in their original order
-> local dot/gradient contributions -> ascending-shard fold -> scalar/vector result
```

  Keep each self pair together; no adjacent-pair reinterpretation. Forward and derivative scratch must stay valid
  across phases. Reuse legitimate mathematical snapshots; account for retained TLS versus per-instance scratch.
- [x] Implement ordered local combination. For every result position, the reference fold is:

```cpp
auto total = 0.0;
for (size_t shard = 0; shard < contributions.size(); ++shard) {
    total += contributions[shard];
}
```

  Add core/identity only once using existing semantics. Do not parallelize this fold with a new floating-point
  association, change `std::fma` policy or remove `[[gnu::noinline]] apply_fused_record_range` without neutrality proof.
- [x] Verify record thresholds, vanishing cosine, restoration after derivative calls, signs in both pictures/bases,
  basis changes, incremental graph axes, repeated evaluation, copies, updates and functional owner lifetimes.
- [x] Run `-R '^(sharded_evaluation_|pare_|combined_|evolution_)'`, the named Python numerical suites and qualified
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

**Outcome:** implemented and **accepted by the owner on 2026-10-01**. Start: `f347e6f`. Handoff with the
integration map, ownership/routing/failure arguments, commands, toolchains, memory accounting, migration ledger and
full evidence: `/home/ubuntu/s4-artifacts/HANDOFF.md`; experiment proposal (not run):
`/home/ubuntu/s4-artifacts/EXPERIMENT-PROPOSAL.md` (both outside the checkout).

- `monoprop_SHARDED_OPENMP_PROTOTYPE` (default OFF) is a PUBLIC definition of `monoprop-objs` and an INTERFACE
  definition of the exported target, so bindings, tests and installed consumers inherit it; `monopropConfig.cmake`
  reports it. ON builds define the root in `detail/monomial_propagator/ShardedPropagator.inl`; the class then has no
  `mp_op_`, `graph_`, matched marks or partition group, so leftover root-storage use cannot compile. The root owns
  configuration, the captured budget (T threads = T shards, copies keep it), a (1, T) router prepared once on the caller
  (`make_router`, never derived from the communicator), the replicated identity, the epoch, validity and T shards.
- Every public operation goes through S1-S3: seeding/copying on owners, `build_graph`/`build_graph_informed` (seed
  angles as `contract_partially` computes them)/`propagate`, `prepare_retained` + `evaluate_shards` for direct and
  retained evaluation (two operation regions for direct evaluation, constant in depth), `replay_shards` for partial
  contraction (blocks in shard order), per-shard export in `evolved_operator_terms()`, owner phases for
  initial-operator updates (routed by the same router), remapping and empty-parameter pictures. Settings stay
  root-only. Outcome mutation flags are transferred before rethrow under `run_operation_`; failures after a team
  invalidate the root and go to `mpi::operation_failed`.
- Rejected, before any mutation or collective: a communicator of more than one rank (no legacy fallback), nondefault
  `partitions` or `child_factory`, any `monoprop_PARTITIONS`. Raw `mp_op()`/`indexing()`/`graph()`/`graph_data()`
  return the sole shard at T = 1 and raise `MultiPartitionUnsupported` with a launch-time T = 1 remedy otherwise; this
  stays the design (S6 renames the exception). `_core.__runtime_identity__` plus an embedded literal identify the
  runtime a binary contains.
- RED: with only the option wired, the ON build still ran legacy (explicit partitions and `monoprop_PARTITIONS`
  accepted, legacy accessor remedy): 16 Python failures and failing `sharded_root_env_*`. GREEN: 20 `sharded_root_*`
  cases in `sharded_root_env_t{1,2,4}` (private shard inspection, actual-worker observation and fault injection via
  `PropagatorTestAccess` and the test-only `detail/sharded/RootObserver.h`), and `tests/test_sharded_openmp.py` in
  fresh fixed-T processes: bitwise equality with a separate legacy build at `monoprop_PARTITIONS=T` for 8 fixtures at
  T = 1/2/4 (energies, gradients, functionals, full decoded maps, contraction blocks, every aggregate and memory field)
  and cross-T full maps, energies and gradients within tolerance.
- 10 S4 mutants (wrong-shard export, omitted owner update, missing invalidation, repeated identity, lost epoch guard,
  wrong-shard picture, functional without invalidation, remap on shard 0, caller-side seeding, legacy selection) are
  all detected; a shared-shard race mutant is reported by TSan/Archer.
- Results (final diff `86ff189b…`):
  - GCC/libgomp: legacy 478/478, pytest 806 passed; prototype 372/372, pytest 856 passed at T = 1 and the full
    suite at T = 2 and 4.
  - GCC + Open MPI 5.0.10: prototype 400/400 plus the two-rank rejection; legacy 508/508 + 16/16 (all
    `mpi_failure_*`), pytest at one and two ranks.
  - Clang 18/libomp: 372/372 and 478/478. GCC ASan/UBSan (prototype): 372/372, no reports. Clang/libomp/Archer TSan,
    qualified with known-safe/racy probes: silent on `sharded_root_*` at T = 1/2/4 and on `openmp_*`.
  - Installed `find_package_smoke` for five packages at T = 1/2/4; a consumer expecting the other runtime fails to
    compile.
- Migration ledger: `cpp/tests/README.md`. Suites needing the legacy root (partition facades, the one-store
  prototype, S1-S3 live-oracle seam suites, multi-rank legacy and failure scenarios) stay in OFF builds; other
  propagator suites run as T = 1 raw-layout tests in ON builds.
- Measurement tooling: the driver, preflight and `benches/conftest.py` bind each arm to the runtime read from the
  measured `_core` file (candidate: positive sharded identity; baseline: legacy, or none for the frozen records).
  Driver `2207529f…` -> `0813b7db…`; campaign and workloads unchanged; the modified compare path re-joins all 250
  frozen baseline cells. The shared builders already used the public constructors; the `partitions=1` was Task 6's
  external adapter, now obsolete. A both-arm overlay is prepared, not applied to any preserved environment.
- Architecture checkpoint (owner-approved proposal, run 2026-10-01; report `/home/ubuntu/s4-artifacts/experiment/`):
  - 32 frozen reference cells, 5 alternating fresh-process pairs each, through the tool overlay on both arms;
    134.7 of 170 measurement minutes; 704 processes, no timeouts or memory kills. Preflight confirmed 96 physical
    cores, one NUMA domain and no SMT; frozen inputs, baseline binaries and the Task 1 tree were unchanged
    (snapshots); both negative controls exited 2.
  - Numerics: all 32 candidate validations equal the formal baseline validations; every memory window exact.
  - off-1x96 (MPI-off, 96 threads): runtime ratios 0.90–1.44 (6/16 <= 1.00), operation peaks 0.95–1.05,
    construction peaks 0.96–1.005; 3/16 cells meet all five ratios. mpi-1x1: runtime 0.93–1.03, memory <= 1.004;
    11/16 meet all five. Task 6's one-store prototype had measured 2.1–61.6 on the same full-machine cells.
- Owner decision: **rework**. Profiling (`perf` and AMD IBS on 5 cells; `/home/ubuntu/s4-artifacts/profile/`)
  showed that barrier waiting was not the gap: the extra work came from a per-partner binary search for slot sizes
  in replay, and from construction's handoff chasing every source's published buffer.
- First rework (`/home/ubuntu/s4-artifacts/rework/`): a per-layer slot-size table in `Evaluation.cpp`, and
  row-written, column-read handoff views in `Construction.h`. Outputs are bitwise identical before and after
  (8 fixtures, T = 1/2/4, both builds); the correctness and sanitizer matrix passes. Profiling-harness ratios
  (not checkpoint samples): energy Heisenberg 1.36 -> 1.12, gradient Heisenberg 1.46 -> 1.22, propagate Pauli
  1.23 -> 1.02, propagate Hubbard 1.20 -> 1.13, build_graph control 0.89 -> 0.91.
- Checkpoint re-run (owner instruction, 2026-10-01; `/home/ubuntu/s4-artifacts/checkpoint2/report.md`): same
  protocol, 134.5 of 170 minutes, 704 processes `ok`, all 32 numerics equal, preserved trees unchanged. off-1x96:
  runtime 0.89–1.16 (8/16 <= 1.00), operation peaks 0.95–1.05 (7/16), construction peaks 0.96–1.005 (11/16);
  5/16 meet all five. mpi-1x1: 11/16 meet all five. The checkpoint still fails.
- Profiling (`/home/ubuntu/s4-artifacts/profile2/FINDINGS.md`): partner reads in replay finishes are latency-exposed
  (the legacy runtime overlaps the same transfers in a bulk copy); every checkpoint costs two 96-thread barriers
  (6.5 µs); the Pauli gradient's operation peak is allocator retention of geometric record growth, with live bytes
  equal to the baseline's.
- Owner decision: **rework** with C (single-barrier checkpoint), D (overlapped partner reads) and E (gradient scratch
  liveness). Second rework (`/home/ubuntu/s4-artifacts/rework2/REWORK2.md`):
  - C: `TeamFailure::checkpoint` is one barrier; a recorded failure lowers a shared first-failed generation, and the
    decision for checkpoint N is `first_failed > N`.
  - D: finishes copy runs of consecutive partner blocks into owner-owned staging (`PartnerStaging`, reached by plain
    reference: a thread-local there cost a TLS lookup per use), then apply them in the unchanged order.
  - E: an exact per-layer cosine-set count (`CosCallbacks::count`) lets the gradient's frame phase reserve its
    records once (`replay::reserve_records`).
  - All 48 probe documents are bitwise identical to the first rework's; the correctness matrix, mutants and sanitizer
    legs pass. Profiling-harness ratios (not checkpoint samples), off-1x96: evaluation 0.61–1.00, propagate and
    build_graph 0.77–0.93; Pauli gradient whole-process peak 3.15 -> 2.05 GiB against the baseline. Hubbard cells keep
    an allocator-level +0.3–1% whole-process peak, with live bytes equal.
- Checkpoint re-run on the second rework (owner-approved, 2026-10-02; `/home/ubuntu/s4-artifacts/checkpoint3/report.md`):
  134.0 of 170 minutes, 704 processes `ok`, all 32 numerics equal, preserved trees unchanged. off-1x96: runtime
  0.33–1.06 (15/16 <= 1.00; energy Heisenberg 1.06 on noisy first calls), operation peaks 0.65–1.007 (8/16), construction
  peaks 0.80–1.007 (12/16); 7/16 meet all five. mpi-1x1: 9/16 meet all five (runtime 0.67–1.026). The checkpoint still
  fails, narrowly and mostly on memory: evaluation operation peaks are +6–12 MiB over the baseline (D's per-owner staging
  adds about 4 MiB to S4's pre-existing allocator-level excess).
- Owner decision (2026-10-02): **proceed**. The remaining deviations "appear entirely reasonable" and are accepted
  for now; they are re-evaluated at a later stage. Accepted deviations (candidate median / baseline median - 1, from
  `checkpoint3/comparison-vs-previous-runs.txt`):
  - off-1x96 runtime: energy Heisenberg +5.8% (single cold calls of about 16 ms; the baseline's own samples span
    15.3–27.4 ms; the warm harness gave -5%). The other 15 cells are faster (0.33–0.96).
  - off-1x96 operation peak: energy Hubbard +0.74%, energy Schrödinger +0.72%, build_graph Hubbard +0.55%, energy
    Heisenberg +0.54%, gradient Hubbard +0.46%, gradient Heisenberg +0.42%, propagate Hubbard +0.35%, gradient
    Schrödinger +0.31% (+6–12 MiB on 0.8–1.7 GiB peaks).
  - off-1x96 construction peak: energy Hubbard +0.65%, build_graph Hubbard +0.46%, gradient Hubbard +0.33%,
    propagate Schrödinger +0.07%.
  - mpi-1x1 runtime: propagate Hubbard +2.6% (present since the first checkpoint), propagate Pauli +0.9%, gradient
    Hubbard +0.6%, energy Schrödinger +0.5%, build_graph Heisenberg +0.2%, energy Pauli +0.1%; mpi-1x1 construction
    peak: energy Hubbard +0.2%.
- What is known about them, for the later re-evaluation:
  - The memory excess has two parts. About 4 MiB is the second rework's per-owner partner staging (one run of at
    least 32 KiB plus positions and blocks, times 96 owners). The rest is a few-MiB excess S4 already had; on Hubbard,
    an `LD_PRELOAD` allocation trace (`profile2/alloc/`) found live tracked bytes equal to the baseline's (0.670
    against 0.669 GiB), so it is allocator-level.
  - Candidate remedies, none started: a smaller staging run or staging into an existing owner buffer (keep only if
    the harness speed holds); attributing the pre-existing excess with the allocation tracer at a small threshold
    inside the operation window; profiling energy Heisenberg's cold first call (thread-local scratch warm-up).
  - These checkpoint results do not change S7's 250-cell campaign or its five per-cell gates; the deviations are
    re-assessed there or earlier.
- Lessons recorded for later tasks:
  - In the shared library, a `thread_local` object on a hot path costs a `__tls_get_addr` call wherever GCC
    rematerializes its address, including inside loops; owner scratch is reached through the frame pointer instead.
  - glibc keeps freed memory from geometric vector growth resident, so reserve exact sizes on peak paths; whole-
    process RSS can differ between Release and RelWithDebInfo builds through allocation timing alone.
  - Timing-shaped tests did not catch a non-generation (sticky) checkpoint flag; the forced-interleaving test does.
    Small fixtures never reach multi-run staging; the test-only `EvaluationObserver::staging_run()` forces it.
- Next step: S5 under its own authorization, starting from this proceed decision.
- Pending: macOS, Linux aarch64, wheels, Nix, minimum-version compiler routes, the Python sanitizer legs and P>1
  candidate runs (S5).

- [x] Add separate-process T=1/2/4 Python cases through the real public API. One concrete core fixture is:

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
- [x] Integrate rank-level state and validity, count/memory aggregation and direct shard iteration in
  `evolved_operator_terms()`. Pair each shard index with its own evolved vector; do not use root `indexing()` at T>1.
  Include retained zeros, identity de-duplication and count/export failure tests before timing any result.
- [x] Record a migration ledger for all old facade/one-store tests: kept at T=1, rewritten shard-aware, or retired with
  equivalent coverage. Prove actual participation inside construction, publication, replay and derivative phases.
  Audit trial selection from binary/build provenance, not `--runtime-shape openmp` or environment labels alone.
- [x] Run integrated R correctness, installed-consumer smoke and M-with-P=1 correctness. Public P>1 candidate support is
  not claimed before S5. Update the candidate builder to use the new root constructor instead of `partitions=1`;
  retain the legacy partition builder for the baseline. Keep `observe`, `validate`, `compare` and the
  allocation-before-import fix. Record/synchronize any approved tool overlay without rebuilding baseline binaries or
  rewriting formal records. If an adapter/schema change cannot consume the frozen evidence honestly, stop for review.
- [x] **Stop for experiment approval.** Propose matched baseline/candidate single-thread controls and actual MPI-off
  full-machine reference cells spanning build_graph, propagate, energy and gradients, both bases/pictures. Use frozen
  builders/parameters and operation/construction memory windows. Specify repetitions, task/per-trial/RAM caps and
  first-touch/placement evidence before running; do not reuse Task 6's allowance.
- [x] If authorized, run through the existing driver against preserved partitions, not just candidate T=1. Report every
  runtime and memory cell plus full numerical validation. Single-kernel speedup is not the checkpoint. Archive results
  outside the checkout; retain natural first-touch effects as architectural evidence.
- [x] Obtain the owner's written proceed/rework/stop decision before S5. Failed parity needs a bounded proposed remedy,
  not an automatic move to runtime deletion. The later frozen campaign is still mandatory after an early pass.
  (2026-10-01: **rework** decided; first rework done. Re-run failed; 2026-10-02: second **rework** (C, D, E) done.
  Its re-run (2026-10-02) left small deviations; the owner accepted them and decided **proceed**, to be re-evaluated
  at a later stage.)

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

**Outcome:** implemented and **accepted by the owner as done on 2026-10-02**. Start: `dc2e102`. Interface
plan, handoff, evidence and the measurement proposal (not run): `/home/ubuntu/s5-artifacts/` (`S5-PLAN.md`, `HANDOFF.md`,
`MEASUREMENT-PROPOSAL.md`), outside the checkout.

- `detail/sharded/Exchange.{h,cpp}`: `PhysicalExchange`, one reusable physical round between this process's T
  owners and those of a peer set.
  - It is move-only and heap-stable. The destructor and move assignment drain.
  - Owners write row-owned count rows and pack disjoint send slices; destinations read received blocks as const
    views, with no scatter copy.
  - Only the primary lays a round out and posts or completes it. Its MPI calls throw `std::logic_error` off
    OpenMP thread 0 or off MPI's initializing thread.
  - Wire layout as `HybridComm`: per peer, destination-shard major and source-shard minor. Counts, totals and
    displacements are checked ints, refused before posting.
  - Transports: pairwise over non-empty legs, or one `MPI_Ialltoallv`. The count round is pairwise.
  - `PhysicalWorld::of()` reads rank, size and the agreed replay transport on the caller.
- Construction: a gate's window reaches another rank exactly when its peer plan does (the linear peer, or every
  rank under splitmix), so every rank branches alike.
  - Each such pass adds Q1-Q3 (lay out, count round, pack, post and wait) and R1-R3 (answers, laid out from rows
    the resolvers write).
  - Answer counts are query counts, not stream words, so they are known only after resolution; the first two-rank
    run found this.
  - Destinations resolve local and received blocks together, in ascending flat-slot order.
- Replay (energy, gradient, contraction, informed P5/P6, seed replay): every step at P > 1 adds a layout phase, a
  pack phase and a post phase in which the callback part (record, cosine or reverse accumulation) runs while the
  transfer is in flight. `EndpointBoard` reads remote partners from the round. P = 1 keeps the S4 phase sequence;
  48/48 probe documents are bitwise identical to S4's final record.
- Failure: after a failed join with live requests, a seam hands the error to `mpi::operation_failed()` before any
  round is destroyed.
- Root: P ranks, router (P, T), flat owner `rank * T + t`, and `world_`.
  - The multi-rank rejection is removed. Linear routing rejects a non-power-of-two P on every rank.
  - Updates apply each rank's own share only.
  - A thread-level failure goes through `operation_failed()`.
  - `routing::Config` loses its partition field; the legacy facade keeps `PartitionCountMismatch`.
- Ledger rulings (`.superpowers/sdd/.../progress.md`):
  - reuse of `begin_flat_exchange`'s semantics, not its code;
  - trailing defaulted `PhysicalWorld` parameters on `evaluate_shards()` and `replay_shards()`, and
    `ConstructionContext.world`;
  - the answer round's legacy-equal count exposure;
  - the local sanitizer settings for Open MPI 5.0.10.
- Results (frozen source `final-s5-source.diff` `3a3d17ec…`; the final matrix is in the handoff):
  - Bitwise against the legacy runtime at P ranks x `partitions = T`:
    - the seams, per shard, at P = 2/4 x T = 1/2/4 and P = 3 splitmix (`sharded_seams_mpi*`);
    - the public API, per rank, at P = 2 x T = 1/2/4 for 8 fixtures (`test_multirank_matches_*`, 24/24).
  - Within tolerance, against one process:
    - the root at P = 2/4 x T = 1/2/4 and P = 3 splitmix (`sharded_root_mpi*`);
    - the gathered public-API maps, energies and gradients over (1, 4), (2, 2), (4, 1) and splitmix (3, 2) against
      (1, 6).
  - Exchange rounds at P = 2/3 x T = 1/2/4.
  - Failure scenarios (`mpi_failure_sharded_*`, 9/9) each end in the library's abort or fail-fast within 0.2-1.6 s.
  - 11 mutants detected (one equivalent mutant replaced).
  - ASan/UBSan (GCC + Open MPI) and TSan with Archer (Clang 18 + libomp + Open MPI, qualified under `mpiexec`):
    0 reports over the multi-rank launches.
  - RED evidence: HEAD binaries plus only the new tests fail at P > 1 with "supports one MPI rank".
- MPI+OpenMP checkpoint, tier A (owner-approved, 2026-10-02; `/home/ubuntu/s5-artifacts/checkpoint/report.md`):
  - 44 frozen reference cells, 5 alternating fresh-process pairs each, through the unchanged S4 overlay and runner
    launch; 161.9 of 210 minutes; 968 processes `ok`.
  - All 44 numerics equal the formal validations, and the preserved trees are unchanged.
  - mpi-2x48: 13/16 meet all five. Runtime 0.49–0.96 apart from three evaluation cells: gradient Schrödinger 1.37,
    energy Pauli 1.21, energy Schrödinger 1.20 (within the baseline's spread). Every memory peak is at or below the
    baseline.
  - mpi-3x32-splitmix, mpi-4x24-linear and mpi-4x24-splitmix: 12/12 meet all five (runtime 0.84–0.96).
  - mpi-1x1: 8/16, with S4-sized deviations up to 1.031 (propagate Hubbard).
  - Tier B and the MPI-only controls were not run.
- Owner decision (2026-10-02): **S5 done**. The three mpi-2x48 runtime misses and the mpi-1x1 deviations are recorded
  here and re-evaluated with the S4 deviations at a later stage; S7's 250-cell campaign and its five per-cell gates are
  unchanged.
- Potential optimizations flagged by S5 (none started; each needs its own authorization and must keep fixed-P/T
  results bitwise):
  - The replay's remote step costs four checkpoints instead of one (layout, pack, post/overlap, finish). Fold the pack
    into the publishing phase (owners can pack once the layout exists) and lay out step p + 1 during step p's post
    phase, bringing a remote step toward two checkpoints. Primary candidates: the three missed cells (gradient
    Schrödinger 1.37, energy Pauli 1.21, energy Schrödinger 1.20 at 2 x 48).
  - The primary lays every round out serially, O(P x T^2) per round. A layer's partner layout is static, so the replay
    layout can be cached per layer for an evaluation (bounded memory: offsets only), or computed owner-parallel (each
    owner its own destination column, then a prefix over peers).
  - Construction adds up to six checkpoints per remote pass (Q1-Q3, R1-R3). Each owner could compute its own send
    offsets from per-peer totals published in P1, merging Q1 into Q2; the answer layout could be planned in the same
    primary phase as the query post.
  - Profile gradient Schrödinger at 2 x 48 first to confirm where the extra ~18 ms per call goes (checkpoints,
    layout, or MPI latency per layer).
  - Robustness, not speed: the answer round trusts one answer per query (as the legacy runtime does); an optional
    count round per remote pass would turn such a bug from a hang into an abort.
  - Minor: each worker rebuilds its gate's peer vector per gate.
- Pending:
  - the whole-suite multi-rank variants and the remaining suites at P > 1 in prototype builds (S6);
  - macOS, Linux aarch64, wheels, Nix and minimum-version compilers;
  - multi-node.

- [x] Add two-process tests with T=1/2/4 for mixed local/remote destinations and canonical sender ordering. Check that
  local sender blocks do not precede lower-numbered remote senders. Verify response source/query alignment, empty
  messages, nonzero slot windows and exact pre-update endpoint values.
- [x] Define one staging layout with checked counts and prefix-sum offsets for `(source shard, destination shard)`
  segments per physical peer. Workers write disjoint payload slices; the primary exchanges metadata/posts/waits and
  publishes received extents; destination owners decode and mutate. Never serialize all payload processing on shard 0.
- [x] Preserve construction's linear peer window and replay's actual multi-peer summaries. Warm/use communicator-agreed
  `routes_pairwise` on the caller/primary, including MPI attribute accesses. Keep mode/seed agreement and remove the
  partition-count agreement field rather than replacing it with T agreement. No geometry-repair fallback.
- [x] Implement request lifetime explicitly. The owner of `pending` and every referenced buffer surrounds the team:

```text
allocate operation frame and request owners
run_team: protected owner/primary phases, post -> independent work -> wait -> dependent work
inspect returned error while request owners are still alive
if error: invalidate after mutation; multi-rank abort, otherwise rethrow
only then permit normal request/buffer destruction
```

  A primary post may fail after some requests became live; those requests still belong to the outer frame. Do not
  return through a draining local destructor before reaching distributed failure handling.
- [x] Adapt failure scenarios for worker throw, before-exchange throw, active-ticket throw, malformed receive and
  wrong-thread entry. Supervised two-process runs must finish in failure before the 30-second timeout; timeout fails the
  test. Single-process rethrow/invalidation and independent-copy survival must also pass. Test actual MPI support, not
  merely the requested level. Keep SERIALIZED until S8 removes executable legacy workers.
- [x] Run linear supported geometries and splitmix with three processes, numerical/full-map comparisons and qualified
  MPI sanitizer runs. Assert MPI occurs only on the initializing primary; no arbitrary `omp single` winner may post.
- [x] **Stop for a separate bounded measurement approval.** (Proposal `/home/ubuntu/s5-artifacts/MEASUREMENT-PROPOSAL.md`;
  tier A approved and run 2026-10-02, report `/home/ubuntu/s5-artifacts/checkpoint/report.md`.) Propose matched MPI-only, MPI+OpenMP and single-thread
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

**Outcome:** implemented and **accepted by the owner on 2026-10-03**. Start: `ef103e9`. Audit and migration ledger
(written first), handoff with identities, commands and evidence: `/home/ubuntu/s6-artifacts/` (`LEDGER.md`,
`HANDOFF.md`), outside the checkout.

- Candidate C++ surface: the constructor ends with `basis_change, logical_num_modes, basis` (public and private
  observer constructors); `partitions`, `child_factory`, `PartitionChildFactory`, the partition forward declaration,
  `PartitionCountMismatch`, the `PartitionGroup.h` include and the `monoprop_PARTITIONS` check are gone from the
  candidate, without aliases. Raw accessors throw `MultiShardUnsupported` (validity first). `EnvConfig.h`'s permissive,
  cached thread parser is legacy-only. The legacy (OFF) surface and binaries are unchanged: its `_core`, `libmonoprop`
  and failure driver hash as at S5.
- Python: the candidate `_core` constructor (callable, annotations, docstring and `_core.pyi`) has no `partitions`;
  removed and never-added keywords and the old positional tail raise `TypeError` at every level.
- Tests: `sharded_api_tests.cpp` (compile-time probes asserted per build, so each must be true in the legacy build
  and false in the candidate); the S1-S3 seam suites, `openmp_runtime_tests.cpp` and `fused_cos_sweep_tests.cpp`
  compile in both builds with legacy-oracle cases OFF-only; a test-only `fused_records` observer hook audits real
  per-shard fused records; extended virtual-hook, raw-access, ignored-variable and unset-budget cases. The runner no
  longer supplies `monoprop_NUM_THREADS`; CTest registration sets T = 1; whole-suite MPI variants run in candidate
  builds; `fused_env_t*`, `openmp_env_root_t4` and `sharded_root_env_openmp_default_t3` are new fixed launches.
- Consumers and CI: the probe and `find_package_smoke` check the constructor shape, derived hooks, raw access, the
  inherited selector and MPI setting, and (MPI) a two-rank comparison with one process; CI gains `runtime: sharded`
  legs with distinct cache and report identities and `just check-runtime`.
- Evidence (GCC 15.2/libgomp, Open MPI 5.0.10, this host): RED first (compile probes, two runtime cases, 24 Python
  cases); 13 mutants and a TSan race mutant detected; final matrix r-off 492/492 + pytest 813, m-off 551/551 +
  pytest 855 + 2 x 742 at two ranks, r-on 487/487 + pytest 874 (T = 1, with the legacy bitwise comparisons) and 813
  (T = 2, 4), m-on 550/550 + pytest 950/853/853 + 2 x 742; GCC ASan/UBSan and Clang 18/libomp/Archer TSan without
  reports; Clang/libomp 487/492; installed consumers for all four packages.
- Pending: macOS, Linux aarch64, wheels, Nix, minimum-version compilers, Python sanitizer legs, multi-node. No
  performance claim: the candidate MPI binaries changed (S5's checkpoint does not cover them); S7 is unchanged.

- [x] Add compile/API tests proving old explicit partition/factory signatures are absent from the candidate, Python
  construction has no new thread/shard argument, and budget parsing/capture/copy use the existing environment contract.
- [x] Implement single-shard access with an explicit T>1 exception and a T=1 remedy. Do not fabricate raw merged
  storage. Update the old `MultiPartitionUnsupported` diagnostic/type as part of the documented C++ break.
- [x] Migrate the audit's real call sites and tests, preserving coverage. Export through shard indices/coefficient
  vectors; fixed-geometry graph-layout tests inspect shards privately. Cross-geometry tests compare full retained maps.
  Retain actual-participation, capacity, malformed-input, fresh-insertion, copy/clone/update and failure-lifetime tests.
- [x] Remove the test main's `monoprop_PARTITIONS=off` assignment; ordinary C++/Python tests launch at T=1. Register
  dedicated fixed-T launches without `ScopedBudget` mutation in integration tests. Do not use `OMP_THREAD_LIMIT<T` to
  test the unsupported sharded configuration, or make all tests serial to avoid migration. (The assignment stays in
  legacy builds only, which still read the variable, until S8.)
- [x] Update `README.md` when relevant, `docs/content/docs/features/parallelism.mdx`,
  `docs/content/docs/building.mdx`, `cpp/tests/README.md`, `benches/LADDER.md` and `CONTEXT.md` if terminology needs
  clarification. Explain required OpenMP, launch-before-import configuration, fixed geometry, single-shard accessors,
  removed extension surface, NUMA/physical-core allocation and new test commands. Update docstrings/API links.
  (`CONTEXT.md` unchanged: its pinned hash is a frozen input.)
- [x] Regenerate bindings/dispatch through the supported build, never by editing generated files. Run full R/M C++ and
  Python suites and installed consumers; rebuild all ABI-coupled artifacts together. Close every migration-ledger row.
- [x] Hand off the source/API coverage audit and pending platforms. Runtime deletion and formal measurements require
  their following gates; this task alone does not establish parity or publication readiness.

### After S6: deviating-cell re-check, B-replay and B-memory (owner-authorized, 2026-10-03/04)

Diagnostic evidence and one optimization between S6 and S7, each step under a written scope with a stop for the
owner. Artifacts: `/home/ubuntu/s6-artifacts/checkpoint-c/`, `b-replay/` and `b-memory/` (outside the checkout).

- **Re-check (C).** The 21 checkpoint cells with any ratio above 1.00 at their latest checkpoint, measured on the S6
  binaries with the S5 runner and protocol (5 pairs, 88 of 120 minutes, numerics equal). Five passed all five ratios.
  The 2×48 runtime misses persisted (energy Pauli 1.13, gradient Schrödinger 1.16), as did the off-1x96 operation peaks
  (+0.3–0.7 %).
- **B-replay, step 0 (profiling).**
  - The 2×48 gap is a first-call effect: in steady state the candidate is within ±3 % of the baseline.
  - About half of the measured gap comes from the runner's own process supervision: a memory and per-thread sampler
    runs unbound on the measured cores, and the candidate's many team barriers per step amplify it.
  - The rest is mostly the primary laying out every remote replay step serially, about a third of the primary's
    time in energy Pauli.
  - MPI itself is under 1 % of samples.
- **B-replay, change A (`bb71b9a`).** Owner-parallel replay-round layout, with offsets and bytes unchanged; the
  multi-rank seams are bitwise against the legacy runtime and the four-build matrix and sanitizers are clean.
  - Energy Pauli: about 7 ms per call faster, 1.03 → 0.97 of the baseline in steady state.
  - The Schrödinger evaluation cells are unchanged.
  - A protocol re-check put all three 2×48 cells at or below 1.00. The Schrödinger passes depend on noisy baseline
    samples, and the owner chose to leave them (option a).
- **B-memory, step 0 (attribution).** The off-1x96 peak excess, candidate minus baseline at the same cell, has four
  parts:
  - the extension's resident code pages, +2.0–2.3 MiB (`_core` 12.8 MB against 9.95 MB);
  - other anonymous mappings, +1–2 MiB;
  - allocator retention with equal live bytes, +4.8 MiB on Schrödinger only (shard 0 allocates from the main
    arena);
  - the per-owner partner staging, about +3.8 MiB in evaluation (96 × 32 KiB plus its lists).
- **B-memory, change 1.** A smaller staging run was rejected. Gradient Hubbard slows monotonically, by 1.8 % at 2048
  values and 4 % at 512, while saving at most 2.6 MiB, which brings no cell to the baseline.
- **Owner decision (2026-10-04).** No structural change for the remaining few MiB. S7 runs with its five gates
  unchanged, and the MPI-off memory cells that fail by these constant overheads are recorded as **known deviations**.
  This records an expected outcome; it relaxes no gate. The runner-supervision interaction was raised with the owner
  and left unchanged; changing it would alter the frozen measurement protocol.

## Task S7: Frozen full-library parity before removal

**Deliverable:** a complete, machine-readable acceptance report for the integrated candidate, or a blocking failure.

**Files:** ordinarily no production changes. Use `tools/benchmark-rank-local-openmp.py`, the two frozen JSON inputs,
`tests/test_rank_local_openmp_benchmark.py` and existing bench-tools. Put campaign artifacts outside the checkout.

**Outcome:** campaign run under option B (Task 1's archived `s01..s05` as the authoritative baseline, fresh repeats
on both arms), approved by the owner on 2026-10-04. Candidate `2ab9b5e`. Artifacts, runner, ledger and report:
`/home/ubuntu/s7-artifacts/` (`campaign/reports/REPORT.md`, `cells.json`, `known-deviations.json`; `LEDGER.md`,
`HANDOFF.md`), outside the checkout. **S7 campaign complete; strict parity not demonstrated.**

- Conduct: 6920 launches, all `ok`, in 12.33 h of the 17 h budget; comparison 2.15 h of 3 h; preserved trees
  byte-identical before and after.
- Evidence complete: all 250 cells compared; numerics equal in all 250; every required window exact; the 196 cells
  whose first five pairs failed a ratio received the prescribed five repeats.
- Verdict (unchanged comparator, exit 1): **102 pass, 148 fail**; 90 memory only, 45 runtime only, 13 both.
  - MPI-off 1x96: 48 fail; 144 (cell, gate) pairs are the known deviation KD-MPIOFF-MEM (+0.7–13.9 MiB within the
    pre-registered bound); 38 cells fail on nothing else.
  - `large--propagate-hubb` has a size-dependent memory regression at every geometry, up to +609 MiB (1.14) on the
    2x48 construction peak sum.
  - MPI 1x96, never measured before: 126 small memory failures (median +1.1 MiB).
  - Runtime: mostly tiny cold calls at MPI 1x1 (up to 2.08, +0.02–0.06 ms) and 2x48 (1.20–1.34), and
    `reference-pared--pare-construct-schr` (1.38–1.50).
  - Multi-rank routing cells (3x32, 4x24 splitmix and linear): all 12 pass.

- [x] Obtain campaign approval with a fresh budget. Reverify frozen file/index identities, baseline binary/environment
  preservation and the actual current target allocation. c8a.metal-24xl previously had one NUMA domain and 96 physical
  cores; verify rather than infer. No SMT-worker campaign or automatic provisioning.
- [x] Before timings, complete the candidate's compiler/build correctness matrix, full retained-map export checks and
  actual-worker/placement evidence. Assert actual R/M build identity. Both arms use the fixed global workloads and
  original timing boundaries; candidate selection needs more than its requested thread count.
- [x] Follow historical Task 1's existing `observe`, `validate`, `compare` schema and distinct timed/construction
  processes. Collect five fresh pairs per required cell, then the specified five more for failing cells. Keep all
  samples and numerical validation. Frozen baseline data cannot be overwritten/recollected to improve a comparison;
  any additional prescribed observations are separately identified records under the approved protocol.
- [x] Check all five per-cell median ratios <=1.00. Reject missing cells in both arms, duplicated/reused runs, extra or
  mismatched sample identities, mismatched provenance, absent validation and inexact required memory windows. Report
  operation, construction and diagnostic outer peaks separately; historical residual peak growth grants no allowance.
- [x] If a gate fails, profile only with separately authorized time/RAM. The permitted refactor-preserving remedies are
  ownership/buffer reuse, redundant copies, packing/layout overhead, barrier placement, cache locality and phase
  balance.
  Keep payload/source order and numerical policy fixed. No hidden backend, new concurrent-index study, smaller workload,
  restored one-store plan or unapproved tuning campaign.
- [x] Stop on unresolved regressions. A pass permits an owner decision about S8, not an automatic deletion/commit/push.
  (S7 did not pass; the owner waived the pass prerequisite for S8 on 2026-10-07, see S8.)
  Record any platform/HPC qualification still pending separately from the target's frozen acceptance scope.

### After S7: remedy rounds (owner-authorized, 2026-10-06/07)

Four bounded rounds under a written scope: profile first, keep a change only if measured better, full verification,
stop for the owner. The owner had the round artifacts (scope, findings, patches, harnesses and results) deleted on
2026-10-07 to free disk space. This section and `/home/ubuntu/s7-artifacts/LEDGER.md` are the remaining record. The
measurements use the figure-1 workloads (60-site Majorana and Pauli circuits), a profiling harness and S7 cells. They
are diagnostics, not S7 evidence. All measurements ran with glibc malloc.

- **Per-gate overhead on splitmix routing.** At 4x24 splitmix every gate runs construction's query and answer rounds,
  and the primary's serial round layout cost +35–40 µs per gate over the baseline. Kept:
  - a contiguous layout with local pointers and one total check per peer;
  - `int` offsets (reverted in round A: they caused a regression in owner-parallel replay planning);
  - count blocks in the narrowest of 1, 2 or 4 bytes per count, since a 4 KiB block at T = 32 crossed Open MPI's
    shared-memory eager limit;
  - prefetch of the round tables before the layout passes.

  Result: 4x24 +3.7 µs per gate, 3x32 −20 µs.
- **Majorana MPI 2x48 memory.** The candidate peaked at 1.155 of the baseline (10.0 GiB vs 8.7 GiB). Cause: frees of
  exchange staging raised glibc's dynamic mmap threshold, and the owners' per-gate transients then stayed in thread
  arenas. Kept:
  - `PhysicalRounds` held by the propagator and reused across operations;
  - `mmap`-backed staging (`std::allocator` under ASan).

  Result: final peak 0.984, every layer at or below the baseline.
- **T = 1 fast path.** A one-worker team runs its body on the caller, with no OpenMP region and no barriers; a
  publication only to self skips the exchange layout. Result: the MPI 1x1 per-call excess shrank by 60–75 % (energy
  Pauli 35 → 17.7 µs against 13.4). It is not closed: still 1.17–1.32×.
- **Rejected (patches deleted with the round artifacts):**
  - Announced-count answer round, which drops two construction checkpoints. Tried three times; it is no better at any
    geometry and up to +70 µs per gate worse at 4x24. The cost is cache-line traffic on the per-gate count tables,
    not the number of checkpoints.
  - Owner-parallel query placement.
  - Partner staging into an owner-held buffer: not implemented. A no-staging probe bounds any saving at 1.1–2.0 MiB,
    while the MPI-off excess is mostly code pages (+3.4–4.8 MiB file-backed) and graph-building memory.
- **Figure-1 harness on the round-3 source:** whole-run time 0.80–1.00 of the baseline, every peak 0.970–0.998,
  numerics identical.
- **Allocators** (`LD_PRELOAD`, 108 runs, numerics equal).
  - mimalloc 3.2.8: peaks 1.35–1.81× the glibc baseline in every arm.
  - jemalloc 5.3.0: removes the S7 candidate's 2x48 excess (0.997 of its same-allocator baseline, confirming the
    diagnosis); peaks 1.03–1.12× and time 0.79–0.99× the glibc baseline.
  - Remedied source with glibc: lowest peaks of any configuration (0.984–0.996), time 0.82–0.97.
  - **Owner decision (2026-10-07):** keep glibc; no allocator change, in the library or in the recommended deployment.
- **Verification of the landed source:**
  - four-build matrix: r-off 495, r-on 491, m-off 525 + 29 MPI, m-on 521 + 33 MPI;
  - pytest T = 1/2/4 (T = 1 with the legacy bitwise comparisons) and 2-rank T = 1/2;
  - GCC ASan/UBSan 521 + 33 and Clang/libomp/Archer TSan 138 + 16, without reports.
- **Left open after these rounds:** the `large--propagate-hubb` memory regression and the
  `reference-pared--pare-construct-schr` runtime (both addressed in round A, below), the MPI 1x1 residual and the
  MPI-off code-page overhead.

### After S7: round A (owner-authorized, 2026-10-07)

A bounded round on the two largest remaining failure classes, under a written scope: profile first, keep a change
only if measured better, full verification, stop for the owner. Budget: 90 min of measurement launches (65 used),
24 GiB per process group under a watchdog. glibc malloc on every arm. Diagnostic harnesses outside the checkout
(`/home/ubuntu/s7-artifacts/round-a/`). The numbers are diagnostics on S7's workloads and launches, not S7 evidence.

- **A1, `large--propagate-hubb` memory, kept.**
  - The remaining excess (about +2.5 % at every geometry) was a transient at the end of every `propagate` call: all T
    owners copied their coefficient and state vectors down to size (`shrink_to_fit` in `initialize_caches`) at the
    same moment, each copy holding old and new blocks. The legacy partitions reach that point staggered.
  - Now at most max(4, T/8) owners copy at once (`SectionLimit` in `detail/sharded/Team.h`); copies under 256 KiB
    skip the limit. `MPOperator::initialize_caches` is split into `warm_caches` and `release_slack`, and the legacy
    path still calls both.
  - Result: peaks 0.946 (MPI-off 1x96), 0.948 (MPI 1x96), 0.962–0.965 (MPI 2x48) of the baseline, against S7's
    1.03–1.15. Step time unchanged.
- **B1, `reference-pared--pare-construct-schr` runtime, kept.**
  - The excess was all in the first call of the process. It made 6,389 thread-arena heap growths (`mprotect`, each
    taking the process's memory-map lock) against the baseline's 273, almost all from paring building every layer's
    full cosine mask by reallocation.
  - `pare_graph_owner` now takes a provider that writes into a buffer, reuses one full-mask and one filtered buffer
    per owner, and stores each kept mask at its exact size (`full_cos_mask_into`, `fold_to_cos_mask_into`). Masks are
    bit-identical.
  - Result: first call 0.997–1.07 of the baseline (S7 1.09–1.50), warm calls 0.87–0.95.
- **C1, regression from the remedy rounds, fixed.**
  - The remedy rounds' 4-byte exchange offsets doubled false sharing in owner-parallel replay planning: neighbouring
    owners write neighbouring entries of a column in `place_column`, and every replay step's packing reads them. Tiny
    evaluation at MPI 2x48 went from 1.31 (S7 candidate) to 1.78 of the baseline.
  - Offsets are 8 bytes again, with separate prefetch strides for counts and offsets.
  - Result: 1.06–1.15; splitmix construction unaffected (0.968 and 1.002 of the 4-byte build).
- **Verification:**
  - four-build matrix: r-off 499, r-on 495, m-off 529 + 29 MPI, m-on 525 + 33 MPI;
  - pytest T = 1/2/4 (T = 1 with the legacy bitwise comparisons) and 2-rank T = 1/2;
  - GCC ASan/UBSan 525 + 33 and Clang/libomp/Archer TSan 141 + 16 (including the new section-limit cases at
    T = 1/2/4/4-passive), without reports.
  - New tests: the section limit (bound, release on exception, minimum one) and paring into one reused buffer; a
    mutant of each is detected.
- **Diagnostic re-measure of S7 failure classes on this source** (representative cells, vs the baseline):

  | Class (S7) | Now |
  | --- | --- |
  | `large--propagate-hubb` memory (1.03–1.14) | 0.946–0.965 |
  | `pare-construct-schr` runtime (1.09–1.50) | first call 0.997–1.07, warm 0.87–0.95 |
  | MPI 1x1 tiny evaluation (1.39–2.08) | repeat calls 1.04–1.36, first calls up to 1.39 |
  | MPI 2x48 tiny evaluation (1.16–1.34) | repeat calls 1.06–1.15, first calls up to 1.52 |
  | MPI-off reference gradient-hubb (1.10) | first call 1.11, repeat calls 0.99 |
  | MPI 1x1 reference construction (1.005–1.032) | `propagate-hubb` 1.037, `build-graph-schr` 1.013 |

### Still open: optimization points for the end of S8

None of these blocks correctness; each is a parity-gate miss that S7's protocol would still report. Owner decision
(2026-10-07): take them up at the end of S8, after the legacy machinery is removed and before S8's final parity
campaign (see S8's checklist), so that the one campaign measures them. Ideas, not measured yet:

1. **First-call costs of small evaluations** (MPI 1x1 and 2x48, and MPI-off reference gradient-hubb's first call).
   Repeat calls are close to the baseline, so this is one-off cost on a process's first functional.
   - Profile it as B1 was profiled: minor faults, `mprotect`/`brk` counts and RSS per call, and attribution of the
     heap growths. B1 found exactly this mechanism, fresh thread arenas growing a page range at a time, in another
     first-call path.
   - Likely sources are per-owner evaluation state allocated on first use: the thread-local owner frames, the
     records reserved per functional, the partner staging. Candidate fixes: hold that scratch in the propagator, as
     the physical rounds already are, sized once per graph instead of per functional; allocate each owner's scratch
     as one block; or warm it in the graph build's last phase, where the heap is already grown.
   - At MPI 1x1 the remaining repeat-call excess (+1–5 µs) is the per-step phase machinery
     (publication, scaling, checkpoints). A T = 1 evaluation loop shaped like the legacy serial loop would remove it.
2. **MPI 1x1 single-thread construction, 1–4 % slower** on calls of seconds (`propagate-hubb` 1.037).
   - Compute-bound, so a per-gate cost difference. Compare perf profiles of legacy and sharded at T = 1 on symbol
     builds of both; the frozen baseline binary is stripped, so build the baseline commit with symbols.
   - Suspects: the sharded gate's phase sequence still runs at T = 1 for cross-owner work that is always empty there
     (inbox and query/answer setup, frame handling), and differences in the fused apply path. Folding the per-gate
     phases into one when T = 1, and skipping empty cross-owner work, would be the first things to try.
3. **Constant memory overheads** (MPI-off +2–8 MiB on tiny and reference cells, MPI 1x96 median +1.1 MiB).
   - Mostly file-backed code pages: the extension carries both runtimes until S8. Re-measure after S8's removal of
     the legacy machinery before anything else.
   - The rest is anonymous memory from graph building at T = 96 (per-owner frames and inbox tables). It could be
     released at the end of the build or allocated on first need.
   - `large--gradient-hubb`'s +9.8 MiB was not re-measured. Its construction ends in the same caches phase, so A1 may
     already cover it.

**Owner decision (2026-10-07):** S7 is not re-run. The S7-pass prerequisite for S8 is waived, and one parity campaign
on S8's final binary (S8's checklist) serves as the parity evidence.

**Taken up in S8 (2026-10-07/08):** see the optimization points in S8's outcome. What stays open is listed there.

## Task S8: Remove legacy machinery, qualify the final binary and close out

**Deliverable:** one sharded OpenMP runtime, correct FUNNELED support and verified final packaging without direct hwloc.

**Prerequisite waiver (owner, 2026-10-07):** S7 did not demonstrate strict parity (102 of 250 cells). The owner waived
the requirement that S7 pass before S8: the removal may proceed on the remedied source, and S8's final parity campaign
on the final binary is the parity evidence. The waiver relaxes no gate: final acceptance still requires all five
per-cell median ratios <= 1.00 under the unchanged protocol, and failures are reported as before. S8 itself still needs
separate authorization.

**Files:** remove `detail/partition/PartitionGroup.h`, `CpuTopology.h`, `CpuTopology.cpp` and obsolete partition CMake
registration; remove `detail/mpi/{ShmComm,HybridComm,PartitionBarrier}.h` and runtime-specific tests after their
coverage has migrated. Update `detail/mpi/{Comm,MPICompat,MPIUtils,Exchange}.h`, `MPICompat.cpp`, parent CMake files,
`cmake/monopropConfig.cmake.in`, `nix/monoprop.nix`, `nix/devshell.nix`, `.github/workflows/nix.yml`,
`pyproject.toml`, installed smoke tests and docs as applicable. Audit other tracked dependency references before
editing; do not touch unrelated caches.

**Outcome (2026-10-07/08, in progress):** cutover implemented and locally verified; optimization rounds run and their
kept changes verified; committed and pushed at the owner's request on 2026-10-08; final campaign not run. Start:
`d023e67`. Ledger, removal/coverage ledger, proposal, findings and handoff: `/home/ubuntu/s8-artifacts/` (`LEDGER.md`,
`REMOVAL-LEDGER.md`, `OPTIMIZATION-PROPOSAL.md`, `opt/FINDINGS-*.md`, `HANDOFF.md`), outside the checkout. **S8
cutover and optimization rounds implemented and locally verified; final-campaign authorization pending.**

- One runtime: the selector, the legacy root, `PartitionGroup`, `CpuTopology`, `ShmComm`, `HybridComm`,
  `PartitionBarrier`, `CpuRelax`, the partition-count agreement, the permissive cached thread parser and every direct
  hwloc/pkg-config input (CMake, package config, Nix, Homebrew/apt lists, `install-deps.sh`, wheel `before-all`) are
  gone. `mpi::Comm` holds only an ordinary communicator; `geometry()` and the in-process verb bundles are gone; the
  real-MPI helpers of the low-level engine and the exported evaluation functions stay. The initializing-thread check
  no longer hangs off the removed `one_store_` flag: every MPI-using operation runs it.
- FUNNELED: `mpi::init` requests it and every construction requires at least it; `MpiThreadLevelUnsupported` moved to
  `MPICompat.h`; diagnostics before an abort no longer allocate unguarded. RED first on the pre-cutover code (unit
  level test, host-FUNNELED acceptance, SINGLE diagnostic, wrong thread at FUNNELED, Python mpi4py subprocesses), then
  GREEN with exact requested/provided levels recorded (`mpi_failure_*`, `tests/test_mpi_thread_support.py`).
- Identity: the extension embeds the unconditional, NUL-terminated `monoprop-runtime=sharded-openmp`; the bench-tools
  preflight reads it and still reads the archived prototype/legacy markers (driver unchanged at `0813b7db…`).
- Tests: legacy-only suites deleted; their plain-MPI arms ported (`mpi_alltoallv_tests.cpp`); the low-level failure
  scenarios moved to the T = 1 root's sole shard; a caller-local-round failure scenario added after a mutant showed the
  gap. Bitwise differential pre/post at fixed (P, T): GCC MPI-off 27/27, GCC MPI 136/136 rank documents, Clang 27/27.
- Pending: Nix, macOS, Linux aarch64, CI wheels (manylinux images), minimum-version compilers, multi-node; TSan of
  Open MPI's own MULTIPLE-level locking (one report inside libopen-pal, no monoprop frame) is inconclusive.
- Optimization points (owner-approved 2026-10-07/08; diagnostics in `s8-artifacts/opt/FINDINGS-STEP{0,1,2}.md` and
  `FINDINGS-ROUND{3,4}.md`):
  - Step 0: removing the legacy machinery changed nothing measurable.
  - Step 1 attributed the remaining misses by profile against symbol builds of both revisions.
  - Step 2 kept three bitwise-neutral changes:
    - **A:** a replay step writes its symmetric rows in one call, plans over every other rank without allocating,
      and packs through its own contiguous column (`write_symmetric_rows`, `plan_totals(transport)`,
      `send_column`). The send offsets are stored by column, so no owner shares a line of another's. Tiny 2x48
      evaluation went from 1.23–1.51 to 0.91–0.99 of the baseline.
    - **B:** no-op scratch growth is tested inline, and reverse factors are built once per call. Gradient-paul 1x1
      went from 1.23 to 1.19; energy at 1x1 is unchanged and stays a miss (1.41, the per-step phase structure).
    - **F:** self hits are reserved from the self-query count. MPI 1x1 propagate-hubb went from 1.040 to 1.014–1.018.
  - D (laying the binding registration out apart from the kernels) lowered import-time code pages only; the kernels'
    own execution brings them back, so it was not kept.
  - Round 3 (owner-approved 2026-10-08):
    - **Page-cache state, not code.** How a shared object's pages entered the page cache moves a process's RSS by up
      to ±10 MiB on this kernel. A freshly written binary maps more than an aged one. That was the whole MPI-off
      code-page excess, and the fresh-against-archived MPI drift is the same effect. RSS gates need equal page-cache
      hygiene on both arms; that is an owner decision for the campaign.
    - **Kept, bitwise-neutral:**
      - C1: partner staging sized by need, at most one run.
      - 3a: a sole shard on one rank, with no observer, is evaluated by the low-level serial evaluator, after the
        sharded per-request checks (`check_shard_request`).
    - **Re-measured with hygiene:** MPI-off tiny memory 0.98–0.99 (was 1.05–1.06). 1x1 tiny evaluation 1.02–1.22
      (energy-paul 1.13, was 1.41). Larger pared 1x1 evaluations 1.05–1.11 (see round 4).
  - Round 4 (owner-approved 2026-10-08):
    - **E-b, kept:** `MPOperator::release_slack()` no longer copies a vector of 256 KiB or more down to its size. It
      keeps the block and discards the pages of its spare capacity (`release_spare_capacity`, `MADV_DONTNEED`,
      Linux only). The memory breakdown counts such a vector's entries (`accounted_bytes`), which at rest is what the
      copy reported. Large propagate-hubb operation peak 0.89 (was 1.034); construction peaks 0.80–0.94.
    - **1x1:** the tiny cells' gap is a cold first call (warm calls equal). The larger cells run the same serial
      evaluator with fewer instructions and equal cache misses, yet need +6 % cycles; a closure-copy fix of the
      cosine-mask kernel removed a reload per element but measured nothing and was reverted. Not attributed.
    - **Bimodal first calls at T = 96:** slow launches wait on the run queue (5–9 ms of summed runnable-not-running
      time against 0.02–1.6 ms). No transparent huge pages (`madvise` mode) or compaction. Host noise on a fully
      bound node, on both arms.
    - **LTO:** a build without nanobind's LTO is within ±3 % on every cell measured. LTO is neither a cause nor a
      measurable benefit.
    - **Campaign protocol (owner decision):** before each batch, evict both arms' shared objects from the page cache
      (after hashing them) and run one warm-up launch per arm; interleave the arms; judge memory against fresh
      baseline samples only, since the archived samples' page-cache state is unknown.
    - **Re-measured under that protocol** (ratios against the fresh baseline):

      | Gate | Result |
      | --- | --- |
      | Memory, every measured cell | ≤ 1.009 (at most +7 MiB) |
      | Runtime, 2x48 tiny / 1x96 tiny | 0.89–0.95 / 0.49–0.72 |
      | Runtime, 1x1 tiny | 1.00–1.21 |
      | Runtime, 1x1 larger | 1.00–1.09 |
      | Runtime, pare-construct 2x48 / MPI-off | 1.11 / 0.93 |

  - Still open, reported with the final evidence: the 1x1 runtime tail (tiny cells' cold first call; the larger cells'
    unattributed +6 % cycles) and the bimodal first calls at T = 96, which hit both arms.
  - Evidence retention: S7's raw candidate term files (about 20 GB) were moved off the host by the owner on
    2026-10-08, verified against a sha256 manifest kept beside S7's reports. The baseline and Task 1's formal evidence
    are untouched.

- [x] Add/activate host-FUNNELED acceptance and insufficient-actual-support tests before lowering initialization/support
  requirements. Test initializing-thread ownership inside the library team and wrong-host-thread rejection. Ensure no
  old worker path remains executable before claiming FUNNELED.
- [x] Remove facade dispatch, custom barriers/queues/thread pinning, local collective tags and the temporary development
  selector. Keep real-MPI communication and source-order data types. Search confirms no production references remain:

```bash
rg -n 'PartitionGroup|ShmComm|HybridComm|PartitionBarrier|PartitionChildFactory|monoprop_PARTITIONS' \
  cpp src CMakeLists.txt cmake pyproject.toml justfile .github
rg -n 'hwloc|pkg.?config|SHARDED_OPENMP_PROTOTYPE' \
  CMakeLists.txt cpp cmake pyproject.toml .github
```

  Explain legitimate negative fixtures/history; production remnants fail the audit. Delete direct hwloc discovery,
  headers, link/export flags and native/wheel/Nix build inputs. An MPI/OpenMP runtime's transitive hwloc is allowed.
- [x] Preserve PUBLIC `OpenMP::OpenMP_CXX` on object and shared targets plus installed
  `find_dependency(OpenMP REQUIRED COMPONENTS CXX)`. Run `just test-find-package` against each installed R/M package,
  link-export probes, wheel-repair imports and relevant Nix/platform builds. Record unavailable qualification as
  pending.
- [x] Rebuild the final candidate, rerun full R/M C++/Python tests, installed consumers, failure drivers, stress and
  qualified sanitizers. Check ordinary aggregate/export semantics and raw-accessor rejection again on the final binary.
- [x] Take up the open optimization points recorded after S7 ("Still open: optimization points for the end of S8"):
  first-call costs of small evaluations, MPI 1x1 single-thread construction, and the constant memory overheads
  (re-measured on the post-removal binary first). Same rules as the post-S7 rounds: a separately approved
  measurement budget, profile before changing, keep a change only if measured better, full verification of what is
  kept. Points that stay open are reported with the final acceptance evidence.
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
