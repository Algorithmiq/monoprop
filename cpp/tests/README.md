# C++ Test Suite

This directory contains the C++ test suite for monoprop, built using Boost.Test.
Every `*.cpp` here is globbed into a single executable, `monoprop_unit_tests.x`.

## Test Organization

Tests carry no labels of their own. The CTest harness (`boostAddTests.cmake`)
discovers every Boost.Test case and registers it twice:

- **`serial`**: the case run in-process with `MPI_COMM_SELF`.
- **`mpi`** (+ rank-specific `mpi-<n>`): the whole suite wrapped in
  `mpiexec -n <n>` for each rank in `monoprop_MPI_TEST_PROCS` (default `2;4`),
  registered when an MPI launcher is detected.

Cases that need multiple ranks check `monoprop::mpi::size(MPI_COMM_WORLD)` and
skip (with a message) when run with too few.

The launch-time thread policy belongs to the CTest registration, not to the runner:
every discovered case (per case, and in the whole-suite MPI variants) runs with
`monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE`, so white-box tests
observe one store per rank. Teams of T > 1 are dedicated fresh-process launches
(`*_env_t*`), and `sharded_root_env_openmp_default_t3` and
`openmp_env_root_budget_openmp_default_3` remove `monoprop_NUM_THREADS` on
purpose, so the custom `main()` in `unit_tests.cpp` only initializes MPI and never
supplies the variable. Running the binary by hand uses your environment; set the
variables above to reproduce a CTest case. Nothing reads `monoprop_PARTITIONS`.

## Building Tests

The supported workflow builds the C++ suite when running `uv sync`:

```bash
uv sync --all-extras -v
ctest --test-dir build/editable/Release
```

For an MPI-enabled tree, rerun `uv sync` with `monoprop_ENABLE_MPI=ON` in the
environment.
## Running Tests

```bash
ctest --test-dir build/editable/Release           # everything
ctest --test-dir build/editable/Release -L serial # serial variants only
ctest --test-dir build/editable/Release -L mpi    # MPI variants
ctest --test-dir build/editable/Release -L mpi-2  # only the 2-rank run
```

Or drive the binary directly:

```bash
build/editable/Release/bin/monoprop_unit_tests.x --list_content
build/editable/Release/bin/monoprop_unit_tests.x --run_test=pauli_algebra_*
mpirun -n 2 build/editable/Release/bin/monoprop_unit_tests.x
```

Because CTest discovery treats each `--list_content` line as a top-level test
name and cannot address suite-nested cases, tests use flat
`BOOST_AUTO_TEST_CASE`s with a shared name prefix (e.g. `pauli_algebra_*`,
`inverted_index_*`) rather than `BOOST_AUTO_TEST_SUITE`.

## Shared Test Utilities

- **`TestUtilities.h`**: fixtures (`ExampleDataFix` = random_exact/n=8,
  `LihFixture` = LiH/n=12), the `build_simulator`/`SimulatorConfig` helpers,
  expectation-value helpers, and the `near()` float comparison used by the
  equivalence suites.
- **`PauliTestOracle.h`**: independent Pauli reference oracle — native/JW
  encoding (`slots_of_string`, `native_bitset`, `jw_basis`), dense Pauli-matrix
  brute force (`matrix_from_string`, `matmul`, ...), and string helpers. Shared
  by the Pauli algebra/build-layer tests and the equivalence suites.
- **`GraphBuildHarness.h`**: direct Layer/MPGraph construction helpers
  (`core_with_gate`, `layer_with_gate`, `graph_with_gates`) for white-box
  MPGraph transform tests.
- **`ExchangeLayoutOracle.h`**: `build_layer_exchange_layout` — the independent
  reference for a layer's exchange counts and displacements, which
  `derive_exchange_layout` in the library is checked against.
- **`dense_query_reference.h`**: the retired dense query record, frozen as the
  independent oracle for `sparse_query_tests.cpp`. Test-only, and not kept in
  sync with the wire format.
- **`PropagatorTestAccess.h`**: white-box access that `MonomialPropagator` befriends but the library
  never defines. Tests use it to read an object's captured thread budget and validity, to inject
  failures through existing members (a throwing cutoff predicate), to inspect the root's actual
  shards, router and physical rounds, to attach the test-only `RootObserver`, and to run one layer
  of the low-level one-owner-per-rank engine on the sole shard of a T = 1 root
  (`sole_shard_layer_observed`, for the failure driver); it adds no public shard interface.
- **`TestData.{h,cpp}`**: the `CaseData` struct and msgpack fixture loader.
- **`boost-test.cmake` / `boostAddTests.cmake`**: CMake test discovery.

## Test Files (by area)

- **Runner**: `unit_tests.cpp`.
- **Containers / algebra / utilities**: `bitset_tests.cpp` (the Bitset container
  vs a std::bitset oracle), `mpfunctions.cpp` (MP utilities + bit-flip helpers),
  `pauli_algebra_tests.cpp`, `majorana_cutoff_tests.cpp` (length/support cutoff,
  CutoffEvaluator, interleave phase, coeff encode/decode, cutoff_sums vs a
  bitwise reference), `validation_tests.cpp`
  (parameter validators), `mpi_utils_tests.cpp` (find_rank, word serialization,
  scan routing agreement), `routing_tests.cpp` (Router term -> flat-slot map:
  splitmix equivalence, the linear shift identity, and the coverage diagnostic),
  `evolution_detail_tests.cpp` (MatchedEpochSet + CutoffContext),
  `row_accessor_tests.cpp` (dense vs OperatorIndex row accessors).
- **Operator store**: `operator_index_tests.cpp`, `inverted_index_tests.cpp`,
  `mp_operator_tests.cpp` (MPOperator get_state Pauli/Majorana scoring,
  get_operator init-map drain, update_initial_operator picture branches,
  insert_absent_terms, inverted-index sync, memory estimate, deep copy, spare capacity
  released in place),
  `bulk_insert_tests.cpp` (the grouped-prefetch insert vs a one-key-at-a-time
  reference: table state and enumeration order).
- **Layer build / evolution**: `build_graph_tests.cpp`,
  `pauli_build_layer_tests.cpp`, `fused_cos_sweep_tests.cpp`,
  `sparse_query_tests.cpp` (the QueryWire wire record against the frozen dense
  oracle, plus the fused value channel), `sparse_resolve_tests.cpp` (probe and
  insert from wire positions vs the dense Monomial-keyed path),
  `combined_recompute_equivalence.cpp` (recompute equivalence +
  snapshot invariance), `exact_upper_atol_rescue.cpp`,
  `large_cosine_storage_tests.cpp`, `gate_boundaries.cpp`.
- **Graph encoding / packing**: `graph_encoding_tests.cpp` (CosineWordBuilder
  coalescer, checked_* overflow guards, packed-phase storage + int8 read,
  build_layer_exchange_layout, and both arms of the D-from-B derivation).
- **Graph / paring**: `pare_graph_tests.cpp`, `mpi_pare.cpp`,
  `mp_graph_tests.cpp` (MPGraph slice_graph/slice_view transforms, the
  front_offset lazy-compaction arms, MPGraphView reverse mapping + OOB throw).
- **Public surface**: `sharded_api_tests.cpp` (compile-time probes of the
  constructor, nested types, extension hooks, exceptions, the removed selector and
  headers, each negative probe paired with a positive control).
- **Transports / distribution**: `mpi_alltoallv_tests.cpp` (the real-MPI request
  helpers of the low-level engine: `begin_alltoallv`, `PendingAlltoallv`,
  `alltoall_counts` and the sparse `PeerPlan`, at world size 1 and in the whole-suite
  MPI variants), `flat_exchange_tests.cpp` (`post_flat_alltoallv`/`Ticket`),
  `mpi_distributed_layer_equivalence.cpp`, `mpi_fresh_insert_equivalence.cpp`
  (serial↔world equivalence of the Schrödinger fused-resolve fresh-insert arms,
  Majorana + native Pauli; self-skips at world size 1).
- **Simulator / operator lifecycle**: `simulator_copy_tests.cpp`,
  `update_initial_operator.cpp`, `ctor_validation_tests.cpp` (constructor guard
  rails + MPGraph bounds).
- **Exchange layout preconditions**: `exchange_layout_precondition_tests.cpp`
  (the layout width check on an ordinary communicator; rank 0 alone runs it on the
  world communicator, so a check that communicated would hang).
- **OpenMP worksharing**: `openmp_workshare_tests.cpp` (`parallel::for_blocks`:
  exactly-once visits, serial and nested fallbacks, budget and bound validation,
  unchanged runtime settings, joined worker exceptions, and actual multi-worker
  participation, which is skipped when the runtime cannot provide two workers).
  Run it in fresh processes under `OMP_DYNAMIC=TRUE`, `OMP_DYNAMIC=FALSE` and
  `OMP_THREAD_LIMIT=1` when changing the helper.
- **Fixed-team phase primitive**: `sharded_team_tests.cpp` covers `detail/sharded/Team.h`
  (`run_team`, `phase`, `TeamFailure`): each owner runs once on its own OpenMP worker, the primary is
  the calling thread, owners keep their worker across phases, empty owners reach every checkpoint,
  writes are visible in the next phase, failures (including non-`std::exception` values) are returned
  after the join from the lowest failing worker and suppress later phases, a fast worker's failure
  cannot change an earlier checkpoint's decision (also with the interleaving forced: the slow workers read
  checkpoint 0's decision only after the fast worker recorded its phase-1 failure, through the test-only
  observer of `TeamFailure::checkpoint`), and OpenMP settings are unchanged. Workers only fill
  preallocated observation slots; all assertions run after the join. The multi-phase failure cases keep
  every worker reaching the same number of checkpoints, so a broken checkpoint shows up as divergent
  decisions instead of a deadlock. `cpp/tests/CMakeLists.txt` reruns them as `sharded_team_env_t1`,
  `_t2`, `_t4` and `_t4_passive`, fresh processes with `monoprop_NUM_THREADS=OMP_NUM_THREADS=T` and
  `OMP_DYNAMIC=FALSE`; cases that need a nonprimary worker skip below T=2 and are left out of the T=1
  launch. The helper assumes exactly T workers, so these cases never run under the limited or dynamic
  teams of `openmp_env_kernels_*`, which test `for_blocks`' reduced-team contract.
- **Owner-initialized shard state**: `sharded_state_tests.cpp` covers `detail/sharded/State.h`
  (`make_shards`, `seed_shards`, `copy_shards`, `ShardState`). Seeding is checked against a reference
  derived in the test from the router, the paired-basis enumeration and the store, and aggregates
  against the root at the same launch, built on the test thread outside the team. Further cases cover flat ownership `rank * T + shard` at several
  (P, T) geometries, identity handling, empty owners, sparse/dense cache selection, pending entries,
  per-shard counts and accounting, exact and independent copies with shared graph cores, the captured
  budget, the rank-level validity guard (preservation evidence), and initializer failures on primary,
  nonprimary and concurrent owners. `AllocationProbe.cpp` replaces the global `operator new`/`delete`
  family for the whole unit-test executable: it counts per-thread allocation, tracks live bytes, and
  can make the n-th allocation of one owner throw `std::bad_alloc`. The seed and copy sweeps use it to fail every allocation site in turn and check for a caught exception,
  no leak and an intact source. The library calls a test-only observer (`NoShardObserver` in
  production) inside each owner's seed or copy body; tests use it to record the executing worker and
  its allocated bytes. `sharded_state_env_t1`, `_t2` and `_t4` rerun the cases in fresh exact-team
  processes, as for the team primitive.
- **Direct-buffer construction**: `sharded_construction_tests.cpp` covers `detail/sharded/Construction.h`
  (`build_graph`, `propagate`, `gather_published`) and the owner-local engine phases it drives
  (`LayerBuildEngine::prepare_exchange`, `resolve_published`, `consume_published`). At geometry (1, T)
  every shard's rows, IDs, coefficients, caches and graph layers equal the root's at the same launch.
  Because both share the phases, the cases also check an independent insertion-order reference, an
  independent coefficient-map propagator and the frozen exact energy of `random_exact.msgpack`; global
  retained maps across T are compared from separate processes (`tests/test_sharded_openmp.py`). Further
  cases cover participation (every
  owner's work runs on its own worker inside one team, kernels serial on the owner), the primary-only
  traversal for opaque cutoffs and basis changes, empty circuits, operators and owners, identity gates,
  the published-view seam (nonzero window bases, empty senders, 255/256/257-query blocks, narrow and
  wide positions, the malformed Plain `QueryWire<128>` fixture), same-shard streams beyond one 4096-query
  window, and failures (an injected throw in every phase, kernel, decode and publication failures,
  concurrent failures, and `AllocationProbe` sweeps) that must join, suppress later phases, report
  whether mutation started and leave earlier copies intact. The test observer records visits,
  streams and kernel ranges per shard. `sharded_construction_env_t1`, `_t2` and `_t4` rerun the cases in
  fresh exact-team processes, as for the team primitive.
- **Snapshot-safe evaluation**: `sharded_evaluation_tests.cpp` covers `detail/sharded/Evaluation.h`
  (`evaluate_shards`, `ev_sharded`, `ev_and_grad_sharded`, `replay_shards`, `prepare_retained`, the
  owner-local forward replay phases and the ascending-shard folds), the owner-local paring seam
  `detail::pare_graph_owner` and coefficient-informed construction (`sharded::build_graph_informed`). At
  geometry (1, T) every shard's evolved operator and local term, the energy and every gradient component
  equal the root's at the same launch bitwise, for every basis/picture fixture, deep-amplification and
  vanishing-cosine parameters, pared and unpared functionals, and partial contraction in both pictures. The
  replay and derivative kernels are shared with the low-level evaluator (`detail/evolution/LayerReplay.h`) and
  the root drives the same seams, so the cases also check an independent coefficient-map propagator, the frozen exact energy of `random_exact.msgpack`,
  single-gate closed-form fits and central finite differences, including the cases of
  `tests/test_deep_circuit_gradient.py`. Further cases cover whole self pairs, other-local-shard and
  multi-peer layouts, empty owners and no-work layers, the identity counted once, empty parameters and
  missing callbacks, repeated parameters and duplicate records, an empty stored pruned mask (replays nothing),
  partner staging one block per run (`EvaluationObserver::staging_run`) bitwise equal to the default runs, exact
  cosine-set counts (`CosCallbacks::count`) with records reserved once and never reallocated,
  interleaved instances, copies and index growth under retained callbacks, participation (every owner's
  work runs on its own worker; opaque callbacks run on the primary in an exclusive phase), and failures in
  every phase, in callbacks, paring and informed construction, plus `AllocationProbe` sweeps run from a fresh
  host thread so every worker starts with cold thread-local scratch. `sharded_evaluation_env_t1`, `_t2` and
  `_t4` rerun the cases in fresh exact-team processes, as for the team primitive.
- **Thread budgets and failed-owner rule**: `openmp_runtime_tests.cpp` covers the budget parser and
  its capture, per-object budgets through copies, the MPI thread-level decisions (FUNNELED
  required) and the initializing-thread guard. It also checks that OpenMP settings and affinity are
  unchanged, that old low-level calls default to serial options, and that a failed mutation
  invalidates the owner while validation errors do not. `cpp/tests/CMakeLists.txt` reruns some of these cases as
  `openmp_env_*` entries, each in a fresh process with a fixed launch environment.
- **Threaded kernels**: `openmp_kernel_tests.cpp` covers the threaded cosine scaling
  (`scale_cos_mask`, `scale_cos_lazy`) and fused rotation apply (`apply_fused_contract`): bitwise
  agreement of every budget with budget one, small work staying serial, joined worker exceptions,
  and participation. Participation is read from a test-only range observer (`KernelTestSupport.h`)
  that each kernel calls at the start of its own logical ranges; production code passes the empty
  `NoRangeObserver`, so a team opened anywhere else never counts. Lazy folds over a synthetic
  multi-block inverted index are checked against the materialised-fold oracle in
  `combined_recompute_equivalence.cpp` (cold parity cache, tails, index growth), and fused records
  from the real build path (one add-owner per slot, both pictures, cross-rank halves under MPI) in
  `fused_cos_sweep_tests.cpp`. `openmp_env_kernels_*` rerun them under `OMP_THREAD_LIMIT=2` and
  `OMP_DYNAMIC=TRUE`.
- **Threaded scan**: the `openmp_scan_*` cases compare the threaded bitmap scan
  (`fused_find_and_collect` over word ranges, merged in range order) with budget one byte for byte:
  every window slot's wire records, sources and values, both self stages, the cosine set and the
  fused-sweep coefficients. `ScanTestSupport.h` builds a synthetic operator of four fold blocks per
  rank and the scenario matrix (one rank, linear zero and non-zero shifts with non-zero window bases,
  dense splitmix at three and four ranks; dense, sparse and empty pivots; Majorana and Pauli; caps,
  cutoffs, atol boundaries, capture, fused sweep). They live in `evolution_detail_tests.cpp`,
  `pauli_build_layer_tests.cpp`, `exact_upper_atol_rescue.cpp`, `majorana_cutoff_tests.cpp` (opaque
  and basis-change cutoffs stay serial), `sparse_query_tests.cpp` (the merge itself) and
  `fused_cos_sweep_tests.cpp` (full construction: rows, row IDs, graph layers and coefficients at
  budgets 1-4). The scan's participation is read from its own ranges through the same observer.
- **Threaded resolution**: `sparse_resolve_tests.cpp` compares the incoming probe (checked
  preparation, decode, `probe_frozen_positions`, predicted IDs, publication) at budgets 2-8 with
  budget one over empty/all-hit/all-miss/mixed inputs, 255/256/257-query boundaries, empty senders,
  non-zero window bases, narrow and wide positions, spilled rows, escaped counts, Plain and Fused
  records and a prebuilt inverted index. It rejects malformed streams (including the noncanonical
  escape `QueryWire<128>` `0x8037e`) before any decode, invalid decoded positions before any probe,
  keeps the scatter serial for sinks without `parallel_resolve` and for the Schrödinger
  `ContractSink`, and checks self windows of 4096 and the publication order across leader/follower
  passes. `sparse_query_tests.cpp` covers `QueryWire::checked_extent` and position validation,
  `operator_index_tests.cpp` pins the duplicate-key behaviour of bulk insertion, and
  `mpi_fresh_insert_equivalence.cpp` compares budgets exactly and reads participation at world size.
  `PhaseLog`/`AccumulatingObserver` keep every call of a phase that runs per window or per pass.
- **Distributed failure driver**: `mpi_failure_driver.cpp` is not part of the unit runner (the glob
  excludes it). The MPI build compiles it into `monoprop_mpi_failure_driver.x`, and
  `run_mpi_failure_scenario.cmake` runs each scenario on two ranks with a 30 s timeout, as the
  `mpi_failure_*` CTest entries. The sharded root's scenarios (`mpi_failure_sharded_*`) inject at two
  threads per rank through its `RootObserver`. The low-level ones run on the sole shard of a root
  launched with one thread per rank: `active_ticket` and `active_ticket_cosine_worker` throw while
  an `evolve_step` replay `Ticket` is posted (the latter from worker 1 of the threaded cosine
  kernel), `active_pending` while a `PendingAlltoallv` is posted, `before_exchange_scan_worker` and
  `resolve_worker_self_probe` throw from a scan or self-probe worker before the query exchange, and
  `resolve_worker_decode|incoming_probe|scatter` from the incoming phases after the query round,
  while the peer enters the response round. The thread-support scenarios require a clean exit
  (`EXPECT_SUCCESS`): `library_initialized_level` (monoprop's own `mpi::init`, which requests
  FUNNELED) and `host_initialized_funneled|serialized|multiple` run real operations at two threads per
  rank on the world and on a reordered split communicator against one process;
  `insufficient_thread_level_single` must be refused and `wrong_thread_entry_*` must fail fast at
  FUNNELED, SERIALIZED and MULTIPLE. Each prints the requested and provided level; a provided level
  that does not exercise the route ends as a skip ("NOT EXERCISED", exit 77).

New `*.cpp` files are auto-discovered on the next configure, with no CMake edit
needed. A file with its own `main()` must be excluded from the glob, as
`mpi_failure_driver.cpp` is.

## Sharded OpenMP runtime suites

There is one runtime, and every suite compiles against it.

- `sharded_api_tests.cpp` checks the public surface at compile time: no constructor tail, `PartitionChildFactory`,
  facade helper, partition-named exception, cached thread-count reader, selector macro or partition/in-process
  communicator header, each probe paired with a positive control, and positive controls pin the eleven-argument
  constructor, `clone_()`, `apply_initial_operator_()` and the virtual destructor.
- `sharded_root_tests.cpp` covers the root (`ShardedPropagator.inl`): T routed shards at the (P, T) router,
  aggregates over every shard with shared metadata counted once, the raw-accessor rule (the actual sole shard at
  T = 1, `MultiShardUnsupported` naming the T = 1 launch otherwise, the validity guard first, the object still usable),
  the obsolete `monoprop_PARTITIONS` ignored bit for bit, the unset budget taking the OpenMP default, actual-worker
  participation in seeding, copies, construction and publication, informed seed replay, propagation, retained
  preparation, forward replay, reverse derivatives, contraction and the root's own owner phases, per-shard export and
  contraction blocks, initial-operator routing, remapping, ordinary virtual hooks, pre-mutation rejections,
  post-mutation invalidation from failures injected in each phase, functional epoch/graph guards, contained
  seed/copy failures and an `AllocationProbe` sweep. `sharded_root_env_t1`, `_t2`, `_t4` and `_openmp_default_t3`
  rerun it in fresh exact-team processes; its `sharded_root_multirank_*` cases run only in `sharded_root_mpi*` (which
  set `monoprop_TEST_SHARDED_RANKS`).
- `fused_cos_sweep_tests.cpp` runs the fused-versus-replay cases at the launch's T, audits the real fused records of
  every shard through the propagation seam's test hook (`RootObserver::fused_records`), and checks shard-by-shard
  determinism at fixed geometry, also as `fused_env_t2` and `_t4`.
- `openmp_runtime_tests.cpp` keeps its parser and capture cases, the launch-budget entries (`openmp_env_root_*`,
  including the unset variable), the thread-level decisions and the invalid-owner cases at the launch's T
  (`openmp_env_root_t4` too).
- `sharded_exchange_tests.cpp`: the physical round's single-process contract (including owner-parallel planning and
  the replay step's bulk path, symmetric rows and column views, against the entry-wise path) and, in
  `sharded_exchange_mpi*`, closed-form blocks between processes.

Bitwise parity with an independently implemented runtime at the same geometry, and parity across geometries, is
checked from separate processes by `tests/test_sharded_openmp.py` (with `monoprop_TEST_LEGACY_PYTHON` naming a
separately built pre-removal legacy environment), because two class definitions must never share a process.

Migration ledger, closed by the removal of the partition runtime (S8). Categories: (1) unchanged at launch-time
T = 1, (2) rewritten with private shard inspection at fixed geometry, (3) replaced by separate-process or
global-map coverage, (4) retired with its subject (the deleted runtime), with the named replacement.

| Suite or registration | Disposition | Coverage now |
| --- | --- | --- |
| `partition_equivalence_tests.cpp`, `partition_group_clone_tests.cpp` | (4) deleted | `sharded_api_tests.cpp` proves the facade and factory absent; ordinary hooks in `sharded_root_virtual_hooks_keep_working`; cross-T maps in Python |
| `shm_comm_tests.cpp`, `hybrid_comm_tests.cpp`, `ThreadHarness.h`, `SlotBlocks.h` | (4) deleted; plain-MPI arms ported | `mpi_alltoallv_tests.cpp` (sparse plans, known counts, empty legs, back-to-back rounds, unroutable plans, request ownership), `flat_exchange_tests.cpp`; in-process verbs have no subject |
| `cpu_topology_tests.cpp` | (4) deleted | no topology code remains; placement is the launcher's |
| `exchange_layout_precondition_tests.cpp` | (2) ShmComm fixture replaced | ordinary communicator; rank-0-only check on the world proves rejection before communication |
| S1-S3 seam suites' legacy-oracle cases (10) and `sharded_{construction,evaluation}_multirank_*` | (3)/(4) retired with the in-process oracle | independent references and the root at the same launch; bitwise per rank against a separately built legacy runtime in `tests/test_sharded_openmp.py` (contraction blocks pin each shard's row order) |
| `openmp_runtime_tests.cpp` one-store cases | (4) | `sharded_root_functional_failures_invalidate_their_owner`; parser, fallback, copy budget and invalid-owner cases kept |
| `fused_cos_sweep_tests.cpp` one-store and cross-budget cases | (3)/(4) | fused shard records and fixed-geometry determinism; kernel budgets in `openmp_scan_*`/`openmp_env_kernels_*`; cross-T maps in Python |
| `mpi_distributed_layer_equivalence.cpp`, `mpi_fresh_insert_equivalence.cpp` hybrid/one-store cases | (4) | `sharded_world_equivalence_mpi*`, `sharded_root_mpi*` |
| `mpi_failure_*` | (2) | sharded scenarios; low-level scenarios migrated to the T = 1 root's sole shard; FUNNELED acceptance added |
| `env_config_tests.cpp` permissive-parser cases | (4) | absence probe in `sharded_api_tests.cpp` |
| Other propagator suites | (1) T = 1 | unchanged assertions on the sole shard |
| `link_export_probe`, `find_package_smoke` | (2) | constructor shape, every operation family, derived hooks, raw-access rule at T = 1 and 2, no selector, the MPI setting, and (MPI) a two-rank comparison with one process |

## MPI Test Configuration

With an MPI launcher on PATH (`MPIEXEC_EXECUTABLE`, `mpiexec`, or `mpirun`),
CMake wraps the whole suite in `mpiexec -n <rank>` for each rank in
`monoprop_MPI_TEST_PROCS` (default `2;4`) — one CTest entry per rank count, not
per case, because the ranks have to reach the same collectives. Each entry has
a 600-second timeout, so a collective deadlock fails instead of occupying the
runner indefinitely. For exhaustive
rank coverage, run `just test-mpi '1;2;4'` or configure
`-Dmonoprop_MPI_TEST_PROCS='1;2;4'`. To run a single case under MPI while
debugging, invoke the binary directly:
`mpirun -n 2 build/editable/Release/bin/monoprop_unit_tests.x --run_test=<case>`.

## Adding New Tests

1. Add a `*.cpp` with flat `BOOST_AUTO_TEST_CASE`s (shared name prefix).
2. Reuse the shared helpers above rather than copying oracle/harness code.
3. For MPI-required scenarios, check `monoprop::mpi::size(MPI_COMM_WORLD)` and
   skip if `< 2`.
4. Rebuild to register the new cases with CTest.
