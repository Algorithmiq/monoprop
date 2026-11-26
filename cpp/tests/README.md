# C++ Test Suite

This directory contains the C++ test suite for monoprop, built using Boost.Test.
Every `*.cpp` here is globbed into a single executable, `monoprop_unit_tests.x`.

## Test Organization

Tests carry no labels of their own. The CTest harness (`boostAddTests.cmake`)
discovers every Boost.Test case and registers it twice:

- **`serial`**: the case run in-process with `MPI_COMM_SELF`.
- **`mpi`** (+ rank-specific `mpi-<n>`): the whole suite wrapped in
  `mpiexec -n <n>` for each rank in `monoprop_MPI_TEST_PROCS` (default `2;4`,
  also labelled `mpi-linear`), plus splitmix-routing entries for each rank in
  `monoprop_MPI_SPLITMIX_TEST_PROCS` (default `2;3;4`, also labelled
  `mpi-splitmix`; a CMake cache variable only, set through `SKBUILD_CMAKE_DEFINE`),
  registered when an MPI launcher is detected.

Cases that need a particular world declare it with a precondition from
`RankPreconditions.h`; Boost reports them as skipped elsewhere, and CTest marks
those serial entries as skipped.

The launch-time thread policy belongs to the CTest registration, not to the runner:
every discovered case (per case, and in the whole-suite MPI variants) runs with
`monoprop_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE monoprop_TEST_EXPECT_TEAM=1`, so
white-box tests observe one store per rank. Exact teams are dedicated fresh-process launches
(`*_env_t*`), and `sharded_root_env_openmp_default_t3` removes `monoprop_NUM_THREADS` on
purpose, so the custom `main()` in `unit_tests.cpp` only initializes MPI and never
supplies the variable. Running the binary by hand uses your environment; set the
variables above to reproduce a CTest case.

## Building Tests

The supported workflow builds the C++ suite when running `uv sync`:

```bash
uv sync --all-extras -v
ctest --test-dir build/editable/Release
```

For an MPI-enabled tree, rerun `uv sync` with `monoprop_ENABLE_MPI=ON` in the
environment. MPI builds go to their own tree, `build/editable/Release-mpi`, so
the MPI variants below are registered there.
## Running Tests

```bash
ctest --test-dir build/editable/Release               # everything
ctest --test-dir build/editable/Release -L serial     # serial variants only
ctest --test-dir build/editable/Release-mpi -L mpi    # MPI variants (MPI tree)
ctest --test-dir build/editable/Release-mpi -L mpi-2  # only the 2-rank run
```

Or drive the binary directly:

```bash
build/editable/Release/bin/monoprop_unit_tests.x --list_content
build/editable/Release/bin/monoprop_unit_tests.x --run_test=pauli_algebra_*
mpirun -n 2 build/editable/Release-mpi/bin/monoprop_unit_tests.x
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
  shards, router and physical rounds, and to attach the test-only `RootObserver`; it adds no public
  shard interface.
- **`RankPreconditions.h`**: Boost preconditions `min_ranks`, `power_of_two_ranks`,
  `even_ranks`, `linear_routing`, `default_routing`.
- **`ShardedTestSupport.h`**: the launch's captured team (`team_options`, `team_size`) and the team it
  names (`expected_team`), the Boost preconditions `has_nonprimary_worker`, `has_allocation_probe` and
  `dedicated_launch` (stacked on the world-size preconditions of `RankPreconditions.h`), `disarm_team`, `ScopedEnv`,
  failure inspection (`InjectedFailure`, `message_of`, `require_success`, `holds_bad_alloc`), the exact-energy
  tolerance `exact_atol`, and exact keys and bit patterns (`key_of`, `bits`) for the sharded and OpenMP runtime
  suites.
- **`ShardedFixtures.h`**: the operator, circuit and shard-seeding fixtures (`fixture_cases`,
  `exact_reference_cases`, `seed`, `Sharded`), graph construction (`run_graph`, `build_checked`,
  `seed_and_build`), phase-visit recording
  (`record_visit`), and the references `reference_root` and `reference_propagate` (a coefficient-map propagator),
  shared by `sharded_state_tests.cpp`, `sharded_construction_tests.cpp` and `sharded_evaluation_tests.cpp`.
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
  scan routing agreement, the sparse peer plan, the physical reductions), `routing_tests.cpp` (Router term -> flat-slot map:
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
- **Transports / distribution**: `flat_exchange_tests.cpp` (`post_flat_alltoallv`/`Ticket`),
  `mpi_distributed_layer_equivalence.cpp`, `mpi_fresh_insert_equivalence.cpp`
  (serial↔world equivalence of the Schrödinger fused-resolve fresh-insert arms,
  Majorana + native Pauli; self-skips at world size 1),
  `mpi_routing_geometry_tests.cpp` (linear routing refuses a 3-rank
  communicator).
- **Simulator / operator lifecycle**: `simulator_copy_tests.cpp`,
  `update_initial_operator.cpp`, `ctor_validation_tests.cpp` (constructor guard
  rails + MPGraph bounds).
- **Per-term coefficients and expectation values**: `evolved_operator_coefficients_tests.cpp` checks
  `evolved_operator_coefficients()` against `evolved_operator_terms()`, and `term_expectation_values_tests.cpp`
  checks `term_expectation_values()` against `expectation_value()`, at the launch's T.
  `evolved_coefficients_env_t2` and `_t4`, and `term_expectation_values_env_t2` and `_t4`, rerun them with
  several shards per rank.
- **Exchange layout preconditions**: `exchange_layout_precondition_tests.cpp`
  (the layout width check on an ordinary communicator; rank 0 alone runs it on the
  world communicator, so a check that communicated would hang).
- **OpenMP worksharing**: `openmp_workshare_tests.cpp` (`parallel::for_blocks`:
  exactly-once visits, serial and nested fallbacks, budget and bound validation,
  unchanged runtime settings, joined worker exceptions, and actual multi-worker
  participation, which is skipped when the runtime cannot provide two workers).
  Run it in fresh processes under `OMP_DYNAMIC=TRUE`, `OMP_DYNAMIC=FALSE` and
  `OMP_THREAD_LIMIT=1` when changing the helper.
- **Fixed-team phase primitive**: `sharded_team_tests.cpp` covers `detail/sharded/Team.h` (`run_team`, `phase`,
  `TeamFailure`): owners on their own workers, checkpoint decisions (an interleaving forced through the test-only
  `TeamFailure::checkpoint` observer) and failures returned after the join. `sharded_team_env_t1`, `_t2`, `_t4` and
  `_t4_passive` (`OMP_WAIT_POLICY=PASSIVE`) rerun it (120 s); the T = 4 entries need four cores.
- **Owner-initialized shard state**: `sharded_state_tests.cpp` covers `detail/sharded/State.h` against a seeding
  reference derived from the router and the store, and the root at the same launch; `AllocationProbe.cpp` (the
  executable's counting, failure-injecting `operator new`) fails every seed and copy allocation in turn.
  `sharded_state_env_t1`, `_t2` and `_t4` rerun it (300 s).
- **Direct-buffer construction**: `sharded_construction_tests.cpp` covers `detail/sharded/Construction.h` and the engine
  phases it drives against the root at geometry (1, T), an insertion-order reference, a coefficient-map propagator and
  the frozen exact energy; the test observer records visits and kernel ranges per shard.
  `sharded_construction_env_t1`, `_t2` and `_t4` rerun it (900 s).
- **Snapshot-safe evaluation**: `sharded_evaluation_tests.cpp` covers `detail/sharded/Evaluation.h`,
  `detail::pare_graph_owner` and `sharded::build_graph_informed`, bitwise against the root at (1, T) and against a
  coefficient-map propagator, closed-form fits and finite differences. `sharded_evaluation_env_t1`, `_t2` and `_t4`
  rerun it (900 s).
- **Root**: `sharded_root_tests.cpp` covers the root (`ShardedPropagator.inl`) through its public operations, with
  shard inspection, phase observation and fault injection (`PropagatorTestAccess.h`), including the raw-accessor rule
  (`MultiShardUnsupported` beyond T = 1). `sharded_root_env_t1`, `_t2` and `_t4` rerun it (900 s), and
  `sharded_root_env_openmp_default_t3` without `monoprop_NUM_THREADS` (`OMP_NUM_THREADS=3`); its
  `sharded_root_multirank_*` cases run only in `sharded_root_mpi*`.
- **Fused records**: `fused_cos_sweep_tests.cpp` checks the fused sweep against graph replay, audits every shard's real
  fused records through `RootObserver::fused_records`, and checks shard-by-shard determinism at fixed geometry.
  `fused_env_t2` and `_t4` rerun it (900 s).
- **Physical round**: `sharded_exchange_tests.cpp` covers `detail/sharded/Exchange.h` in one process (the bulk replay
  path against the entry-wise path) and, in `sharded_exchange_mpi*`, closed-form blocks between processes.
- **Thread budgets and failed-owner rule**: `openmp_runtime_tests.cpp` covers the budget parser and capture, the MPI
  thread-level decisions (FUNNELED required), the initializing-thread guard and the invalid-owner rule. Fresh-process
  reruns: `openmp_env_root_budget_1` and `_3` (`monoprop_NUM_THREADS=1`, `=3`), `openmp_env_root_t4` (exact team of
  four) and `openmp_env_no_affinity_change` (`OMP_PROC_BIND=false`).
- **Threaded kernels**: `openmp_kernel_tests.cpp` checks `scale_cos_mask`, `scale_cos_lazy` and `apply_fused_contract`
  bitwise against budget one, with participation read from the range observer of `KernelTestSupport.h`; multi-block
  lazy folds are checked against the materialised fold in `combined_recompute_equivalence.cpp`.
  `openmp_env_kernels_thread_limit_2` and `_dynamic_teams` rerun the kernel, scan and resolve cases under
  `OMP_THREAD_LIMIT=2` and `OMP_DYNAMIC=TRUE`.
- **Threaded scan**: the `openmp_scan_*` cases (in `evolution_detail_tests.cpp`, `pauli_build_layer_tests.cpp`,
  `exact_upper_atol_rescue.cpp`, `majorana_cutoff_tests.cpp`, `sparse_query_tests.cpp` and `fused_cos_sweep_tests.cpp`)
  compare the threaded `fused_find_and_collect` with budget one byte for byte over the scenarios of `ScanTestSupport.h`.
- **Threaded resolution**: `sparse_resolve_tests.cpp` checks the incoming decode, `probe_frozen_positions`, publication
  and scatter at budgets 2-8 against budget one, and malformed streams refused before any decode; `PhaseLog` keeps
  every call of a per-window phase. `sparse_query_tests.cpp` covers `QueryWire::checked_extent`.
- **Distributed failure driver**: `mpi_failure_driver.cpp` has its own `main()` (excluded from the glob) and builds
  `monoprop_mpi_failure_driver.x` in MPI builds; `run_mpi_failure_scenario.cmake` runs each scenario on two ranks with a
  30 s timeout as an `mpi_failure_*` entry (labels `mpi`, `mpi_failure`). A failure scenario passes only on a prompt
  nonzero exit with the expected diagnostic, an acceptance scenario (`EXPECT_SUCCESS`) only on exit 0; a timeout means
  a rank was left blocked, and fails.
  - `mpi_failure_sharded_*`, at two threads per rank through `RootObserver` (rank 0 fails, its peer follows the normal
    phase): `worker_throw` (worker 1 throws in a replay finish), `before_exchange` (the primary throws before any
    physical round), `active_round` and `active_local_round` (a cosine pass while the replay round, or a caller-local
    replay round, is posted), `active_counts` (packing queries while the count round is posted) and
    `malformed_receive` (rank 1 sends malformed query blocks, which rank 0's owners must refuse).
  - `active_ticket` and `active_ticket_cosine_worker`: the exported replay functions on the sole shard of a root at one
    thread per rank throw while an `evolve_step` replay `Ticket` is posted (the latter from worker 1 of the threaded
    cosine kernel).
  - Thread support: `library_initialized_level` (monoprop's `mpi::init`, requesting FUNNELED) and
    `host_initialized_funneled|serialized|multiple` must complete real operations at two threads per rank, on the
    world and a reordered split communicator; `insufficient_thread_level_single` must be refused; `wrong_thread_entry_*`
    (a guarded call from a `std::thread`) must fail fast at every level, relying on the launcher ending the peers of an
    aborting process (Open MPI's default). A provided level that does not exercise the route ends as a skip
    ("NOT EXERCISED", exit 77).

Every fixed-T entry is a fresh process, since a propagator reads its thread budget once, when it is constructed; no
case changes T inside a process, and results across T are compared from separate processes
(`tests/test_sharded_openmp.py`). The sharded suites keep an `_env_t1` launch because the per-case entries, which
also run at T = 1, get only CTest's default timeout.

New `*.cpp` files are auto-discovered on the next configure, with no CMake edit
needed. A file with its own `main()` must be excluded from the glob, as
`mpi_failure_driver.cpp` is.

## MPI Test Configuration

With an MPI launcher on PATH (`MPIEXEC_EXECUTABLE`, `mpiexec`, or `mpirun`),
CMake wraps the whole suite in `mpiexec -n <rank>` for each rank in
`monoprop_MPI_TEST_PROCS` (default `2;4`) — one CTest entry per rank count, not
per case, because the ranks have to reach the same collectives. Each entry has
a 600-second timeout, so a collective deadlock fails instead of occupying the
runner indefinitely. `just test-mpi` runs the same entries CI does. To run a single case under MPI while
debugging, invoke the binary directly:
`mpirun -n 2 build/editable/Release-mpi/bin/monoprop_unit_tests.x --run_test=<case>`.

Multi-process cases of the sharded runtime have their own launches, one per P ranks x T threads, labelled `mpi`
and `mpi-<P>`, with a 300-second timeout. Each sets `monoprop_NUM_THREADS`, `OMP_NUM_THREADS` and
`monoprop_TEST_EXPECT_TEAM` to T with `OMP_DYNAMIC=FALSE` and `OMP_WAIT_POLICY=PASSIVE`, leaves binding to OpenMP
(`OMPI_MCA_hwloc_base_binding_policy=none`), allows oversubscription, and names the world size in
`monoprop_TEST_SHARDED_RANKS`; a multi-rank case skips without it, so the whole-suite variants never run one at
their default budget.

| Entries | What they run |
| --- | --- |
| `sharded_exchange_mpi2_t*`, `_mpi3_t*` | the physical round (`detail/sharded/Exchange.h`): closed-form blocks over both transports, with exchanged and known counts (count blocks of every wire width), the XOR peer, empty rounds, staging reuse, request moves |
| `sharded_root_mpi2_t*`, `_mpi4_t*`, `_mpi3_t2_splitmix`, `_mpi3_t2_linear_rejected` | the root over P ranks against one process, raw access and derived hooks per rank, MPI on the primary only, the owner's physical rounds reused across operations and never shared by a copy, construction rounds unmapping their staging when a construction ends while the replay round keeps its staging, and the unroutable-geometry rejection |
| `sharded_world_equivalence_mpi2_t*`, `_mpi4_t*` | the serial-against-world cases of `mpi_distributed_layer_equivalence.cpp` and `mpi_fresh_insert_equivalence.cpp` |
| `monoprop_unit_tests.x_mpi_<n>`, `monoprop_unit_tests.x_mpi_<n>_splitmix` | the whole suite at T = 1 per rank |

A `_t*` suffix covers T = 1, 2 and 4. Every MPI build with a launcher registers all of them.

## Adding New Tests

1. Add a `*.cpp` with flat `BOOST_AUTO_TEST_CASE`s (shared name prefix).
2. Reuse the shared helpers above rather than copying oracle/harness code.
3. For a case that needs several ranks, add a precondition:
   `BOOST_AUTO_TEST_CASE(name, *test_utils::min_ranks(2))`. Never `return` early:
   that passes without checking anything.
4. Rebuild to register the new cases with CTest.
