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
white-box tests observe one store per rank. The cases that depend on T rerun in one fresh-process
launch at T = 4 (`sharded_env_t4`), so the custom `main()` in `unit_tests.cpp` only initializes MPI
and never supplies the variable. Running the binary by hand uses your environment; set the
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
  failures through existing members (a throwing cutoff predicate) and to inspect the root's actual
  shards and router; it adds no public shard interface.
- **`RankPreconditions.h`**: Boost preconditions `min_ranks`, `power_of_two_ranks`,
  `even_ranks`, `linear_routing`, `default_routing`.
- **`ShardedTestSupport.h`**: the launch's captured team (`team_options`) and the team it names (`expected_team`),
  `ScopedEnv` and `message_of`.
- **`KernelTestSupport.h`**: `make_op`, `column_rows`, `distinct_values` and `bitwise_equal`.
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
  oracle, the fused value channel, and `QueryWire::checked_extent` refusing malformed records before any
  decode), `sparse_resolve_tests.cpp` (probe and
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
  checks `term_expectation_values()` against `expectation_value()`, at the launch's T; `sharded_env_t4` reruns
  them with several shards per rank.
- **Exchange layout preconditions**: `exchange_layout_precondition_tests.cpp`
  (the layout width check on an ordinary communicator; rank 0 alone runs it on the
  world communicator, so a check that communicated would hang).
- **OpenMP worksharing**: `openmp_workshare_tests.cpp` (`parallel::for_blocks` rejoins its workers before it
  rethrows a worker's exception).
- **Root**: `sharded_root_tests.cpp` covers the root (`ShardedPropagator.inl`) through its public operations, with
  shard inspection (`PropagatorTestAccess.h`): the raw-accessor rule (`MultiShardUnsupported` beyond T = 1), owner
  routing, rejections before mutation, empty operators, and (`sharded_root_multirank_*`) P ranks against one process.
- **Reference energies at T > 1**: `build_graph_tests.cpp` and `fused_cos_sweep_tests.cpp` check the frozen exact
  energy of `random_exact.msgpack` at T = 1 per case and at T = 4 in `sharded_env_t4`.
- **Physical round**: `sharded_exchange_tests.cpp` covers `detail/sharded/Exchange.h` between processes: closed-form
  blocks over both transports, with exchanged and known counts, count blocks of every wire width, the XOR peer, empty
  rounds and the per-peer `int` bound.
- **Thread budgets and failed-owner rule**: `openmp_runtime_tests.cpp` covers the strict budget parser, its capture at
  construction, unchanged OpenMP settings and the invalid-owner rule.
- **Threaded kernels**: `openmp_kernel_tests.cpp` checks `scale_cos_mask` at budgets 2-64 bitwise against budget one
  and a hand oracle.
- **Distributed failure driver**: `mpi_failure_driver.cpp` has its own `main()` (excluded from the glob) and builds
  `monoprop_mpi_failure_driver.x` in MPI builds; `run_mpi_failure_scenario.cmake` runs each scenario on two ranks with a
  30 s timeout as an `mpi_failure_*` entry (labels `mpi`, `mpi_failure`). A failure scenario passes only on a prompt
  nonzero exit with the expected diagnostic, an acceptance scenario (`EXPECT_SUCCESS`) only on exit 0; a timeout means
  a rank was left blocked, and fails.
  - `build_failure`: rank 0's cutoff predicate throws in a build at two threads per rank that has started mutating.
  - `active_ticket`: the exported `evolve_step` on the sole shard of a root at one thread per rank throws while its
    replay `Ticket` is posted.
  - Thread support: `host_initialized_funneled` must complete real operations at two threads per rank, on the world
    and a reordered split communicator; `insufficient_thread_level_single` must be refused;
    `wrong_thread_entry_funneled` (a guarded call from a `std::thread`) must fail fast, relying on the launcher ending
    the peers of an aborting process (Open MPI's default). A provided level that does not exercise the route ends as a
    skip ("NOT EXERCISED", exit 77).

`sharded_env_t4` is a fresh process, since a propagator reads its thread budget once, when it is constructed;
results across T are compared from separate processes (`tests/test_sharded_openmp.py`).

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

The multi-rank cases of the sharded runtime (`sharded_root_multirank_*`, `sharded_exchange_*` and the
serial-against-world cases of `mpi_distributed_layer_equivalence.cpp` and `mpi_fresh_insert_equivalence.cpp`) run in
the whole-suite variants at T = 1 and again in `sharded_mpi2_t2`: two ranks of two threads each, labelled `mpi` and
`mpi-2`, with a 300-second timeout, `OMP_WAIT_POLICY=PASSIVE` and binding left to OpenMP.

## Adding New Tests

1. Add a `*.cpp` with flat `BOOST_AUTO_TEST_CASE`s (shared name prefix).
2. Reuse the shared helpers above rather than copying oracle/harness code.
3. For a case that needs several ranks, add a precondition:
   `BOOST_AUTO_TEST_CASE(name, *test_utils::min_ranks(2))`. Never `return` early:
   that passes without checking anything.
4. Rebuild to register the new cases with CTest.
