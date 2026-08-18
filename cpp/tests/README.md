# C++ Test Suite

This directory contains the C++ test suite for monoprop, built using Boost.Test.
Every `*.cpp` here is globbed into a single executable, `monoprop_unit_tests.x`.

## Test Organization

Tests carry no labels of their own. The CTest harness (`boostAddTests.cmake`)
discovers every Boost.Test case and registers it twice:

- **`serial`**: the case run in-process with `MPI_COMM_SELF`.
- **`mpi`** (+ rank-specific `mpi-<n>`): the whole suite wrapped in
  `mpiexec -n <n>` for each rank in `monoprop_MPI_TEST_PROCS` (default `2`),
  registered when an MPI launcher is detected.

Cases that need multiple ranks check `monoprop::mpi::size(MPI_COMM_WORLD)` and
skip (with a message) when run with too few.

The custom `main()` in `unit_tests.cpp` initializes MPI and forces
`monoprop_PARTITIONS=off`, so white-box tests observe the single-partition engine.
A test that needs the partition runtime must pass an explicit `partitions=` argument
(see `partition_equivalence_tests.cpp`).

## Building Tests

The supported workflow builds the C++ suite when running `uv sync`:

```bash
uv sync --all-extras -v
ctest --test-dir build/editable/Release
```

For an MPI-enabled tree, rerun `uv sync` with
`--config-settings=cmake.define.monoprop_ENABLE_MPI=ON`.
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
- **`ThreadHarness.h`**: `run_comm_threads` — spawn S partition threads over a
  transport and capture per-thread exceptions (used by the ShmComm/HybridComm
  suites).
- **`GraphBuildHarness.h`**: direct Layer/MPGraph construction helpers
  (`core_with_gate`, `layer_with_gate`, `graph_with_gates`) for white-box
  MPGraph transform tests.
- **`TestData.{h,cpp}`**: the `CaseData` struct and msgpack fixture loader.
- **`boost-test.cmake` / `boostAddTests.cmake`**: CMake test discovery.

## Test Files (by area)

- **Runner**: `unit_tests.cpp`.
- **Containers / algebra / utilities**: `bitset_tests.cpp` (the Bitset container
  vs a std::bitset oracle), `mpfunctions.cpp` (MP utilities + bit-flip helpers),
  `pauli_algebra_tests.cpp`, `majorana_cutoff_tests.cpp` (length/support cutoff,
  CutoffEvaluator, interleave phase, coeff encode/decode), `validation_tests.cpp`
  (parameter validators), `mpi_utils_tests.cpp` (find_rank + word serialization),
  `evolution_detail_tests.cpp` (MatchedEpochSet + CutoffContext),
  `row_accessor_tests.cpp` (dense vs OperatorIndex row accessors).
- **Operator store**: `operator_index_tests.cpp`, `inverted_index_tests.cpp`,
  `mp_operator_tests.cpp` (MPOperator get_state Pauli/Majorana scoring,
  get_operator init-map drain, update_initial_operator picture branches,
  insert_absent_terms, inverted-index sync, memory estimate, deep copy).
- **Layer build / evolution**: `build_graph_tests.cpp`,
  `pauli_build_layer_tests.cpp`, `fused_cos_sweep_tests.cpp`,
  `fused_query_codec_tests.cpp`, `combined_recompute_equivalence.cpp`
  (recompute equivalence + snapshot invariance), `exact_upper_atol_rescue.cpp`,
  `large_cosine_storage_tests.cpp`, `gate_boundaries.cpp`.
- **Graph encoding / packing**: `graph_encoding_tests.cpp` (CosineWordBuilder
  coalescer, checked_* overflow guards, packed-phase storage + int8 read,
  build_layer_exchange_layout, and both arms of the D-from-B derivation).
- **Graph / paring**: `pare_graph_tests.cpp`, `mpi_pare.cpp`,
  `mp_graph_tests.cpp` (MPGraph layer indexing under either arrival order,
  replay_view/contraction_view, replace_layer, clear, MPGraphView reverse
  mapping + OOB throw).
- **Transports / distribution**: `shm_comm_tests.cpp`, `hybrid_comm_tests.cpp`
  (MPI-only), `partition_equivalence_tests.cpp`,
  `mpi_distributed_layer_equivalence.cpp`, `mpi_fresh_insert_equivalence.cpp`
  (serial↔world equivalence of the Schrödinger fused-resolve fresh-insert arms,
  Majorana + native Pauli; self-skips at world size 1).
- **Simulator / operator lifecycle**: `simulator_copy_tests.cpp`,
  `update_initial_operator.cpp`, `ctor_validation_tests.cpp` (constructor guard
  rails + MPGraph bounds).

New `*.cpp` files are auto-discovered on the next configure — no CMake edit
needed.

## MPI Test Configuration

With an MPI launcher on PATH (`MPIEXEC_EXECUTABLE`, `mpiexec`, or `mpirun`),
CMake wraps the whole suite in `mpiexec -n <rank>` for each rank in
`monoprop_MPI_TEST_PROCS` (default `2`) — one CTest entry per rank count, not
per case, because the ranks have to reach the same collectives. For exhaustive
rank coverage: `-Dmonoprop_MPI_TEST_PROCS='1;2;4'`. To run a single case under
MPI while debugging, invoke the binary directly:
`mpirun -n 2 build/editable/Release/bin/monoprop_unit_tests.x --run_test=<case>`.

## Adding New Tests

1. Add a `*.cpp` with flat `BOOST_AUTO_TEST_CASE`s (shared name prefix).
2. Reuse the shared helpers above rather than copying oracle/harness code.
3. For MPI-required scenarios, check `monoprop::mpi::size(MPI_COMM_WORLD)` and
   skip if `< 2`.
4. Rebuild to register the new cases with CTest.
