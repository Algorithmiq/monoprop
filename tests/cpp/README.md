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
`monoprop_SHARDS=off`, so white-box tests observe the single-partition engine.
A test that needs the shard runtime must pass an explicit `shards=` argument
(see `shard_equivalence_tests.cpp`).

## Building Tests

Built automatically with CMake (skipped under scikit-build wheels):

```bash
cmake --preset release-gcc-mpi && cmake --build --preset release-gcc-mpi
```

or the plain form:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -Dmonoprop_ENABLE_MPI=ON
cmake --build build/release
```

## Running Tests

```bash
ctest --preset release-gcc-mpi              # everything
ctest --test-dir build/... -L serial        # serial variants only
ctest --test-dir build/... -L mpi           # MPI variants
ctest --test-dir build/... -L mpi-2         # only the 2-rank run
```

Or drive the binary directly:

```bash
./tests/cpp/monoprop_unit_tests.x --list_content
./tests/cpp/monoprop_unit_tests.x --run_test=pauli_algebra_*
mpirun -n 2 ./tests/cpp/monoprop_unit_tests.x
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
- **`ThreadHarness.h`**: `run_comm_threads` — spawn S shard threads over a
  transport and capture per-thread exceptions (used by the ShmComm/HybridComm
  suites).
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
- **Operator store**: `operator_index_tests.cpp`, `inverted_index_tests.cpp`.
- **Layer build / evolution**: `build_graph_tests.cpp`,
  `pauli_build_layer_tests.cpp`, `fused_cos_sweep_tests.cpp`,
  `fused_query_codec_tests.cpp`, `combined_recompute_equivalence.cpp`
  (recompute equivalence + snapshot invariance), `exact_upper_atol_rescue.cpp`,
  `large_cosine_storage_tests.cpp`, `gate_boundaries.cpp`.
- **Graph encoding / packing**: `graph_encoding_tests.cpp` (CosineWordBuilder
  coalescer, checked_* overflow guards, packed-phase storage + int8 read,
  build_layer_exchange_layout_impl, and both arms of the D-from-B derivation).
- **Graph / paring**: `pare_graph_tests.cpp`, `mpi_pare.cpp`.
- **Transports / distribution**: `shm_comm_tests.cpp`, `hybrid_comm_tests.cpp`
  (MPI-only), `shard_equivalence_tests.cpp`,
  `mpi_distributed_layer_equivalence.cpp`.
- **Simulator / operator lifecycle**: `simulator_copy_tests.cpp`,
  `update_initial_operator.cpp`, `ctor_validation_tests.cpp` (constructor guard
  rails + MPGraph bounds).

New `*.cpp` files are auto-discovered on the next configure — no CMake edit
needed.

## MPI Test Configuration

With an MPI launcher on PATH (`MPIEXEC_EXECUTABLE`, `mpiexec`, or `mpirun`),
CMake wraps the suite in `mpiexec -n <rank>` for each rank in
`monoprop_MPI_TEST_PROCS` (default `2`). For exhaustive rank coverage:
`-Dmonoprop_MPI_TEST_PROCS='1;2;4'`. For per-test MPI expansion (debugging):
`-Dmonoprop_MPI_TEST_LAYOUT=per-test`.

## Adding New Tests

1. Add a `*.cpp` with flat `BOOST_AUTO_TEST_CASE`s (shared name prefix).
2. Reuse the shared helpers above rather than copying oracle/harness code.
3. For MPI-required scenarios, check `monoprop::mpi::size(MPI_COMM_WORLD)` and
   skip if `< 2`.
4. Rebuild to register the new cases with CTest.
