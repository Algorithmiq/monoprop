# C++ Test Suite

This directory contains the C++ test suite for monoprop, built using Boost.Test.

## Test Organization

Tests are organized using Boost.Test labels to indicate their MPI requirements:

- **`serial`**: Tests that don't use MPI and run with `MPI_COMM_SELF`
- **`mpi-optional`**: Tests that work with both serial and MPI execution (test both `MPI_COMM_SELF` and `MPI_COMM_WORLD`)
- **`mpi-required`**: Tests that require multiple MPI ranks to function correctly

## Building Tests

Tests are built automatically when building monoprop with CMake (unless using scikit-build):

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -Dmonoprop_ENABLE_MPI=ON
cmake --build build/release
```

## Running Tests

### Run All Tests

To run all discovered tests (serial + MPI variants when `mpiexec` is available):

```bash
cd build/release
ctest --output-on-failure
```

### Run Tests by Label

Every discovered Boost.Test case is registered in serial mode and as MPI variants
for ranks configured by `monoprop_MPI_TEST_PROCS` (default: `2`). By default,
MPI runs are registered as suite-level tests (`unit_tests.x_mpi_<rank>`) to
avoid per-case test explosion. Use CTest labels to target the desired execution mode:

```bash
# Run only the serial variants
ctest -L serial --output-on-failure

# Run only the MPI variants
ctest -L mpi --output-on-failure

# Run only a specific MPI rank variant (if configured)
ctest -L mpi-2 --output-on-failure

# Run everything (default)
ctest --output-on-failure
```

### Run Specific Test Directly

You can also run the test executable directly:

```bash
# List all available tests
./tests/cpp/unit_tests.x --list_content

# Run a specific test
./tests/cpp/unit_tests.x --run_test=random_exact_infinite_cutoff_expval

# Run all tests with a specific label
./tests/cpp/unit_tests.x --run_test=@serial
./tests/cpp/unit_tests.x --run_test=@mpi-optional

# Run with MPI (single test case)
mpirun -n 2 ./tests/cpp/unit_tests.x --run_test=random_exact_infinite_cutoff_expval
```

## Test Files

- **`unit_tests.cpp`**: Main test runner with MPI initialization
- **`build_graph_tests.cpp`**: Tests for graph construction and layer storage behaviour
- **`infinite_cutoff.cpp`**: Tests with infinite cutoff across threading/MPI modes (MPI-optional)
- **`large_cosine_storage_tests.cpp`**: Coverage for compressed cosine and execution-plan storage edge cases
- **`mpi_compat.cpp`**: MPI compatibility wrappers and size_t collective coverage
- **`mpi_pare.cpp`**: MPI-specific tests requiring multiple ranks (MPI-required)
- **`mpfunctions.cpp`**: Unit tests for MP utility functions (serial)
- **`update_initial_operator.cpp`**: Tests for updating the initial operator in both pictures
- **`utilities.cpp`**: Unit tests for general utilities (serial)
- **`word_width_mpi_equivalence.cpp`**: Regression tests for extended word-width and MPI equivalence

## Test Utilities

- **`TestUtilities.h`**: Shared test helpers and fixtures
- **`boost-test.cmake`**: CMake integration for test discovery
- **`boostAddTests.cmake`**: Script to automatically discover Boost.Test cases

## MPI Test Configuration

As long as an MPI launcher is detected (`MPIEXEC_EXECUTABLE`, or `mpiexec`/`mpirun` on PATH), CMake automatically creates MPI
suite tests by wrapping `unit_tests.x` with `mpiexec -n <rank>` for each configured rank in
`monoprop_MPI_TEST_PROCS` (default: `2`). MPI suite variants carry the `mpi` label and
rank-specific labels (`mpi-2`, etc.), so `ctest -L mpi` runs all distributed
variants while `ctest -L mpi-2` targets only rank-2 runs.

For exhaustive rank coverage, configure with:
`-Dmonoprop_MPI_TEST_PROCS='1;2;4'`

If you need old per-test MPI expansion for debugging, configure with
`-Dmonoprop_MPI_TEST_LAYOUT=per-test`.

## Design Principles

1. **Tests remain label-free** – the CTest harness assigns `serial`/`mpi` labels automatically
2. **MPI-optional tests gracefully handle both serial and parallel execution**
3. **MPI-required tests skip with a message when run with insufficient ranks**
4. **The test suite uses a single executable** (`unit_tests.x`) for simplicity
5. **CTest provides flexible filtering** by test name or label
6. **MPI initialization/finalization is handled once** in the main test runner

## Adding New Tests

When adding new tests:

1. For MPI-required scenarios, check `monoprop::mpi::size(MPI_COMM_WORLD)` and skip if < 2
2. For MPI-optional tests, exercise both `MPI_COMM_SELF` and `MPI_COMM_WORLD` communicators when practical
3. Rebuild to register new tests with CTest

Example:

```cpp
BOOST_AUTO_TEST_CASE(my_serial_test) {
    // Test that doesn't use MPI
}

BOOST_AUTO_TEST_CASE(my_mpi_optional_test) {
    // Test that works with both MPI_COMM_SELF and MPI_COMM_WORLD
    MPI_Comm comm = MPI_COMM_WORLD;
    // ...
}

BOOST_AUTO_TEST_CASE(my_mpi_required_test) {
    const int world_size = monoprop::mpi::size(MPI_COMM_WORLD);
    if (world_size < 2) {
        BOOST_TEST_MESSAGE("Skipping test: requires MPI world size >= 2");
        return;
    }
    // Test that requires multiple MPI ranks
}
```
