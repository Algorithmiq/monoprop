find_package(Boost 1.85 COMPONENTS unit_test_framework REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

cpmaddpackage(
  NAME "msgpack-cxx"
  GIT_REPOSITORY "https://github.com/msgpack/msgpack-c"
  GIT_TAG "cpp-7.0.0"
  OPTIONS "MSGPACK_USE_BOOST OFF" "MSGPACK_CXX20 ON"
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/TestUtilities.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/TestUtilities.h
  @ONLY
)

file(
  GLOB _tests_cpps
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

add_executable(monoprop_unit_tests.x ${_tests_cpps})

target_compile_definitions(
  monoprop_unit_tests.x
  PUBLIC
    BOOST_TEST_DYN_LINK
    BOOST_TEST_NO_MAIN
)

target_include_directories(
  monoprop_unit_tests.x
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(
  monoprop_unit_tests.x
  PRIVATE
    monoprop-objs
    Boost::unit_test_framework
    msgpack-cxx
    PkgConfig::HWLOC
)

include(${CMAKE_CURRENT_LIST_DIR}/boost-test.cmake)

# Under monoprop_ENABLE_TSAN every test process needs the suppressions file, so it
# is attached as a test property rather than left to the caller's shell.
# history_size raises the shadow-memory depth so a report about a barrier word
# can still name the thread that wrote the previous generation; the spin loops
# retire far more accesses between two interesting ones than the default 2 keeps.
set(_monoprop_test_env_entries)
if(monoprop_ENABLE_TSAN)
  list(
    APPEND _monoprop_test_env_entries
    "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_LIST_DIR}/tsan.supp:history_size=4:second_deadlock_stack=1"
  )
endif()

# CTest runs every Boost case as its own process, so each one pays a full MPI_Init --
# and MPI_Init initialises every fabric device it finds whether or not the process will
# ever send a message. Measured on a Deucalion login node (8 Mellanox HCAs): 8.8 s per
# process, against 0.2 s of user CPU, so 224 cases spend over half an hour opening
# adapters for single-process tests. Excluding the fabric drops that to 1.9 s.
#
# A per-case launch is ONE process: its world size is 1, the `*_World` cases skip
# themselves, and every remaining communicator is MPI_COMM_SELF. So the fabric cannot
# affect the outcome, only the startup cost -- which is why this is scoped to the
# `serial` variants through SERIAL_ENVIRONMENT and must never reach the MPI ones. On
# this node `OMPI_MCA_pml=^ucx` makes a 2-rank run of
# zero_cutoff_upper_atol_zero_is_exact_World hang indefinitely where it otherwise
# passes in 29 ms; the same hang reproduces on the pre-branch commit, so it belongs to
# the component selection, not to the engine.
#
# Exclusions (`^`), never a positive component list -- naming a component that must
# exist is how this breaks on the next machine: `vader` became `sm` in Open MPI 5, and
# `OMPI_MCA_btl=self,vader` there silently reduces to `self` alone. Unknown OMPI_MCA_*
# variables are ignored by other MPIs, so this is inert under MPICH.
option(
  monoprop_TEST_EXCLUDE_MPI_FABRIC
  "Skip fabric init in the single-process (serial) test variants to cut MPI_Init cost"
  ON
)
set(_monoprop_serial_env_entries)
if(monoprop_TEST_EXCLUDE_MPI_FABRIC AND monoprop_ENABLE_MPI)
  list(
    APPEND _monoprop_serial_env_entries
    "OMPI_MCA_pml=^ucx"
    "OMPI_MCA_btl=^openib,ofi,uct"
  )
endif()

# ENVIRONMENT takes ONE value holding every VAR=value, so the separators must be
# ESCAPED semicolons. A plain "${list}" does not work: a CMake list *is* a
# semicolon-joined string, so it flattens straight back out and every entry after the
# first is read as the name of the next property -- verified by inspecting
# `ctest --show-only=json-v1`, where the second entry vanished from ENVIRONMENT and
# reappeared as a stray property. Keep the escape, and re-check that output if this
# block is ever edited.
set(_monoprop_test_env)
if(_monoprop_test_env_entries)
  string(
    REPLACE ";"
    "\;"
    _monoprop_test_env_joined
    "${_monoprop_test_env_entries}"
  )
  set(
    _monoprop_test_env
    ENVIRONMENT
    "${_monoprop_test_env_joined}"
  )
endif()

# Automatic discovery of unit tests.
# Default CTest run includes per-case serial tests plus suite-level MPI variants
# for monoprop_MPI_TEST_PROCS.
discover_tests(
  monoprop_unit_tests.x
  PROPERTIES
  LABELS
  "unit"
  ${_monoprop_test_env}
  SERIAL_ENVIRONMENT
  ${_monoprop_serial_env_entries}
)
