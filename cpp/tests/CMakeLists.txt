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

# CTest runs every Boost case as its own process, so each pays a full MPI_Init -- which
# initialises every fabric device present even though a single-process test never sends a
# message. Measured on a compute node: 2.03 s per case without the exclusion, 0.61 s with it.
#
# A per-case launch is ONE process (world size 1, `*_World` cases skip themselves, every
# other communicator is MPI_COMM_SELF), so the fabric can only cost startup time. The MPI
# variants exchange real messages and MUST keep the full component set -- `OMPI_MCA_pml=^ucx`
# hangs a 2-rank run indefinitely -- hence SERIAL_ENVIRONMENT rather than a global setting.
#
# Exclusions (`^`), never a positive component list: `vader` became `sm` in Open MPI 5, so
# naming a component breaks on the next machine. Unknown OMPI_MCA_* variables are ignored by
# other MPIs, so this is inert under MPICH.
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

# Automatic discovery of unit tests.
# Default CTest run includes per-case serial tests plus suite-level MPI variants
# for monoprop_MPI_TEST_PROCS.
discover_tests(
  monoprop_unit_tests.x
  PROPERTIES
  LABELS
  "unit"
  SERIAL_ENVIRONMENT
  ${_monoprop_serial_env_entries}
)
