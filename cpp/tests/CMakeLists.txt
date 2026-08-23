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
    ${monoprop_TIERED_TARGETS}
    Boost::unit_test_framework
    msgpack-cxx
    PkgConfig::HWLOC
)

# In tier-dso mode the tiers are shared objects that resolve their monoprop_TIER_ABI symbols out of
# whatever module loaded them -- which here is the executable, and an executable puts nothing in
# .dynsym without being asked. Without this the binary links and then dies at startup on a symbol
# lookup, so it is not a nicety.
if(monoprop_FAT_TIER_DSO)
  set_target_properties(
    monoprop_unit_tests.x
    PROPERTIES
      ENABLE_EXPORTS
        ON
  )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/boost-test.cmake)

# CTest launches each Boost case as a world-size-1 process, so excluding fabric components cuts
# MPI_Init from 2.03 s to 0.61 s without affecting communication; real MPI tests must keep the
# full component set.
#
# Use SERIAL_ENVIRONMENT with exclusions (^), not a positive component list: Open MPI component
# names change across versions, and unknown OMPI_MCA_* variables are harmless under other MPIs.
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
