include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

cpmfindpackage(
  NAME Catch2
  VERSION 3
  GIT_REPOSITORY "https://github.com/catchorg/Catch2.git"
  GIT_TAG "v3.15.3"
  OPTIONS "CATCH_INSTALL_DOCS OFF"
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)

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
    Catch2::Catch2
    msgpack-cxx
    PkgConfig::HWLOC
)

if(Catch2_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
endif()
include(Catch)

# CTest launches each Catch2 case as a world-size-1 process, so excluding fabric components cuts
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

catch_discover_tests(
  monoprop_unit_tests.x
  PROPERTIES
    LABELS
    unit
)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/catch-properties.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/catch-properties.cmake
  @ONLY
)
set_property(
  DIRECTORY
  APPEND
  PROPERTY
    TEST_INCLUDE_FILES
      ${CMAKE_CURRENT_BINARY_DIR}/catch-properties.cmake
)

set(
  monoprop_MPI_TEST_PROCS
  "2"
  CACHE STRING
  "Semicolon-separated list of ranks for MPI test variants"
)

set(_monoprop_mpiexec "${MPIEXEC_EXECUTABLE}")
if(NOT _monoprop_mpiexec)
  find_program(
    _monoprop_mpiexec
    NAMES
      mpiexec
      mpirun
  )
endif()

set(_monoprop_mpiexec_numproc_flag "${MPIEXEC_NUMPROC_FLAG}")
if(NOT _monoprop_mpiexec_numproc_flag)
  set(_monoprop_mpiexec_numproc_flag "-n")
endif()

if(monoprop_ENABLE_MPI AND _monoprop_mpiexec)
  set(_monoprop_mpi_ranks ${monoprop_MPI_TEST_PROCS})
  list(REMOVE_DUPLICATES _monoprop_mpi_ranks)
  foreach(_monoprop_mpi_rank IN LISTS _monoprop_mpi_ranks)
    if(NOT _monoprop_mpi_rank MATCHES "^[1-9][0-9]*$")
      message(
        FATAL_ERROR
        "Invalid MPI rank '${_monoprop_mpi_rank}' in monoprop_MPI_TEST_PROCS='${monoprop_MPI_TEST_PROCS}'"
      )
    endif()

    add_test(
      NAME "monoprop_unit_tests.x_mpi_${_monoprop_mpi_rank}"
      COMMAND
        "${_monoprop_mpiexec}" "${_monoprop_mpiexec_numproc_flag}"
        "${_monoprop_mpi_rank}" ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:monoprop_unit_tests.x> --order lex --rng-seed 1
        ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
      "monoprop_unit_tests.x_mpi_${_monoprop_mpi_rank}"
      PROPERTIES
        WORKING_DIRECTORY
          "${CMAKE_CURRENT_BINARY_DIR}"
        LABELS
          "unit;cxx;mpi;mpi-${_monoprop_mpi_rank}"
        ENVIRONMENT
          "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    )
  endforeach()
endif()
