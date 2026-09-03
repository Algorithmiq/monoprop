set(
  monoprop_MPI_TEST_PROCS
  "2"
  CACHE STRING
  "Semicolon-separated list of ranks for MPI test variants"
)
set(
  monoprop_MPI_SPARSE_ROWS_TEST_PROCS
  "2"
  CACHE STRING
  "Semicolon-separated list of ranks for the sparse-row-backend MPI test variants (kept separate from monoprop_MPI_TEST_PROCS so growing dense-backend MPI coverage does not silently multiply how many sparse-row mpiexec launches CI pays for)"
)

set(_monoprop_mpiexec "${MPIEXEC_EXECUTABLE}")
if(NOT _monoprop_mpiexec)
  find_program(
    _monoprop_mpiexec_fallback
    NAMES
      mpiexec
      mpirun
  )
  set(_monoprop_mpiexec "${_monoprop_mpiexec_fallback}")
endif()

set(_monoprop_mpiexec_numproc_flag "${MPIEXEC_NUMPROC_FLAG}")
if(NOT _monoprop_mpiexec_numproc_flag)
  set(_monoprop_mpiexec_numproc_flag "-n")
endif()

# SERIAL_ENVIRONMENT: VAR=value entries applied to the per-case `serial` variants only.
function(discover_tests TARGET)
  cmake_parse_arguments(
    ""
    ""
    "WORKING_DIRECTORY"
    "EXTRA_ARGS;PROPERTIES;SERIAL_ENVIRONMENT"
    ${ARGN}
  )

  if(NOT _WORKING_DIRECTORY)
    set(_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  if(NOT _TEST_LIST)
    set(_TEST_LIST ${TARGET}_TESTS)
  endif()

  ## Generate a unique name based on the extra arguments
  string(SHA1 args_hash "${_TEST_SPEC} ${_EXTRA_ARGS}")
  string(SUBSTRING ${args_hash} 0 7 args_hash)

  # Define rule to generate test list for aforementioned test executable
  set(
    ctest_include_file
    "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_include-${args_hash}.cmake"
  )
  set(
    ctest_tests_file
    "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_tests-${args_hash}.cmake"
  )
  if(_monoprop_mpiexec AND monoprop_ENABLE_MPI)
    set(_enable_mpi_variants "ON")
  else()
    set(_enable_mpi_variants "OFF")
  endif()

  add_custom_command(
    TARGET ${TARGET}
    POST_BUILD
    BYPRODUCTS
      "${ctest_tests_file}"
    COMMAND
      "${CMAKE_COMMAND}" -D "TEST_TARGET=${TARGET}" -D
      "TEST_EXECUTABLE=$<TARGET_FILE:${TARGET}>" -D
      "TEST_WORKING_DIR=${_WORKING_DIRECTORY}" -D
      "TEST_EXTRA_ARGS=${_EXTRA_ARGS}" -D "TEST_PROPERTIES=${_PROPERTIES}" -D
      "TEST_SERIAL_ENVIRONMENT=${_SERIAL_ENVIRONMENT}" -D
      "TEST_LIST=${_TEST_LIST}" -D "CTEST_FILE=${ctest_tests_file}" -D
      "TEST_ENABLE_MPI_VARIANTS=${_enable_mpi_variants}" -D
      "TEST_MPI_NUMPROCS=${monoprop_MPI_TEST_PROCS}" -D
      "TEST_MPI_SPARSE_ROWS_NUMPROCS=${monoprop_MPI_SPARSE_ROWS_TEST_PROCS}" -D
      "MPIEXEC_EXECUTABLE=${_monoprop_mpiexec}" -D
      "MPIEXEC_NUMPROC_FLAG=${_monoprop_mpiexec_numproc_flag}" -D
      "MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" -D
      "MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}" -P "${_DISCOVER_TESTS_SCRIPT}"
    VERBATIM
  )

  file(
    WRITE "${ctest_include_file}"
    "if(EXISTS \"${ctest_tests_file}\")\n"
    "  include(\"${ctest_tests_file}\")\n"
    "else()\n"
    "  add_test(${TARGET}_NOT_BUILT-${args_hash} ${TARGET}_NOT_BUILT-${args_hash})\n"
    "endif()\n"
  )

  # Add discovered tests to directory TEST_INCLUDE_FILES
  set_property(
    DIRECTORY
    APPEND
    PROPERTY
      TEST_INCLUDE_FILES
        "${ctest_include_file}"
  )
endfunction()

###############################################################################

set(
  _DISCOVER_TESTS_SCRIPT
  ${CMAKE_CURRENT_LIST_DIR}/boostAddTests.cmake
  CACHE INTERNAL
  "The location of the boostAddTests script"
)
mark_as_advanced(_DISCOVER_TESTS_SCRIPT)
