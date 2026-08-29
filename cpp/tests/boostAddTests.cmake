if(NOT DEFINED TEST_ENABLE_MPI_VARIANTS)
  set(TEST_ENABLE_MPI_VARIANTS "OFF")
endif()
if(NOT DEFINED TEST_MPI_NUMPROCS)
  set(TEST_MPI_NUMPROCS "2")
endif()
if(NOT DEFINED TEST_MPI_SPARSE_ROWS_NUMPROCS)
  set(TEST_MPI_SPARSE_ROWS_NUMPROCS "2")
endif()
if(TEST_ENABLE_MPI_VARIANTS AND NOT MPIEXEC_EXECUTABLE)
  message(
    WARNING
    "Requested MPI variants for tests but MPIEXEC_EXECUTABLE is not set"
  )
  set(TEST_ENABLE_MPI_VARIANTS "OFF")
endif()
if(TEST_ENABLE_MPI_VARIANTS AND NOT MPIEXEC_NUMPROC_FLAG)
  message(
    FATAL_ERROR
    "MPI variants require MPIEXEC_NUMPROC_FLAG to select rank counts"
  )
endif()

set(_mpi_ranks)
if(TEST_ENABLE_MPI_VARIANTS)
  if("${TEST_MPI_NUMPROCS}" STREQUAL "")
    set(TEST_MPI_NUMPROCS 2)
  endif()

  foreach(_rank IN LISTS TEST_MPI_NUMPROCS)
    if(NOT _rank MATCHES "^[1-9][0-9]*$")
      message(
        FATAL_ERROR
        "Invalid MPI rank '${_rank}' in TEST_MPI_NUMPROCS='${TEST_MPI_NUMPROCS}'. Use positive integers."
      )
    endif()
  endforeach()

  set(_mpi_ranks ${TEST_MPI_NUMPROCS})
  list(REMOVE_DUPLICATES _mpi_ranks)
endif()

# Validated the same way as _mpi_ranks above, but kept in its own list (TEST_MPI_SPARSE_ROWS_NUMPROCS)
# rather than reusing _mpi_ranks: growing dense-backend rank coverage must not silently multiply how
# many sparse-row mpiexec launches CI pays for.
set(_mpi_sparse_ranks)
if(TEST_ENABLE_MPI_VARIANTS)
  if("${TEST_MPI_SPARSE_ROWS_NUMPROCS}" STREQUAL "")
    set(TEST_MPI_SPARSE_ROWS_NUMPROCS 2)
  endif()

  foreach(_rank IN LISTS TEST_MPI_SPARSE_ROWS_NUMPROCS)
    if(NOT _rank MATCHES "^[1-9][0-9]*$")
      message(
        FATAL_ERROR
        "Invalid MPI rank '${_rank}' in TEST_MPI_SPARSE_ROWS_NUMPROCS='${TEST_MPI_SPARSE_ROWS_NUMPROCS}'. Use positive integers."
      )
    endif()
  endforeach()

  set(_mpi_sparse_ranks ${TEST_MPI_SPARSE_ROWS_NUMPROCS})
  list(REMOVE_DUPLICATES _mpi_sparse_ranks)
endif()

set(extra_args ${TEST_EXTRA_ARGS})
set(properties ${TEST_PROPERTIES})
set(serial_env ${TEST_SERIAL_ENVIRONMENT})
set(script)
set(tests)

# LABELS and ENVIRONMENT are split off because each variant extends them; every other caller
# property is inherited verbatim.
set(common_properties)
set(common_labels_list)
set(common_env_list)
list(LENGTH properties _common_prop_len)
set(_common_prop_idx 0)
while(_common_prop_idx LESS _common_prop_len)
  math(EXPR _common_prop_next "${_common_prop_idx} + 1")
  list(GET properties ${_common_prop_idx} _common_prop_key)
  set(_common_prop_value "")
  if(_common_prop_next LESS _common_prop_len)
    list(GET properties ${_common_prop_next} _common_prop_value)
  endif()
  if(_common_prop_key STREQUAL "LABELS")
    if(NOT _common_prop_value STREQUAL "")
      list(APPEND common_labels_list ${_common_prop_value})
    endif()
  elseif(_common_prop_key STREQUAL "ENVIRONMENT")
    if(NOT _common_prop_value STREQUAL "")
      list(APPEND common_env_list ${_common_prop_value})
    endif()
  else()
    list(
      APPEND common_properties
      "${_common_prop_key}"
      "${_common_prop_value}"
    )
  endif()
  math(EXPR _common_prop_idx "${_common_prop_idx} + 2")
endwhile()

list(APPEND common_labels_list cxx)
list(REMOVE_DUPLICATES common_labels_list)
list(REMOVE_DUPLICATES common_env_list)

function(add_command NAME)
  set(_args "")
  foreach(_arg ${ARGN})
    if(_arg MATCHES "^\\[=+\\[.*\\]=+\\]$")
      set(_args "${_args} ${_arg}")
    elseif(_arg MATCHES "[^-./:a-zA-Z0-9_]")
      set(_args "${_args} [==[${_arg}]==]") # form a bracket_argument
    else()
      set(_args "${_args} ${_arg}")
    endif()
  endforeach()
  set(script "${script}${NAME}(${_args})\n" PARENT_SCOPE)
endfunction()

# `script` and `tests` are written back because add_command's PARENT_SCOPE write lands here.
function(register_variant NAME)
  cmake_parse_arguments(
    ""
    ""
    ""
    "COMMAND;LABELS;ENVIRONMENT;PROPERTIES"
    ${ARGN}
  )

  set(
    _variant_labels
    ${common_labels_list}
    ${_LABELS}
  )
  list(REMOVE_DUPLICATES _variant_labels)
  set(
    _variant_env
    ${common_env_list}
    ${_ENVIRONMENT}
  )
  list(REMOVE_DUPLICATES _variant_env)

  set(
    _variant_properties
    ${common_properties}
    ${_PROPERTIES}
  )
  # Bracket-quote the joined values: they hold the `;` separators CTest expects, which would
  # otherwise be re-split when the generated script is included.
  if(_variant_labels)
    list(JOIN _variant_labels ";" _labels_value)
    list(
      APPEND _variant_properties
      "LABELS"
      "[==[${_labels_value}]==]"
    )
  endif()
  if(_variant_env)
    list(JOIN _variant_env ";" _env_value)
    list(
      APPEND _variant_properties
      "ENVIRONMENT"
      "[==[${_env_value}]==]"
    )
  endif()

  add_command(add_test "${NAME}" ${_COMMAND})
  add_command(set_tests_properties
    "${NAME}"
    PROPERTIES
    WORKING_DIRECTORY "${TEST_WORKING_DIR}"
    ${_variant_properties}
  )

  set(script "${script}" PARENT_SCOPE)
  set(
    tests
    ${tests}
    "${NAME}"
    PARENT_SCOPE
  )
endfunction()

# Run test executable to get list of available tests. Boost.Test has no CMake-side discovery
# module (no analogue of gtest_discover_tests), so parsing --list_content is the only option.
if(NOT EXISTS "${TEST_EXECUTABLE}")
  message(
    FATAL_ERROR
    "Specified test executable '${TEST_EXECUTABLE}' does not exist"
  )
endif()

execute_process(
  COMMAND
    "${TEST_EXECUTABLE}" --list_content=HRF --report_sink=stdout
  OUTPUT_VARIABLE output
  ERROR_VARIABLE err # it prints to stderr...
  RESULT_VARIABLE result
  WORKING_DIRECTORY "${TEST_WORKING_DIR}"
)
if(NOT ${result} EQUAL 0)
  message(
    FATAL_ERROR
    "Error running test executable '${TEST_EXECUTABLE}':\n"
    "  Result: ${result}\n"
    "  Output: ${output}\n"
    "  Error: ${err}\n"
  )
endif()

# Convert the raw output to a list of lines
string(
  REPLACE "\n"
  ";"
  LINES
  "${output}"
)

# process each line
foreach(LINE ${LINES})
  # Remove trailing asterisk
  string(
    REGEX REPLACE "\\*$"
    ""
    CLEANED_LINE
    "${LINE}"
  )

  # Trim whitespace
  string(STRIP "${CLEANED_LINE}" test)

  # Check if the line doesn't contain _0, _1, etc. and is not empty
  if(NOT test MATCHES "_[0-9]+" AND NOT "${test}" STREQUAL "")
    register_variant("${test}"
      COMMAND
        "${TEST_EXECUTABLE}"
        "--run_test=${test}"
        "--report_level=detailed"
        "--catch_system_errors=yes"
        ${extra_args}
      LABELS
        serial
      ENVIRONMENT
        ${serial_env}
    )
    # Run the same case again with the support-form row backend forced. The suite is below
    # SparseRowStore::preferred_for_modes()'s crossover, so the automatic choice would compile
    # that backend but never run it, even though it is the one used for wide systems. Running each
    # case separately makes any divergence easy to identify.
    #
    # Keep serial_env for the same reason as the variant above: this is another world-size-1 run
    # of the same case, so without it half of `-L serial` would pay the MPI_Init setup cost that
    # the other half avoids.
    register_variant("${test}_sparse_rows"
      COMMAND
        "${TEST_EXECUTABLE}"
        "--run_test=${test}"
        "--report_level=detailed"
        "--catch_system_errors=yes"
        ${extra_args}
      LABELS
        serial
        sparse-rows
      ENVIRONMENT
        ${serial_env}
        "monoprop_ROW_STORE=sparse"
    )
  endif()
endforeach()

# The MPI variants wrap the WHOLE suite in one mpiexec per rank count: the ranks must reach the
# same collectives, which per-case launches cannot guarantee.
if(TEST_ENABLE_MPI_VARIANTS AND MPIEXEC_EXECUTABLE)
  foreach(_mpi_rank IN LISTS _mpi_ranks)
    set(mpi_cmd "${MPIEXEC_EXECUTABLE}")
    list(
      APPEND mpi_cmd
      "${MPIEXEC_NUMPROC_FLAG}"
      "${_mpi_rank}"
    )
    if(MPIEXEC_PREFLAGS)
      list(APPEND mpi_cmd ${MPIEXEC_PREFLAGS})
    endif()
    list(
      APPEND mpi_cmd
      "${TEST_EXECUTABLE}"
      "--report_level=detailed"
      "--catch_system_errors=yes"
      ${extra_args}
    )
    if(MPIEXEC_POSTFLAGS)
      list(APPEND mpi_cmd ${MPIEXEC_POSTFLAGS})
    endif()

    register_variant("${TEST_TARGET}_mpi_${_mpi_rank}"
      COMMAND
        ${mpi_cmd}
      LABELS
        mpi
        "mpi-${_mpi_rank}"
      ENVIRONMENT
        "OMPI_ALLOW_RUN_AS_ROOT=1"
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
      PROPERTIES
        TIMEOUT
        600
    )
  endforeach()

  # MPI counterpart of the "_sparse_rows" serial variant above. The cross-rank resolve inserts absent
  # terms into whichever backend is live, and the sparse one is what wide (MPI-scale) systems actually
  # resolve to -- so it needs its own multi-rank coverage, not just the single-rank one above. Runs over
  # _mpi_sparse_ranks, not _mpi_ranks, so it stays cheap by default regardless of how wide the dense rank
  # list grows.
  foreach(_mpi_rank IN LISTS _mpi_sparse_ranks)
    set(mpi_cmd "${MPIEXEC_EXECUTABLE}")
    list(
      APPEND mpi_cmd
      "${MPIEXEC_NUMPROC_FLAG}"
      "${_mpi_rank}"
    )
    if(MPIEXEC_PREFLAGS)
      list(APPEND mpi_cmd ${MPIEXEC_PREFLAGS})
    endif()
    list(
      APPEND mpi_cmd
      "${TEST_EXECUTABLE}"
      "--report_level=detailed"
      "--catch_system_errors=yes"
      ${extra_args}
    )
    if(MPIEXEC_POSTFLAGS)
      list(APPEND mpi_cmd ${MPIEXEC_POSTFLAGS})
    endif()

    register_variant("${TEST_TARGET}_mpi_${_mpi_rank}_sparse_rows"
      COMMAND
        ${mpi_cmd}
      LABELS
        mpi
        "mpi-${_mpi_rank}"
        sparse-rows
      ENVIRONMENT
        "OMPI_ALLOW_RUN_AS_ROOT=1"
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
        "monoprop_ROW_STORE=sparse"
    )
  endforeach()
endif()

# Create a list of all discovered tests, which users may use to e.g. set
# properties on the tests
add_command(set ${TEST_LIST} ${tests})

# Write CTest script
file(WRITE "${CTEST_FILE}" "${script}")
