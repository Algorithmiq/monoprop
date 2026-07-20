if(NOT DEFINED TEST_ENABLE_MPI_VARIANTS)
  set(TEST_ENABLE_MPI_VARIANTS "OFF")
endif()
if(NOT DEFINED TEST_MPI_NUMPROCS)
  set(TEST_MPI_NUMPROCS "2")
endif()
if(NOT DEFINED TEST_MPI_LAYOUT)
  set(TEST_MPI_LAYOUT "suite")
endif()
if(
  NOT
    TEST_MPI_LAYOUT
      STREQUAL
      "suite"
  AND
    NOT
      TEST_MPI_LAYOUT
        STREQUAL
        "per-test"
)
  message(
    FATAL_ERROR
    "TEST_MPI_LAYOUT must be either 'suite' or 'per-test', got '${TEST_MPI_LAYOUT}'"
  )
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

set(extra_args ${TEST_EXTRA_ARGS})
set(properties ${TEST_PROPERTIES})
set(script)
set(suite)
set(tests)

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

function(
  build_mpi_properties
  OUT_VAR
  BASE_PROPERTIES_VAR
  BASE_LABELS_VAR
  BASE_ENV_VAR
  MPI_RANK
)
  set(_mpi_labels_list ${${BASE_LABELS_VAR}})
  list(
    APPEND _mpi_labels_list
    mpi
    "mpi-${MPI_RANK}"
  )
  list(REMOVE_DUPLICATES _mpi_labels_list)
  list(JOIN _mpi_labels_list ";" _mpi_labels_value)

  set(_mpi_env_list ${${BASE_ENV_VAR}})
  list(
    APPEND _mpi_env_list
    "OMPI_ALLOW_RUN_AS_ROOT=1"
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  )
  list(REMOVE_DUPLICATES _mpi_env_list)
  list(JOIN _mpi_env_list ";" _mpi_env_value)

  set(_mpi_properties ${${BASE_PROPERTIES_VAR}})
  if(_mpi_labels_value)
    set(_mpi_labels_arg "[==[${_mpi_labels_value}]==]")
    list(
      APPEND _mpi_properties
      "LABELS"
      "${_mpi_labels_arg}"
    )
  endif()
  if(_mpi_env_value)
    set(_mpi_env_arg "[==[${_mpi_env_value}]==]")
    list(
      APPEND _mpi_properties
      "ENVIRONMENT"
      "${_mpi_env_arg}"
    )
  endif()

  set(${OUT_VAR} ${_mpi_properties} PARENT_SCOPE)
endfunction()

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

# Run test executable to get list of available tests
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
    # ...and add to script
    set(base_properties ${common_properties})
    set(base_labels_list ${common_labels_list})
    set(base_env_list ${common_env_list})

    set(serial_labels_list ${base_labels_list})
    list(APPEND serial_labels_list serial)
    list(REMOVE_DUPLICATES serial_labels_list)
    list(JOIN serial_labels_list ";" serial_labels_value)

    set(serial_env_list ${base_env_list})
    list(REMOVE_DUPLICATES serial_env_list)
    list(JOIN serial_env_list ";" serial_env_value)

    set(serial_properties ${base_properties})
    if(serial_labels_value)
      set(serial_labels_arg "[==[${serial_labels_value}]==]")
      list(
        APPEND serial_properties
        "LABELS"
        "${serial_labels_arg}"
      )
    endif()
    if(serial_env_value)
      set(serial_env_arg "[==[${serial_env_value}]==]")
      list(
        APPEND serial_properties
        "ENVIRONMENT"
        "${serial_env_arg}"
      )
    endif()

    add_command(add_test
      "${test}"
      "${TEST_EXECUTABLE}"
      "--run_test=${test}"
      "--report_level=detailed"
      "--catch_system_errors=yes"
    )

    add_command(set_tests_properties
      "${test}"
      PROPERTIES
      WORKING_DIRECTORY "${TEST_WORKING_DIR}"
      ${serial_properties}
    )

    list(APPEND tests "${test}")

    if(
      TEST_ENABLE_MPI_VARIANTS
      AND
        MPIEXEC_EXECUTABLE
      AND
        TEST_MPI_LAYOUT
          STREQUAL
          "per-test"
    )
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
          "--run_test=${test}"
          "--report_level=detailed"
          "--catch_system_errors=yes"
        )
        if(MPIEXEC_POSTFLAGS)
          list(APPEND mpi_cmd ${MPIEXEC_POSTFLAGS})
        endif()

        build_mpi_properties(mpi_properties base_properties base_labels_list base_env_list ${_mpi_rank})

        set(_mpi_test_name "${test}_mpi_${_mpi_rank}")
        add_command(add_test
          "${_mpi_test_name}"
          ${mpi_cmd}
        )
        add_command(set_tests_properties
          "${_mpi_test_name}"
          PROPERTIES
          WORKING_DIRECTORY "${TEST_WORKING_DIR}"
          ${mpi_properties}
        )

        list(APPEND tests "${_mpi_test_name}")
      endforeach()
    endif()
  endif()
endforeach()

if(
  TEST_ENABLE_MPI_VARIANTS
  AND
    MPIEXEC_EXECUTABLE
  AND
    TEST_MPI_LAYOUT
      STREQUAL
      "suite"
)
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
    )
    if(MPIEXEC_POSTFLAGS)
      list(APPEND mpi_cmd ${MPIEXEC_POSTFLAGS})
    endif()

    build_mpi_properties(mpi_properties common_properties common_labels_list common_env_list ${_mpi_rank})

    set(_suite_test_name "${TEST_TARGET}_mpi_${_mpi_rank}")
    add_command(add_test
      "${_suite_test_name}"
      ${mpi_cmd}
    )
    add_command(set_tests_properties
      "${_suite_test_name}"
      PROPERTIES
      WORKING_DIRECTORY "${TEST_WORKING_DIR}"
      ${mpi_properties}
    )

    list(APPEND tests "${_suite_test_name}")
  endforeach()
endif()

# Create a list of all discovered tests, which users may use to e.g. set
# properties on the tests
add_command(set ${TEST_LIST} ${tests})

# Write CTest script
file(WRITE "${CTEST_FILE}" "${script}")
