if(NOT DEFINED TEST_ENABLE_MPI_VARIANTS)
  set(TEST_ENABLE_MPI_VARIANTS "OFF")
endif()
if(NOT DEFINED TEST_MPI_NUMPROCS)
  set(TEST_MPI_NUMPROCS "2")
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

# Positive integers, deduplicated; an empty list is legal and registers nothing.
function(validated_ranks OUT LIST_NAME)
  set(_ranks ${${LIST_NAME}})
  foreach(_rank IN LISTS _ranks)
    if(NOT _rank MATCHES "^[1-9][0-9]*$")
      message(
        FATAL_ERROR
        "Invalid MPI rank '${_rank}' in ${LIST_NAME}='${${LIST_NAME}}'. Use positive integers."
      )
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _ranks)
  set(${OUT} ${_ranks} PARENT_SCOPE)
endfunction()

set(_mpi_ranks)
set(_mpi_splitmix_ranks)
if(TEST_ENABLE_MPI_VARIANTS)
  if("${TEST_MPI_NUMPROCS}" STREQUAL "")
    set(TEST_MPI_NUMPROCS 2)
  endif()
  validated_ranks(_mpi_ranks TEST_MPI_NUMPROCS)
  validated_ranks(_mpi_splitmix_ranks TEST_MPI_SPLITMIX_NUMPROCS)
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

# Every variant's launch environment (the caller's ENVIRONMENT), ahead of what a variant adds.
list(APPEND common_env_list ${TEST_ENVIRONMENT})

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

foreach(LINE ${LINES})
  # A data-driven case lists its children (_0, _1, ...) indented beneath it, and running the parent
  # runs them all. Matching the name instead would also drop a case like `..._32_bit`.
  if(LINE MATCHES "^[ \t]")
    continue()
  endif()
  string(
    REGEX REPLACE "\\*$"
    ""
    test
    "${LINE}"
  )
  string(STRIP "${test}" test)
  if(test STREQUAL "")
    continue()
  endif()
  # Boost reports a case whose precondition failed (e.g. too few ranks) as skipped with exit 0;
  # without this CTest would count it as a pass.
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
    PROPERTIES
      SKIP_REGULAR_EXPRESSION
      "Test case \"[^\"]+\" was skipped"
  )
endforeach()

# The MPI variants wrap the WHOLE suite in one mpiexec per rank count: the ranks must reach the
# same collectives, which per-case launches cannot guarantee. No skip regex here: a case may skip at
# one rank count (e.g. it needs 4) while the rest of the suite runs.
#
# Open MPI refuses more ranks than slots, and 4 ranks exceed a 3-core runner; both spellings
# cover ORTE (4.x) and PRRTE (5.x), and other MPIs ignore them.
set(
  _mpi_env
  "OMPI_ALLOW_RUN_AS_ROOT=1"
  "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  "OMPI_MCA_rmaps_base_oversubscribe=1"
  "PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe"
)

function(register_mpi_variant RANKS ROUTING)
  set(
    mpi_cmd
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}"
    "${RANKS}"
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

  set(_name "${TEST_TARGET}_mpi_${RANKS}")
  # Set for linear too: an inherited monoprop_ROUTING=splitmix would otherwise run splitmix under
  # the mpi-linear label and leave linear routing untested.
  set(
    _env
    ${_mpi_env}
    "monoprop_ROUTING=${ROUTING}"
  )
  if(ROUTING STREQUAL "splitmix")
    string(APPEND _name "_splitmix")
  endif()
  register_variant("${_name}"
    COMMAND
      ${mpi_cmd}
    LABELS
      mpi
      "mpi-${RANKS}"
      "mpi-${ROUTING}"
    ENVIRONMENT
      ${_env}
    PROPERTIES
      TIMEOUT
      600
  )
  set(script "${script}" PARENT_SCOPE)
  set(tests ${tests} PARENT_SCOPE)
endfunction()

if(TEST_ENABLE_MPI_VARIANTS AND MPIEXEC_EXECUTABLE)
  foreach(_mpi_rank IN LISTS _mpi_ranks)
    register_mpi_variant("${_mpi_rank}" linear)
  endforeach()
  foreach(_mpi_rank IN LISTS _mpi_splitmix_ranks)
    register_mpi_variant("${_mpi_rank}" splitmix)
  endforeach()
endif()

# Create a list of all discovered tests, which users may use to e.g. set
# properties on the tests
add_command(set ${TEST_LIST} ${tests})

# Write CTest script
file(WRITE "${CTEST_FILE}" "${script}")
