# Runs one mpi_failure_driver scenario under the MPI launcher and checks how the job ends.
#
# Inputs (-D): MPIEXEC, NUMPROC_FLAG, PREFLAGS (MPIEXEC_PREFLAGS, may be empty), DRIVER, SCENARIO, OPTION
# (may be empty), EXPECT (regex that must appear in the combined output), FORBID (optional regex that must
# not), EXPECT_SUCCESS (optional; ON for an acceptance scenario), TIMEOUT_SECONDS (default 30).
#
# A failure scenario passes on a nonzero exit before the timeout with EXPECT in the output; an acceptance
# scenario (EXPECT_SUCCESS) passes on exit 0 before the timeout with EXPECT in the output. A timeout is a
# failure, never a pass: it means some rank was left blocked. The launcher must end the peers of a rank that
# exits abnormally (Open MPI's default); the wrong-thread case depends on that policy. "NOT EXERCISED" in the
# output (the MPI library's provided thread level does not exercise the route) is reported as a skip.

if(NOT DEFINED TIMEOUT_SECONDS)
  set(TIMEOUT_SECONDS 30)
endif()

set(
  _command
  "${MPIEXEC}"
  "${NUMPROC_FLAG}"
  2
  ${PREFLAGS}
  "${DRIVER}"
  "${SCENARIO}"
)
if(NOT "${OPTION}" STREQUAL "")
  list(APPEND _command "${OPTION}")
endif()

string(TIMESTAMP _start "%s" UTC)
execute_process(
  COMMAND
    ${_command}
  TIMEOUT ${TIMEOUT_SECONDS}
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr
)
string(TIMESTAMP _stop "%s" UTC)
math(EXPR _elapsed "${_stop} - ${_start}")

message("command: ${_command}")
message(
  "elapsed: ${_elapsed} s (timeout ${TIMEOUT_SECONDS} s); result: ${_result}"
)
message("--- stdout ---\n${_stdout}--- stderr ---\n${_stderr}--------------")

set(_output "${_stdout}${_stderr}")
if(_result MATCHES "timeout")
  message(
    FATAL_ERROR
    "FAILED: the job did not terminate within ${TIMEOUT_SECONDS} s"
  )
endif()
if(_output MATCHES "NOT EXERCISED")
  message(
    "NOT EXERCISED: the MPI library supplied more thread support than requested"
  )
  if(CMAKE_VERSION VERSION_LESS 3.29)
    message(FATAL_ERROR "NOT EXERCISED (CMake < 3.29 cannot report a skip)")
  endif()
  cmake_language(EXIT 77)
endif()
if(EXPECT_SUCCESS)
  if(NOT "${_result}" STREQUAL "0")
    message(
      FATAL_ERROR
      "FAILED: the acceptance scenario exited with ${_result}, not 0"
    )
  endif()
elseif("${_result}" STREQUAL "0")
  message(
    FATAL_ERROR
    "FAILED: the job exited 0; the injected failure did not end it"
  )
endif()
if(NOT _output MATCHES "${EXPECT}")
  message(FATAL_ERROR "FAILED: expected diagnostic /${EXPECT}/ not found")
endif()
if(
  DEFINED
    FORBID
  AND
    NOT
      "${FORBID}"
        STREQUAL
        ""
  AND
    _output
      MATCHES
      "${FORBID}"
)
  message(FATAL_ERROR "FAILED: forbidden output /${FORBID}/ found")
endif()
if(EXPECT_SUCCESS)
  message("PASSED: exit 0 after ${_elapsed} s with the expected completion")
else()
  message(
    "PASSED: nonzero exit after ${_elapsed} s with the expected diagnostic"
  )
endif()
