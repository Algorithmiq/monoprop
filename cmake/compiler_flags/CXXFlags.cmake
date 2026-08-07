#.rst:
#
# Manages C++ compiler flags.
#
# There is one user-facing option to enable architecture-specific compiler
# flags.
# The complete list of flags is built as:
#
#   CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_<CONFIG> ARCH_FLAG monoprop_CXX_FLAGS EXTRA_CXXFLAGS
#
# where:
#
# - ``CMAKE_CXX_FLAGS`` is initialized by the contents of the ``CXXFLAGS``
#   environment variable when configuring. Default is empty.
# - ``CMAKE_CXX_FLAGS_<CONFIG>`` are build-type specific compiler flags.
#   The defaults are compiler-dependent: have a look at the ``GNU.CXX.cmake``,
#   ``Clang.CXX.cmake``, and ``Intel.CXX.cmake`` files.
# - ``ARCH_FLAG`` is the architecture-dependent optimization flag, *e.g.*
#   vectorization. Default is empty.
# - ``monoprop_CXX_FLAGS`` are monoprop-specific flags to be used for all builds.
#   The defaults are compiler-dependent: have a look at the ``GNU.CXX.cmake``,
#   ``Clang.CXX.cmake``, and ``Intel.CXX.cmake`` files.
# - ``EXTRA_CXXFLAGS`` useful if you need to append certain flags to the full
#   list, *e.g.* to override previous compiler flags without touching the CMake
#   scripts.  Default is empty.
#
# Variables used::
#
#   monoprop_ENABLE_ARCH_FLAGS
#   EXTRA_CXXFLAGS
#
# Variables modified::
#
#   CMAKE_CXX_FLAGS
#
# Environment variables used::
#
#   CXXFLAGS

option(
  monoprop_ENABLE_ARCH_FLAGS
  "Enable architecture-specific compiler flags"
  ON
)

# code needs C++23 at least
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# do not use compiler extensions to the C++ standard
set(CMAKE_CXX_EXTENSIONS FALSE)
# disable scanning for C++20 modules (unused)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
# generate a JSON database of compiler commands (useful for LSP IDEs)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
# position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
# visibility levels
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

set(ARCH_FLAG "")
if(monoprop_ENABLE_ARCH_FLAGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(ARCH_FLAG "-march=native")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(ARCH_FLAG "-march=native")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(ARCH_FLAG "-xHost")
  endif()
endif()

# Query the machine-dependent flags for a given -march value and store the
# cleaned, space-separated string in the variable named by OUTPUT_VARIABLE. A
# MARCH of "default" queries the default target (no -march flag).
#
# Usage:
#   _monoprop_query_machine_flags(MARCH <arch> OUTPUT_VARIABLE <var>)
function(_monoprop_query_machine_flags)
  set(
    _one_value_args
    MARCH
    OUTPUT_VARIABLE
  )
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "")

  if(NOT _arg_OUTPUT_VARIABLE)
    message(
      FATAL_ERROR
      "_monoprop_query_machine_flags: OUTPUT_VARIABLE is required"
    )
  endif()
  if(NOT _arg_MARCH)
    message(FATAL_ERROR "_monoprop_query_machine_flags: MARCH is required")
  endif()

  if(_arg_MARCH STREQUAL "default")
    set(_march_args "")
  else()
    set(_march_args "-march=${_arg_MARCH}")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    # AppleClang does not support `-Q --help=target`. Query the driver with
    # `-###` and normalize CPU/march flags from the reported invocation.
    execute_process(
      COMMAND
        # gersemi: off
        ${CMAKE_CXX_COMPILER} ${_march_args} -### -x c++ -c /dev/null
      # gersemi: on
      ERROR_VARIABLE _query_output
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _query_result
    )
    if(NOT _query_result EQUAL 0)
      message(
        WARNING
        "Failed to query machine-dependent flags for '${_arg_MARCH}' with AppleClang (exit code ${_query_result}). Continuing with empty machine flags."
      )
      set(_flags "")
    else()
      execute_process(
        COMMAND
          ${CMAKE_COMMAND} -E echo "${_query_output}"
        COMMAND
          ${Python_EXECUTABLE}
          "${PROJECT_SOURCE_DIR}/tools/clang-target-help-clean.py"
        OUTPUT_VARIABLE _flags
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _parse_result
      )
      if(NOT _parse_result EQUAL 0)
        message(
          WARNING
          "Failed to parse AppleClang machine-dependent flags for '${_arg_MARCH}' (exit code ${_parse_result}). Continuing with empty machine flags."
        )
        set(_flags "")
      endif()
    endif()
  else()
    # Report the machine-dependent flags GCC uses for each target variant by
    # querying `gcc -march=<arch> -Q --help=target` and cleaning the output
    # with tools/gcc-target-help-clean.py.
    execute_process(
      COMMAND
        ${CMAKE_CXX_COMPILER} ${_march_args} -Q --help=target
      COMMAND
        ${Python_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/tools/gcc-target-help-clean.py"
      OUTPUT_VARIABLE _flags
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
      message(
        FATAL_ERROR
        "Failed to query machine-dependent flags for '${_arg_MARCH}' (exit code ${_result})"
      )
    endif()
  endif()
  set(${_arg_OUTPUT_VARIABLE} "${_flags}" PARENT_SCOPE)
endfunction()

set(monoprop_DEFAULT_VARIANT_FLAGS "")
if(monoprop_ENABLE_ARCH_FLAGS)
  _monoprop_query_machine_flags(MARCH native OUTPUT_VARIABLE monoprop_DEFAULT_VARIANT_FLAGS)
else()
  _monoprop_query_machine_flags(MARCH default OUTPUT_VARIABLE monoprop_DEFAULT_VARIANT_FLAGS)
endif()

set(monoprop_VARIANTS "")
set(monoprop_VARIANT_FLAGS "")

# generate a header file with the macros needed to describe the variant
configure_file(
  ${PROJECT_SOURCE_DIR}/cpp/include/monoprop/Variants.h.in
  ${PROJECT_BINARY_DIR}/include/monoprop/Variants.h
  @ONLY
)

set(monoprop_CXX_FLAGS "")
include(${CMAKE_CURRENT_LIST_DIR}/GNU.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Intel.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Clang.CXX.cmake)
