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
# - ``monoprop_CXX_FLAGS`` are monorop-specific flags to be used for all builds.
#   The defaults are compiler-dependent: have a look at the ``GNU.CXX.cmake``,
#   ``Clang.CXX.cmake``, and ``Intel.CXX.cmake`` files.
# - ``EXTRA_CXXFLAGS`` useful if you need to append certain flags to the full
#   list, *e.g.* to override previous compiler flags without touching the CMake
#   scripts.  Default is empty.
#
# Variables used::
#
#   monoprop_ENABLE_MULTIVERSIONING
#   EXTRA_CXXFLAGS
#
# Variables modified::
#
#   CMAKE_CXX_FLAGS
#
# Environment variables used::
#
#   CXXFLAGS

option_with_print(monoprop_ENABLE_MULTIVERSIONING "Enable function multiversioning" ON)

# code needs C++23 at least
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# do not use compiler extensions to the C++ standard
set(CMAKE_CXX_EXTENSIONS FALSE)
# generate a JSON database of compiler commands (useful for LSP IDEs)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
# position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
# visibility levels
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

# Query the machine-dependent flags for a given -march value and store the
# cleaned, space-separated string in the variable named by OUTPUT_VARIABLE. A
# MARCH of "default" queries the default target (no -march flag).
#
# Usage:
#   _monoprop_query_machine_flags(MARCH <arch> OUTPUT_VARIABLE <var>)
function(_monoprop_query_machine_flags)
  set(_one_value_args MARCH OUTPUT_VARIABLE)
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "")

  if(NOT _arg_OUTPUT_VARIABLE)
    message(FATAL_ERROR "_monoprop_query_machine_flags: OUTPUT_VARIABLE is required")
  endif()
  if(NOT _arg_MARCH)
    message(FATAL_ERROR "_monoprop_query_machine_flags: MARCH is required")
  endif()

  if(_arg_MARCH STREQUAL "default")
    set(_march_args "")
  else()
    set(_march_args "-march=${_arg_MARCH}")
  endif()
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} ${_march_args} -Q --help=target
    COMMAND ${Python_EXECUTABLE} ${_machine_flags_cleaner}
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
  set(${_arg_OUTPUT_VARIABLE} "${_flags}" PARENT_SCOPE)
endfunction()

# Report the machine-dependent flags GCC uses for each target variant by
# querying `gcc -march=<arch> -Q --help=target` and cleaning the output with
# tools/gcc-target-help-clean.py.
find_package(Python COMPONENTS Interpreter REQUIRED QUIET)
set(_machine_flags_cleaner "${PROJECT_SOURCE_DIR}/tools/gcc-target-help-clean.py")

set(monoprop_MACHINE_FLAGS_DEFAULT "")
_monoprop_query_machine_flags(MARCH default OUTPUT_VARIABLE monoprop_MACHINE_FLAGS_DEFAULT)

set(monoprop_TARGET_CLONES "")
set(monoprop_VARIANTS "")
set(monoprop_MACHINE_FLAGS_VARIANTS "")
# function multiversioning is only enabled in release builds on x86_64 architectures
if(
  monoprop_ENABLE_MULTIVERSIONING
  AND
    NOT
      CMAKE_BUILD_TYPE
        STREQUAL
        "Debug"
  AND
    CMAKE_HOST_SYSTEM_PROCESSOR
      MATCHES
      "x86_64"
)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES GNU)
    message(
      FATAL_ERROR
      "Multiversioning is only supported with GNU compilers. Please disable monoprop_ENABLE_MULTIVERSIONING or use a GNU compiler."
    )
  else()
    list(
      APPEND _targets
      "x86-64-v2"
      "x86-64-v3"
      "x86-64-v4"
      "icelake-server"
      "sapphirerapids"
      "graniterapids"
      "znver3"
      "znver4"
      "znver5"
    )

    set(_targets_arch "")
    foreach(_target IN LISTS _targets)
      list(APPEND _targets_arch "\"arch=${_target}\"")
    endforeach()
    list(JOIN _targets_arch ", " TARGETS)

    message(STATUS "Function multiversioning targets: \"default\", ${TARGETS}")
    set(monoprop_TARGET_CLONES "[[using gnu: flatten, target_clones(\"default\", ${TARGETS})]]")
    set(_variants "")
    foreach(_target IN LISTS _targets)
      string(APPEND _variants "monoprop_FMV_VARIANT(\"${_target}\")\n")
    endforeach()
    set(monoprop_VARIANTS "${_variants}")

    set(_machine_flags_variants "")
    foreach(_target IN LISTS _targets)
      _monoprop_query_machine_flags(MARCH "${_target}" OUTPUT_VARIABLE _target_machine_flags)
      string(
        APPEND _machine_flags_variants
        "monoprop_FMV_MACHINE_FLAGS(\"${_target}\", \"${_target_machine_flags}\")\n"
      )
    endforeach()
    set(monoprop_MACHINE_FLAGS_VARIANTS "${_machine_flags_variants}")
  endif()
endif()

# generate a header file with a macro containing (or not) the function multiversioning attribute
configure_file(
  ${PROJECT_SOURCE_DIR}/include/monoprop/FunctionMultiversioning.h.in
  ${PROJECT_BINARY_DIR}/include/monoprop/FunctionMultiversioning.h
  @ONLY
)

set(monoprop_CXX_FLAGS "")
include(${CMAKE_CURRENT_LIST_DIR}/GNU.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Intel.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Clang.CXX.cmake)
