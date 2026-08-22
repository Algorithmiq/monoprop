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

# The single place an architecture is chosen. Everything downstream reads monoprop_ARCH_MARCH rather
# than re-deciding, because the three sites that used to decide independently disagreed: ARCH_FLAG was
# additionally suppressed in Debug, while the provenance query and the sparse-row crossover were gated
# on the option alone. A Debug build therefore compiled portable code, advertised the native ISA and
# took the native-tuned crossover.
#
# monoprop_ARCH_MARCH is the -march *value*, or "default" for "no -march flag", which is the spelling
# _monoprop_query_machine_flags takes.
set(ARCH_FLAG "")
set(monoprop_ARCH_MARCH "default")
if(monoprop_ENABLE_ARCH_FLAGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(ARCH_FLAG "-march=native")
    set(monoprop_ARCH_MARCH "native")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(ARCH_FLAG "-xHost")
    set(monoprop_ARCH_MARCH "native")
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
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "FLAGS")

  if(NOT _arg_OUTPUT_VARIABLE)
    message(
      FATAL_ERROR
      "_monoprop_query_machine_flags: OUTPUT_VARIABLE is required"
    )
  endif()
  if(NOT _arg_MARCH AND NOT _arg_FLAGS)
    message(
      FATAL_ERROR
      "_monoprop_query_machine_flags: one of MARCH or FLAGS is required"
    )
  endif()

  # FLAGS is the general form: a fat-binary tier is a whole flag list, not a single -march token, and
  # the reported flags have to reflect all of it or the provenance is a guess.
  if(_arg_FLAGS)
    set(_march_args "${_arg_FLAGS}")
  elseif(_arg_MARCH STREQUAL "default")
    set(_march_args "")
  else()
    set(_march_args "-march=${_arg_MARCH}")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    execute_process(
      COMMAND
        # gersemi: off
        ${CMAKE_CXX_COMPILER} ${_march_args} -\#\#\# -x c++ -c /dev/null
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
          "${PROJECT_SOURCE_DIR}/tools/target-help-clean.py" --mode clang
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
    execute_process(
      COMMAND
        ${CMAKE_CXX_COMPILER} ${_march_args} -Q --help=target
      COMMAND
        ${Python_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tools/target-help-clean.py"
        --mode gcc
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

# Write a Variants.h reporting one variant's identity and the machine flags the compiler actually
# resolved for it. Called once per fat-binary tier, into a per-tier include directory that the tier's
# object library puts ahead of the shared one, plus once for the plain single-ISA build.
#
# This is the fix for a provenance bug worth naming: the header used to be configured once, from a
# query of -march=native whenever monoprop_ENABLE_ARCH_FLAGS was ON regardless of what was actually
# compiled. So monoprop.__variant__, monoprop.__compiler_flags__ and every benchmark artifact's
# machine-flags entry reported the host's ISA even when the build had been pointed somewhere else --
# which is exactly the metadata a fat binary needs to be trustworthy, since it is how you tell which
# tier got loaded.
#
# Usage:
#   _monoprop_generate_variant_header(VARIANT_ID <id> OUTPUT_DIR <dir> [MARCH <arch>] [FLAGS <flags>])
function(_monoprop_generate_variant_header)
  set(
    _one_value_args
    VARIANT_ID
    OUTPUT_DIR
    MARCH
  )
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "FLAGS")

  if(NOT _arg_VARIANT_ID OR NOT _arg_OUTPUT_DIR)
    message(
      FATAL_ERROR
      "_monoprop_generate_variant_header: VARIANT_ID and OUTPUT_DIR are required"
    )
  endif()

  # Unquoted on purpose: cmake_parse_arguments(PARSE_ARGV) escapes the semicolons inside a single
  # argument, so a quoted list arrives as one flag spelled "-march=x86-64\;-mtune=skylake\;..." and
  # the query silently reports the compiler's defaults instead of the variant's.
  _monoprop_query_machine_flags(
    MARCH "${_arg_MARCH}"
    FLAGS ${_arg_FLAGS}
    OUTPUT_VARIABLE monoprop_VARIANT_MACHINE_FLAGS
  )
  set(monoprop_VARIANT_ID "${_arg_VARIANT_ID}")

  configure_file(
    ${PROJECT_SOURCE_DIR}/cpp/include/monoprop/Variants.h.in
    ${_arg_OUTPUT_DIR}/monoprop/Variants.h
    @ONLY
  )
endfunction()

_monoprop_generate_variant_header(
  VARIANT_ID "${monoprop_ARCH_MARCH}"
  MARCH "${monoprop_ARCH_MARCH}"
  OUTPUT_DIR "${PROJECT_BINARY_DIR}/include"
)

set(monoprop_CXX_FLAGS "")
include(${CMAKE_CURRENT_LIST_DIR}/GNU.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Intel.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Clang.CXX.cmake)

# Must come last: with the fat binary enabled this overwrites ARCH_FLAG with the baseline tier's flags
# and defines the per-tier engine targets.
include(${CMAKE_CURRENT_LIST_DIR}/FatBinary.cmake)
