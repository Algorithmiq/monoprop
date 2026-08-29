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

include(CMakeDependentOption)

# Disable machine-specific code generation for debug and sanitizer builds.
set(_monoprop_arch_flags_supported TRUE)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT monoprop_SANITIZER STREQUAL "none")
  set(_monoprop_arch_flags_supported FALSE)
endif()

cmake_dependent_option(
  monoprop_ENABLE_ARCH_FLAGS
  "Enable architecture-specific compiler flags"
  ON
  "_monoprop_arch_flags_supported"
  OFF
)

# What CXXFLAGS actually contained, recorded before anything appends to CMAKE_CXX_FLAGS. The fat
# binary appends its baseline ISA floor there (see FatBinary.cmake), and the status report has to
# be able to tell the two apart or it attributes our flags to the user's environment.
set(monoprop_CXX_FLAGS_FROM_ENV "${CMAKE_CXX_FLAGS}")

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
# than deciding again from monoprop_ENABLE_ARCH_FLAGS, so that what is compiled and what is reported
# cannot disagree -- a build that advertises an ISA it did not compile for makes every benchmark
# artifact and every monoprop.__compiler_flags__ a guess.
#
# monoprop_ARCH_MARCH is the variant *id* a single-ISA build reports as monoprop.__variant__:
# "native", or "default" for a build with no -march flag. The flags themselves are ARCH_FLAG, which is
# what the provenance query reads.
set(ARCH_FLAG "")
set(monoprop_ARCH_MARCH "default")
if(monoprop_ENABLE_ARCH_FLAGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(ARCH_FLAG "-march=native")
    set(monoprop_ARCH_MARCH "native")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(ARCH_FLAG "-xHost")
    set(monoprop_ARCH_MARCH "native")
  endif()
endif()

# Query the machine-dependent flags the compiler applies under a given architecture selection and
# store the cleaned, space-separated string in the variable named by OUTPUT_VARIABLE.
#
# Usage:
#   _monoprop_query_machine_flags(ARCH_FLAGS <flags...> OUTPUT_VARIABLE <var>)
function(_monoprop_query_machine_flags)
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "OUTPUT_VARIABLE" "ARCH_FLAGS")

  if(NOT _arg_OUTPUT_VARIABLE)
    message(
      FATAL_ERROR
      "_monoprop_query_machine_flags: OUTPUT_VARIABLE is required"
    )
  endif()

  set(_arch_args ${_arg_ARCH_FLAGS})
  if(_arch_args)
    string(JOIN " " _arch_label ${_arch_args})
  else()
    set(_arch_label "the default target")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    execute_process(
      COMMAND
        # gersemi: off
        ${CMAKE_CXX_COMPILER} ${_arch_args} -\#\#\# -x c++ -c /dev/null
      # gersemi: on
      ERROR_VARIABLE _query_output
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _query_result
    )
    if(NOT _query_result EQUAL 0)
      message(
        WARNING
        "Failed to query machine-dependent flags for ${_arch_label} with AppleClang (exit code ${_query_result}). Continuing with empty machine flags."
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
          "Failed to parse AppleClang machine-dependent flags for ${_arch_label} (exit code ${_parse_result}). Continuing with empty machine flags."
        )
        set(_flags "")
      endif()
    endif()
  else()
    execute_process(
      COMMAND
        ${CMAKE_CXX_COMPILER} ${_arch_args} -Q --help=target
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
        "Failed to query machine-dependent flags for ${_arch_label} (exit code ${_result})"
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
#   _monoprop_generate_variant_header(VARIANT_ID <id> OUTPUT_DIR <dir> [ARCH_FLAGS <flags...>])
function(_monoprop_generate_variant_header)
  cmake_parse_arguments(
    PARSE_ARGV 0
    _arg
    ""
    "VARIANT_ID;OUTPUT_DIR"
    "ARCH_FLAGS"
  )

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
    ARCH_FLAGS ${_arg_ARCH_FLAGS}
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
  ARCH_FLAGS ${ARCH_FLAG}
  OUTPUT_DIR "${PROJECT_BINARY_DIR}/include"
)

set(monoprop_CXX_FLAGS "")
include(${CMAKE_CURRENT_LIST_DIR}/GNU.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Intel.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Clang.CXX.cmake)

# Must come last: with the fat binary enabled this overwrites ARCH_FLAG with the baseline tier's flags
# and defines the per-tier engine targets.
include(${CMAKE_CURRENT_LIST_DIR}/FatBinary.cmake)
