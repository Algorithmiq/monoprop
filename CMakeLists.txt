cmake_minimum_required(VERSION 3.28...4.0)

project(monoprop LANGUAGES CXX)

# warn if the user invokes CMake directly
if(NOT SKBUILD)
  message(
    WARNING
    "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files."
  )
endif()

if(APPLE)
  set(THREADS_FOUND TRUE CACHE BOOL "")
  set(CMAKE_THREAD_LIBS_INIT "-lpthread" CACHE STRING "")
  set(CMAKE_HAVE_THREADS_LIBRARY TRUE CACHE BOOL "")
  set(CMAKE_USE_PTHREADS_INIT TRUE CACHE BOOL "")
  set(THREADS_PREFER_PTHREAD_FLAG ON CACHE BOOL "")
endif()

# if CMAKE_BUILD_TYPE undefined, we set it to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(monoprop_ENABLE_MPI "Enable MPI parallelization" OFF)
option(
  monoprop_WIDE_TERM_INDEX
  "Use 64-bit term indices (support > 2^32 terms per partition)"
  OFF
)

option(monoprop_ENABLE_CXX_UNIT_TESTS "Enable C++ unit test suite" ON)

# Compile the engine once per x86-64 ISA tier and pick one when monoprop is imported. This is what
# published wheels want and what a source build does not: a source build has -march=native, which is
# strictly better than any tier, so the default is OFF. See cmake/compiler_flags/FatBinary.cmake and
# docs/content/docs/fat-binary.mdx.
option(
  monoprop_ENABLE_FAT_BINARY
  "Build one copy of the engine per x86-64 ISA tier and dispatch at import time"
  OFF
)

# Empty means "take the default for this ISA", resolved below once CXXFlags.cmake has defined
# monoprop_ENABLE_ARCH_FLAGS. A cache entry so a build can pin the crossover explicitly.
set(
  monoprop_SPARSE_ROW_MIN_MODES
  ""
  CACHE STRING
  "Logical mode count at or above which the sparse row store is preferred (empty = ISA default)"
)

# How many storage words the scan's per-term kernel is specialized for. The gain is confined to narrow
# registers -- it is the per-term word loops getting a compile-time trip count, which matters least
# where the loop is longest -- so this is a cap and not a switch: 0 turns the specialization off
# entirely and gives the unspecialized kernel at every width, Bitset::kInlineWords specializes the whole
# inline regime. See cpp/monoprop/detail/evolution/layer_build/TermProduct.h.
set(
  monoprop_NARROW_KERNEL_MAX_WORDS
  4
  CACHE STRING
  "Storage word counts 1..N for which the per-term scan kernel is specialized (0 disables)"
)

# The upper bound is Bitset::kInlineWords -- a specialized kernel's precondition is that every operand
# is inline -- read out of the header rather than restated here. TermProduct.h static_asserts the same
# bound, but only after a compile has started; this is the check that names the variable the user set.
file(
  STRINGS
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/monoprop/Bitset.h"
  monoprop_INLINE_WORDS_DECL
  REGEX "kInlineWords[ \t]*=[ \t]*[0-9]+"
  LIMIT_COUNT 1
)
string(REGEX MATCH "[0-9]+" monoprop_INLINE_WORDS "${monoprop_INLINE_WORDS_DECL}")
if(NOT monoprop_INLINE_WORDS)
  message(
    FATAL_ERROR
    "could not read Bitset::kInlineWords out of cpp/monoprop/Bitset.h; the declaration this scrapes moved"
  )
endif()

if(
  NOT monoprop_NARROW_KERNEL_MAX_WORDS MATCHES "^[0-9]+$"
  OR monoprop_NARROW_KERNEL_MAX_WORDS GREATER monoprop_INLINE_WORDS
)
  message(
    FATAL_ERROR
    "monoprop_NARROW_KERNEL_MAX_WORDS must be 0..${monoprop_INLINE_WORDS} (Bitset::kInlineWords), got '${monoprop_NARROW_KERNEL_MAX_WORDS}'"
  )
endif()

set(Python_FIND_VIRTUALENV FIRST)
find_package(
  Python
  3.11
  REQUIRED
  COMPONENTS
    Interpreter
    Development.Module
    ${SKBUILD_SABI_COMPONENT}
)

find_package(Threads REQUIRED)

if(monoprop_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/compiler_flags/CXXFlags.cmake)

# The sparse/dense crossover is set by whether the target has a vector popcount, so it follows the
# arch flags rather than the mode count alone: dense costs one pass per storage word, sparse is flat
# in the width, and without a vector popcount the dense pass degrades sooner.
#
# Both values are the *first whole 32-mode storage block where sparse rows win by more than the
# run-to-run spread*, measured end to end through the propagator (not on the Stage 3 representation
# bench, which predates the codes algebra and the support-form query record and modelled dense as a
# word-at-a-time scan rather than the packed position lists that actually ship). The workload is held
# identical across widths by relabelling one operator into a wider register, so only the storage width
# varies; sparse comes out flat in the width and dense monotone, and the crossing sits in the same
# block at cutoffs 4, 6 and 10. See notes/monomial-storage §6e for the tables.
#
# Expect +-1 block of machine dependence: the crossing is where a dense word pass costs what a lane
# read does, so cache and popcount throughput move it. Both directions cost only a few percent near
# the crossing, which is why the first *clear* win is the threshold rather than the first win.
#
# A fat binary pins one value across every tier rather than following each tier's capability, even
# though only the top tier has a vector popcount and 768 is the measured crossing there. The two
# backends agree on term sets and values but not on term *order*, so a per-tier threshold would make a
# wide run's accumulation order -- and hence its energy in the last bits -- depend on which CPU the
# wheel landed on. 256 is the value three of the four tiers want and the value today's wheels already
# use, so nothing regresses; the top tier gives up a few percent on systems at or above 256 modes.
# One consequence to expect: a fat capture and a monoprop_ENABLE_ARCH_FLAGS=ON capture disagree on
# any case at or above 256 storage modes, because they pick different backends there. Compare fat
# against fat -- `just diff-baseline-variants`, not `just diff-baseline`.
if(monoprop_SPARSE_ROW_MIN_MODES STREQUAL "")
  if(monoprop_ENABLE_FAT_BINARY)
    set(monoprop_SPARSE_ROW_MIN_MODES 256)
  elseif(monoprop_ENABLE_ARCH_FLAGS)
    set(monoprop_SPARSE_ROW_MIN_MODES 768)
  else()
    set(monoprop_SPARSE_ROW_MIN_MODES 256)
  endif()
endif()

# report on compiler flags in use
message(STATUS "Configuring a ${CMAKE_BUILD_TYPE} build")
string(TOUPPER ${CMAKE_BUILD_TYPE} _cmake_build_type_upper)

message(STATUS "Compiler flags for ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "   From environment       : ${monoprop_CXX_FLAGS_FROM_ENV}")
set(
  _cmake_build_type_specific_flags
  "${CMAKE_CXX_FLAGS_${_cmake_build_type_upper}}"
)
message(
  STATUS
  "   Build-type-specific    : ${_cmake_build_type_specific_flags}"
)
message(STATUS "   Vectorization flag     : ${ARCH_FLAG}")
message(STATUS "   Fat binary             : ${monoprop_ENABLE_FAT_BINARY}")
if(monoprop_ENABLE_FAT_BINARY)
  message(STATUS "   ISA tiers              : ${monoprop_FAT_TIERS}")
  message(STATUS "   Tier tuning            : -mtune=${monoprop_FAT_MTUNE}")
  message(STATUS "   Baseline ISA floor     : ${_monoprop_baseline_flags}")
endif()
message(
  STATUS
  "   Project defaults       : ${CMAKE_CXX${CMAKE_CXX_STANDARD}_STANDARD_COMPILE_OPTION} ${monoprop_CXX_FLAGS}"
)
message(STATUS "   User-appended          : ${EXTRA_CXXFLAGS}")

message(STATUS "   MPI parallelization    : ${monoprop_ENABLE_MPI}")
message(STATUS "   Wide term index        : ${monoprop_WIDE_TERM_INDEX}")
message(STATUS "   C++ unit tests         : ${monoprop_ENABLE_CXX_UNIT_TESTS}")
message(
  STATUS
  "   Sparse rows from       : ${monoprop_SPARSE_ROW_MIN_MODES} modes"
)
message(
  STATUS
  "   Narrow scan kernel     : up to ${monoprop_NARROW_KERNEL_MAX_WORDS} storage words"
)

# Testing is enabled from the *top-level* list file on purpose: CTest's root is wherever
# enable_testing() was called, so called from cpp/ it wrote no top-level CTestTestfile.cmake and every
# documented entry point (the CMakePresets test presets, `just test-mpi`, `just test-wide`) pointed
# ctest at a directory with no tests. ctest reports "No tests were found" and exits 0 for that, so those
# commands were silently running nothing.
if(monoprop_ENABLE_CXX_UNIT_TESTS)
  enable_testing()
  include(CTest)
endif()

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

add_library(monoprop-objs OBJECT "")
add_library(monoprop SHARED $<TARGET_OBJECTS:monoprop-objs>)

# must run before add_subdirectory(cpp): CTest's enabled-ness does not propagate
# back up to a parent directory that has already been added as a subdirectory.
if(monoprop_ENABLE_CXX_UNIT_TESTS)
  enable_testing()
  include(CTest)
endif()

add_subdirectory(cpp)
add_subdirectory(src/monoprop/bindings)
