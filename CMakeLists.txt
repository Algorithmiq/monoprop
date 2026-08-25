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

include(${PROJECT_SOURCE_DIR}/cmake/compiler_flags/Sanitizers.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/compiler_flags/CXXFlags.cmake)

# Sparse/dense crossover depends on vector popcount support.
# Dense scales with storage-word passes, while sparse is mostly width-flat.
# Without vector popcount, dense degrades earlier.
#
# This is intentionally not a cache variable: it should track the flags actually used
# for compilation. monoprop_ROW_STORE=dense|sparse already lets users force a backend
# at runtime without rebuilding.
#
# Thresholds are the first full 32-mode block where sparse is clearly faster than dense
# beyond run-to-run noise, measured end-to-end in the propagator. Expect about +/-1 block
# variation across machines due to cache and popcount throughput.
if(ARCH_FLAG)
  set(monoprop_SPARSE_ROW_MIN_MODES 768)
else()
  set(monoprop_SPARSE_ROW_MIN_MODES 256)
endif()

# report on compiler flags in use
message(STATUS "Configuring a ${CMAKE_BUILD_TYPE} build")
string(TOUPPER ${CMAKE_BUILD_TYPE} _cmake_build_type_upper)

message(STATUS "Compiler flags for ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "   From environment       : ${CMAKE_CXX_FLAGS}")
set(
  _cmake_build_type_specific_flags
  "${CMAKE_CXX_FLAGS_${_cmake_build_type_upper}}"
)
message(
  STATUS
  "   Build-type-specific    : ${_cmake_build_type_specific_flags}"
)
message(STATUS "   Vectorization flag     : ${ARCH_FLAG}")
message(
  STATUS
  "   Project defaults       : ${CMAKE_CXX${CMAKE_CXX_STANDARD}_STANDARD_COMPILE_OPTION} ${monoprop_CXX_FLAGS}"
)
message(STATUS "   User-appended          : ${EXTRA_CXXFLAGS}")
message(STATUS "   Sanitizer profile      : ${monoprop_SANITIZER}")

message(STATUS "   MPI parallelization    : ${monoprop_ENABLE_MPI}")
message(STATUS "   Wide term index        : ${monoprop_WIDE_TERM_INDEX}")
message(STATUS "   C++ unit tests         : ${monoprop_ENABLE_CXX_UNIT_TESTS}")
message(
  STATUS
  "   Sparse rows from       : ${monoprop_SPARSE_ROW_MIN_MODES} modes"
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
