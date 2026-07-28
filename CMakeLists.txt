cmake_minimum_required(VERSION 3.28...4.0)

include(cmake/CPM.cmake)

if(SKBUILD)
  # this value doesn't really matter, as it will be set by scikit-build-core
  set(PROJECT_VERSION "0.1.0")
else()
  cpmaddpackage("gh:LecrisUT/CMakeExtraUtils@0.4.1")
  include(${CMakeExtraUtils_BINARY_DIR}/DynamicVersion.cmake)
  dynamic_version(PROJECT_PREFIX monoprop_)
endif()

project(monoprop LANGUAGES CXX VERSION ${PROJECT_VERSION})

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

# A cache STRING rather than an option(): it holds a mode count, not a boolean. An explicitly
# empty -Dmonoprop_MAX_NUM_MODES= is left empty and fails in the binding generators, instead of
# silently reverting to the default. Every option below is echoed in the configuration report.
set(
  monoprop_MAX_NUM_MODES
  "250"
  CACHE STRING
  "Maximum number of simulable Fermionic modes with Python bindings"
)
option(monoprop_ENABLE_MPI "Enable MPI parallelization" OFF)
option(
  monoprop_WIDE_TERM_INDEX
  "Use 64-bit term indices (support > 2^32 terms per partition)"
  OFF
)
if(monoprop_WIDE_TERM_INDEX)
  add_compile_definitions(monoprop_WIDE_TERM_INDEX)
endif()

find_package(Threads REQUIRED)

if(monoprop_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/compiler_flags/CXXFlags.cmake)

# PROJECT_VERSION_FULL comes from dynamic_version(), which is skipped under scikit-build-core.
if(NOT DEFINED SKBUILD)
  message(STATUS "${PROJECT_NAME} version: ${PROJECT_VERSION_FULL}")
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
message(STATUS "   MPI parallelization    : ${monoprop_ENABLE_MPI}")
message(STATUS "   Wide term index        : ${monoprop_WIDE_TERM_INDEX}")
message(STATUS "   Max simulable modes    : ${monoprop_MAX_NUM_MODES}")

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

configure_file(
  ${PROJECT_SOURCE_DIR}/include/monoprop/Info.h.in
  ${PROJECT_BINARY_DIR}/include/monoprop/Info.h
)

add_subdirectory(src)

if(NOT DEFINED SKBUILD)
  enable_testing()
  include(CTest)

  # This must come last!!
  add_subdirectory(tests/cpp)
endif()
