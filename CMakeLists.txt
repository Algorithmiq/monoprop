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

# Options handling utilities Macro for printing an option in a consistent manner
# Written by Lori A. Burns (@loriab) and Ryan M. Richard (@ryanmrichard) Syntax:
# print_option(<option to print> <was specified>)
macro(print_option variable default)
  if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
    message(STATUS "Setting (unspecified) option ${variable}: ${default}")
  else()
    message(STATUS "Setting option ${variable}: ${${variable}}")
  endif()
endmacro()

# Wraps an option with default ON/OFF. Adds nice messaging to option() Written
# by Lori A. Burns (@loriab) and Ryan M. Richard (@ryanmrichard) Syntax:
# option_with_print(<option name> <description> <default value>)
macro(option_with_print variable msge default)
  print_option(${variable} ${default})
  option(${variable} ${msge} ${default})
endmacro()

# Wraps an option with a default other than ON/OFF and prints it Written by Lori
# A. Burns (@loriab) and Ryan M. Richard (@ryanmrichard) NOTE: Can't combine
# with above b/c CMake handles ON/OFF options specially NOTE2: CMake variables
# are always defined so need to further check for if they are the NULL string.
# This is also why we need the force Syntax: option_with_default(<option name>
# <description> <default value>)
macro(option_with_default variable msge default)
  print_option(${variable} "${default}")
  if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
    set(${variable} "${default}" CACHE STRING ${msge} FORCE)
  endif()
endmacro()

option_with_default(monoprop_MAX_NUM_MODES "Maximum number of simulable Fermionic modes with Python bindings" 250)
option_with_print(monoprop_ENABLE_MPI "Enable MPI parallelization" OFF)
option_with_print(monoprop_WIDE_TERM_INDEX "Use 64-bit term indices (support > 2^32 local terms)" OFF)
if(monoprop_WIDE_TERM_INDEX)
  add_compile_definitions(monoprop_WIDE_TERM_INDEX)
endif()

# The in-repo thread pool (src/monoprop/Threading.cpp) uses std::thread.
find_package(Threads REQUIRED)

if(monoprop_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/compiler_flags/CXXFlags.cmake)

if(DEFINED SKBUILD)
  message(
    STATUS
    "Maximum simulable Fermionic modes with Python bindings: ${monoprop_MAX_NUM_MODES}"
  )
else()
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
