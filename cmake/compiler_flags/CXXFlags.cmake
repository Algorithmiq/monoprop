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

option_with_print(monoprop_ENABLE_ARCH_FLAGS "Enable architecture-specific compiler flags" ON)

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

set(ARCH_FLAG "")
if(monoprop_ENABLE_ARCH_FLAGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(ARCH_FLAG "-march=native")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    if(WIN32) # use AVX2 on Windows
      set(ARCH_FLAG "/arch:AVX2")
    else()
      set(ARCH_FLAG "-march=native")
    endif()
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(ARCH_FLAG "/arch:AVX2")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(ARCH_FLAG "-xHost")
  endif()
endif()

set(monoprop_CXX_FLAGS "")
include(${CMAKE_CURRENT_LIST_DIR}/GNU.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Intel.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Clang.CXX.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/MSVC.CXX.cmake)
