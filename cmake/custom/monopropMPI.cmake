# MPI setup
# =========
#
# Included from the top-level CMakeLists.txt. Sets monoprop_MPI_ABI_NAMES: the MPI ABI families
# this build compiles MPI variants for (empty without MPI).

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/monopropMPIABI.cmake)

set(
  monoprop_MPI_ABIS
  ""
  CACHE STRING
  "MPI ABI variants to build from install prefixes, as ';'-separated <abi>=<prefix> entries (wheel builds)"
)

set(monoprop_MPI_ABI_NAMES "")

#[=[
_monoprop_add_prefix_mpi_target(<abi>=<prefix>)

Creates monoprop::mpi_<abi> from an MPI installed under <prefix> (include/mpi.h, lib/libmpi or
lib/libmpi_abi) without compiler wrappers, and appends <abi> to monoprop_MPI_ABI_NAMES.
#]=]
function(_monoprop_add_prefix_mpi_target entry)
  if(NOT entry MATCHES "^(mpich|openmpi|mpiabi)=(.+)$")
    message(
      FATAL_ERROR
      "monoprop_MPI_ABIS entry '${entry}' is not <abi>=<prefix> with <abi> one of mpich, openmpi, mpiabi"
    )
  endif()
  set(_abi "${CMAKE_MATCH_1}")
  set(_root "${CMAKE_MATCH_2}")
  if(_abi IN_LIST monoprop_MPI_ABI_NAMES)
    message(FATAL_ERROR "monoprop_MPI_ABIS names '${_abi}' twice")
  endif()
  if(_abi STREQUAL "mpiabi")
    # MPICH's mpi.h switches to mpi_abi.h on MPI_ABI
    set(_library mpi_abi)
    set(_definitions MPI_ABI)
  else()
    set(_library mpi)
    set(_definitions "")
  endif()
  if(NOT EXISTS "${_root}/include/mpi.h")
    message(
      FATAL_ERROR
      "monoprop_MPI_ABIS entry '${entry}': no ${_root}/include/mpi.h"
    )
  endif()
  find_library(
    _library_path
    NAMES
      ${_library}
    PATHS
      "${_root}/lib"
    NO_DEFAULT_PATH
    NO_CACHE
    REQUIRED
  )
  monoprop_inspect_mpi(
    _found _problem
    INCLUDE_DIRS "${_root}/include"
    DEFINITIONS ${_definitions}
  )
  if(NOT _found STREQUAL _abi)
    message(
      FATAL_ERROR
      "monoprop_MPI_ABIS entry '${entry}': the mpi.h under ${_root}/include is '${_found}', not '${_abi}'"
    )
  endif()
  if(_problem)
    message(FATAL_ERROR "monoprop_MPI_ABIS entry '${entry}': ${_problem}")
  endif()
  add_library(monoprop::mpi_${_abi} INTERFACE IMPORTED)
  target_include_directories(
    monoprop::mpi_${_abi}
    SYSTEM
    INTERFACE
      "${_root}/include"
  )
  target_compile_definitions(
    monoprop::mpi_${_abi}
    INTERFACE
      ${_definitions}
      MPICH_SKIP_MPICXX
      OMPI_SKIP_MPICXX
  )
  target_link_libraries(monoprop::mpi_${_abi} INTERFACE "${_library_path}")
  message(STATUS "MPI variant ${_abi}: ${_library_path}")
  set(
    monoprop_MPI_ABI_NAMES
    ${monoprop_MPI_ABI_NAMES}
    ${_abi}
    PARENT_SCOPE
  )
endfunction()

if(monoprop_MPI_ABIS AND NOT monoprop_ENABLE_MPI)
  message(
    FATAL_ERROR
    "monoprop_MPI_ABIS is set but monoprop_ENABLE_MPI is OFF; enable MPI (monoprop.mpi) to build MPI variants"
  )
endif()

if(monoprop_ENABLE_MPI AND monoprop_MPI_ABIS)
  foreach(_entry IN LISTS monoprop_MPI_ABIS)
    _monoprop_add_prefix_mpi_target("${_entry}")
  endforeach()
elseif(monoprop_ENABLE_MPI)
  # The MPI-2 C++ bindings would add libmpicxx/libmpi_cxx, outside the soname contract the wheels
  # rely on; monoprop only uses the C API.
  set(MPI_CXX_SKIP_MPICXX ON)
  find_package(MPI REQUIRED COMPONENTS CXX)
  monoprop_inspect_mpi(
    _monoprop_mpi_abi _monoprop_mpi_problem
    INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS}
    DEFINITIONS ${MPI_CXX_COMPILE_DEFINITIONS}
  )
  if(_monoprop_mpi_problem)
    message(
      FATAL_ERROR
      "${_monoprop_mpi_problem} (the MPI CMake found: ${MPI_CXX_INCLUDE_DIRS})"
    )
  endif()
  set(monoprop_MPI_ABI_NAMES ${_monoprop_mpi_abi})
  add_library(monoprop::mpi_${_monoprop_mpi_abi} INTERFACE IMPORTED)
  target_link_libraries(
    monoprop::mpi_${_monoprop_mpi_abi}
    INTERFACE
      MPI::MPI_CXX
  )
endif()
