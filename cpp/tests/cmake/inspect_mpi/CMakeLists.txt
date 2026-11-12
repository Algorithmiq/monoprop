# Inspect an installed MPI the way configure does:
#   cmake --fresh -S cpp/tests/cmake/inspect_mpi -B build/inspect-mpi \
#     -DMPI_INCLUDE_DIR=<dir> -DEXPECTED_ABI=<abi>
# Fails unless the mpi.h there has ABI family EXPECTED_ABI and meets monoprop's version floor.
# A project rather than a script: the inspection compiles a probe with try_compile.

cmake_minimum_required(VERSION 3.28...4.0)
project(monoprop_inspect_mpi LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/custom/monopropMPIABI.cmake)

monoprop_mpi_macros(_macros INCLUDE_DIRS "${MPI_INCLUDE_DIR}")
monoprop_mpi_vendor_from_macros(_name _version "${_macros}")
monoprop_inspect_mpi(_abi _problem INCLUDE_DIRS "${MPI_INCLUDE_DIR}")
message(STATUS "${MPI_INCLUDE_DIR}: ABI family ${_abi}, ${_name} ${_version}")
if(NOT _abi STREQUAL EXPECTED_ABI)
  message(FATAL_ERROR "expected the ${EXPECTED_ABI} ABI family, found ${_abi}")
endif()
if(_problem)
  message(FATAL_ERROR "${_problem}")
endif()
