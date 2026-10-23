# Inspect an installed MPI the way configure does:
#   cmake -DMPI_INCLUDE_DIR=<dir> -DEXPECTED_ABI=<abi> [-DCMAKE_CXX_COMPILER=<c++>] -P inspect_mpi.cmake
# Fails unless the mpi.h there has ABI family EXPECTED_ABI and meets monoprop's version floor.

include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/custom/monopropMPIABI.cmake)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER c++)
endif()
# monoprop_mpi_macros() writes its probe source under CMAKE_BINARY_DIR
set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../build/inspect-mpi")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles")

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
