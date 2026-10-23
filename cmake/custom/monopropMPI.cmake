# MPI setup
# =========
#
# Included from the top-level CMakeLists.txt. Sets monoprop_MPI_ABI_NAMES: the MPI ABI families
# this build compiles MPI variants for (empty without MPI).

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/monopropMPIABI.cmake)

set(monoprop_MPI_ABI_NAMES "")

if(monoprop_ENABLE_MPI)
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
endif()
