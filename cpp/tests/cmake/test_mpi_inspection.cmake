# Script-mode test of the MPI inspection helpers: cmake -P cpp/tests/cmake/test_mpi_inspection.cmake

include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/custom/monopropMPIABI.cmake)

function(expect_abi macros expected)
  monoprop_mpi_abi_from_macros(_abi "${macros}")
  if(NOT _abi STREQUAL expected)
    message(
      FATAL_ERROR
      "ABI of [${macros}]: got '${_abi}', expected '${expected}'"
    )
  endif()
endfunction()

function(expect_vendor macros name version)
  monoprop_mpi_vendor_from_macros(_name _version "${macros}")
  if(NOT _name STREQUAL name OR NOT _version STREQUAL version)
    message(
      FATAL_ERROR
      "vendor of [${macros}]: got '${_name}' '${_version}', expected '${name}' '${version}'"
    )
  endif()
endfunction()

function(expect_problem name version expected)
  monoprop_mpi_version_problem(_problem "${name}" "${version}")
  if(NOT _problem STREQUAL expected)
    message(
      FATAL_ERROR
      "${name} ${version}: got '${_problem}', expected '${expected}'"
    )
  endif()
endfunction()

expect_abi("#define MPI_ABI_VERSION 1\n#define MPICH_VERSION \"5.0.2\"\n" mpiabi
)
expect_abi("#define MPI_ABI_VERSION 0\n#define MPICH_VERSION \"5.0.2\"\n" mpich)
expect_abi("#define OPEN_MPI 1\n#define OMPI_MAJOR_VERSION 5\n" openmpi)
expect_abi("#define MPICH_VERSION \"4.2.0\"\n" mpich)

expect_vendor(
  "#define OPEN_MPI 1\n#define OMPI_MAJOR_VERSION 4\n#define OMPI_MINOR_VERSION 0\n#define OMPI_RELEASE_VERSION 7\n"
  "Open MPI" "4.0.7"
)
expect_vendor("#define MPICH_VERSION \"4.0\"\n#define MPICH_NUMVERSION 40000300\n" "MPICH" "4.0"
)
# derivatives also define MPICH_VERSION (their MPICH base), so they must win over it
expect_vendor("#define MPICH_VERSION \"3.4a2\"\n#define CRAY_MPICH_VERSION 8.1.29\n" "Cray MPICH" "8.1.29"
)
expect_vendor("#define MPICH_VERSION \"4.1.2\"\n#define MVAPICH_VERSION \"3.0\"\n" "MVAPICH" "3.0"
)
expect_vendor("#define MVAPICH2_VERSION \"2.3.7\"\n" "MVAPICH" "2.3.7")
expect_vendor("#define MPICH_VERSION \"3.4\"\n#define I_MPI_NUMVERSION 20211900300\n" "Intel MPI" "2021.19.0"
)
expect_vendor("#define MPI_VERSION 3\n" "unknown" "")

expect_problem("Open MPI" "4.0.7" "monoprop needs Open MPI >= 4.1, found 4.0.7")
expect_problem("Open MPI" "4.1.0" "")
expect_problem("MPICH" "4.0" "")
expect_problem("MPICH" "3.4.3" "monoprop needs MPICH >= 4.0, found 3.4.3")
expect_problem("Intel MPI" "2021.1.1" "")
expect_problem("Intel MPI" "2019.12.0" "monoprop needs Intel MPI >= 2021.0, found 2019.12.0")
expect_problem("Cray MPICH" "8.1.29" "")
expect_problem("MVAPICH" "2.3.7" "monoprop needs MVAPICH >= 3.0, found 2.3.7")
# an unrecognised implementation only warns
expect_problem("unknown" "" "")

message(STATUS "MPI inspection: all checks passed")
