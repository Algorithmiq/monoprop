# MPI inspection
# ==============
#
# Names the MPI ABI family of an mpi.h ("mpiabi": the MPI-5 standard ABI; "openmpi": Open MPI's
# libmpi.so.40; "mpich": the MPICH ABI of MPICH, Intel MPI, MVAPICH and Cray MPICH), identifies the
# implementation, and checks it against monoprop's version floors. Used by monoprop's build and by
# monopropConfig.cmake. Everything except monoprop_mpi_macros() and monoprop_inspect_mpi(), which
# compile a probe, works on text, so it runs in script mode
# (cpp/tests/cmake/test_mpi_inspection.cmake).

include_guard(GLOBAL)

#[=[
monoprop_mpi_min_versions(<out-var>)

Sets <out-var> to monoprop's version floors as <vendor>=<floor> entries, keyed by implementation as
mpi4py.MPI.get_vendor() names it. monoprop supports each from its floor to the latest release;
configure time and the Python loader (through the generated monoprop/_mpi_abis.py) enforce them.

A function rather than a variable because of include_guard(GLOBAL): monopropConfig.cmake includes
this file in the consumer's scope, so a second find_package(monoprop) from a sibling directory skips
the include and would not see a variable, while function definitions are global.
#]=]
function(monoprop_mpi_min_versions out_var)
  set(
    ${out_var}
    "Open MPI=4.1"
    "MPICH=4.0"
    "Intel MPI=2021.0"
    "MVAPICH=3.0"
    "Cray MPICH=8.1"
    PARENT_SCOPE
  )
endfunction()

#[=[
monoprop_mpi_macros(<out-var> INCLUDE_DIRS <dir>... [DEFINITIONS <definition>...])

Sets <out-var> to "#define <name> <value>" lines for the macros the functions below read, with the
values the compiler sees after including mpi.h, so macros from headers that mpi.h includes count
too. As FindMPI and CMake's compiler identification do, a probe is compiled with try_compile, never
linked or run, and the values are read back from it with file(STRINGS). DEFINITIONS are given
without -D, as in MPI_CXX_COMPILE_DEFINITIONS; MPICH's standard-ABI mode needs MPI_ABI among them.
#]=]
function(monoprop_mpi_macros out_var)
  cmake_parse_arguments(PARSE_ARGV 1 _arg "" "" "INCLUDE_DIRS;DEFINITIONS")
  set(
    _names
    MPI_ABI_VERSION
    OPEN_MPI
    OMPI_MAJOR_VERSION
    OMPI_MINOR_VERSION
    OMPI_RELEASE_VERSION
    MPICH_VERSION
    MVAPICH_VERSION
    MVAPICH2_VERSION
    CRAY_MPICH_VERSION
    I_MPI_NUMVERSION
  )
  set(
    _source
    "#include <mpi.h>\n#define MONOPROP_STR_(x) #x\n#define MONOPROP_STR(x) MONOPROP_STR_(x)\n"
  )
  foreach(_name IN LISTS _names)
    # extern: a namespace-scope const array has internal linkage and could be dropped
    string(
      APPEND _source
      "#ifdef ${_name}\n"
      "extern char const monoprop_${_name}[];\n"
      "char const monoprop_${_name}[] =\n"
      "  \"INFO:monoprop-mpi[${_name}=\" MONOPROP_STR(${_name}) \"]\";\n"
      "#endif\n"
    )
  endforeach()
  set(_definitions "")
  foreach(
    _definition
    IN
    LISTS
      _arg_DEFINITIONS
    ITEMS
      MPICH_SKIP_MPICXX
      OMPI_SKIP_MPICXX
  )
    list(APPEND _definitions "-D${_definition}")
  endforeach()
  # A static library is never linked, so the probe needs no MPI library.
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(
    _probe
    "${CMAKE_BINARY_DIR}/CMakeFiles/monoprop_mpi_probe${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
  try_compile(
    _compiled
    SOURCE_FROM_CONTENT
      monoprop_mpi_probe.cpp
      "${_source}"
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES=${_arg_INCLUDE_DIRS}"
    COMPILE_DEFINITIONS
      ${_definitions}
    NO_CACHE
    COPY_FILE "${_probe}"
    OUTPUT_VARIABLE _output
  )
  if(NOT _compiled)
    message(
      FATAL_ERROR
      "monoprop: cannot compile mpi.h with include dirs '${_arg_INCLUDE_DIRS}':\n${_output}"
    )
  endif()
  file(
    STRINGS "${_probe}"
    _entries
    REGEX "INFO:monoprop-mpi\\[[A-Z0-9_]+=[^]]*\\]"
  )
  set(_macros "")
  foreach(_entry IN LISTS _entries)
    if(_entry MATCHES "INFO:monoprop-mpi\\[([A-Z0-9_]+)=([^]]*)\\]")
      string(APPEND _macros "#define ${CMAKE_MATCH_1} ${CMAKE_MATCH_2}\n")
    endif()
  endforeach()
  set(${out_var} "${_macros}" PARENT_SCOPE)
endfunction()

#[=[
monoprop_mpi_abi_from_macros(<out-var> <macros>)

Sets <out-var> to mpiabi, openmpi or mpich.
#]=]
function(monoprop_mpi_abi_from_macros out_var macros)
  set(_abi "mpich")
  if(macros MATCHES "#define MPI_ABI_VERSION ([0-9]+)")
    if(CMAKE_MATCH_1 GREATER_EQUAL 1)
      set(_abi "mpiabi")
    endif()
  endif()
  if(_abi STREQUAL "mpich" AND macros MATCHES "#define OPEN_MPI ")
    set(_abi "openmpi")
  endif()
  set(${out_var} "${_abi}" PARENT_SCOPE)
endfunction()

#[=[
monoprop_mpi_vendor_from_macros(<name-var> <version-var> <macros>)

Sets <name-var> to the implementation, named as mpi4py.MPI.get_vendor() names it, or "unknown",
and <version-var> to its dotted version (empty when unknown). MPICH derivatives also define
MPICH_VERSION, their MPICH base, so they are recognised first.
#]=]
function(monoprop_mpi_vendor_from_macros name_var version_var macros)
  set(_number "[0-9]+(\\.[0-9]+)*")
  set(_name "unknown")
  set(_version "")
  if(macros MATCHES "#define CRAY_MPICH_VERSION (${_number})")
    set(_name "Cray MPICH")
    set(_version "${CMAKE_MATCH_1}")
  elseif(macros MATCHES "#define MVAPICH2?_VERSION \"(${_number})")
    set(_name "MVAPICH")
    set(_version "${CMAKE_MATCH_1}")
  elseif(macros MATCHES "#define I_MPI_NUMVERSION ([0-9]+)")
    # major * 10^7 + minor * 10^5 + micro * 10^3 + build, decoded as mpi4py's pympivendor.h does
    math(EXPR _packed "${CMAKE_MATCH_1} / 1000")
    math(EXPR _major "${_packed} / 10000")
    math(EXPR _minor "(${_packed} % 10000) / 100")
    math(EXPR _micro "${_packed} % 100")
    set(_name "Intel MPI")
    set(_version "${_major}.${_minor}.${_micro}")
  elseif(macros MATCHES "#define OPEN_MPI ")
    set(_parts "")
    foreach(_part MAJOR MINOR RELEASE)
      string(
        REGEX MATCH "#define OMPI_${_part}_VERSION ([0-9]+)"
        _match
        "${macros}"
      )
      list(APPEND _parts "${CMAKE_MATCH_1}")
    endforeach()
    set(_name "Open MPI")
    list(JOIN _parts "." _version)
  elseif(macros MATCHES "#define MPICH_VERSION \"(${_number})")
    set(_name "MPICH")
    set(_version "${CMAKE_MATCH_1}")
  endif()
  set(${name_var} "${_name}" PARENT_SCOPE)
  set(${version_var} "${_version}" PARENT_SCOPE)
endfunction()

#[=[
monoprop_mpi_version_problem(<out-var> <name> <version>)

Sets <out-var> to an error message when <name> <version> is below monoprop's floor, and to an
empty string otherwise. An implementation without a floor only warns.
#]=]
function(monoprop_mpi_version_problem out_var name version)
  monoprop_mpi_min_versions(_floors)
  set(_problem "")
  set(_known FALSE)
  foreach(_entry IN LISTS _floors)
    if(_entry MATCHES "^(.+)=(.+)$")
      set(_vendor "${CMAKE_MATCH_1}")
      set(_floor "${CMAKE_MATCH_2}")
      if(_vendor STREQUAL name)
        set(_known TRUE)
        if(version VERSION_LESS _floor)
          set(_problem "monoprop needs ${name} >= ${_floor}, found ${version}")
        endif()
      endif()
    endif()
  endforeach()
  if(NOT _known)
    list(JOIN _floors ", " _supported)
    message(
      WARNING
      "monoprop has not been validated with this MPI implementation (${name} ${version}); it supports ${_supported}"
    )
  endif()
  set(${out_var} "${_problem}" PARENT_SCOPE)
endfunction()

#[=[
monoprop_inspect_mpi(<abi-var> <problem-var> INCLUDE_DIRS <dir>... [DEFINITIONS <definition>...])

Sets <abi-var> to the ABI family of the mpi.h found in INCLUDE_DIRS and <problem-var> to the
version-floor error (empty when there is none). Callers decide whether that is fatal.
#]=]
function(monoprop_inspect_mpi abi_var problem_var)
  cmake_parse_arguments(PARSE_ARGV 2 _arg "" "" "INCLUDE_DIRS;DEFINITIONS")
  monoprop_mpi_macros(
    _macros
    INCLUDE_DIRS ${_arg_INCLUDE_DIRS}
    DEFINITIONS ${_arg_DEFINITIONS}
  )
  monoprop_mpi_abi_from_macros(_abi "${_macros}")
  set(_problem "")
  # The standard ABI header names no implementation, and every implementation of it is recent.
  if(NOT _abi STREQUAL "mpiabi")
    monoprop_mpi_vendor_from_macros(_name _version "${_macros}")
    monoprop_mpi_version_problem(_problem "${_name}" "${_version}")
  endif()
  set(${abi_var} "${_abi}" PARENT_SCOPE)
  set(${problem_var} "${_problem}" PARENT_SCOPE)
endfunction()
