if(monoprop_ENABLE_MPI)
  include(${PROJECT_SOURCE_DIR}/cmake/custom/FindPythonModule.cmake)
  find_python_module(mpi4py QUIET)

  if(NOT mpi4py_FOUND)
    message(
      FATAL_ERROR
      "MPI is enabled but mpi4py is not importable from ${Python_EXECUTABLE}. "
      "mpi4py is a build-time input of the bindings: its headers hand the MPI "
      "communicator across the nanobind boundary.\n"
      "Enable MPI with the environment switch, which provisions it automatically:\n"
      "    monoprop_ENABLE_MPI=ON uv sync --all-extras\n"
      "If you drive the build with config-settings, request it explicitly too, as "
      "config-settings cannot trigger a scikit-build-core override:\n"
      "    uv sync --all-extras \\\n"
      "        --config-settings=cmake.define.monoprop_ENABLE_MPI=ON \\\n"
      "        --config-settings=build.requires=mpi4py>=4.1.0"
    )
  endif()

  # we also need the include directories for mpi4py
  if(mpi4py_FOUND)
    execute_process(
      COMMAND
        "${Python_EXECUTABLE}" "-c"
        "import mpi4py as m; print(m.__version__); print(m.get_include());"
      RESULT_VARIABLE _mpi4py_SEARCH_SUCCESS
      OUTPUT_VARIABLE _mpi4py_VALUES
      ERROR_VARIABLE _mpi4py_ERROR_VALUE
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Convert the process output into a list
    string(
      REGEX REPLACE ";"
      "\\\\;"
      _mpi4py_VALUES
      ${_mpi4py_VALUES}
    )
    string(
      REGEX REPLACE "\n"
      ";"
      _mpi4py_VALUES
      ${_mpi4py_VALUES}
    )
    list(GET _mpi4py_VALUES 0 mpi4py_VERSION)
    list(GET _mpi4py_VALUES 1 mpi4py_INCLUDE_DIRS)

    # Make sure all directory separators are '/'
    string(
      REGEX REPLACE "\\\\"
      "/"
      mpi4py_INCLUDE_DIRS
      ${mpi4py_INCLUDE_DIRS}
    )

    # Get the major and minor version numbers
    string(
      REGEX REPLACE "\\."
      ";"
      _mpi4py_VERSION_LIST
      ${mpi4py_VERSION}
    )
    list(GET _mpi4py_VERSION_LIST 0 mpi4py_VERSION_MAJOR)
    list(GET _mpi4py_VERSION_LIST 1 mpi4py_VERSION_MINOR)
    list(GET _mpi4py_VERSION_LIST 2 mpi4py_VERSION_PATCH)
    string(
      REGEX MATCH "[0-9]*"
      mpi4py_VERSION_PATCH
      ${mpi4py_VERSION_PATCH}
    )
    math(
      EXPR
      mpi4py_VERSION_DECIMAL
      "(${mpi4py_VERSION_MAJOR} * 10000) + (${mpi4py_VERSION_MINOR} * 100) + ${mpi4py_VERSION_PATCH}"
    )
  endif()
endif()

find_package(nanobind CONFIG REQUIRED)
message(STATUS "Using nanobind: ${nanobind_DIR} (version ${nanobind_VERSION})")

# generate binding
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/generate-binders.py"
    "--max-num-modes" "${monoprop_MAX_NUM_MODES}" "--batch-size" "8"
    "--output-dir" "${CMAKE_CURRENT_BINARY_DIR}/generated" "--binding-kind"
    "core"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)

file(
  GLOB _bind_cpps
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_BINARY_DIR}/generated/bind_up_to_[0-9]*.cpp"
)
set(_bind_cpps_print "")
foreach(_cpp IN LISTS _bind_cpps)
  file(
    RELATIVE_PATH
    _rel_cpp
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${_cpp}"
  )
  list(APPEND _bind_cpps_print "${_rel_cpp}")
endforeach()
message(STATUS "Generated binding source files: ${_bind_cpps_print}")

set(_BINDINGS_BODY_ "")
file(READ "${CMAKE_CURRENT_BINARY_DIR}/generated/call.txt" _BINDINGS_BODY_)
string(STRIP "${_BINDINGS_BODY_}" _BINDINGS_BODY_)
set(_BINDINGS_BODY_ "${_BINDINGS_BODY_}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp"
  @ONLY
)

if(APPLE)
  set(_rpath "@loader_path/${CMAKE_INSTALL_LIBDIR}")
else()
  set(_rpath "\$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
endif()

#[=[
monoprop_add_python_core(<target> OBJS <objects> MODULE_NAME <name> OUTPUT_NAME <stem> [MPI])

One nanobind extension over a core variant. Every MPI module is named _core_mpi and lives in the
nanobind domain monoprop_mpi, so one of them can share a process with the serial _core.
#]=]
function(monoprop_add_python_core target)
  cmake_parse_arguments(
    PARSE_ARGV 1
    _arg
    "MPI"
    "OBJS;MODULE_NAME;OUTPUT_NAME"
    ""
  )
  if(_arg_MPI)
    set(_bridge comm_mpi.cpp)
    set(
      _domain
      NB_DOMAIN
      monoprop_mpi
    )
  else()
    set(_bridge comm_serial.cpp)
    set(_domain "")
  endif()

  nanobind_add_module(${target}
    NB_SUPPRESS_WARNINGS  # suppress warnings from nanobind and Python headers
    NOMINSIZE  # don’t perform optimizations to minimize binary size
    LTO  # perform link-time optimization
    BACKEND_MODULE nanobind_backend
    ${_domain}
    ${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp
    ${_bind_cpps}  # List of generated binders
    ${CMAKE_CURRENT_SOURCE_DIR}/${_bridge}
  )
  target_compile_definitions(
    ${target}
    PRIVATE
      monoprop_PY_MODULE_NAME=${_arg_MODULE_NAME}
  )
  target_include_directories(
    ${target}
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  if(_arg_MPI)
    target_include_directories(${target} SYSTEM PRIVATE ${mpi4py_INCLUDE_DIRS})
  endif()
  target_link_libraries(
    ${target}
    PRIVATE
      ${_arg_OBJS}
      monoprop-sanitizers
  )
  if(monoprop_WHEEL_BUILD)
    set(_use_link_path OFF)
  else()
    set(_use_link_path ON)
  endif()
  set(_module_rpath "${_rpath}")
  if(APPLE AND monoprop_WHEEL_BUILD AND _arg_MPI)
    list(APPEND _module_rpath ${monoprop_MACOS_MPI_RPATHS})
  endif()
  set_target_properties(
    ${target}
    PROPERTIES
      OUTPUT_NAME
        "${_arg_OUTPUT_NAME}"
      MACOSX_RPATH
        ON
      SKIP_BUILD_RPATH
        OFF
      BUILD_WITH_INSTALL_RPATH
        OFF
      INSTALL_RPATH
        "${_module_rpath}"
      INSTALL_RPATH_USE_LINK_PATH
        ${_use_link_path}
  )
  install(TARGETS ${target} LIBRARY DESTINATION ${PROJECT_NAME})
endfunction()

monoprop_add_python_core(_core OBJS monoprop-objs MODULE_NAME _core OUTPUT_NAME _core)

set(monoprop_MPI_MODULE_FILES_PY "")
foreach(_abi IN LISTS monoprop_MPI_ABI_NAMES)
  monoprop_add_python_core(
    _core_mpi_${_abi}
    OBJS monoprop-objs-${_abi}
    MODULE_NAME _core_mpi
    OUTPUT_NAME _core_mpi.${_abi}
    MPI
  )
  string(
    APPEND monoprop_MPI_MODULE_FILES_PY
    "\"${_abi}\": \"_core_mpi.${_abi}\", "
  )
endforeach()

if(monoprop_SANITIZER STREQUAL "asan-ubsan")
  # GCC bounds-strict misdiagnoses nanobind's valid access to a tuple's trailing allocation.
  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set_source_files_properties(
      "${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp"
      ${_bind_cpps}
      PROPERTIES
        COMPILE_OPTIONS
          "-fno-sanitize=bounds-strict"
    )
  endif()
endif()

# generate dispatch function module
install(
  CODE
    "execute_process(
      COMMAND \"${Python_EXECUTABLE}\" \"${PROJECT_SOURCE_DIR}/tools/generate-dispatch.py\" \"--max-num-modes\" \"${monoprop_MAX_NUM_MODES}\" \"--class-name\" \"MonomialPropagator\" \"--module-name\" \"monoprop\"
WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
COMMAND_ERROR_IS_FATAL ANY
)"
)

# monoprop/_mpi_abis.py records the shipped MPI ABI families for monoprop._backend.
set(monoprop_MPI_ABIS_PY "")
foreach(_abi IN LISTS monoprop_MPI_ABI_NAMES)
  string(APPEND monoprop_MPI_ABIS_PY "\"${_abi}\", ")
endforeach()
set(monoprop_MPI_MIN_VERSIONS_PY "")
monoprop_mpi_min_versions(_floors)
foreach(_entry IN LISTS _floors)
  string(
    REGEX MATCH "^(.+)=(.+)$"
    _match
    "${_entry}"
  )
  string(
    REPLACE "."
    ", "
    _floor
    "${CMAKE_MATCH_2}"
  )
  string(
    APPEND monoprop_MPI_MIN_VERSIONS_PY
    "\"${CMAKE_MATCH_1}\": (${_floor}), "
  )
endforeach()
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/_mpi_abis.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/_mpi_abis.py
  @ONLY
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/_mpi_abis.py
  DESTINATION ${PROJECT_NAME}
)

# generation of Python typing stubs
# NOTE must come after installing the _core target
#
# Skip sanitizer builds: INSTALL_TIME stub generation imports _core and requires a preloaded
# sanitizer runtime throughout installation.
if(monoprop_SANITIZER STREQUAL "none")
  nanobind_add_stub(_core
    INSTALL_TIME  # Stub generation postponed to the installation phase.
    MODULE _core  # Specifies the name of the module that should be imported.
    OUTPUT "${PROJECT_NAME}/_core.pyi"  # Specifies the name of the stub file that should be written.
    PYTHON_PATH "${PROJECT_NAME}"  # List of search paths - relative to CMAKE_INSTALL_PREFIX - that should be considered when importing the module.
    VERBOSE  # Show status messages generated by stubgen.
    MARKER_FILE "${PROJECT_NAME}/py.typed"  # Automatically generate an empty marker file.
  )
endif()
