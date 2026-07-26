if(monoprop_ENABLE_MPI)
  include(${PROJECT_SOURCE_DIR}/cmake/custom/FindPythonModule.cmake)
  find_python_module(mpi4py REQUIRED)

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

nanobind_add_module(_core
  STABLE_ABI  # Perform a stable ABI build
  NB_STATIC  # Compile the core nanobind library as a static library
  NOMINSIZE  # Don’t perform optimizations to minimize binary size
  ${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp
  ${_bind_cpps}  # List of generated binders
)

target_include_directories(
  _core
  SYSTEM
  PRIVATE
    $<IF:$<BOOL:${mpi4py_FOUND}>,${mpi4py_INCLUDE_DIRS},>
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_core PRIVATE monoprop)

# The hot MonomialPropagator<N> kernels are header templates instantiated in this
# target's TUs, so ARCH_FLAG must be applied here too: on the library alone it never
# reached them and they compiled to scalar x86-64.
target_compile_options(_core BEFORE PRIVATE "${ARCH_FLAG}")

file(
  RELATIVE_PATH
  _rel
  ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
  ${CMAKE_INSTALL_PREFIX}
)
if(APPLE)
  set(_rpath "@loader_path/${_rel}")
else()
  set(_rpath "\$ORIGIN/${_rel}")
endif()
file(
  TO_NATIVE_PATH
  "${_rpath}/${CMAKE_INSTALL_LIBDIR}"
  monoprop_RPATH
)

set_target_properties(
  _core
  PROPERTIES
    MACOSX_RPATH
      ON
    SKIP_BUILD_RPATH
      OFF
    BUILD_WITH_INSTALL_RPATH
      OFF
    INSTALL_RPATH
      "${monoprop_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH
      ON
)

# generate dispatch function module
install(
  CODE
    "execute_process(
      COMMAND \"${Python_EXECUTABLE}\" \"${PROJECT_SOURCE_DIR}/tools/generate-dispatch.py\" \"--max-num-modes\" \"${monoprop_MAX_NUM_MODES}\" \"--class-name\" \"MonomialPropagator\" \"--module-name\" \"monoprop\"
WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
COMMAND_ERROR_IS_FATAL ANY
)"
)

install(TARGETS _core LIBRARY DESTINATION ${PROJECT_NAME})

# generation of Python typing stubs
# NOTE must come after installing the _core target
nanobind_add_stub(_core
  INSTALL_TIME  # Stub generation postponed to the installation phase.
  MODULE _core  # Specifies the name of the module that should be imported.
  OUTPUT "${PROJECT_NAME}/_core.pyi"  # Specifies the name of the stub file that should be written.
  PYTHON_PATH "${PROJECT_NAME}"  # List of search paths - relative to CMAKE_INSTALL_PREFIX - that should be considered when importing the module.
  VERBOSE  # Show status messages generated by stubgen.
  MARKER_FILE "${PROJECT_NAME}/py.typed"  # Automatically generate an empty marker file.
)
