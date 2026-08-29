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

if(APPLE)
  set(_rpath_origin "@loader_path")
else()
  set(_rpath_origin "$ORIGIN")
endif()

# Build one binding module.
#
# The module's sources are the configured bindings.cpp plus the generated per-mode-width batches, and
# every tier compiles all of them: `objs` is the engine object library this module links, and an engine
# object library carries its arch flags as PUBLIC compile options, so these translation units are
# compiled at the tier rather than merely linked against it. That is what makes the tier worth
# anything here -- MonomialPropagator<NumModes> is header-resident, so the scan, the fold and every
# template they instantiate land in these TUs and nowhere else.
#
# Every module is named `_core` on disk whatever its target is called, because CPython derives the
# init function it looks for from the last component of the module name it is asked to load, and the
# loader asks for `monoprop._core` for all of them. `destination` and `output_dir` are what keep the
# identically-named files apart.
function(_monoprop_add_binding_module)
  set(
    _one_value_args
    TARGET
    OBJS
    DESTINATION
    OUTPUT_DIR
    INSTALL_RPATH
  )
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "")

  nanobind_add_module(${_arg_TARGET}
    NB_SUPPRESS_WARNINGS  # suppress warnings from nanobind and Python headers
    NOMINSIZE  # don’t perform optimizations to minimize binary size
    LTO  # perform link-time optimization
    BACKEND_MODULE nanobind_backend  # nanobind 3 split mode; rules out STABLE_ABI and NB_STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp
    ${_bind_cpps}  # List of generated binders
  )

  target_include_directories(
    ${_arg_TARGET}
    SYSTEM
    PRIVATE
      $<IF:$<BOOL:${mpi4py_FOUND}>,${mpi4py_INCLUDE_DIRS},>
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(
    ${_arg_TARGET}
    PRIVATE
      ${_arg_OBJS}
      monoprop-sanitizers
  )

  set_target_properties(
    ${_arg_TARGET}
    PROPERTIES
      OUTPUT_NAME
        "_core"
      LIBRARY_OUTPUT_DIRECTORY
        "${_arg_OUTPUT_DIR}"
      MACOSX_RPATH
        ON
      SKIP_BUILD_RPATH
        OFF
      BUILD_WITH_INSTALL_RPATH
        OFF
      INSTALL_RPATH
        "${_arg_INSTALL_RPATH}"
      INSTALL_RPATH_USE_LINK_PATH
        ON
  )

  install(TARGETS ${_arg_TARGET} LIBRARY DESTINATION ${_arg_DESTINATION})
endfunction()

if(monoprop_ENABLE_FAT_BINARY)
  foreach(_tier IN LISTS monoprop_FAT_TIERS)
    monoprop_tier_targets(TIER "${_tier}" OBJS_VAR _tier_objs)
    _monoprop_tier_slug("${_tier}" _slug)
    _monoprop_add_binding_module(
      TARGET _core-${_slug}
      OBJS ${_tier_objs}
      DESTINATION ${PROJECT_NAME}/_variants/${_tier}
      OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/_variants/${_tier}
      # two levels deeper than a single-variant build: monoprop/_variants/<tier>/ -> monoprop/lib
      INSTALL_RPATH "${_rpath_origin}/../../${CMAKE_INSTALL_LIBDIR}"
    )
  endforeach()

  # The probe the import-time dispatch runs before it can load anything else, so it is the one
  # extension that must execute on every machine the wheel installs on: baseline ISA, appended last so
  # a CXXFLAGS from the environment cannot widen it, and linked against no engine object library so it
  # cannot inherit a tier's flags either.
  #
  # Split mode like every other module here, and not by preference: wheel.py-api = "cp311" sets
  # SKBUILD_SABI_VERSION, and nanobind 3 refuses a linked (NB_STATIC) module under any py-api below
  # cp312. It costs the probe nothing -- the shared backend is not an engine object library, so the
  # probe still inherits no tier's arch flags, and the fat binary's baseline floor in CMAKE_CXX_FLAGS
  # is what keeps the backend itself at x86-64.
  nanobind_add_module(_isa
    NB_SUPPRESS_WARNINGS
    NOMINSIZE
    BACKEND_MODULE nanobind_backend
    ${CMAKE_CURRENT_SOURCE_DIR}/isa.cpp
  )
  target_include_directories(_isa PRIVATE ${PROJECT_BINARY_DIR}/include)
  target_compile_options(
    _isa
    PRIVATE
      -march=x86-64
      -mtune=${monoprop_FAT_MTUNE}
  )
  install(TARGETS _isa LIBRARY DESTINATION ${PROJECT_NAME})
else()
  _monoprop_add_binding_module(
    TARGET _core
    OBJS monoprop-objs
    DESTINATION ${PROJECT_NAME}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_RPATH "${_rpath_origin}/${CMAKE_INSTALL_LIBDIR}"
  )
endif()

# Which module the typing stub is generated from. Every tier exposes the same API -- they differ only
# in the instructions behind it -- so the baseline is as good as any, and it is the one guaranteed to
# import on the machine doing the build.
if(monoprop_ENABLE_FAT_BINARY)
  list(GET monoprop_FAT_TIERS 0 _stub_tier)
  set(_stub_python_path "${PROJECT_NAME}/_variants/${_stub_tier}")
else()
  set(_stub_python_path "${PROJECT_NAME}")
endif()

# generation of Python typing stubs
# NOTE must come after installing the modules above
#
# Skip sanitizer builds: INSTALL_TIME stub generation imports _core and requires a preloaded
# sanitizer runtime throughout installation.
if(monoprop_SANITIZER STREQUAL "none")
  nanobind_add_stub(_core_stub
    INSTALL_TIME  # Stub generation postponed to the installation phase.
    MODULE _core  # Specifies the name of the module that should be imported.
    OUTPUT "${PROJECT_NAME}/_core.pyi"  # Specifies the name of the stub file that should be written.
    PYTHON_PATH "${_stub_python_path}"  # List of search paths - relative to CMAKE_INSTALL_PREFIX - that should be considered when importing the module.
    VERBOSE  # Show status messages generated by stubgen.
    MARKER_FILE "${PROJECT_NAME}/py.typed"  # Automatically generate an empty marker file.
  )
endif()
