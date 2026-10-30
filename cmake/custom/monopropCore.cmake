# Core object libraries
# =====================
#
# monoprop_ENABLE_MPI changes class layouts (cpp/monoprop/detail/mpi/Comm.h), so every MPI ABI
# variant compiles the core sources again into an OBJECT library of its own. Subdirectories
# register their translation units once with monoprop_add_core_sources();
# monoprop_add_core_objects() then stamps out one variant.

include_guard(GLOBAL)

#[=[
monoprop_add_core_sources(<source>...)

Registers core translation units, given relative to the calling directory.
#]=]
function(monoprop_add_core_sources)
  foreach(_source IN LISTS ARGN)
    cmake_path(
      ABSOLUTE_PATH _source
      BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE _absolute
    )
    set_property(
      GLOBAL
      APPEND
      PROPERTY
        monoprop_CORE_SOURCES
          "${_absolute}"
    )
  endforeach()
endfunction()

#[=[
monoprop_add_core_objects(<name> [MPI <target>])

Compiles every registered core source into the OBJECT library <name>. With MPI, <target> provides
the MPI headers and library, and monoprop_ENABLE_MPI is defined for <name> and its consumers.
#]=]
function(monoprop_add_core_objects name)
  cmake_parse_arguments(PARSE_ARGV 1 _arg "" "MPI" "")
  get_property(_sources GLOBAL PROPERTY monoprop_CORE_SOURCES)
  add_library(${name} OBJECT ${_sources})
  target_link_libraries(
    ${name}
    PUBLIC
      monoprop-core-usage
    PRIVATE
      PkgConfig::HWLOC
  )
  if(_arg_MPI)
    target_link_libraries(${name} PUBLIC ${_arg_MPI})
    target_compile_definitions(${name} PUBLIC monoprop_ENABLE_MPI)
  endif()
  set_target_properties(
    ${name}
    PROPERTIES
      CXX_VISIBILITY_PRESET
        hidden
      VISIBILITY_INLINES_HIDDEN
        YES
      POSITION_INDEPENDENT_CODE
        ON
  )
endfunction()
