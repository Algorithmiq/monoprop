find_package(Boost 1.85 CONFIG REQUIRED)
message(STATUS "Using Boost: ${Boost_DIR} (version ${Boost_VERSION})")

find_package(PkgConfig REQUIRED QUIET)
pkg_check_modules(HWLOC REQUIRED QUIET IMPORTED_TARGET GLOBAL "hwloc>=2.9")
message(
  STATUS
  "Using hwloc: ${HWLOC_LINK_LIBRARIES} (version ${HWLOC_VERSION})"
)

monoprop_engine_sources(
  Evolution.cpp
  MPFunctions.cpp
  MPGraph.cpp
  Validation.cpp
)

# Everything an engine object library needs beyond its sources. Factored out because the fat binary
# compiles the engine once per ISA tier and the tiers must differ in exactly two things -- the arch
# flags and which Variants.h they see -- with nothing else drifting apart between them.
#
# variant_include_dir goes in BEFORE the shared binary include dir, which is what makes a tier report
# its own identity instead of the build's default one.
#
# `scope` is PUBLIC for a library whose flags the binding module should inherit, and PRIVATE for one
# whose flags it must not. In whole-library mode the module *is* the tier, so it wants the tier's
# -march; in narrow-seam mode it links every tier at once, so inheriting any of them would mean
# compiling the shared code at whichever tier the linker happened to order last.
function(
  _monoprop_configure_engine_objs
  target
  arch_flags
  variant_include_dir
  scope
)
  target_compile_definitions(
    ${target}
    ${scope}
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:monoprop_ENABLE_MPI>
    $<$<BOOL:${monoprop_WIDE_TERM_INDEX}>:monoprop_WIDE_TERM_INDEX>
    monoprop_SPARSE_ROW_MIN_MODES=${monoprop_SPARSE_ROW_MIN_MODES}
    monoprop_NARROW_KERNEL_MAX_WORDS=${monoprop_NARROW_KERNEL_MAX_WORDS}
  )

  # flags to prepend
  target_compile_options(
    ${target}
    BEFORE
    ${scope}
    "${monoprop_CXX_FLAGS}"
    "${arch_flags}"
  )

  # flags to append
  target_compile_options(
    ${target}
    ${scope}
    "${EXTRA_CXXFLAGS}"
  )

  target_include_directories(
    ${target}
    BEFORE
    ${scope}
    $<BUILD_INTERFACE:${variant_include_dir}>
  )

  target_include_directories(
    ${target}
    ${scope}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
  )

  target_link_libraries(
    ${target}
    ${scope}
    Boost::boost
    Threads::Threads
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:MPI::MPI_CXX>
    PRIVATE
      PkgConfig::HWLOC
  )

  set_target_properties(
    ${target}
    PROPERTIES
      CXX_VISIBILITY_PRESET
        hidden
      VISIBILITY_INLINES_HIDDEN
        YES
  )
endfunction()

if(monoprop_FAT_NARROW_SEAM)
  # The shared build, at the baseline. ARCH_FLAG is already the baseline tier's flags (FatBinary.cmake
  # overwrites it), and PUBLIC so bindings.cpp compiles at the same floor.
  _monoprop_configure_engine_objs(
    monoprop-objs
    "${ARCH_FLAG}"
    "${PROJECT_BINARY_DIR}/include"
    PUBLIC
  )
  # Reaches TierDispatch.cpp, which needs the generated tier table, and bindings.cpp, which reports the
  # tier that was selected rather than the one it was itself compiled for.
  target_compile_definitions(monoprop-objs PUBLIC monoprop_FAT_NARROW_SEAM)

  foreach(_tier IN LISTS monoprop_FAT_TIERS)
    _monoprop_tier_spec(TIER "${_tier}" MARCH_VAR _tier_flags)
    _monoprop_tier_slug("${_tier}" _slug)
    _monoprop_configure_engine_objs(
      monoprop-tier-${_slug}
      "${_tier_flags}"
      "${PROJECT_BINARY_DIR}/variants/${_tier}/include"
      PRIVATE
    )
    # The namespace the tiered TU's entry points land in. TierDispatch.cpp reaches them by the same
    # slug, derived from the same tier id, so the two cannot disagree without the tier list changing.
    target_compile_definitions(
      monoprop-tier-${_slug}
      PRIVATE
        monoprop_TIER_SLUG=${_slug}
    )
  endforeach()
elseif(monoprop_ENABLE_FAT_BINARY)
  foreach(_tier IN LISTS monoprop_FAT_TIERS)
    monoprop_tier_targets(
      TIER "${_tier}"
      OBJS_VAR _tier_objs
      VARIANT_INCLUDE_DIR_VAR _tier_variant_include_dir
    )
    _monoprop_tier_spec(TIER "${_tier}" MARCH_VAR _tier_flags)
    _monoprop_configure_engine_objs(
      "${_tier_objs}"
      "${_tier_flags}"
      "${_tier_variant_include_dir}"
      PUBLIC
    )
  endforeach()
else()
  _monoprop_configure_engine_objs(
    monoprop-objs
    "${ARCH_FLAG}"
    "${PROJECT_BINARY_DIR}/include"
    PUBLIC
  )
endif()

include(GenerateExportHeader)

generate_export_header(
  monoprop-objs
  BASE_NAME "${PROJECT_NAME}"
  EXPORT_FILE_NAME
    "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}Export.h"
  EXPORT_MACRO_NAME "${PROJECT_NAME}_EXPORT"
  DEPRECATED_MACRO_NAME "${PROJECT_NAME}_DEPRECATED"
  NO_EXPORT_MACRO_NAME "${PROJECT_NAME}_NO_EXPORT"
  STATIC_DEFINE "${PROJECT_NAME}_STATIC_DEFINE"
  NO_DEPRECATED_MACRO_NAME "${PROJECT_NAME}_NO_DEPRECATED"
  DEFINE_NO_DEPRECATED
)

target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      BASE_DIRS
        ${PROJECT_SOURCE_DIR}/cpp
      TYPE HEADERS
      FILES
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/Bitset.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/MPGraphEncoding.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/TypeAliases.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/Utilities.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/Validation.h"
)

target_compile_definitions(
  monoprop
  INTERFACE
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:monoprop_ENABLE_MPI>
    $<$<BOOL:${monoprop_WIDE_TERM_INDEX}>:monoprop_WIDE_TERM_INDEX>
    monoprop_SPARSE_ROW_MIN_MODES=${monoprop_SPARSE_ROW_MIN_MODES}
    monoprop_NARROW_KERNEL_MAX_WORDS=${monoprop_NARROW_KERNEL_MAX_WORDS}
)

target_link_libraries(
  monoprop
  INTERFACE
    Boost::boost
    Threads::Threads
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:MPI::MPI_CXX>
  PRIVATE
    PkgConfig::HWLOC
)

# The per-tier object libraries, in narrow-seam mode only (the list is empty otherwise). They are
# linked rather than spliced in with $<TARGET_OBJECTS:...> at the top level because an object library's
# own usage requirements -- Boost's include dirs among them -- come with the link and not with the
# splice. Every consumer of the engine needs this: the shared library here, the binding module, and the
# C++ test binary, since build_layer_dispatch has no definition without them.
target_link_libraries(monoprop PRIVATE ${monoprop_TIERED_OBJ_TARGETS})

add_subdirectory(algebra)
add_subdirectory(core)
add_subdirectory(detail)

install(
  TARGETS
    monoprop
  EXPORT monopropTargets
  LIBRARY
    DESTINATION ${PROJECT_NAME}/${CMAKE_INSTALL_LIBDIR}
    COMPONENT lib
  FILE_SET api_headers
    DESTINATION ${PROJECT_NAME}/${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT dev
  FILE_SET headers
    DESTINATION ${PROJECT_NAME}/${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT dev
)
