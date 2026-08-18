find_package(Boost 1.85 CONFIG REQUIRED)
message(STATUS "Using Boost: ${Boost_DIR} (version ${Boost_VERSION})")

find_package(PkgConfig REQUIRED QUIET)
pkg_check_modules(HWLOC REQUIRED QUIET IMPORTED_TARGET GLOBAL "hwloc>=2.9")
message(
  STATUS
  "Using hwloc: ${HWLOC_LINK_LIBRARIES} (version ${HWLOC_VERSION})"
)

target_sources(
  monoprop-objs
  PRIVATE
    Evolution.cpp
    MPFunctions.cpp
    MPGraph.cpp
    Validation.cpp
)

target_compile_definitions(
  monoprop-objs
  PUBLIC
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:monoprop_ENABLE_MPI>
    $<$<BOOL:${monoprop_WIDE_TERM_INDEX}>:monoprop_WIDE_TERM_INDEX>
)

# flags to prepend
target_compile_options(
  monoprop-objs
  BEFORE
  PUBLIC
    "${monoprop_CXX_FLAGS}"
    "${ARCH_FLAG}"
)

# flags to append
target_compile_options(monoprop-objs PUBLIC "${EXTRA_CXXFLAGS}")

target_include_directories(
  monoprop-objs
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
)

target_link_libraries(
  monoprop-objs
  PUBLIC
    Boost::boost
    Threads::Threads
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:MPI::MPI_CXX>
  PRIVATE
    PkgConfig::HWLOC
)

set_target_properties(
  monoprop-objs
  PROPERTIES
    CXX_VISIBILITY_PRESET
      hidden
    VISIBILITY_INLINES_HIDDEN
      YES
)

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
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/Indirect.h"
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
