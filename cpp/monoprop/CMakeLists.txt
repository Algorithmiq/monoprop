find_package(Boost 1.85 CONFIG REQUIRED)
message(STATUS "Using Boost: ${Boost_DIR} (version ${Boost_VERSION})")

find_package(PkgConfig REQUIRED QUIET)
pkg_check_modules(HWLOC REQUIRED QUIET IMPORTED_TARGET GLOBAL "hwloc>=2.9")
message(
  STATUS
  "Using hwloc: ${HWLOC_LINK_LIBRARIES} (version ${HWLOC_VERSION})"
)

include(${PROJECT_SOURCE_DIR}/cmake/custom/monopropCore.cmake)

# Usage requirements every core variant compiles with and hands on to its consumers.
add_library(monoprop-core-usage INTERFACE)

# flags to prepend
target_compile_options(
  monoprop-core-usage
  BEFORE
  INTERFACE
    "${monoprop_CXX_FLAGS}"
    "${ARCH_FLAG}"
)

# flags to append
target_compile_options(monoprop-core-usage INTERFACE "${EXTRA_CXXFLAGS}")

target_include_directories(
  monoprop-core-usage
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
)

target_link_libraries(
  monoprop-core-usage
  INTERFACE
    monoprop-sanitizers
    Boost::boost
    Threads::Threads
)

monoprop_add_core_sources(
  Evolution.cpp
  MPFunctions.cpp
  MPGraph.cpp
  Validation.cpp
)

include(GenerateExportHeader)

generate_export_header(
  monoprop
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
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/DefaultInitAllocator.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/MPGraphEncoding.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/TypeAliases.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/Utilities.h"
        "${PROJECT_SOURCE_DIR}/cpp/monoprop/Validation.h"
)

target_compile_definitions(
  monoprop
  INTERFACE
    $<$<BOOL:${monoprop_ENABLE_MPI}>:monoprop_ENABLE_MPI>
)

target_compile_features(monoprop INTERFACE cxx_std_23)

target_link_libraries(
  monoprop
  PUBLIC
    $<$<BOOL:${monoprop_ENABLE_MPI}>:MPI::MPI_CXX>
  INTERFACE
    Boost::boost
    Threads::Threads
  PRIVATE
    # Keep the non-exported sanitizer target out of monopropTargets.
    $<BUILD_INTERFACE:monoprop-sanitizers>
    PkgConfig::HWLOC
)

add_subdirectory(algebra)
add_subdirectory(core)
add_subdirectory(detail)

# Every subdirectory has registered its sources by now.
if(monoprop_ENABLE_MPI)
  monoprop_add_core_objects(monoprop-objs MPI MPI::MPI_CXX)
else()
  monoprop_add_core_objects(monoprop-objs)
endif()

target_sources(monoprop PRIVATE $<TARGET_OBJECTS:monoprop-objs>)

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
