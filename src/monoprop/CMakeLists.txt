find_package(Boost 1.85 CONFIG REQUIRED)
message(STATUS "Using Boost: ${Boost_DIR} (version ${Boost_VERSION})")

add_library(
  monoprop-objs
  OBJECT
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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_link_libraries(
  monoprop-objs
  PUBLIC
    Boost::boost
    Threads::Threads
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:MPI::MPI_CXX>
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

add_library(monoprop SHARED $<TARGET_OBJECTS:monoprop-objs>)

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
)

target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      BASE_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
      FILES
        "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/Evolution.h"
        "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/MPFunctions.h"
        "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/MPGraph.h"
        "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/MonomialPropagator.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/Bitset.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/MPGraphEncoding.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/TypeAliases.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utilities.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/Validation.h"
        "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}Export.h"
        "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/Info.h"
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
  FILE_SET headers
    DESTINATION ${PROJECT_NAME}/${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT dev
)

add_subdirectory(bindings)
