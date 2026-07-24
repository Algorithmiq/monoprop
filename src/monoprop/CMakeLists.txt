find_package(Boost 1.85 CONFIG REQUIRED)
message(STATUS "Using Boost: ${Boost_DIR} (version ${Boost_VERSION})")

add_library(
  monoprop
  SHARED
  Evolution.cpp
  MPFunctions.cpp
  detail/pare/PareGraph.cpp
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

file(
  GLOB_RECURSE monoprop_PUBLIC_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/*.h"
  "${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}/*.h"
  "${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}/detail/*.h"
  "${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}/detail/**/*.h"
)

list(
  APPEND monoprop_PUBLIC_HEADERS
  ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/Info.h
  ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}Export.h
)

set(monoprop_PRIVATE_HEADERS)

set_target_properties(
  monoprop
  PROPERTIES
    CXX_VISIBILITY_PRESET
      hidden
    VISIBILITY_INLINES_HIDDEN
      YES
    PUBLIC_HEADER
      "${monoprop_PUBLIC_HEADERS}"
    PRIVATE_HEADER
      "${monoprop_PRIVATE_HEADERS}"
)

target_include_directories(
  monoprop
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
  monoprop
  PUBLIC
    Boost::boost
    Threads::Threads
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:MPI::MPI_CXX>
)

target_compile_definitions(
  monoprop
  PUBLIC
    $<$<TARGET_EXISTS:MPI::MPI_CXX>:monoprop_ENABLE_MPI>
)

# flags to prepend
target_compile_options(
  monoprop
  BEFORE
  PRIVATE
    "${monoprop_CXX_FLAGS}"
    "${ARCH_FLAG}"
)

# flags to append
target_compile_options(monoprop PRIVATE "${EXTRA_CXXFLAGS}")

install(
  TARGETS
    monoprop
  EXPORT monopropTargets
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/monoprop/private
)

install(
  DIRECTORY
    ${PROJECT_SOURCE_DIR}/include/monoprop
    ${PROJECT_SOURCE_DIR}/src/monoprop
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(
  FILES
    ${PROJECT_BINARY_DIR}/include/monoprop/Info.h
    ${PROJECT_BINARY_DIR}/include/monoprop/monopropExport.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/monoprop
)

add_subdirectory(bindings)
