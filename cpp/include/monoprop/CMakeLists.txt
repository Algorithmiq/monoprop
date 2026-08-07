configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Info.h.in
  ${PROJECT_BINARY_DIR}/include/monoprop/Info.h
)

target_sources(
  monoprop
  PUBLIC
    FILE_SET api_headers
      TYPE HEADERS
      BASE_DIRS
        ${PROJECT_SOURCE_DIR}/cpp/include
        ${PROJECT_BINARY_DIR}/include
      FILES
        "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}Export.h"
        "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/Info.h"
        "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/Variants.h"
        "${PROJECT_SOURCE_DIR}/cpp/include/${PROJECT_NAME}/Evolution.h"
        "${PROJECT_SOURCE_DIR}/cpp/include/${PROJECT_NAME}/MPFunctions.h"
        "${PROJECT_SOURCE_DIR}/cpp/include/${PROJECT_NAME}/MPGraph.h"
        "${PROJECT_SOURCE_DIR}/cpp/include/${PROJECT_NAME}/MonomialPropagator.h"
)
