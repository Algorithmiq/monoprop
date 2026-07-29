target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "PareGraph.h"
)

target_sources(monoprop PRIVATE PareGraph.cpp)
