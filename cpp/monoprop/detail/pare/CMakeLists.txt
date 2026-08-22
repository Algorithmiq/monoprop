target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "PareGraph.h"
)

monoprop_engine_sources(PareGraph.cpp)
