target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "PareGraph.h"
)

monoprop_add_core_sources(PareGraph.cpp)
