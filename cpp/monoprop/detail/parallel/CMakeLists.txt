target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Options.h"
        "ThreadBudget.h"
        "Workshare.h"
)

monoprop_add_core_sources(ThreadBudget.cpp)
