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

target_sources(monoprop-objs PRIVATE ThreadBudget.cpp)
