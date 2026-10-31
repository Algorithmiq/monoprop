target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Construction.h"
        "Evaluation.h"
        "Exchange.h"
        "RootObserver.h"
        "State.h"
        "Team.h"
)

monoprop_add_core_sources(
  Evaluation.cpp
  Exchange.cpp
)
