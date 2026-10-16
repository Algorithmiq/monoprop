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

target_sources(
  monoprop-objs
  PRIVATE
    Evaluation.cpp
    Exchange.cpp
)
