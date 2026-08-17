target_sources(monoprop-objs PRIVATE MonomialPropagator.cpp)

target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "MonomialPropagatorCommon.h"
)
