target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "MonomialPropagatorCommon.h"
        "MonomialPropagator.inl"
)
