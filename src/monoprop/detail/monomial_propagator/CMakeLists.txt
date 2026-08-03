target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "MonomialPropagatorHelpers.h"
        "MonomialPropagatorCommon.h"
        "MonomialPropagatorImpl.h"
)
