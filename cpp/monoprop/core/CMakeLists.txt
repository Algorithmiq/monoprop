monoprop_engine_sources(Monomial.cpp)

target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Monomial.h"
)
