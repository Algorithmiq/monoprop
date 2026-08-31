target_sources(monoprop-objs PRIVATE Monomial.cpp)

target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Monomial.h"
)
