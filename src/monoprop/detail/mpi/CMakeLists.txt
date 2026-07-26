target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "MPICompat.h"
        "MPIUtils.h"
)
