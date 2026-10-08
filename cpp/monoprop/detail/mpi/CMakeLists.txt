target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CheckedCount.h"
        "Comm.h"
        "Exchange.h"
        "MPICompat.h"
        "MPIUtils.h"
        "OperationFailure.h"
        "Pairwise.h"
        "Routing.h"
)

target_sources(monoprop-objs PRIVATE MPICompat.cpp)
