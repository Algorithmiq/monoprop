target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CheckedCount.h"
        "Comm.h"
        "CpuRelax.h"
        "Exchange.h"
        "HybridComm.h"
        "MPICompat.h"
        "MPIUtils.h"
        "Pairwise.h"
        "PartitionBarrier.h"
        "Routing.h"
        "ShmComm.h"
)

target_sources(monoprop-objs PRIVATE MPICompat.cpp)
