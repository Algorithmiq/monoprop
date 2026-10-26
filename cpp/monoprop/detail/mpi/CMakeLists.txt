target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CheckedCount.h"
        "Comm.h"
        "CpuRelax.h"
        "DuplicatedComm.h"
        "Exchange.h"
        "HybridComm.h"
        "MPICompat.h"
        "MPIUtils.h"
        "Pairwise.h"
        "PartitionBarrier.h"
        "Routing.h"
        "ShmComm.h"
)

monoprop_add_core_sources(MPICompat.cpp)
