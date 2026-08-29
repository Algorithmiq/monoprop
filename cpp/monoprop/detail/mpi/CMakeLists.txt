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
        "PartitionBarrier.h"
        "ShmComm.h"
)

monoprop_engine_sources(MPICompat.cpp)
