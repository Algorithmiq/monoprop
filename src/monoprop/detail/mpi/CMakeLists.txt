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
        "RecvLayout.h"
        "ShmComm.h"
)
