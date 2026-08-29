target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CpuTopology.h"
        "PartitionGroup.h"
        "StagedCollect.h"
)

target_sources(monoprop-objs PRIVATE CpuTopology.cpp)
