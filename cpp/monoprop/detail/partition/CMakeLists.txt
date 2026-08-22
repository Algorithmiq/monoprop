target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CpuTopology.h"
        "PartitionGroup.h"
)

monoprop_engine_sources(CpuTopology.cpp)
