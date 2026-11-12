target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CpuTopology.h"
        "PartitionGroup.h"
)

monoprop_add_core_sources(CpuTopology.cpp)
