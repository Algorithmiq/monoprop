target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "InvertedIndex.h"
        "MPOperator.h"
        "OperatorIndex.h"
        "SparseRowStore.h"
)
