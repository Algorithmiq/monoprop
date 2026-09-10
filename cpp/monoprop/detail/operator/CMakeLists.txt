target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "ChunkedArray.h"
        "InvertedIndex.h"
        "MPOperator.h"
        "OperatorIndex.h"
        "RowAccess.h"
        "RowKey.h"
        "TermTable.h"
)
