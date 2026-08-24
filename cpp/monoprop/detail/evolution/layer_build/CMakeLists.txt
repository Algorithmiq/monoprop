target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "Common.h"
        "Engine.h"
        "FusedApply.h"
        "PartnerMerge.h"
        "QueryCodec.h"
        "Resolve.h"
        "Scan.h"
        "SparseQuery.h"
)
