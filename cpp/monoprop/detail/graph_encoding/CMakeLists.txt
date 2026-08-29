target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "MPGraphEncodingStorage.h"
        "MPGraphEncodingTypes.h"
)

monoprop_engine_sources(MPGraphEncoding.cpp)
