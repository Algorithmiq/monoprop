target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "MPGraphEncodingStorage.h"
        "MPGraphEncodingTypes.h"
)

target_sources(monoprop-objs PRIVATE MPGraphEncoding.cpp)
