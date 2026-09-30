target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CosineRecompute.h"
        "CosineRecomputeCallbacks.h"
        "CutoffContext.h"
        "LayerBuilder.h"
)

# Library-internal: included only by compiled library sources (so the library's floating-point flags apply to every
# instantiation), never by an installed header, and therefore not installed.
target_sources(monoprop-objs PRIVATE LayerReplay.h)

add_subdirectory(layer_build)
