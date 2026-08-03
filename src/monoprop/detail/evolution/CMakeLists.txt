target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
        "CosineRecompute.h"
        "CosineRecomputeCallbacks.h"
        "EvolutionHelpers.h"
        "LayerBuilder.h"
)

add_subdirectory(layer_build)
