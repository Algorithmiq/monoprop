# The tiered translation unit: one copy per ISA tier under narrow-seam mode, one copy otherwise. It is
# the only source in the project that gets more than one copy, and adding a second is a real cost --
# see monoprop_tiered_engine_sources in cmake/compiler_flags/FatBinary.cmake.
monoprop_tiered_engine_sources(TieredLayerBuild.cpp)

# Shared, and it must stay shared: it runs the ISA predicates, so it is the one file that executes on
# machines no tier above the baseline can run on.
monoprop_engine_sources(TierDispatch.cpp)

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
        "TieredLayerBuild.h"
)

add_subdirectory(layer_build)
