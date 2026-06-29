#pragma once

// CosineRecomputeCallbacks.h — lightweight callback type aliases for cos recompute.
//
// Split out from CosineRecompute.h (which pulls in the heavy DistributedLayerBuilder.h) so the public
// evolution headers can name LayerCosScale / LayerCosAccumulate in their declarations WITHOUT dragging
// the build-pipeline templates (and the Evolution.h <-> EvolutionHelpers.h include cycle) into every
// translation unit. The full prepared-fold machinery lives in CosineRecompute.h and is included only
// where the callbacks are constructed/used (the .cpp files and make_functional).

#include <cstddef>
#include <functional>

namespace monoprop::detail {

// Per-layer forward scale and reverse accumulate. `layer` selects the layer's cos source.
using LayerCosScale = std::function<void(size_t layer, double *coeff, double cos_val)>;
using LayerCosAccumulate =
    std::function<double(size_t layer, double *state, double *ham, double cos_val, double sec_val)>;

} // namespace monoprop::detail
