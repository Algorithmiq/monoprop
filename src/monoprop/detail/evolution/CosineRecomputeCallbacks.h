// Copyright 2026 Algorithmiq
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// CosineRecomputeCallbacks.h — lightweight callback type aliases for cos recompute.
//
// Split out from CosineRecompute.h (which pulls in the heavy LayerBuilder.h) so the public
// evolution headers can name LayerCosScale / LayerCosAccumulate in their declarations WITHOUT dragging
// the build-pipeline templates (and the Evolution.h <-> EvolutionHelpers.h include cycle) into every
// translation unit. The full prepared-fold machinery lives in CosineRecompute.h and is included only
// where the callbacks are constructed/used (the .cpp files and make_functional).

#include <cstddef>
#include <functional>

namespace monoprop::detail {

// Per-layer cosine callbacks used by the forward evolution and reverse-gradient walks. `layer` selects
// which layer's cosine set to replay (a stored pruned cos, or one recomputed from the generator fold).
//   LayerCosScale      — forward: scale the operator coefficients `coeff` in place by the layer's
//                        per-term cosine factors (`cos_val` = the gate's cos θ).
//   LayerCosAccumulate — reverse: apply the same cosine to `state` and `ham` in place
//                        (`cos_val` = cos θ, `sec_val` = sec θ = 1/cos θ) and return this layer's
//                        contribution to the gradient.
using LayerCosScale = std::function<void(size_t layer, double *coeff, double cos_val)>;
using LayerCosAccumulate =
    std::function<double(size_t layer, double *state, double *ham, double cos_val, double sec_val)>;

} // namespace monoprop::detail
