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

// Cos-recompute callback type aliases, split out from CosineRecompute.h so public headers can name them
// without dragging in the build-pipeline templates.

#include <cstddef>
#include <functional>

namespace monoprop::detail {

// Per-layer cosine callbacks; `layer` selects the cosine set to replay.
//   LayerCosScale      — forward: scale operator coefficients in place by cos θ.
//   LayerCosAccumulate — reverse: apply cos θ / sec θ to state and ham, returning the layer's gradient term.
using LayerCosScale = std::function<void(size_t layer, double *coeff, double cos_val)>;
using LayerCosAccumulate =
    std::function<double(size_t layer, double *state, double *ham, double cos_val, double sec_val)>;

} // namespace monoprop::detail
