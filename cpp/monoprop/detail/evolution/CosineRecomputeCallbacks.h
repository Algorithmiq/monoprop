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
#include <memory>
#include <vector>

namespace monoprop::detail {

// Per-layer cosine callbacks; `layer` selects the cosine set to replay. Scale is the forward path
// (coeff *= cos, writing the pre-scale coefficients to `record` in sweep order when it is non-null);
// Accumulate is the reverse path (state *= cos, ham read back from `record` in that same order, or
// ham *= sec when `record` is null) and returns Σ state·ham.
//
// The two sweeps walk one layer's cosine set in the same order off the same fold, so the record needs
// no index map -- but that also means a record written for layer L is only readable back at layer L.
using LayerCosScale = std::function<void(size_t layer, double *coeff, double cos_val, double *record)>;
using LayerCosAccumulate = std::function<
    double(size_t layer, double *state, double *ham, const double *record, double cos_val, double sec_val)>;
struct CosCallbacks {
    LayerCosScale scale;           ///< forward path; required whenever the parameters are non-empty
    LayerCosAccumulate accumulate; ///< reverse path; required by the gradient only
    /// Per-layer cosine-set size, i.e. each layer's record length. Required by the gradient only.
    std::shared_ptr<const std::vector<size_t>> cos_counts;
};

} // namespace monoprop::detail
