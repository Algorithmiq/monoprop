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
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

/// Below 2^-20 a layer's cosine scaling is not invertible in double, so the reverse pass reads its whole
/// cosine set back from the record instead of dividing; see Evolution.cpp.
inline constexpr double kVanishingCos = 0x1p-20;

/// The reverse pass amplifies an error injected at layer L by Sum_{M>L} log2|sec_M| bits; below one
/// significand it cannot reach the leading bit, so the layer above L has nothing worth recording.
inline constexpr double kRecordSpreadBits = 53.0;

/// Pre-layer coefficients the forward pass kept for one reverse layer, in place of dividing them back out.
struct CosRecordView {
    const TermIndex *indices = nullptr; ///< coefficient indices, ascending within a source but not merged
    const double *values = nullptr;     ///< the pre-layer coefficient at each of those indices
    size_t count = 0;
};

// Stand the recorded coefficients where the layer's scaling would have left them, so the unchanged reverse
// kernel's own ×sec lands back on the recorded value. An index the kernel skips is fixed up by the restore.
inline auto predivide_cos_record(double *ham, const CosRecordView &record, double cos_val) -> void {
    for (size_t k = 0; k < record.count; ++k) {
        ham[record.indices[k]] = record.values[k] * cos_val;
    }
}

// Put the exact pre-layer values back. Idempotent, so the two record sources may name the same index.
inline auto restore_cos_record(double *ham, const CosRecordView &record) -> void {
    for (size_t k = 0; k < record.count; ++k) {
        ham[record.indices[k]] = record.values[k];
    }
}

// Per-layer cosine callbacks; `layer` selects the cosine set to replay. Scale is the forward path
// (coeff *= cos); Accumulate is the reverse path (state *= cos, ham *= sec) and returns Σ state·ham.
using LayerCosScale = std::function<void(size_t layer, double *coeff, double cos_val)>;
using LayerCosAccumulate =
    std::function<double(size_t layer, double *state, double *ham, double cos_val, double sec_val)>;
/// Appends one layer's cosine-set indices to `out`; the gradient calls it only where the cosine vanishes.
using LayerCosIndices = std::function<void(size_t layer, std::vector<TermIndex> &out)>;
/// How many indices LayerCosIndices appends for `layer`, without materialising them.
using LayerCosCount = std::function<size_t(size_t layer)>;

struct CosCallbacks {
    LayerCosScale scale;           ///< forward path; required whenever the parameters are non-empty
    LayerCosAccumulate accumulate; ///< reverse path; required by the gradient only
    LayerCosIndices indices;       ///< reverse path; required by the gradient only
    /// Optional: exactly what `indices` appends per layer. Lets records reserve once; only memory depends on it.
    LayerCosCount count;
    /// Provenance, not a property of the closures: set only by detail::make_cos_callbacks() (behind
    /// build_cos_callbacks()), whose closures touch only their own inverted index (including its lazy caches), graph
    /// layers and the calling thread's scratch. Distinct owners' sets marked this way may then run concurrently, each
    /// on one thread at a time. Anything else is opaque and runs one owner at a time, on the primary thread of a
    /// sharded evaluation.
    bool owner_parallel = false;
};

} // namespace monoprop::detail
