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

#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include <format>
#include <print>

#include "monoprop/detail/graph/MPGraphLayers.h"

namespace monoprop {

// A layer index at or past the end of the graph window being indexed.
class LayerIndexOutOfRange : public std::out_of_range {
public:
    using std::out_of_range::out_of_range;
};

/// The optimizer slot of stored layer `layer_idx` in an `n`-layer graph.
// Its own inverse, so the slot-to-layer direction calls it too. The one spelling of MPGraph's storage
// invariant: every conversion between store order and optimizer order goes through it. It lives here,
// below MPGraph, so the views over a graph reach the same spelling the graph itself uses.
constexpr auto slot_of_layer(size_t layer_idx, size_t n) -> size_t {
    return n - 1 - layer_idx;
}

static_assert(slot_of_layer(0, 4) == 3);
static_assert(slot_of_layer(slot_of_layer(1, 4), 4) == 1);

// The store offset of layer `layer_idx` in a `count`-layer store. `reverse` means the store runs against
// layer order, so layer 0 is its last element. A graph and the views over it both index through here, which
// is what keeps them agreeing on the mapping and on the diagnostic.
inline auto checked_layer_offset(size_t layer_idx, size_t count, bool reverse) -> size_t {
    if (layer_idx >= count) {
        throw LayerIndexOutOfRange(std::format("Layer {} is out of range (layers={})", layer_idx, count));
    }

    return reverse ? slot_of_layer(layer_idx, count) : layer_idx;
}

// One rank's own graph memory only.
struct GraphMemoryBreakdown final {
    size_t layer_descriptor_bytes = 0;
    size_t layer_storage_object_bytes = 0;
    size_t cos_data_bytes = 0;
    size_t cross_rank_bytes = 0;
    size_t exchange_layout_bytes = 0;

    auto total_bytes() const -> size_t {
        return layer_descriptor_bytes + layer_storage_object_bytes + cos_data_bytes + cross_rank_bytes
               + exchange_layout_bytes;
    }

    // Lets a partitioned propagator aggregate its per-partition graph breakdowns.
    auto operator+=(const GraphMemoryBreakdown &o) -> GraphMemoryBreakdown & {
        layer_descriptor_bytes += o.layer_descriptor_bytes;
        layer_storage_object_bytes += o.layer_storage_object_bytes;
        cos_data_bytes += o.cos_data_bytes;
        cross_rank_bytes += o.cross_rank_bytes;
        exchange_layout_bytes += o.exchange_layout_bytes;
        return *this;
    }
};

// A bounds-checked, optionally reversed index space over a window of layers. `reverse` traverses it
// newest-first. Non-owning: the layer storage must outlive the view.
//
// Not a std::ranges adaptor, and not iterable, for three reasons:
//  - Index i addresses three parallel things at once -- the layer, params[i] in evolve_operator(), and
//    the recipe cache in build_cos_callbacks(). ev_and_grad() passes a computed index, not an iteration
//    step. The consumers need an index space, not a sequence.
//  - One type must serve both directions. views::counted(...) and views::reverse(views::counted(...))
//    are different types, and std::ranges::any_view is C++26.
//  - The type appears in the exported signatures of ev(), ev_and_grad(), evolve_operator() and
//    state_operator_derivative_local(). A ranges adaptor would put a library-version-dependent
//    template soup into their mangled names.
class MPGraphView {
public:
    MPGraphView(std::span<const Layer> layers, bool reverse) : layers_(layers), reverse_(reverse) {}

    auto layers() const -> size_t { return layers_.size(); }

    auto get_layer(size_t layer_idx) const -> const Layer & {
        return layers_[checked_layer_offset(layer_idx, layers_.size(), reverse_)];
    }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

private:
    std::span<const Layer> layers_;
    bool reverse_ = false;
};

} // namespace monoprop
