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

#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "monoprop/detail/graph/MPGraphLayers.h"
#include "monoprop/detail/graph/MPGraphViews.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

/// The slope of the optimizer slots a build hands to append(), which is Picture::gate_slot's slope.
// This one bit is all the graph needs to know about the simulation that drives it.
enum class ArrivalOrder : uint8_t {
    DescendingSlot, ///< each arriving gate takes a lower slot than the last
    AscendingSlot,  ///< each arriving gate takes a higher slot than the last
};

/// The optimizer slot of stored layer `layer_idx` in an `n`-layer graph.
// Its own inverse, so the slot-to-layer direction calls it too. The one spelling of MPGraph's storage
// invariant: every conversion between store order and optimizer order goes through it.
constexpr auto slot_of_layer(size_t layer_idx, size_t n) -> size_t {
    return n - 1 - layer_idx;
}

static_assert(slot_of_layer(0, 4) == 3);
static_assert(slot_of_layer(slot_of_layer(1, 4), 4) == 1);

/// Ordered per-rank record of the evolution circuit, one Layer per generator.
// LAYER INDICES ARE IN DESCENDING OPTIMIZER-SLOT ORDER under either arrival order: layer i is optimizer
// slot slot_of_layer(i, layers()). That one order is what lets everything reconstructing optimizer order
// from a graph -- the evolved-operator setup and the gradient loop in MPFunctions,
// MonomialPropagator::graph_gate_arrays_ -- do it without knowing how this graph was built. Changing it
// means changing those too.
//
// layers_ holds arrival order, so a push_back is all an append costs; reverse_indexing_() maps a layer
// index onto it. An AscendingSlot build therefore stores its layers backwards, and get_layer() is the only
// place that knows.
class monoprop_EXPORT MPGraph {
private:
    ArrivalOrder arrival_;
    std::vector<Layer> layers_;

    // True when arrival order runs against layer order, so layer 0 is the last element.
    auto reverse_indexing_() const -> bool { return arrival_ == ArrivalOrder::AscendingSlot; }

    auto stored_offset_(size_t layer_idx) const -> size_t {
        check_layer_index_(layer_idx);
        return reverse_indexing_() ? slot_of_layer(layer_idx, layers_.size()) : layer_idx;
    }

    auto check_layer_index_(size_t layer_idx) const -> void {
        if (layer_idx >= layers()) {
            throw LayerIndexOutOfRange(std::format("Layer {} is out of range (layers={})", layer_idx, layers()));
        }
    }

public:
    explicit MPGraph(ArrivalOrder arrival) : arrival_(arrival) {}

    /// Gate info (param_index, gen_coeff, gate_index) is written onto `storage` here while it is still
    /// mutable, before it is frozen into the Layer's shared const core.
    auto append(std::shared_ptr<LayerCore> storage,
                size_t param_index = 0,
                double gen_coeff = 0.0,
                size_t gate_index = 0) -> void {
        storage->param_index = param_index;
        storage->gen_coeff = gen_coeff;
        storage->gate_index = gate_index;
        layers_.push_back(Layer(std::move(storage)));
    }

    /// Swap in a rebuilt layer, addressed the same way get_layer() addresses it.
    auto replace_layer(size_t layer_idx, Layer layer) -> void { layers_[stored_offset_(layer_idx)] = std::move(layer); }

    /// Drop every layer. The graph stays usable, and a later append() starts from an empty store.
    auto clear() -> void { layers_.clear(); }

    auto layers() const -> size_t { return layers_.size(); }

    auto get_layer(size_t layer_idx) -> Layer& { return layers_[stored_offset_(layer_idx)]; }

    auto get_layer(size_t layer_idx) const -> const Layer& { return layers_[stored_offset_(layer_idx)]; }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

    /// Non-owning replay view over the layers, in layer (descending optimizer-slot) order.
    auto replay_view() const -> MPGraphView { return {layers_, reverse_indexing_()}; }

    /// The layers in the order this graph's own build walked them, which is the order a contraction must
    /// replay them in.
    // Never reversed, whichever way the build ran: arrival order IS build order, and a contraction drives
    // the live coefficient vector, so it follows the simulation. The picture-free evaluation order is the
    // layer order instead, which is why replay_view() is the one that carries the flag.
    auto contraction_view() const -> MPGraphView { return {layers_, false}; }

    /// A normally-built layer stores no cosine set, so the companion cosine-index count cannot come from
    /// the graph: only the operator's inverted index can supply it.
    auto total_cycles() const -> size_t;
    auto storage_memory_usage() const -> GraphMemoryBreakdown;
};
} // namespace monoprop
