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

/// Which end of the layer store a newly appended gate attaches to.
// This one bit is all the graph needs to know about the simulation that drives it.
enum class LayerGrowth : uint8_t {
    Back,  ///< a new gate takes the lowest optimizer slot, so it attaches at the back
    Front, ///< a new gate takes the highest optimizer slot, so it attaches at the front
};

/// Ordered per-rank record of the evolution circuit, one Layer per generator.
// A graph built by append() stores its layers in DESCENDING optimizer-slot order under either growth end:
// layer i is optimizer slot layers()-1-i. append_layer() normalizes the two arrival orders into that one
// storage order, which is what lets everything reconstructing optimizer order from a graph -- the
// evolved-operator setup and the gradient loop in MPFunctions, MonomialPropagator::graph_gate_arrays_ -- do
// it without knowing which growth end this graph has. Changing the storage order means changing those too.
class monoprop_EXPORT MPGraph {
private:
    LayerGrowth growth_;
    std::vector<Layer> layers_;

    auto append_position() -> std::vector<Layer>::iterator {
        return grows_at_front() ? layers_.begin() : layers_.end();
    }

    auto append_layer(Layer layer) -> void { layers_.emplace(append_position(), std::move(layer)); }

    auto check_layer_index_(size_t layer_idx) const -> void {
        if (layer_idx >= layers()) {
            throw LayerIndexOutOfRange(std::format("Layer {} is out of range (layers={})", layer_idx, layers()));
        }
    }

public:
    explicit MPGraph(LayerGrowth growth) : growth_(growth) {}

    explicit MPGraph(LayerGrowth growth, std::vector<Layer> layers) : growth_(growth), layers_(std::move(layers)) {}

    /// Gate info (param_index, gen_coeff, gate_index) is written onto `storage` here while it is still
    /// mutable, before it is frozen into the Layer's shared const core.
    auto append(std::shared_ptr<LayerCore> storage,
                size_t param_index = 0,
                double gen_coeff = 0.0,
                size_t gate_index = 0) -> void {
        storage->param_index = param_index;
        storage->gen_coeff = gen_coeff;
        storage->gate_index = gate_index;
        append_layer(Layer(std::move(storage)));
    }

    /// Drop every layer. The graph stays usable, and a later append() starts from an empty store.
    auto clear() -> void { layers_.clear(); }

    auto layers() const -> size_t { return layers_.size(); }

    auto get_layer(size_t layer_idx) -> Layer& {
        check_layer_index_(layer_idx);
        return layers_[layer_idx];
    }

    auto get_layer(size_t layer_idx) const -> const Layer& {
        check_layer_index_(layer_idx);
        return layers_[layer_idx];
    }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

    /// Non-owning replay view over the layers, in stored (descending optimizer-slot) order.
    auto replay_view() const -> MPGraphView { return {layers_, false}; }

    /// The layers in the order this graph's own build walked them, which is the order a contraction must
    /// replay them in.
    // Not replay_view(): a contraction drives the live coefficient vector, so it follows the simulation
    // direction, where the picture-free evaluation order is always the stored one.
    auto contraction_view() const -> MPGraphView { return {layers_, grows_at_front()}; }

    /// Carried so a pared copy keeps its source's layer order.
    auto growth() const -> LayerGrowth { return growth_; }

    /// Whether new layers attach at the front, so the oldest operation is at the back.
    // The one spelling of the growth question: every ordering-sensitive site, here and in pare_graph, asks
    // it this way rather than comparing enumerators.
    auto grows_at_front() const -> bool { return growth_ == LayerGrowth::Front; }

    /// A normally-built layer stores no cosine set, so the companion cosine-index count cannot come from
    /// the graph: only the operator's inverted index can supply it.
    auto total_cycles() const -> size_t;
    auto storage_memory_usage() const -> GraphMemoryBreakdown;
};
} // namespace monoprop
