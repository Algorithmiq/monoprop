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
// active layer i is optimizer slot layers()-1-i. append_layer() normalizes the two arrival orders into that
// one storage order, which is what lets everything reconstructing optimizer order from a graph -- the
// evolved-operator setup and the gradient loop in MPFunctions, MonomialPropagator::graph_gate_arrays_ -- do
// it without knowing which growth end this graph has. Changing the storage order means changing those too.
class monoprop_EXPORT MPGraph {
private:
    using LayerIterator = std::vector<Layer>::iterator;
    using ConstLayerIterator = std::vector<Layer>::const_iterator;

    LayerGrowth growth_;
    std::vector<Layer> layers_;
    size_t front_offset_ = 0;

    auto active_begin_index() const -> size_t { return front_offset_; }

    auto active_end_index() const -> size_t { return layers_.size(); }

    auto active_begin_iterator() -> LayerIterator {
        return layers_.begin() + static_cast<std::ptrdiff_t>(active_begin_index());
    }

    auto active_end_iterator() -> LayerIterator { return layers_.end(); }

    auto active_begin_iterator() const -> ConstLayerIterator {
        return layers_.begin() + static_cast<std::ptrdiff_t>(active_begin_index());
    }

    auto active_end_iterator() const -> ConstLayerIterator { return layers_.end(); }

    auto append_position() -> LayerIterator {
        return grows_at_front() ? active_begin_iterator() : active_end_iterator();
    }

    auto append_layer(Layer layer) -> void { layers_.emplace(append_position(), std::move(layer)); }

    auto checked_layer_offset(size_t layer_idx) const -> size_t {
        if (layer_idx >= layers()) {
            throw LayerIndexOutOfRange(std::format("Layer {} is out of range (layers={})", layer_idx, layers()));
        }
        return active_begin_index() + layer_idx;
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

    /// Slice the graph at `key` (the number of earliest operations to include); `contract` also removes
    /// the sliced part from this graph.
    // Layers come back in application order, earliest first, which matches this class's storage order only
    // for LayerGrowth::Back. A slice is for replay_view() and must not be fed to anything that
    // reconstructs optimizer order from it.
    auto slice_graph(size_t key, bool contract = false) -> MPGraph;

    auto slice_view(size_t key) const -> MPGraphView;

    auto layers() const -> size_t { return active_end_index() - active_begin_index(); }

    auto get_layer(size_t layer_idx) -> Layer& { return layers_[checked_layer_offset(layer_idx)]; }

    auto get_layer(size_t layer_idx) const -> const Layer& { return layers_[checked_layer_offset(layer_idx)]; }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

    /// Non-owning replay view over the active layers, in stored (descending optimizer-slot) order.
    auto replay_view() const -> MPGraphView { return {layers_, active_begin_index(), layers(), false}; }

    /// Carried so a derived graph (a slice, a pared copy) keeps its source's layer order.
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
