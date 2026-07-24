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
#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "monoprop/detail/graph/MPGraphLayers.h"
#include "monoprop/detail/graph/MPGraphViews.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

/// @brief Ordered per-rank record of the evolution circuit, one Layer per gate.
/// Each Layer holds a shared immutable LayerCore plus an optional pruned cosine list; the per-layer
/// cosine set is not stored but recomputed from the operator's inverted index (except pruned layers).
class monoprop_EXPORT MPGraph {
private:
    using LayerIterator = std::vector<Layer>::iterator;
    using ConstLayerIterator = std::vector<Layer>::const_iterator;

    bool schrodinger_;
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

    auto append_position() -> LayerIterator { return schrodinger_ ? active_begin_iterator() : active_end_iterator(); }

    auto append_layer(Layer layer) -> void { layers_.emplace(append_position(), std::move(layer)); }

    auto checked_layer_offset(size_t layer_idx) const -> size_t {
        if (layer_idx >= layers()) {
            throw std::out_of_range(std::format("Layer {} is out of range (layers={})", layer_idx, layers()));
        }
        return active_begin_index() + layer_idx;
    }

public:
    /// @brief Initialize the Majorana graph.
    explicit MPGraph(bool schrodinger) : schrodinger_(schrodinger) {}

    /// @brief Initialize the Majorana graph with existing layers.
    explicit MPGraph(bool schrodinger, std::vector<Layer> layers)
        : schrodinger_(schrodinger),
          layers_(std::move(layers)) {}

    /// @brief Append a new layer to the graph.
    /// @param storage The layer's LayerCore; gate info (param_index, gen_coeff, gate_index) is written
    ///   onto it here while still mutable, before it is frozen into the Layer's shared const core.
    auto append(std::shared_ptr<LayerCore> storage,
                size_t param_index = 0,
                double gen_coeff = 0.0,
                size_t gate_index = 0) -> void {
        storage->param_index = param_index;
        storage->gen_coeff = gen_coeff;
        storage->gate_index = gate_index;
        append_layer(Layer(std::move(storage)));
    }

    /// @brief Slice the graph at `key` (the number of earliest operations to include).
    /// @param contract If true, remove the sliced part from this graph.
    auto slice_graph(size_t key, bool contract = false) -> MPGraph;

    auto slice_view(size_t key) const -> MPGraphView;

    /// @brief The number of layers in the graph.
    auto layers() const -> size_t { return active_end_index() - active_begin_index(); }

    /// @brief Get the layer at `layer_idx`.
    auto get_layer(size_t layer_idx) -> Layer & { return layers_[checked_layer_offset(layer_idx)]; }

    auto get_layer(size_t layer_idx) const -> const Layer & { return layers_[checked_layer_offset(layer_idx)]; }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

    /// @brief Non-owning replay view over the active layers, in build order.
    auto replay_view() const -> MPGraphView { return MPGraphView(layers_, active_begin_index(), layers(), false); }

    /// @brief Whether the graph is in the Schrodinger picture.
    auto is_schrodinger() const -> bool { return schrodinger_; }

    /// @brief The number of (cos_inds, cycles) across all layers.
    auto num_cos_inds_and_cycles() const -> std::pair<size_t, size_t>;
    auto storage_memory_usage() const -> GraphMemoryBreakdown;
};
} // namespace monoprop
