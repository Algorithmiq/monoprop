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

#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/detail/graph/MPGraphLayers.h"
#include "monoprop/detail/graph/MPGraphViews.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

/**
 * @brief Class for storing and manipulating the Monomial Propagator graph.
 *
 * This class represents the graph for a single rank. Each layer contains:
 * - Cosine indices (local)
 * - Local cycles (both indices on this rank)
 * - Outgoing sources (source indices to send to each target rank)
 * - Incoming targets (target indices to update from each source rank)
 */
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

    auto append_active_layers_to(std::vector<Layer> &target) const -> void {
        target.insert(target.end(), active_begin_iterator(), active_end_iterator());
    }

public:
    /**
     * @brief Initialize the Majorana graph.
     *
     * @param schrodinger Whether the simulation is in the Schrodinger picture.
     */
    explicit MPGraph(bool schrodinger) : schrodinger_(schrodinger) {}

    /**
     * @brief Initialize the Majorana graph with existing layers.
     *
     * @param schrodinger Whether the simulation is in the Schrodinger picture.
     * @param layers Vector of Layer objects.
     */
    explicit MPGraph(bool schrodinger, std::vector<Layer> layers)
        : schrodinger_(schrodinger),
          layers_(std::move(layers)) {}

    /**
     * @brief Append a new layer to the graph with flattened cross-rank storage.
     *
     * @param cos_inds Cosine indices for this layer.
     * @param local_cycles Local cycles (source, target, phase on this rank).
     * @param cross_rank Cross-rank cycles indexed by remote rank.
     */
    auto append(VecZ cos_inds, std::vector<LocalCycle> local_cycles, std::vector<CrossRankCycles> cross_rank) -> void {
        append_layer(Layer(std::move(cos_inds), std::move(local_cycles), std::move(cross_rank)));
    }

    auto append(CompressedCosineData cos_data,
                std::vector<LocalCycle> local_cycles,
                std::vector<CrossRankCycles> cross_rank) -> void {
        append_layer(Layer(std::move(cos_data), std::move(local_cycles), std::move(cross_rank)));
    }

    /**
     * @brief Slice the graph at the given key.
     *
     * @param key Number of earliest operations to include in the slice.
     * @param contract If true, modify this graph to remove the sliced part.
     * @return A new graph containing the sliced layers.
     */
    auto slice_graph(size_t key, bool contract = false) -> MPGraph;

    auto slice_view(size_t key) const -> MPGraphView;

    auto consume_prefix(size_t key) -> void;

    /**
     * @brief Create a union of two graphs without copying layer data.
     *
     * @param other The other graph to union with.
     * @return A new graph containing references to all layers from both graphs.
     */
    auto union_with(const MPGraph &other) const -> MPGraph;

    /**
     * @brief Get the number of layers.
     *
     * @return The number of layers in the graph.
     */
    auto layers() const -> size_t { return active_end_index() - active_begin_index(); }

    /**
     * @brief Get a specific layer from the graph.
     *
     * @param layer_idx The layer index.
     * @return Reference to the Layer object.
     */
    auto get_layer(size_t layer_idx) -> Layer & { return layers_[checked_layer_offset(layer_idx)]; }

    auto get_layer(size_t layer_idx) const -> const Layer & { return layers_[checked_layer_offset(layer_idx)]; }

    auto get_layer_traversal(size_t layer_idx) const -> LayerTraversal { return get_layer(layer_idx).traversal(); }

    /**
     * @brief Check if the graph is in Schrodinger picture.
     *
     * @return True if the graph is in Schrodinger picture, false otherwise.
     */
    auto is_schrodinger() const -> bool { return schrodinger_; }

    /**
     * @brief Return the number of cos_inds and cycles across all layers.
     *
     * @return Pair containing the number of (cos_inds, cycles).
     */
    auto num_cos_inds_and_cycles() const -> std::pair<size_t, size_t>;
    auto storage_memory_usage() const -> GraphMemoryBreakdown;
};
} // namespace monoprop

namespace std {
template <>
struct formatter<monoprop::Layer> {
    constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
    template <typename FormatContext>
    auto format(const monoprop::Layer &layer, FormatContext &ctx) const {
        size_t outgoing_count = 0, incoming_count = 0;
        for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
            outgoing_count += layer.cross_rank_out_size(rank);
            incoming_count += layer.cross_rank_in_size(rank);
        }
        return std::format_to(ctx.out(),
                              "Layer{{cos_inds={}, local={}, outgoing={}, incoming={}}}",
                              layer.num_cos_inds(),
                              layer.local_cycle_count(),
                              outgoing_count,
                              incoming_count);
    }
};
} // namespace std
