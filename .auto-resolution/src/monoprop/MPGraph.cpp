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

#include "monoprop/MPGraph.h"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop {

namespace {

auto maybe_compact_layers(std::vector<Layer> &layers, size_t &front_offset) -> void {
    if (front_offset == 0) {
        return;
    }

    if (front_offset >= layers.size()) {
        layers.clear();
        front_offset = 0;
        return;
    }

    if (front_offset >= 4096 && front_offset * 2 >= layers.size()) {
        layers.erase(layers.begin(), layers.begin() + static_cast<std::ptrdiff_t>(front_offset));
        front_offset = 0;
    }
}

auto layer_storage_memory_usage(const LayerStorage &storage) -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_storage_object_bytes = sizeof(LayerStorage);
    breakdown.cos_data_bytes = detail::compressed_cosine_data_storage_bytes(storage.cos_data);
    breakdown.local_cycle_bytes = detail::local_cycle_storage_bytes(storage.local_cycles);
    breakdown.cross_rank_bytes = detail::cross_rank_storage_bytes(storage.cross_rank);
    breakdown.exchange_layout_bytes = detail::layer_exchange_layout_storage_bytes(storage.evolution_exchange_layout);
    return breakdown;
}

auto execution_storage_memory_breakdown(const detail::ExecutionPlanStorage &storage) -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.execution_plan_overhead_bytes =
        sizeof(detail::ExecutionPlanStorage) + storage.cos_data_blocks.capacity() * sizeof(CompressedCosineData)
        + storage.local_cycle_position_blocks.capacity() * sizeof(CompressedPositionData)
        + storage.cross_rank_out_position_blocks.capacity() * sizeof(CompressedPositionData)
        + storage.cross_rank_in_position_blocks.capacity() * sizeof(CompressedPositionData);
    for (const auto &block : storage.cos_data_blocks) {
        breakdown.execution_plan_cos_data_bytes += detail::compressed_cosine_data_storage_bytes(block);
    }
    for (const auto &block : storage.local_cycle_position_blocks) {
        breakdown.execution_plan_local_cycle_position_bytes += detail::compressed_position_data_storage_bytes(block);
    }
    for (const auto &block : storage.cross_rank_out_position_blocks) {
        breakdown.execution_plan_cross_rank_position_bytes += detail::compressed_position_data_storage_bytes(block);
    }
    for (const auto &block : storage.cross_rank_in_position_blocks) {
        breakdown.execution_plan_cross_rank_position_bytes += detail::compressed_position_data_storage_bytes(block);
    }
    breakdown.execution_plan_bytes = breakdown.execution_plan_overhead_bytes + breakdown.execution_plan_cos_data_bytes
                                     + breakdown.execution_plan_local_cycle_position_bytes
                                     + breakdown.execution_plan_cross_rank_position_bytes;
    return breakdown;
}

auto add_breakdown(GraphMemoryBreakdown &target, const GraphMemoryBreakdown &source) -> void {
    target.layer_descriptor_bytes += source.layer_descriptor_bytes;
    target.layer_storage_object_bytes += source.layer_storage_object_bytes;
    target.cos_data_bytes += source.cos_data_bytes;
    target.local_cycle_bytes += source.local_cycle_bytes;
    target.cross_rank_bytes += source.cross_rank_bytes;
    target.exchange_layout_bytes += source.exchange_layout_bytes;
    target.execution_plan_overhead_bytes += source.execution_plan_overhead_bytes;
    target.execution_plan_cos_data_bytes += source.execution_plan_cos_data_bytes;
    target.execution_plan_local_cycle_position_bytes += source.execution_plan_local_cycle_position_bytes;
    target.execution_plan_cross_rank_position_bytes += source.execution_plan_cross_rank_position_bytes;
    target.execution_plan_bytes += source.execution_plan_bytes;
}

} // namespace

auto MPGraph::slice_graph(size_t key, bool contract) -> MPGraph {
    std::vector<Layer> sliced_layers;
    const auto k = std::min(key, layers());
    sliced_layers.reserve(k);

    if (schrodinger_) {
        const size_t active_end = active_end_index();
        for (size_t i = 0; i < k; ++i) {
            sliced_layers.push_back(layers_[active_end - 1 - i]);
        }

        if (contract && k != 0) {
            layers_.resize(active_end - k);
        }
    }
    else {
        const size_t active_begin = active_begin_index();
        const size_t slice_end = active_begin + k;
        const auto begin = active_begin_iterator();
        sliced_layers.insert(sliced_layers.end(), begin, begin + static_cast<std::ptrdiff_t>(k));

        if (contract && k != 0) {
            front_offset_ = slice_end;
            maybe_compact_layers(layers_, front_offset_);
        }
    }

    return MPGraph(schrodinger_, std::move(sliced_layers));
}

auto MPGraph::slice_view(size_t key) const -> MPGraphView {
    const auto k = std::min(key, layers());
    if (schrodinger_) {
        return MPGraphView(layers_, active_end_index() - k, k, true);
    }
    return MPGraphView(layers_, active_begin_index(), k, false);
}

auto MPGraph::consume_prefix(size_t key) -> void {
    const auto k = std::min(key, layers());
    if (k == 0) {
        return;
    }

    if (schrodinger_) {
        layers_.resize(active_end_index() - k);
        return;
    }

    front_offset_ = active_begin_index() + k;
    maybe_compact_layers(layers_, front_offset_);
}

auto MPGraph::union_with(const MPGraph &other) const -> MPGraph {
    if (schrodinger_ != other.schrodinger_) {
        throw std::runtime_error("Cannot union graphs with different Schrodinger/Heisenberg settings");
    }

    std::vector<Layer> combined_layers;
    combined_layers.reserve(layers() + other.layers());

    if (schrodinger_) {
        // Schrödinger picture stores layers newest-first.
        other.append_active_layers_to(combined_layers);
        append_active_layers_to(combined_layers);
    }
    else {
        // In Heisenberg picture, this graph's operations are applied first.
        append_active_layers_to(combined_layers);
        other.append_active_layers_to(combined_layers);
    }

    return MPGraph(schrodinger_, std::move(combined_layers));
}

auto MPGraph::num_cos_inds_and_cycles() const -> std::pair<size_t, size_t> {
    size_t total_cy = 0;
    size_t total_ci = 0;

    for (auto it = active_begin_iterator(); it != active_end_iterator(); ++it) {
        const auto &layer = *it;
        total_cy += layer.total_cycles();
        total_ci += layer.num_cos_inds();
    }

    return {total_ci, total_cy};
}

auto MPGraph::storage_memory_usage() const -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_descriptor_bytes = layers_.capacity() * sizeof(Layer);

    std::unordered_set<const LayerStorage *> seen_storage;
    for (auto it = active_begin_iterator(); it != active_end_iterator(); ++it) {
        const auto storage = it->shared_storage();
        if (storage == nullptr || !seen_storage.insert(storage.get()).second) {
            continue;
        }
        add_breakdown(breakdown, layer_storage_memory_usage(*storage));
    }

    return breakdown;
}

auto MPExecutionPlan::storage_memory_usage() const -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_descriptor_bytes = layers_.capacity() * sizeof(LayerExecutionPlan);

    std::unordered_set<const LayerStorage *> seen_storage;
    std::unordered_set<const detail::ExecutionPlanStorage *> seen_execution_storage;
    for (const auto &layer : layers_) {
        const auto storage = layer.shared_storage();
        if (storage != nullptr && seen_storage.insert(storage.get()).second) {
            add_breakdown(breakdown, layer_storage_memory_usage(*storage));
        }

        const auto execution_storage = layer.shared_execution_storage();
        if (execution_storage != nullptr && seen_execution_storage.insert(execution_storage.get()).second) {
            add_breakdown(breakdown, execution_storage_memory_breakdown(*execution_storage));
        }
    }

    return breakdown;
}

} // namespace monoprop
