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
#include <array>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <format>
#include "monoprop/detail/print_compat.h"

#include "monoprop/TypeAliases.h"

namespace monoprop {

namespace {

// Consumed front layers are tracked by front_offset rather than erased eagerly; physically drop them
// only once the dead prefix is both large and at least half the vector, to bound the amortized cost.
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

auto layer_storage_memory_usage(const LayerCore &storage) -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_storage_object_bytes = sizeof(LayerCore);
    breakdown.cross_rank_bytes = detail::cross_rank_storage_bytes(storage.cross_rank);
    breakdown.exchange_layout_bytes = detail::layer_exchange_layout_storage_bytes(storage.evolution_exchange_layout);
    return breakdown;
}

auto add_breakdown(GraphMemoryBreakdown &target, const GraphMemoryBreakdown &source) -> void {
    target.layer_descriptor_bytes += source.layer_descriptor_bytes;
    target.layer_storage_object_bytes += source.layer_storage_object_bytes;
    target.cos_data_bytes += source.cos_data_bytes;
    target.cross_rank_bytes += source.cross_rank_bytes;
    target.exchange_layout_bytes += source.exchange_layout_bytes;
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

auto MPGraph::num_cos_inds_and_cycles() const -> std::pair<size_t, size_t> {
    size_t total_cy = 0;
    size_t total_ci = 0;

    for (auto it = active_begin_iterator(); it != active_end_iterator(); ++it) {
        const auto &layer = *it;
        total_cy += layer.total_cycles();
        // num_cos_inds counts cosine-ONLY terms (cos-scaled but not rotation endpoints) = total anti
        // endpoints minus rotation endpoints; saturate to guard the unsigned subtract.
        const size_t cos_total = layer.num_cos_inds();
        const size_t endpoints = layer.total_rotation_endpoints();
        total_ci += (cos_total > endpoints) ? (cos_total - endpoints) : 0;
    }

    return {total_ci, total_cy};
}

auto MPGraph::storage_memory_usage() const -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_descriptor_bytes = layers_.capacity() * sizeof(Layer);

    std::unordered_set<const LayerCore *> seen_storage;
    for (auto it = active_begin_iterator(); it != active_end_iterator(); ++it) {
        const auto storage = it->shared_core();
        if (storage != nullptr && seen_storage.insert(storage.get()).second) {
            add_breakdown(breakdown, layer_storage_memory_usage(*storage));
        }
        // Pruned cos is stored per-layer (on PrunedLayer), not on the shared core, so accumulate it
        // per active layer without the shared-core dedup. FoldLayer stores no cos.
        if (const CosMask *cos = it->pruned_cos(); cos != nullptr) {
            breakdown.cos_data_bytes += cos->blocks.capacity() * sizeof(std::pair<size_t, uint64_t>);
        }
    }

    return breakdown;
}

} // namespace monoprop
