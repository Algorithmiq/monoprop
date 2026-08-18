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

#include <unordered_set>
#include <utility>

#include "monoprop/TypeAliases.h"

namespace monoprop {

namespace {

auto layer_storage_memory_usage(const LayerCore &storage) -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_storage_object_bytes = sizeof(LayerCore);
    breakdown.cross_rank_bytes = detail::cross_rank_storage_bytes(storage.cross_rank);
    breakdown.exchange_layout_bytes = detail::layer_exchange_layout_storage_bytes(storage.evolution_exchange_layout);
    return breakdown;
}

} // namespace

auto MPGraph::total_cycles() const -> size_t {
    size_t total = 0;
    for (const auto &layer : layers_) {
        total += layer.traversal().total_cycles();
    }
    return total;
}

auto MPGraph::storage_memory_usage() const -> GraphMemoryBreakdown {
    GraphMemoryBreakdown breakdown;
    breakdown.layer_descriptor_bytes = layers_.capacity() * sizeof(Layer);

    std::unordered_set<const LayerCore *> seen_storage;
    for (const auto &layer : layers_) {
        if (const auto storage = layer.shared_core(); storage != nullptr && seen_storage.insert(storage.get()).second) {
            breakdown += layer_storage_memory_usage(*storage);
        }
        // Pruned cos is owned per-layer, not by the shared core, so it accumulates without the dedup.
        if (const CosMask *cos = layer.pruned_cos(); cos != nullptr) {
            breakdown.cos_data_bytes += cos->blocks.capacity() * sizeof(std::pair<size_t, uint64_t>);
        }
    }

    return breakdown;
}

} // namespace monoprop
