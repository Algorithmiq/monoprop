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

#include "PareGraph.h"

#include <algorithm>
#include <bit>
#include <utility>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop {

namespace {

// Bits of `present` whose absolute index is in the keep set.
inline auto keep_mask_for_block(const std::vector<char> &keep, size_t base, uint64_t present) -> uint64_t {
    uint64_t mask = 0;
    uint64_t b = present;
    while (b) {
        const auto t = static_cast<size_t>(std::countr_zero(b));
        if (const size_t idx = base + t; idx < keep.size() && keep[idx] != 0) {
            mask |= (uint64_t{1} << t);
        }
        b &= b - 1;
    }
    return mask;
}

// Filter the cosine set to the kept nodes into `filtered` (contents replaced, capacity reused); true means nothing was
// pruned (replay folds the full set) and leaves `filtered` unspecified.
auto filter_layer_cosine_data(const CosMask &cos, const std::vector<char> &nodes_to_keep, CosMask &filtered) -> bool {
    const size_t n = cos.blocks.size();
    filtered.blocks.clear();
    filtered.total_count = 0;
    bool preserves = true;
    for (size_t k = 0; k < n; ++k) {
        const auto [base, bits] = cos.blocks[k];
        const uint64_t kept = bits & keep_mask_for_block(nodes_to_keep, base, bits);
        if (kept != bits) {
            preserves = false;
        }
        if (kept) {
            filtered.blocks.emplace_back(base, kept);
            filtered.total_count += static_cast<size_t>(std::popcount(kept));
        }
    }
    return preserves;
}

// The cos pass (not the D-apply) scales every D target, since cos holds all anticommuting indices, so no
// D target may be pruned; a B source is kept exactly when its partner D target is, i.e. always. Every rank
// reaches the same conclusion about its own endpoints, so no cross-rank agreement is needed.
//
// B sources are marked for remote ranks only: the self-rank slot carries local cycles, whose sources
// follow the ordinary backward reachability instead of being force-kept.
auto mark_cross_rank_endpoints_kept(const LayerTraversal &layer, size_t my_rank, std::vector<char> &nodes_to_keep)
    -> void {
    const auto mark = [&nodes_to_keep](size_t idx) {
        if (idx < nodes_to_keep.size()) {
            nodes_to_keep[idx] = 1;
        }
    };
    layer.for_each_occupied_slot([&mark, my_rank](size_t rank, const detail::CrossRankSlotView &slot) {
        for (size_t k = 0; k < slot.sin_send_count; ++k) {
            mark(detail::slot_sin_recv_index(slot, k));
            if (rank != my_rank) {
                mark(detail::slot_sin_send_index(slot, k));
            }
        }
    });
}

} // namespace

} // namespace monoprop

namespace monoprop::detail {

auto pare_graph_owner(const MPGraph &graph,
                      const VecZ &nonzero_inds,
                      size_t local_index_count,
                      bool schrodinger,
                      size_t flat_owner,
                      const std::function<void(size_t, CosMask &)> &full_cos_of_layer) -> MPGraph {
    const size_t num_layers = graph.layers();
    const size_t my_rank = flat_owner;

    std::vector<char> nodes_to_keep(local_index_count, 0);
    for (auto idx : nonzero_inds) {
        if (idx < nodes_to_keep.size()) {
            nodes_to_keep[idx] = 1;
        }
    }

    std::vector<Layer> layers(num_layers);
    /*
     * One buffer for every layer's full set and one for its filtered set: growing a fresh vector per layer, interleaved
     * with the stored sets, extends the owner's heap one reallocation at a time, and at T owners those heap-growth
     * calls serialize on the process's memory map. Only the stored sets are allocated, each at its exact size.
     */
    CosMask full;
    CosMask filtered;

    // Single backward sweep, entirely rank-local: every cross-rank endpoint is force-kept (see
    // mark_cross_rank_endpoints_kept), so nodes_to_keep stays consistent across ranks with no exchange.
    // Cross-rank lists are never pruned; the keep-set only has to be right so cos pruning stays exact.
    for (size_t iter = 0; iter < num_layers; ++iter) {
        const size_t layer_idx = schrodinger ? iter : (num_layers - 1 - iter);
        const auto &layer = graph.get_layer(layer_idx);
        const auto lt = layer.traversal();

        // Order is load-bearing for bit-exact pruning: endpoints kept before the cosine filter.
        mark_cross_rank_endpoints_kept(lt, my_rank, nodes_to_keep);

        full_cos_of_layer(layer_idx, full);
        if (filter_layer_cosine_data(full, nodes_to_keep, filtered)) {
            layers[layer_idx] = Layer(layer.shared_core());
        }
        else {
            layers[layer_idx] = Layer(layer.shared_core(),
                                      CosMask{.blocks = {filtered.blocks.begin(), filtered.blocks.end()},
                                              .total_count = filtered.total_count});
        }
    }

    return MPGraph(graph.is_schrodinger(), std::move(layers));
}

} // namespace monoprop::detail

namespace monoprop {

auto pare_graph(const MPGraph &graph,
                const VecZ &nonzero_inds,
                size_t local_index_count,
                bool schrodinger,
                mpi::Comm comm,
                const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph {
    return detail::pare_graph_owner(graph,
                                    nonzero_inds,
                                    local_index_count,
                                    schrodinger,
                                    static_cast<size_t>(mpi::rank(comm)),
                                    [&full_cos_of_layer](size_t i, CosMask &out) { out = full_cos_of_layer(i); });
}

} // namespace monoprop
