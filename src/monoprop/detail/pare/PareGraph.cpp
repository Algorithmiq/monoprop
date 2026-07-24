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
#include <cmath>
#include <utility>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop {

namespace {

// Cross-rank exchange layout. The exchange ROUNDS are load-bearing for multi-rank correctness even
// though we store NO positions — they propagate the keep-set across ranks.
struct BuilderExchangeLayout final {
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    size_t total_send = 0;
    size_t total_recv = 0;
};

struct BuilderExchangeBuffers final {
    VecI send_buffer;
    VecI recv_buffer;
};

enum class BuilderExchangeDirection {
    Outgoing,
    Incoming,
};

template <typename Func>
auto for_each_remote_rank(const LayerTraversal &layer, size_t my_rank, Func &&func) -> void {
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        if (rank != my_rank) {
            func(rank);
        }
    }
}

auto has_remote_cross_rank_edges(const LayerTraversal &layer, size_t my_rank) -> bool {
    bool has_remote_edges = false;
    for_each_remote_rank(layer, my_rank, [&layer, &has_remote_edges](size_t rank) {
        if (layer.cross_rank_sin_send_size(rank) != 0 || layer.cross_rank_sin_recv_size(rank) != 0) {
            has_remote_edges = true;
        }
    });
    return has_remote_edges;
}

auto build_builder_exchange_layout(const LayerTraversal &layer, size_t my_rank, BuilderExchangeDirection direction)
    -> BuilderExchangeLayout {
    BuilderExchangeLayout layout;
    layout.send_counts.resize(layer.cross_rank_rank_count(), 0);
    layout.send_displs.resize(layer.cross_rank_rank_count(), 0);
    layout.recv_counts.resize(layer.cross_rank_rank_count(), 0);
    layout.recv_displs.resize(layer.cross_rank_rank_count(), 0);

    size_t total_send = 0;
    size_t total_recv = 0;
    for_each_remote_rank(layer, my_rank, [&layer, &direction, &layout, &total_send, &total_recv](size_t rank) {
        // In this layout the "outgoing" direction maps to B (send) and the "incoming" to D (recv).
        const size_t send_count = direction == BuilderExchangeDirection::Outgoing
                                      ? layer.cross_rank_sin_send_size(rank)
                                      : layer.cross_rank_sin_recv_size(rank);
        const size_t recv_count = direction == BuilderExchangeDirection::Outgoing
                                      ? layer.cross_rank_sin_recv_size(rank)
                                      : layer.cross_rank_sin_send_size(rank);
        layout.send_counts[rank] = detail::checked_mpi_int(send_count, "Pare builder send count");
        layout.recv_counts[rank] = detail::checked_mpi_int(recv_count, "Pare builder receive count");
        total_send += send_count;
        total_recv += recv_count;
    });

    size_t send_displacement = 0;
    size_t recv_displacement = 0;
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        layout.send_displs[rank] = detail::checked_mpi_int(send_displacement, "Pare builder send displacement");
        layout.recv_displs[rank] = detail::checked_mpi_int(recv_displacement, "Pare builder receive displacement");
        send_displacement += static_cast<size_t>(layout.send_counts[rank]);
        recv_displacement += static_cast<size_t>(layout.recv_counts[rank]);
    }

    layout.total_send = total_send;
    layout.total_recv = total_recv;
    return layout;
}

auto resize_builder_exchange_buffers(const BuilderExchangeLayout &layout, BuilderExchangeBuffers &buffers) -> void {
    // Size >= 1 so data() is never nullptr (some MPI impls reject nullptr buffers even at zero count).
    buffers.send_buffer.resize(layout.total_send == 0 ? 1 : layout.total_send);
    buffers.recv_buffer.resize(layout.total_recv == 0 ? 1 : layout.total_recv);
}

auto build_empty_builder_exchange_layout(size_t num_ranks) -> BuilderExchangeLayout {
    BuilderExchangeLayout layout;
    layout.send_counts.assign(num_ranks, 0);
    layout.send_displs.assign(num_ranks, 0);
    layout.recv_counts.assign(num_ranks, 0);
    layout.recv_displs.assign(num_ranks, 0);
    layout.total_send = 0;
    layout.total_recv = 0;
    return layout;
}

auto pack_source_keep_flags(const LayerTraversal &layer,
                            const std::vector<char> &nodes_to_keep,
                            const BuilderExchangeLayout &layout,
                            size_t my_rank,
                            VecI &send_buffer) -> void {
    for_each_remote_rank(layer, my_rank, [&layer, &layout, &nodes_to_keep, &send_buffer](size_t rank) {
        const auto base = static_cast<size_t>(layout.send_displs[rank]);
        const size_t count = layer.cross_rank_sin_send_size(rank);
        layer.for_each_cross_rank_sin_send_range(
            rank,
            0,
            count,
            [&send_buffer, &base, &nodes_to_keep](size_t logical_idx, size_t src_idx) {
                send_buffer[base + logical_idx] = (src_idx < nodes_to_keep.size() && nodes_to_keep[src_idx]) ? 1 : 0;
            });
    });
}

auto execute_builder_exchange(const BuilderExchangeLayout &layout,
                              BuilderExchangeBuffers &buffers,
                              const mpi::Comm &comm) -> void {
    // Blocking one-shot keep-flag exchange; recv counts are the per-rank transpose of the send counts,
    // so no count round is needed. All ranks must participate (facade discipline); buffers are >= 1.
    mpi::post_flat_alltoallv<int>(buffers.send_buffer.data(),
                                  layout.send_counts.data(),
                                  layout.send_displs.data(),
                                  buffers.recv_buffer.data(),
                                  layout.recv_counts.data(),
                                  layout.recv_displs.data(),
                                  static_cast<int>(layout.send_counts.size()),
                                  comm)
        .wait();
}

// Filter the full cosine set to the kept nodes: {{}, true} when nothing was pruned (replay folds the
// full set), else {filtered, false}.
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

auto filter_layer_cosine_data(const CosMask &cos, const std::vector<char> &nodes_to_keep) -> std::pair<CosMask, bool> {
    const size_t n = cos.blocks.size();
    CosMask filtered;
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
    if (preserves) {
        return {{}, true};
    }
    return {std::move(filtered), false};
}

// The cos pass (not the D-apply) scales every D target, since cos holds ALL anticommuting indices, so
// mark every D target kept BEFORE the cosine filter — the pruned cos and its backward-reachable
// producers must retain them.
auto mark_replayed_d_targets(const LayerTraversal &layer, std::vector<char> &nodes_to_keep) -> void {
    const size_t rank_count = layer.cross_rank_rank_count();
    for (size_t rank = 0; rank < rank_count; ++rank) {
        layer.for_each_cross_rank_sin_recv_range(rank,
                                                 0,
                                                 layer.cross_rank_sin_recv_size(rank),
                                                 [&nodes_to_keep](size_t /*logical_idx*/, size_t tgt_idx, int) {
                                                     if (tgt_idx < nodes_to_keep.size()) {
                                                         nodes_to_keep[tgt_idx] = 1;
                                                     }
                                                 });
    }
}

// Per-rank body of propagate_cross_rank_d: applies D backward reachability to one remote rank's
// cross-rank D entries — fills the per-edge selection flags and marks surviving D targets kept.
auto propagate_cross_rank_d_for_rank(const LayerTraversal &layer,
                                     size_t rank,
                                     const BuilderExchangeLayout &source_keep_layout,
                                     const VecI &remote_src_keep,
                                     const BuilderExchangeLayout &selection_layout,
                                     VecI &selected_incoming_flags,
                                     std::vector<char> &nodes_to_keep) -> void {
    const auto remote_base = static_cast<size_t>(source_keep_layout.recv_displs[rank]);
    const auto notify_base = static_cast<size_t>(selection_layout.send_displs[rank]);
    layer.for_each_cross_rank_sin_recv_range(
        rank,
        0,
        layer.cross_rank_sin_recv_size(rank),
        [&nodes_to_keep, &remote_base, &remote_src_keep, &notify_base, &selected_incoming_flags](size_t logical_idx,
                                                                                                 size_t tgt_idx,
                                                                                                 int) {
            const bool keep_tgt = tgt_idx < nodes_to_keep.size() && nodes_to_keep[tgt_idx];
            const bool keep_src = remote_base + logical_idx < remote_src_keep.size()
                                      ? remote_src_keep[remote_base + logical_idx] != 0
                                      : false;
            if (keep_src || keep_tgt) {
                if (notify_base + logical_idx < selected_incoming_flags.size()) {
                    selected_incoming_flags[notify_base + logical_idx] = 1;
                }
                if (!keep_tgt && tgt_idx < nodes_to_keep.size()) {
                    nodes_to_keep[tgt_idx] = 1;
                }
            }
        });
}

// Phase 1 (before the selection exchange): propagate the keep-set backward across this rank's
// cross-rank D entries and record a per-edge selection flag for each B source the partner must keep,
// so both endpoints of a cross-rank edge agree. Stores no positions.
auto propagate_cross_rank_d(const LayerTraversal &layer,
                            size_t my_rank,
                            const BuilderExchangeLayout &source_keep_layout,
                            const VecI &remote_src_keep,
                            const BuilderExchangeLayout &selection_layout,
                            VecI &selected_incoming_flags,
                            std::vector<char> &nodes_to_keep) -> void {
    // Size >= 1 so a non-null send pointer exists even with outgoing-only edges (total_send == 0); the
    // padding slot is never indexed nor sent.
    selected_incoming_flags.assign(std::max<size_t>(1, selection_layout.total_send), 0);
    for_each_remote_rank(
        layer,
        my_rank,
        [&layer, &source_keep_layout, &remote_src_keep, &selection_layout, &selected_incoming_flags, &nodes_to_keep](
            size_t rank) {
            propagate_cross_rank_d_for_rank(layer,
                                            rank,
                                            source_keep_layout,
                                            remote_src_keep,
                                            selection_layout,
                                            selected_incoming_flags,
                                            nodes_to_keep);
        });
}

// Phase 2 (after the selection exchange): for each B entry the partner selected, mark its source node
// so its producers in earlier (later-processed) layers are kept. Stores no positions.
auto propagate_cross_rank_b(const LayerTraversal &layer,
                            size_t my_rank,
                            const BuilderExchangeLayout &selection_layout,
                            const VecI &selection_recv,
                            std::vector<char> &nodes_to_keep) -> void {
    for_each_remote_rank(layer, my_rank, [&selection_layout, &layer, &selection_recv, &nodes_to_keep](size_t rank) {
        const auto base = static_cast<size_t>(selection_layout.recv_displs[rank]);
        layer.for_each_cross_rank_sin_send_range(
            rank,
            0,
            layer.cross_rank_sin_send_size(rank),
            [&base, &selection_recv, &nodes_to_keep](size_t logical_idx, size_t src_idx) {
                const bool selected =
                    base + logical_idx < selection_recv.size() && selection_recv[base + logical_idx] != 0;
                if (selected && src_idx < nodes_to_keep.size()) {
                    nodes_to_keep[src_idx] = 1;
                }
            });
    });
}

} // namespace

// Prune the graph to the subgraph reaching the surviving output nodes: sweep the keep-set backward,
// emitting each layer unchanged (all cosines kept) or with its cosine list filtered. full_cos_of_layer
// supplies a layer's full cosine set lazily, so only filtered layers are materialized.
auto pare_graph(const MPGraph &graph,
                const VecZ &nonzero_inds,
                size_t local_index_count,
                bool schrodinger,
                mpi::Comm comm,
                const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph {
    const size_t num_layers = graph.layers();
    const int num_ranks = mpi::size(comm);
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));

    std::vector<char> nodes_to_keep(local_index_count, 0);
    for (auto idx : nonzero_inds) {
        if (idx < nodes_to_keep.size()) {
            nodes_to_keep[idx] = 1;
        }
    }

    std::vector<Layer> layers(num_layers);
    BuilderExchangeBuffers source_keep_buffers;
    BuilderExchangeBuffers selection_buffers;

    // Single backward sweep. Per cross-rank layer, two phases keep nodes_to_keep exact across ranks:
    // (1) decide surviving D entries + flag the partner B source needed; (2) after exchange, keep every
    // B source a partner selected (both endpoints agree). Cross-rank lists are NEVER pruned — they replay
    // unmasked at >1 rank; the exchange rounds only keep nodes_to_keep correct so cos pruning stays exact.
    for (size_t iter = 0; iter < num_layers; ++iter) {
        const size_t layer_idx = schrodinger ? iter : (num_layers - 1 - iter);
        const auto &layer = graph.get_layer(layer_idx);
        const auto lt = layer.traversal();
        const bool has_remote_cross_rank = has_remote_cross_rank_edges(lt, my_rank);
        BuilderExchangeLayout source_keep_layout;
        BuilderExchangeLayout selection_layout;

        if (num_ranks > 1) {
            // All ranks must call MPI_Alltoallv even without local remote edges (asymmetric
            // participation deadlocks); build an empty layout when there are none.
            if (has_remote_cross_rank) {
                source_keep_layout = build_builder_exchange_layout(lt, my_rank, BuilderExchangeDirection::Outgoing);
                selection_layout = build_builder_exchange_layout(lt, my_rank, BuilderExchangeDirection::Incoming);
            }
            else {
                const size_t rank_count = lt.cross_rank_rank_count();
                source_keep_layout = build_empty_builder_exchange_layout(rank_count);
                selection_layout = build_empty_builder_exchange_layout(rank_count);
            }
            resize_builder_exchange_buffers(source_keep_layout, source_keep_buffers);
            resize_builder_exchange_buffers(selection_layout, selection_buffers);
            if (has_remote_cross_rank) {
                pack_source_keep_flags(lt,
                                       nodes_to_keep,
                                       source_keep_layout,
                                       my_rank,
                                       source_keep_buffers.send_buffer);
            }
            // Round 1: exchange source-keep flags so each rank knows which remote D sources are kept.
            execute_builder_exchange(source_keep_layout, source_keep_buffers, comm);
        }
        else {
            source_keep_buffers.recv_buffer.clear();
            selection_buffers.send_buffer.clear();
            selection_buffers.recv_buffer.clear();
        }

        // Order is load-bearing for bit-exact pruning: mark_replayed_d_targets → cosine filter →
        // cross-rank D pass. The D targets are already forced kept by mark_replayed_d_targets, so the cos
        // filter sees the same nodes_to_keep regardless of D-pass order; keeping the sequencing is exact.
        mark_replayed_d_targets(lt, nodes_to_keep);

        // Materialize THIS layer's full cos lazily, prune to nodes_to_keep, discard it. preserves ⇒
        // nothing trimmed ⇒ emit a FoldLayer (cos recomputed at replay); else a PrunedLayer.
        const CosMask full = full_cos_of_layer(layer_idx);
        auto [filtered, preserves] = filter_layer_cosine_data(full, nodes_to_keep);

        // Phase 1: cross-rank D backward reachability; fills selection_buffers.send_buffer.
        if (has_remote_cross_rank) {
            propagate_cross_rank_d(lt,
                                   my_rank,
                                   source_keep_layout,
                                   source_keep_buffers.recv_buffer,
                                   selection_layout,
                                   selection_buffers.send_buffer,
                                   nodes_to_keep);
        }

        if (num_ranks > 1) {
            // Round 2: exchange selections, then keep the surviving B sources backward. Cross-rank
            // entries replay UNMASKED; this only keeps nodes_to_keep exact (no positions stored).
            execute_builder_exchange(selection_layout, selection_buffers, comm);
            if (has_remote_cross_rank) {
                propagate_cross_rank_b(lt, my_rank, selection_layout, selection_buffers.recv_buffer, nodes_to_keep);
            }
        }

        layers[layer_idx] = preserves ? Layer(layer.shared_core()) : Layer(layer.shared_core(), std::move(filtered));
    }

    return MPGraph(graph.is_schrodinger(), std::move(layers));
}

// Threshold wrapper over pare_graph: keep only the indices whose amplitude exceeds `threshold` in the
// relevant vector (the Hamiltonian in the Schrödinger picture, the state otherwise), then pare.
auto get_pared_graph(const VecD &state,
                     const VecD &hamiltonian,
                     double threshold,
                     const MPGraph &graph,
                     bool schrodinger,
                     mpi::Comm comm,
                     const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph {
    const auto &source = schrodinger ? hamiltonian : state;
    VecZ nonzero_inds;
    nonzero_inds.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        if (std::abs(source[i]) > threshold) {
            nonzero_inds.push_back(i);
        }
    }

    return pare_graph(graph, nonzero_inds, source.size(), schrodinger, comm, full_cos_of_layer);
}

} // namespace monoprop
