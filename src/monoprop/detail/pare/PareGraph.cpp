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

// ── Cross-rank exchange layout. The exchange ROUNDS are load-bearing for multi-rank correctness
// even though we store NO positions — they propagate the keep-set across ranks. ──

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

inline auto cross_rank_sin_send_size(const Layer &layer, size_t rank) -> size_t {
    return layer.cross_rank_sin_send_size(rank);
}
inline auto cross_rank_sin_recv_size(const Layer &layer, size_t rank) -> size_t {
    return layer.cross_rank_sin_recv_size(rank);
}

template <typename Func>
auto for_each_remote_rank(const Layer &layer, size_t my_rank, Func &&func) -> void {
    for (size_t rank = 0; rank < layer.cross_rank_rank_count(); ++rank) {
        if (rank != my_rank) {
            func(rank);
        }
    }
}

auto has_remote_cross_rank_edges(const Layer &layer, size_t my_rank) -> bool {
    bool has_remote_edges = false;
    for_each_remote_rank(layer, my_rank, [&](size_t rank) {
        if (cross_rank_sin_send_size(layer, rank) != 0 || cross_rank_sin_recv_size(layer, rank) != 0) {
            has_remote_edges = true;
        }
    });
    return has_remote_edges;
}

auto build_builder_exchange_layout(const Layer &layer, size_t my_rank, BuilderExchangeDirection direction)
    -> BuilderExchangeLayout {
    BuilderExchangeLayout layout;
    layout.send_counts.resize(layer.cross_rank_rank_count(), 0);
    layout.send_displs.resize(layer.cross_rank_rank_count(), 0);
    layout.recv_counts.resize(layer.cross_rank_rank_count(), 0);
    layout.recv_displs.resize(layer.cross_rank_rank_count(), 0);

    size_t total_send = 0;
    size_t total_recv = 0;
    for_each_remote_rank(layer, my_rank, [&](size_t rank) {
        // In this layout the "outgoing" direction maps to B (send) and the "incoming" to D (recv).
        const size_t send_count = direction == BuilderExchangeDirection::Outgoing
                                      ? cross_rank_sin_send_size(layer, rank)
                                      : cross_rank_sin_recv_size(layer, rank);
        const size_t recv_count = direction == BuilderExchangeDirection::Outgoing
                                      ? cross_rank_sin_recv_size(layer, rank)
                                      : cross_rank_sin_send_size(layer, rank);
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
    // Always allocate at least 1 element so data() is never nullptr (some MPI
    // implementations reject nullptr send/recv buffers even for zero-count calls).
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

auto pack_source_keep_flags(const Layer &layer,
                            const std::vector<char> &nodes_to_keep,
                            const BuilderExchangeLayout &layout,
                            size_t my_rank,
                            VecI &send_buffer) -> void {
    for_each_remote_rank(layer, my_rank, [&](size_t rank) {
        const size_t base = static_cast<size_t>(layout.send_displs[rank]);
        const size_t count = cross_rank_sin_send_size(layer, rank);
        layer.for_each_cross_rank_sin_send_range(rank, 0, count, [&](size_t logical_idx, size_t src_idx) {
            send_buffer[base + logical_idx] = (src_idx < nodes_to_keep.size() && nodes_to_keep[src_idx]) ? 1 : 0;
        });
    });
}

auto execute_builder_exchange(const BuilderExchangeLayout &layout, BuilderExchangeBuffers &buffers, mpi::Comm comm)
    -> void {
    // Blocking one-shot keep-flag exchange. Recv counts are known locally (the per-rank transpose of
    // the send counts), so no count round is needed — just post + wait via the facade, which also
    // owns the "all ranks participate / never skip on zero counts" deadlock discipline. Buffers are
    // always sized >= 1 (see resize_builder_exchange_buffers).
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

// ── Cosine filter ─────────────────────────────────────────────────────────────
// Filter the full cosine set to the kept nodes. Returns {{}, true} when nothing was pruned (the
// replay then folds the full set); otherwise {filtered, false}. Blocks are independent, so large
// layers filter in parallel over block ranges (per-range local lists concatenated in order).
inline auto keep_mask_for_block(const std::vector<char> &keep, size_t base, uint64_t present) -> uint64_t {
    uint64_t mask = 0;
    uint64_t b = present;
    while (b) {
        const size_t t = static_cast<size_t>(std::countr_zero(b));
        const size_t idx = base + t;
        if (idx < keep.size() && keep[idx] != 0) {
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

// Every D rotation the pared replay applies (`op[i] += sin·φ·partner`) requires its target i to have
// been cos-scaled first, because cos now holds ALL anticommuting indices (endpoints included) and the
// cos pass — not the D-apply — performs that scaling. The self slot (my_rank) is never pruned
// (for_each_remote_rank skips it) and cross-rank D replays unmasked at >1 rank, so EVERY D target is
// replayed. Mark them all into nodes_to_keep BEFORE the cosine filter runs so the pruned cos keeps
// them (and so backward reachability keeps their pre-cos producers too).
auto mark_replayed_d_targets(const Layer &layer, std::vector<char> &nodes_to_keep) -> void {
    const size_t rank_count = layer.cross_rank_rank_count();
    for (size_t rank = 0; rank < rank_count; ++rank) {
        layer.for_each_cross_rank_sin_recv_range(rank,
                                                 0,
                                                 cross_rank_sin_recv_size(layer, rank),
                                                 [&](size_t /*logical_idx*/, size_t tgt_idx, int) {
                                                     if (tgt_idx < nodes_to_keep.size()) {
                                                         nodes_to_keep[tgt_idx] = 1;
                                                     }
                                                 });
    }
}

// Phase 1 (before the selection exchange): propagate the keep-set across this rank's cross-rank D
// entries by backward reachability and record per-edge selection flags. A D entry survives if its
// target (local update index) is active, or the remote source is active (keep_src). For every
// surviving D entry we set a selection flag telling the partner we need its B source — the
// authoritative per-edge keep signal the partner uses to filter its matching B entry, guaranteeing
// both endpoints of a cross-rank edge agree. NO positions are stored: this only mutates
// nodes_to_keep and the selection send buffer.
auto propagate_cross_rank_d(const Layer &layer,
                            size_t my_rank,
                            const BuilderExchangeLayout &source_keep_layout,
                            const VecI &remote_src_keep,
                            const BuilderExchangeLayout &selection_layout,
                            VecI &selected_incoming_flags,
                            std::vector<char> &nodes_to_keep) -> void {
    // Keep the buffer sized >= 1 so execute_builder_exchange can post a non-null send pointer even
    // when this rank has outgoing-only cross-rank edges (total_send == 0); the padding slot is never
    // indexed by a real edge nor sent (send_counts sum to total_send). Mirrors resize_builder_exchange_buffers.
    selected_incoming_flags.assign(std::max<size_t>(1, selection_layout.total_send), 0);
    for_each_remote_rank(layer, my_rank, [&](size_t rank) {
        const size_t remote_base = static_cast<size_t>(source_keep_layout.recv_displs[rank]);
        const size_t notify_base = static_cast<size_t>(selection_layout.send_displs[rank]);
        layer.for_each_cross_rank_sin_recv_range(
            rank,
            0,
            cross_rank_sin_recv_size(layer, rank),
            [&](size_t logical_idx, size_t tgt_idx, int) {
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
    });
}

// Phase 2 (after the selection exchange): for each cross-rank B entry whose D the partner selected,
// mark its source node so its own producers are kept in earlier (later-processed) layers. NO
// positions are stored.
auto propagate_cross_rank_b(const Layer &layer,
                            size_t my_rank,
                            const BuilderExchangeLayout &selection_layout,
                            const VecI &selection_recv,
                            std::vector<char> &nodes_to_keep) -> void {
    for_each_remote_rank(layer, my_rank, [&](size_t rank) {
        const size_t base = static_cast<size_t>(selection_layout.recv_displs[rank]);
        layer.for_each_cross_rank_sin_send_range(rank,
                                                 0,
                                                 cross_rank_sin_send_size(layer, rank),
                                                 [&](size_t logical_idx, size_t src_idx) {
                                                     const bool selected = base + logical_idx < selection_recv.size()
                                                                           && selection_recv[base + logical_idx] != 0;
                                                     if (selected && src_idx < nodes_to_keep.size()) {
                                                         nodes_to_keep[src_idx] = 1;
                                                     }
                                                 });
    });
}

} // namespace

// Prune the graph to the subgraph that can reach the surviving output nodes (nonzero_inds): sweep the
// keep-set backward through the layers, then emit each layer either unchanged (all cosines kept) or
// with its cosine list filtered to the kept subset. full_cos_of_layer supplies a layer's full cosine
// set lazily, so only the layers actually being filtered are materialized.
auto pare_graph(const MPGraph &graph,
                const VecZ &nonzero_inds,
                size_t local_index_count,
                bool schrodinger,
                mpi::Comm comm,
                const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph {
    const size_t num_layers = graph.layers();
    const int num_ranks = mpi::size(comm);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));

    std::vector<char> nodes_to_keep(local_index_count, 0);
    for (auto idx : nonzero_inds) {
        if (idx < nodes_to_keep.size()) {
            nodes_to_keep[idx] = 1;
        }
    }

    std::vector<Layer> layers(num_layers);
    BuilderExchangeBuffers source_keep_buffers;
    BuilderExchangeBuffers selection_buffers;

    // Single backward sweep. Per cross-rank layer the keep-set crosses ranks in two phases:
    //   (1) D-phase: each rank decides which of its cross-rank D entries survive (backward
    //       reachability) and records a per-edge selection flag for the partner B source it needs;
    //   (2) after exchanging selections, each rank keeps the source of every B entry a partner
    //       selected.
    // Keying B-entry survival on the partner's selection (not on whether the source node happens to
    // be kept for some other reason) makes both endpoints of every cross-rank edge agree. Marking
    // the surviving source nodes then propagates the dependency to earlier (later-processed) layers,
    // so one reverse sweep suffices — matching the single-rank pure-backward-reachability semantics.
    // The cross-rank lists themselves are NEVER pruned here (cross-rank replays unmasked at >1 rank);
    // the exchange rounds exist only to keep nodes_to_keep correct across ranks so the per-layer cos
    // pruning stays exact.
    for (size_t iter = 0; iter < num_layers; ++iter) {
        const size_t layer_idx = schrodinger ? iter : (num_layers - 1 - iter);
        const auto &layer = graph.get_layer(layer_idx);
        const bool has_remote_cross_rank = has_remote_cross_rank_edges(layer, my_rank);
        BuilderExchangeLayout source_keep_layout;
        BuilderExchangeLayout selection_layout;

        if (num_ranks > 1) {
            // All ranks must participate in MPI_Alltoallv regardless of whether this rank has local
            // remote edges. Asymmetric participation (one rank skips while another calls) causes a
            // deadlock. Build an empty layout for ranks with no local remote edges so counts arrays
            // are sized correctly.
            if (has_remote_cross_rank) {
                source_keep_layout = build_builder_exchange_layout(layer, my_rank, BuilderExchangeDirection::Outgoing);
                selection_layout = build_builder_exchange_layout(layer, my_rank, BuilderExchangeDirection::Incoming);
            }
            else {
                const size_t rank_count = layer.cross_rank_rank_count();
                source_keep_layout = build_empty_builder_exchange_layout(rank_count);
                selection_layout = build_empty_builder_exchange_layout(rank_count);
            }
            resize_builder_exchange_buffers(source_keep_layout, source_keep_buffers);
            resize_builder_exchange_buffers(selection_layout, selection_buffers);
            if (has_remote_cross_rank) {
                pack_source_keep_flags(layer,
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

        // Order is load-bearing for bit-exact pruning:
        //   mark_replayed_d_targets → cosine filter → cross-rank D pass (phase 1).
        // The D pass also sets nodes_to_keep for surviving cross-rank D targets, but those targets
        // were already forced kept by mark_replayed_d_targets, so the cos filter sees the same
        // nodes_to_keep whether the D pass runs before or after it — except for the keep_src
        // backward marks (D source nodes feed EARLIER layers, never this layer's own cos). Keeping
        // the original sequencing makes this bit-exact by construction.
        mark_replayed_d_targets(layer, nodes_to_keep);

        // Cosine filter: materialize THIS layer's full cos lazily, prune to nodes_to_keep, discard
        // the full set immediately. `preserves` ⇒ nothing trimmed ⇒ emit a FoldLayer (cos recomputed
        // at replay); otherwise emit a PrunedLayer carrying the trimmed list.
        const CosMask full = full_cos_of_layer(layer_idx);
        auto [filtered, preserves] = filter_layer_cosine_data(full, nodes_to_keep);

        // Phase 1: cross-rank D backward reachability; fills selection_buffers.send_buffer.
        if (has_remote_cross_rank) {
            propagate_cross_rank_d(layer,
                                   my_rank,
                                   source_keep_layout,
                                   source_keep_buffers.recv_buffer,
                                   selection_layout,
                                   selection_buffers.send_buffer,
                                   nodes_to_keep);
        }

        if (num_ranks > 1) {
            // Round 2: exchange selections, then propagate the surviving B sources backward (exact
            // per-edge agreement). Cross-rank entries replay UNMASKED in multi-rank — the cross-rank
            // D/B propagation above is still required (it keeps nodes_to_keep exact across ranks so
            // the cosine pruning stays exact) but no positions are stored.
            execute_builder_exchange(selection_layout, selection_buffers, comm);
            if (has_remote_cross_rank) {
                propagate_cross_rank_b(layer, my_rank, selection_layout, selection_buffers.recv_buffer, nodes_to_keep);
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
