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

#include "monoprop/Evolution.h"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop {
namespace {

// Rotation-endpoint accumulators for the gradient identity: cos_terms = Σ s_old·h_old over the
// endpoints, sin_terms = Σ φ·s_old·h_partner.
struct EndpointContrib {
    double cos_terms = 0.0;
    double sin_terms = 0.0;
};

struct TrigValues {
    double cos_val;
    double sin_val;
    double sec_val;
    double g_val; // 2·gen_coeff
    double tan_val;

    explicit TrigValues(double param, double gen_coeff = 1.0) {
        const double g = 2.0 * gen_coeff;
        cos_val = std::cos(g * param);
        sin_val = std::sin(g * param);
        sec_val = 1.0 / cos_val;
        g_val = g;
        tan_val = sin_val * sec_val;
    }
};

auto combine_endpoint_contrib(const EndpointContrib &a, const EndpointContrib &b) -> EndpointContrib {
    return {.cos_terms = a.cos_terms + b.cos_terms, .sin_terms = a.sin_terms + b.sin_terms};
}

struct FlatExchangeBuffers {
    VecD send_buffer;
    VecD recv_buffer;
    // Derived per layer for the exchange being posted, and reused, so it allocates once per thread per
    // world size; one exchange in flight per thread, as send_buffer above has always required. ONE
    // layout, not two: it describes the recv side as well. See derive_layer_exchange.
    LayerExchangeLayout layout;
};

auto &acquire_flat_exchange_buffers() {
    struct Scratch {
        FlatExchangeBuffers buffers;
    };
    static thread_local Scratch scratch;
    return scratch.buffers;
}

void resize_flat_exchange_buffers(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers) {
    const size_t alloc = layout.total_count == 0 ? 1 : layout.total_count;
    buffers.send_buffer.resize(alloc);
}

// A property of the communicator, not the layer: all ranks participate even at local total_count 0.
auto layer_exchange_participates(const mpi::Comm &comm) -> bool {
    return mpi::size(comm) != 1;
}

// Derives both sides at once: the count matrix is symmetric, so the recv layout is the send layout.
auto derive_layer_exchange(const LayerTraversal &layer, const mpi::Comm &comm, int scale, FlatExchangeBuffers &buffers)
    -> void {
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));
    const char *what = scale == 1 ? "Layer exchange" : "Layer derivative exchange";
    detail::derive_exchange_layout(layer.cross_rank(), my_rank, scale, buffers.layout, what);
    mpi::check_exchange_layout_width(buffers.layout.counts, comm);
}

// The completed alltoallv payload as an apply pass sees it: peer `rank`'s entries start at
// recv_buffer[recv_displs[rank]], and `my_rank`'s own slot is absent (the self slot is handled locally).
struct ExchangePayload {
    const VecD &recv_buffer;
    const std::vector<int> &recv_displs;
    int my_rank;
};

// An empty ticket means nothing is in flight.
struct CrossRankExchangeHandle {
    const LayerExchangeLayout *layout = nullptr;
    FlatExchangeBuffers *buffers = nullptr;
    [[no_unique_address]] mpi::Ticket ticket;
};

inline auto begin_flat_exchange(FlatExchangeBuffers &buffers, const mpi::Comm &comm) -> CrossRankExchangeHandle {
    const LayerExchangeLayout &layout = buffers.layout;
    CrossRankExchangeHandle handle;
    handle.layout = &layout;
    handle.buffers = &buffers;
    buffers.recv_buffer.resize(layout.total_count == 0 ? 1 : layout.total_count);
    // Same arrays on both sides: MPI reads recvcounts/recvdispls and never writes them.
    handle.ticket = mpi::post_flat_alltoallv<double>({.send = buffers.send_buffer.data(),
                                                      .send_counts = layout.counts.data(),
                                                      .send_displs = layout.displs.data(),
                                                      .recv = buffers.recv_buffer.data(),
                                                      .recv_counts = layout.counts.data(),
                                                      .recv_displs = layout.displs.data()},
                                                     mpi::size(comm),
                                                     comm);
    return handle;
}

inline auto wait_flat_exchange(CrossRankExchangeHandle &handle) -> void {
    handle.ticket.wait();
}

// !active means the caller's participation guard declined and no transfer was posted.
struct InFlightExchange {
    CrossRankExchangeHandle handle;
    int my_rank = 0;
    bool active = false;
};

// Callers run layer_exchange_participates first, so no layout is derived for a transfer that never posts.
template <typename Pack>
inline auto begin_layer_exchange(const LayerTraversal &layer, int scale, const mpi::Comm &comm, Pack pack)
    -> InFlightExchange {
    InFlightExchange in_flight;
    in_flight.my_rank = mpi::rank(comm);
    in_flight.active = true;

    auto &buffers = acquire_flat_exchange_buffers();
    derive_layer_exchange(layer, comm, scale, buffers);
    resize_flat_exchange_buffers(buffers.layout, buffers);
    pack(in_flight.my_rank, buffers.layout, buffers.send_buffer);
    in_flight.handle = begin_flat_exchange(buffers, comm);
    return in_flight;
}

// An inactive round yields a default-constructed result without waiting.
template <typename Apply>
inline auto finish_layer_exchange(InFlightExchange &in_flight, Apply apply)
    -> std::invoke_result_t<Apply &, const ExchangePayload &> {
    using Result = std::invoke_result_t<Apply &, const ExchangePayload &>;
    if (!in_flight.active || in_flight.handle.layout == nullptr) {
        if constexpr (std::is_void_v<Result>) {
            return;
        }
        else {
            return Result{};
        }
    }
    wait_flat_exchange(in_flight.handle);
    return apply(ExchangePayload{.recv_buffer = in_flight.handle.buffers->recv_buffer,
                                 .recv_displs = in_flight.handle.buffers->layout.displs,
                                 .my_rank = in_flight.my_rank});
}

// Per-thread pre-cos snapshot buffers, reused across layers to avoid a malloc/free per layer. Passed whole
// to the pack and apply passes: each reads two of the four vectors, and re-splitting them at every hand-off
// is what lets a send buffer be mistaken for a recv one.
// Indexed by occupied position, not world slot, so nothing here grows with P.
struct DerivativeSnapshotScratch {
    std::vector<VecD> sin_send_state;
    std::vector<VecD> sin_send_op;
    std::vector<VecD> sin_recv_state;
    std::vector<VecD> sin_recv_op;
};

auto derivative_snapshot_scratch() -> DerivativeSnapshotScratch & {
    static thread_local DerivativeSnapshotScratch scratch;
    return scratch;
}

// Pack sin_send entries from the pre-cos snapshots; the live state/op there are clobbered by the cos pass.
void pack_cross_rank_derivative_payload_impl(const DerivativeSnapshotScratch &snap,
                                             const LayerTraversal &layer,
                                             int my_rank,
                                             const LayerExchangeLayout &layout,
                                             VecD &send_buffer) {
    layer.for_each_occupied_slot(
        [my_rank, &layout, &snap, &send_buffer](size_t pos, size_t rank, const detail::CrossRankSlotView &slot) {
            if (static_cast<int>(rank) == my_rank) {
                return;
            }
            // The send layout stays dense in the world; only the snapshot is indexed by position.
            const auto base = static_cast<size_t>(layout.displs[rank]);
            const auto &bs = snap.sin_send_state[pos];
            const auto &bh = snap.sin_send_op[pos];
            for (size_t k = 0; k < slot.sin_send_count; ++k) {
                send_buffer[base + 2 * k] = bs[k];
                send_buffer[base + 2 * k + 1] = bh[k];
            }
        });
}

// Remote endpoint pass: own pre-cos values come from the sin_recv snapshots, partner values from the
// received sin_send payload.
auto apply_cross_rank_derivative_exchange_impl(VecD &state,
                                               VecD &op,
                                               const LayerTraversal &layer,
                                               const DerivativeSnapshotScratch &snap,
                                               const TrigValues &trig,
                                               const ExchangePayload &payload) -> EndpointContrib {
    EndpointContrib local{};
    layer.for_each_occupied_slot(
        [&payload, &snap, &trig, &op, &state, &local](size_t pos, size_t rank, const detail::CrossRankSlotView &slot) {
            if (static_cast<int>(rank) == payload.my_rank) {
                return;
            }
            const auto *rv = payload.recv_buffer.data() + payload.recv_displs[rank];
            const auto &ds = snap.sin_recv_state[pos];
            const auto &dh = snap.sin_recv_op[pos];
            for (size_t k = 0; k < slot.sin_send_count; ++k) {
                const size_t i = detail::slot_sin_recv_index(slot, k);
                const auto phi = static_cast<double>(detail::slot_sin_recv_phase(slot, k));
                // Inverse-rotation write-back (−sin): un-evolves state/op for the next reverse layer.
                const double ps = -trig.sin_val * phi;
                const double s_old = ds[k];
                const double h_old = dh[k];
                const double s_p = rv[2 * k];
                const double h_p = rv[2 * k + 1];
                local.cos_terms += s_old * h_old;
                local.sin_terms += phi * s_old * h_p;
                op[i] = (h_old * trig.cos_val) + (ps * h_p);
                state[i] = (s_old * trig.cos_val) + (ps * s_p);
            }
        });
    return local;
}

// Pack + Ialltoallv fire up front so the transfer overlaps the cos pass and the self-slot.
inline auto begin_cross_rank_derivative_exchange(const DerivativeSnapshotScratch &snap,
                                                 const LayerTraversal &layer,
                                                 const mpi::Comm &comm) -> InFlightExchange {
    // Single-rank (or no peer participating): nothing to exchange — the self slot covers everything.
    if (!layer_exchange_participates(comm)) {
        return {};
    }
    // Safe to fire before the cos pass: pack reads pre-cos snapshots and the transfer touches only buffers.
    // Scale 2: each rotation endpoint carries both the op and the state payload.
    return begin_layer_exchange(layer,
                                2,
                                comm,
                                [&snap, &layer](int my_rank, const LayerExchangeLayout &layout, VecD &send_buffer) {
                                    pack_cross_rank_derivative_payload_impl(snap, layer, my_rank, layout, send_buffer);
                                });
}

// Must run after the cos pass — the ordering is floating-point significant.
inline auto finish_cross_rank_derivative_exchange(VecD &state,
                                                  VecD &op,
                                                  const LayerTraversal &layer,
                                                  const DerivativeSnapshotScratch &snap,
                                                  const TrigValues &trig,
                                                  InFlightExchange &in_flight) -> EndpointContrib {
    return finish_layer_exchange(in_flight, [&state, &op, &layer, &snap, &trig](const ExchangePayload &payload) {
        return apply_cross_rank_derivative_exchange_impl(state, op, layer, snap, trig, payload);
    });
}

void pack_cross_rank_evolution_payload_impl(VecD &op,
                                            const LayerTraversal &layer,
                                            int my_rank,
                                            const LayerExchangeLayout &layout,
                                            VecD &send_buffer) {
    layer.for_each_occupied_slot(
        [my_rank, &layout, &send_buffer, &op](size_t rank, const detail::CrossRankSlotView &slot) {
            if (static_cast<int>(rank) == my_rank) {
                return;
            }
            const auto base = static_cast<size_t>(layout.displs[rank]);
            for (size_t k = 0; k < slot.sin_send_count; ++k) {
                send_buffer[base + k] = op[detail::slot_sin_send_index(slot, k)];
            }
        });
}

void apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              double sin_val,
                                              const ExchangePayload &payload) {
    // op[i] is already cos-scaled, so only the sine term is added; rv[k] is the partner's pre-cos value.
    layer.for_each_occupied_slot([&payload, sin_val, &op](size_t rank, const detail::CrossRankSlotView &slot) {
        if (static_cast<int>(rank) == payload.my_rank) {
            return;
        }
        const auto *rv = payload.recv_buffer.data() + payload.recv_displs[rank];
        for (size_t k = 0; k < slot.sin_send_count; ++k) {
            const size_t i = detail::slot_sin_recv_index(slot, k);
            op[i] += sin_val * static_cast<double>(detail::slot_sin_recv_phase(slot, k)) * rv[k];
        }
    });
}

inline auto begin_cross_rank_evolution_exchange(VecD &op, const LayerTraversal &layer, const mpi::Comm &comm)
    -> InFlightExchange {
    if (!layer_exchange_participates(comm)) {
        return {};
    }
    return begin_layer_exchange(
        layer,
        1,
        comm,
        [&op, &layer](int my_rank, const LayerExchangeLayout &active_layout, VecD &send_buffer) {
            pack_cross_rank_evolution_payload_impl(op, layer, my_rank, active_layout, send_buffer);
        });
}

inline auto finish_cross_rank_evolution_exchange(VecD &op,
                                                 const LayerTraversal &layer,
                                                 double sin_val,
                                                 InFlightExchange &in_flight) -> void {
    finish_layer_exchange(in_flight, [&op, &layer, sin_val](const ExchangePayload &payload) {
        apply_cross_rank_evolution_exchange_impl(op, layer, sin_val, payload);
    });
}

// Snapshot-free self-slot endpoint pass: sin_recv entries k and k+pairs are the two endpoints of one
// rotation, so reading both (pre-cos recovered from the post-cos slots) before writing either avoids the
// read-after-write hazard.
auto apply_self_slot_derivative_paired(VecD &state, VecD &op, const LayerTraversal &layer, const TrigValues &trig)
    -> EndpointContrib {
    // O(1) and rank-free: the layer records where its own slot sits at build time.
    const auto slot = layer.cross_rank_self_slot();
    const size_t self_d_count = slot.sin_send_count;
    if (self_d_count == 0) {
        return {};
    }
    const auto pairs = self_d_count / 2;
    EndpointContrib local{};
    for (size_t k = 0; k < pairs; ++k) {
        const size_t i1 = detail::slot_sin_recv_index(slot, k);
        const auto phi1 = static_cast<double>(detail::slot_sin_recv_phase(slot, k));
        const size_t i2 = detail::slot_sin_recv_index(slot, k + pairs);
        const auto phi2 = static_cast<double>(detail::slot_sin_recv_phase(slot, k + pairs));
        // Recover pre-cos values.
        const double s1 = state[i1] * trig.sec_val;
        const double h1 = op[i1] * trig.cos_val;
        const double s2 = state[i2] * trig.sec_val;
        const double h2 = op[i2] * trig.cos_val;
        local.cos_terms += (s1 * h1) + (s2 * h2);
        local.sin_terms += (phi1 * s1 * h2) + (phi2 * s2 * h1);
        // Inverse-rotation write-back (−sin); see apply_cross_rank_derivative_exchange_impl.
        const double ps1 = -trig.sin_val * phi1;
        const double ps2 = -trig.sin_val * phi2;
        op[i1] = (h1 * trig.cos_val) + (ps1 * h2);
        state[i1] = (s1 * trig.cos_val) + (ps1 * s2);
        op[i2] = (h2 * trig.cos_val) + (ps2 * h1);
        state[i2] = (s2 * trig.cos_val) + (ps2 * s1);
    }
    return local;
}

// The self slot recovers live, so it is emptied rather than filled: a previous layer's snapshot at
// this position must not be readable.
void clear_slot_snapshot(DerivativeSnapshotScratch &snap, size_t pos) {
    snap.sin_send_state[pos].clear();
    snap.sin_send_op[pos].clear();
    snap.sin_recv_state[pos].clear();
    snap.sin_recv_op[pos].clear();
}

void fill_slot_snapshot(DerivativeSnapshotScratch &snap,
                        size_t pos,
                        const VecD &state,
                        const VecD &op,
                        const detail::CrossRankSlotView &slot) {
    const size_t count = slot.sin_send_count;
    auto &bs = snap.sin_send_state[pos];
    auto &bh = snap.sin_send_op[pos];
    auto &ds = snap.sin_recv_state[pos];
    auto &dh = snap.sin_recv_op[pos];
    bs.resize(count);
    bh.resize(count);
    ds.resize(count);
    dh.resize(count);
    for (size_t k = 0; k < count; ++k) {
        const size_t bi = detail::slot_sin_send_index(slot, k);
        bs[k] = state[bi];
        bh[k] = op[bi];
        const size_t di = detail::slot_sin_recv_index(slot, k);
        ds[k] = state[di];
        dh[k] = op[di];
    }
}

// Snapshot pre-cos (state, op) at every remote rank's sin_send/sin_recv endpoints before the cos pass
// clobbers them. The self slot needs none — it recovers live — so it is cleared.
void snapshot_remote_endpoints(const VecD &state,
                               const VecD &op,
                               const LayerTraversal &layer,
                               size_t my_rank,
                               DerivativeSnapshotScratch &snap) {
    // Sized by the slots that carry traffic. Grow-only: shrinking would free the allocations this
    // scratch exists to reuse, and positions past the occupancy are never visited.
    const size_t occupied = layer.occupied_slot_count();
    const auto grow = [occupied](std::vector<VecD> &v) {
        if (v.size() < occupied) {
            v.resize(occupied);
        }
    };
    grow(snap.sin_send_state);
    grow(snap.sin_send_op);
    grow(snap.sin_recv_state);
    grow(snap.sin_recv_op);
    layer.for_each_occupied_slot(
        [&snap, my_rank, &state, &op](size_t pos, size_t r, const detail::CrossRankSlotView &slot) {
            if (r == my_rank) {
                clear_slot_snapshot(snap, pos);
                return;
            }
            fill_slot_snapshot(snap, pos, state, op, slot);
        });
}

} // namespace

auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraphView &graph,
                                     size_t layer_idx,
                                     LayerAngle angle,
                                     mpi::Comm comm,
                                     const detail::LayerCosAccumulate &cos_acc) -> double {
    const TrigValues trig(angle.param, angle.gen_coeff);
    const auto layer = graph.get_layer_traversal(layer_idx);
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = layer.cross_rank_rank_count();

    auto &snap = derivative_snapshot_scratch();
    snapshot_remote_endpoints(state, op, layer, my_rank, snap);

    // No-op at single rank; the transfer touches only buffers, so the cos pass below may mutate state/op.
    auto in_flight = begin_cross_rank_derivative_exchange(snap, layer, comm);

    // A = Σ s_old·h_old over all anticommuting indices, endpoints included — hence the subtraction below.
    const double A = cos_acc(layer_idx, state.data(), op.data(), trig.cos_val, trig.sec_val);

    EndpointContrib ep;
    if (my_rank < R) {
        ep = apply_self_slot_derivative_paired(state, op, layer, trig);
    }
    const auto remote = finish_cross_rank_derivative_exchange(state, op, layer, snap, trig, in_flight);
    ep = combine_endpoint_contrib(ep, remote);

    // dE/dθ = −g·(tan·(A − ep.cos_terms) − ep.sin_terms); note the plus sign on ep.sin_terms.
    return -trig.g_val * (trig.tan_val * (A - ep.cos_terms) - ep.sin_terms);
}

auto evolve_step_traversal_impl(VecD &op,
                                const LayerTraversal &layer,
                                double param,
                                size_t layer_idx,
                                const mpi::Comm &comm,
                                const detail::LayerCosScale &cos_scale) -> void {
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);

    auto *const op_data = op.data();

    // Snapshot this rank's own sin_send values before the cos pass; runs unconditionally (the remote
    // pack skips the self slot) so single-rank works. Rank-free: the layer records where its slot sits.
    const auto self_slot = layer.cross_rank_self_slot();
    const size_t self_b_count = self_slot.sin_send_count;
    VecD self_b_snapshot;
    self_b_snapshot.resize(self_b_count);
    for (size_t k = 0; k < self_b_count; ++k) {
        self_b_snapshot[k] = op[detail::slot_sin_send_index(self_slot, k)];
    }

    // Pack + start the exchange before the cos scan so partner values are pre-cos and the transfer overlaps.
    auto in_flight = begin_cross_rank_evolution_exchange(op, layer, comm);
    cos_scale(layer_idx, op_data, cos_val);
    finish_cross_rank_evolution_exchange(op, layer, sin_val, in_flight);

    // Self-slot sin_recv entries: op[i] is already cos-scaled, so only the sine term is added. B and D
    // have the same count, so self_b_count serves both.
    for (size_t k = 0; k < self_b_count; ++k) {
        const size_t i = detail::slot_sin_recv_index(self_slot, k);
        op[i] += sin_val * static_cast<double>(detail::slot_sin_recv_phase(self_slot, k)) * self_b_snapshot[k];
    }
}

auto evolve_step_impl(VecD &op,
                      const MPGraphView &graph,
                      double param,
                      size_t layer_idx,
                      const mpi::Comm &comm,
                      const detail::LayerCosScale &cos_scale) -> void {
    evolve_step_traversal_impl(op, graph.get_layer_traversal(layer_idx), param, layer_idx, comm, cos_scale);
}

// A standalone Layer replays as a one-layer graph, so layer_idx 0 is the only cosine set to select.
auto evolve_step(VecD &op, const Layer &layer, double param, mpi::Comm comm, const detail::LayerCosScale &cos_scale)
    -> void {
    evolve_step_traversal_impl(op, layer.traversal(), param, 0, comm, cos_scale);
}

auto evolve_operator(VecD &&coeffs,
                     const MPGraphView &graph,
                     const VecD &params,
                     mpi::Comm comm,
                     const detail::LayerCosScale &cos_scale) -> VecD {
    // Evolved in place in the caller's moved-from vector, then handed back: no per-layer copy.
    VecD evolved = std::move(coeffs);
    for (size_t i = 0; i < graph.layers(); ++i) {
        evolve_step_impl(evolved, graph, params[i], i, comm, cos_scale);
    }
    return evolved;
}

} // namespace monoprop
