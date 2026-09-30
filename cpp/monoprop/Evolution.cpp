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

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/evolution/LayerReplay.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/OperationFailure.h"

namespace monoprop {
namespace {

using detail::replay::DerivativeSnapshotScratch;
using detail::replay::EndpointContrib;
using detail::replay::TrigValues;

struct FlatExchangeBuffers {
    VecD send_buffer;
    VecD recv_buffer;
    LayerExchangeLayout layout;
};

auto &acquire_flat_exchange_buffers() {
    struct Scratch {
        FlatExchangeBuffers buffers;
    };
    static thread_local Scratch scratch;
    return scratch.buffers;
}

// A property of the communicator, not the layer: all ranks participate even at local total_count 0.
// Also under the pairwise arm: a rank that skips the round strands the others either way.
auto layer_exchange_participates(const mpi::Comm &comm) -> bool {
    return mpi::size(comm) != 1;
}

// Derives both sides at once: the count matrix is symmetric, so the recv layout is the send layout.
auto derive_layer_exchange(const LayerTraversal &layer, const mpi::Comm &comm, int scale, LayerExchangeLayout &layout)
    -> void {
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));
    const char *what = scale == 1 ? "Layer exchange" : "Layer derivative exchange";
    detail::derive_exchange_layout(layer.cross_rank(), my_rank, scale, layout, what);
    mpi::check_exchange_layout_width(layout.counts, comm);
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
                                                     comm,
                                                     mpi::routes_pairwise(comm));
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

template <typename Pack>
inline auto begin_layer_exchange(const LayerTraversal &layer, int scale, const mpi::Comm &comm, Pack pack)
    -> InFlightExchange {
    InFlightExchange in_flight;
    in_flight.my_rank = mpi::rank(comm);
    in_flight.active = true;

    auto &buffers = acquire_flat_exchange_buffers();
    derive_layer_exchange(layer, comm, scale, buffers.layout);
    buffers.send_buffer.resize(buffers.layout.total_count == 0 ? 1 : buffers.layout.total_count);
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

auto derivative_snapshot_scratch() -> DerivativeSnapshotScratch & {
    static thread_local DerivativeSnapshotScratch scratch;
    return scratch;
}

// Pack sin_send entries from the pre-cos snapshots; the live state/op there are clobbered by the cos pass.
// The send layout stays dense in the world; only the snapshot is indexed by position.
void pack_cross_rank_derivative_payload_impl(const DerivativeSnapshotScratch &snap,
                                             const LayerTraversal &layer,
                                             int my_rank,
                                             const LayerExchangeLayout &layout,
                                             VecD &send_buffer) {
    detail::replay::pack_derivative_payload(snap, layer, static_cast<size_t>(my_rank), [&](size_t rank) {
        return send_buffer.data() + layout.displs[rank];
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
    return detail::replay::apply_derivative_payload(
        state,
        op,
        layer,
        snap,
        trig,
        static_cast<size_t>(payload.my_rank),
        [&](size_t rank) { return payload.recv_buffer.data() + payload.recv_displs[rank]; });
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
    detail::replay::pack_evolution_payload(op, layer, static_cast<size_t>(my_rank), [&](size_t rank) {
        return send_buffer.data() + layout.displs[rank];
    });
}

void apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              double sin_val,
                                              const ExchangePayload &payload) {
    detail::replay::apply_evolution_payload(op, layer, sin_val, static_cast<size_t>(payload.my_rank), [&](size_t rank) {
        return payload.recv_buffer.data() + payload.recv_displs[rank];
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

} // namespace

auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraphView &graph,
                                     size_t layer_idx,
                                     LayerAngle angle,
                                     mpi::Comm comm,
                                     const detail::LayerCosAccumulate &cos_acc,
                                     const detail::CosRecordView &record,
                                     [[maybe_unused]] detail::parallel::Options options) -> double {
    const TrigValues trig(angle.param, angle.gen_coeff);
    const auto layer = graph.get_layer_traversal(layer_idx);
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = layer.cross_rank_rank_count();

    auto &snap = derivative_snapshot_scratch();
    // Snapshots and packing may throw on this rank alone before it posts: guarded, since peers post regardless.
    auto in_flight = mpi::guard_distributed(comm, [&] {
        detail::replay::snapshot_remote_endpoints(state, op, layer, my_rank, snap);
        // The self slot reads its entry op off the post-cos slots, which a record overwrites first.
        if (record.count > 0 && my_rank < R) {
            detail::replay::snapshot_self_recv_op(op, layer, snap);
        }
        // No-op at single rank; the transfer touches only buffers, so the cos pass below may mutate state/op.
        return begin_cross_rank_derivative_exchange(snap, layer, comm);
    });
    const bool self_pre = record.count > 0 && my_rank < R;

    // Caught here, while in_flight still owns posted requests: unwinding past it would wait on peers.
    EndpointContrib ep;
    const double A = mpi::guard_distributed(comm, [&] {
        const double a = detail::replay::accumulate_layer(state, op, layer_idx, trig, record, cos_acc);
        if (my_rank < R) {
            ep = detail::replay::apply_self_slot_derivative_paired(state,
                                                                   op,
                                                                   layer,
                                                                   trig,
                                                                   self_pre ? snap.self_recv_op.data() : nullptr);
        }
        return a;
    });
    const auto remote = finish_cross_rank_derivative_exchange(state, op, layer, snap, trig, in_flight);
    ep = detail::replay::combine_endpoint_contrib(ep, remote);

    // Note the plus sign in layer_derivative. Both sums are in pre-layer units, so the cancellation happens
    // before any ×sec rather than after it.
    return detail::replay::layer_derivative(trig, A, ep);
}

namespace {
auto evolve_step_traversal_impl(VecD &op,
                                const LayerTraversal &layer,
                                double param,
                                size_t layer_idx,
                                const mpi::Comm &comm,
                                const detail::LayerCosScale &cos_scale) -> void {
    const double cos_val = std::cos(2 * param);
    const double sin_val = std::sin(2 * param);

    auto *const op_data = op.data();

    // This rank's own sin_send values before the cos pass. Unconditional, since the remote pack skips
    // the self slot, so single-rank works.
    VecD self_b_snapshot;
    // Pack + start the exchange before the cos scan so partner values are pre-cos and the transfer overlaps.
    // A throw before posting is guarded too: the peers post and wait regardless.
    auto in_flight = mpi::guard_distributed(comm, [&] {
        detail::replay::snapshot_self_sources(op, layer, self_b_snapshot);
        return begin_cross_rank_evolution_exchange(op, layer, comm);
    });
    // Caught here, while in_flight still owns posted requests: unwinding past it would wait on peers.
    mpi::guard_distributed(comm, [&] { cos_scale(layer_idx, op_data, cos_val); });
    finish_cross_rank_evolution_exchange(op, layer, sin_val, in_flight);

    // Self-slot sin_recv entries: op[i] is already cos-scaled, so only the sine term is added.
    detail::replay::apply_self_sources(op, layer, sin_val, self_b_snapshot);
}
} // namespace

// The layer's cosine scaling runs threaded through the budget captured in `cos_scale`; the endpoint work
// below it is still serial, so `options` itself is not yet consumed at this level.
auto evolve_step(VecD &op,
                 const MPGraphView &graph,
                 double param,
                 size_t layer_idx,
                 mpi::Comm comm,
                 const detail::LayerCosScale &cos_scale,
                 [[maybe_unused]] detail::parallel::Options options) -> void {
    evolve_step_traversal_impl(op, graph.get_layer_traversal(layer_idx), param, layer_idx, comm, cos_scale);
}

// A standalone Layer replays as a one-layer graph, so layer_idx 0 is the only cosine set to select.
auto evolve_step(VecD &op,
                 const Layer &layer,
                 double param,
                 mpi::Comm comm,
                 const detail::LayerCosScale &cos_scale,
                 [[maybe_unused]] detail::parallel::Options options) -> void {
    evolve_step_traversal_impl(op, layer.traversal(), param, 0, comm, cos_scale);
}

auto evolve_operator(VecD &&coeffs,
                     const MPGraphView &graph,
                     const VecD &params,
                     mpi::Comm comm,
                     const detail::LayerCosScale &cos_scale,
                     detail::parallel::Options options) -> VecD {
    // Evolved in place in the caller's moved-from vector, then handed back: no per-layer copy.
    VecD evolved = std::move(coeffs);
    for (size_t i = 0; i < graph.layers(); ++i) {
        evolve_step(evolved, graph, params[i], i, comm, cos_scale, options);
    }
    return evolved;
}

} // namespace monoprop
