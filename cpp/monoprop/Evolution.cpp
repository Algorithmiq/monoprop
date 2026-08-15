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
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    // The send layout for the exchange currently being posted, derived per layer rather than
    // read from one retained per layer. Reused, so it allocates once per thread per world size.
    // Sharing one instance across layers is only sound because at most one exchange is in flight
    // per thread -- the same invariant send_buffer above has always required.
    LayerExchangeLayout layout;
    int recv_total = 0;
};

auto &acquire_flat_exchange_buffers() {
    struct Scratch {
        FlatExchangeBuffers buffers;
    };
    static thread_local Scratch scratch;
    return scratch.buffers;
}

void resize_flat_exchange_buffers(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers) {
    // Only the send side: the recv counts/displs are filled by derive_layer_exchange before this
    // runs, so clearing them here (as this did when the recv side was resolved afterwards) would
    // throw away the layout the transfer is about to be posted with.
    const size_t send_alloc = layout.total_count == 0 ? 1 : layout.total_count;
    buffers.send_buffer.resize(send_alloc);
}

// Nothing to exchange at one rank. All ranks must participate even at local total_count 0, else
// MPI_Alltoallv deadlocks, so this is a property of the communicator and not of the layer.
auto layer_exchange_participates(const mpi::Comm &comm) -> bool {
    return mpi::size(comm) != 1;
}

// Derive this layer's send layout into `buffers.layout` at `scale`, and its recv side into
// buffers.recv_counts/recv_displs at the same scale.
//
// The recv side is resolved ONCE per layer, at scale 1, into the layer's own cache. Scaling
// commutes with the transpose -- every rank multiplies its counts by the same literal, so peer q
// sends me exactly `scale` times what it sent at scale 1, and displacements are prefix sums of
// counts so they scale with them. That is what lets the derivative round reuse the evolution
// round's collective instead of paying an alltoall of its own.
auto derive_layer_exchange(const LayerTraversal &layer,
                           const mpi::Comm &comm,
                           int scale,
                           FlatExchangeBuffers &buffers) -> void {
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));

    // Resolve the transpose at scale 1, so evolution and derivative rounds share one cache entry
    // and therefore one collective.
    detail::derive_exchange_layout(layer.cross_rank(), my_rank, 1, buffers.layout);
    const auto &recv =
        mpi::resolve_recv(buffers.layout.counts, comm, layer.evolution_recv_cache(), layer.exchange_generation());

    const size_t n = recv.counts.size();
    buffers.recv_counts.resize(n);
    buffers.recv_displs.resize(n);
    for (size_t i = 0; i < n; ++i) {
        buffers.recv_counts[i] = detail::checked_mpi_int(static_cast<size_t>(recv.counts[i]) * static_cast<size_t>(scale),
                                                         "Layer exchange recv count");
        buffers.recv_displs[i] = detail::checked_mpi_int(static_cast<size_t>(recv.displs[i]) * static_cast<size_t>(scale),
                                                         "Layer exchange recv displacement");
    }
    buffers.recv_total = detail::checked_mpi_int(static_cast<size_t>(recv.total) * static_cast<size_t>(scale),
                                                 "Layer exchange recv total");

    // Now the send side at the requested scale. Re-derived rather than scaled in place so the
    // overflow check runs against the value MPI actually receives.
    if (scale != 1) {
        detail::derive_exchange_layout(layer.cross_rank(), my_rank, scale, buffers.layout, "Layer derivative exchange");
    }
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
    buffers.recv_buffer.resize(buffers.recv_total == 0 ? 1 : static_cast<size_t>(buffers.recv_total));
    handle.ticket = mpi::post_flat_alltoallv<double>({.send = buffers.send_buffer.data(),
                                                      .send_counts = layout.counts.data(),
                                                      .send_displs = layout.displs.data(),
                                                      .recv = buffers.recv_buffer.data(),
                                                      .recv_counts = buffers.recv_counts.data(),
                                                      .recv_displs = buffers.recv_displs.data()},
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

// Callers run layer_exchange_participates first, so no layout is derived at a single rank -- where
// deriving one would allocate a P-int pair and resolve a transpose for a transfer that never posts.
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
                                 .recv_displs = in_flight.handle.buffers->recv_displs,
                                 .my_rank = in_flight.my_rank});
}

// Per-thread pre-cos snapshot buffers, reused across layers to avoid a malloc/free per layer. Passed whole
// to the pack and apply passes: each reads two of the four vectors, and re-splitting them at every hand-off
// is what lets a send buffer be mistaken for a recv one.
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
    const size_t num_ranks = layer.cross_rank_rank_count();
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == my_rank) {
            continue;
        }
        const size_t end = layer.cross_rank_sin_send_size(rank);
        if (end == 0) {
            continue;
        }
        const auto base = static_cast<size_t>(layout.displs[rank]);
        const auto &bs = snap.sin_send_state[rank];
        const auto &bh = snap.sin_send_op[rank];
        layer.for_each_cross_rank_sin_send_range(rank, 0, end, [&send_buffer, &base, &bs, &bh](size_t k, size_t /*i*/) {
            send_buffer[base + 2 * k] = bs[k];
            send_buffer[base + 2 * k + 1] = bh[k];
        });
    }
}

// Remote endpoint pass: own pre-cos values come from the sin_recv snapshots, partner values from the
// received sin_send payload.
auto apply_cross_rank_derivative_exchange_impl(VecD &state,
                                               VecD &op,
                                               const LayerTraversal &layer,
                                               const DerivativeSnapshotScratch &snap,
                                               const TrigValues &trig,
                                               const ExchangePayload &payload) -> EndpointContrib {
    const size_t num_ranks = layer.cross_rank_rank_count();
    EndpointContrib local{};
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == payload.my_rank) {
            continue;
        }
        const size_t end = layer.cross_rank_sin_recv_size(rank);
        if (end == 0) {
            continue;
        }
        const auto *rv = payload.recv_buffer.data() + payload.recv_displs[rank];
        const auto &ds = snap.sin_recv_state[rank];
        const auto &dh = snap.sin_recv_op[rank];
        layer.for_each_cross_rank_sin_recv_range(
            rank,
            0,
            end,
            [&trig, &ds, &dh, &rv, &local, &op, &state](size_t k, size_t i, int phi_signed) {
                const auto phi = static_cast<double>(phi_signed);
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
            });
    }
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
    const size_t num_ranks = layer.cross_rank_rank_count();
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == my_rank) {
            continue;
        }
        const size_t end = layer.cross_rank_sin_send_size(rank);
        if (end == 0) {
            continue;
        }
        const auto base = static_cast<size_t>(layout.displs[rank]);
        layer.for_each_cross_rank_sin_send_range(rank, 0, end, [&send_buffer, &base, &op](size_t k, size_t i) {
            send_buffer[base + k] = op[i];
        });
    }
}

void apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              double sin_val,
                                              const ExchangePayload &payload) {
    // op[i] is already cos-scaled, so only the sine term is added; rv[k] is the partner's pre-cos value.
    const size_t num_ranks = layer.cross_rank_rank_count();
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == payload.my_rank) {
            continue;
        }
        const size_t end = layer.cross_rank_sin_recv_size(rank);
        if (end == 0) {
            continue;
        }
        const auto *rv = payload.recv_buffer.data() + payload.recv_displs[rank];
        layer.for_each_cross_rank_sin_recv_range(rank,
                                                 0,
                                                 end,
                                                 [&op, &sin_val, &rv](size_t k, size_t i, int phi_signed) {
                                                     op[i] += sin_val * static_cast<double>(phi_signed) * rv[k];
                                                 });
    }
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
auto apply_self_slot_derivative_paired(VecD &state,
                                       VecD &op,
                                       const LayerTraversal &layer,
                                       size_t my_rank,
                                       const TrigValues &trig) -> EndpointContrib {
    const size_t self_d_count = layer.cross_rank_sin_recv_size(my_rank);
    if (self_d_count == 0) {
        return {};
    }
    const auto pairs = self_d_count / 2;
    // my_rank is loop-invariant, so resolve the slot once: the four fetches below are four
    // lookups into the P-sized record array per rotation pair otherwise, in the innermost
    // gradient loop.
    const auto slot = layer.cross_rank_slot(my_rank);
    EndpointContrib local{};
    for (size_t k = 0; k < pairs; ++k) {
        const size_t i1 = detail::slot_sin_recv_index(slot, k);
        const double phi1 = static_cast<double>(detail::slot_sin_recv_phase(slot, k));
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

// Snapshot pre-cos (state, op) at every remote rank's sin_send/sin_recv endpoints before the cos pass
// clobbers them. The self slot needs none — it recovers live — so it is cleared.
void snapshot_remote_endpoints(const VecD &state,
                               const VecD &op,
                               const LayerTraversal &layer,
                               size_t my_rank,
                               size_t R,
                               DerivativeSnapshotScratch &snap) {
    snap.sin_send_state.resize(R);
    snap.sin_send_op.resize(R);
    snap.sin_recv_state.resize(R);
    snap.sin_recv_op.resize(R);
    for (size_t r = 0; r < R; ++r) {
        if (r == my_rank) {
            snap.sin_send_state[r].clear();
            snap.sin_send_op[r].clear();
            snap.sin_recv_state[r].clear();
            snap.sin_recv_op[r].clear();
            continue;
        }
        const size_t bc = layer.cross_rank_sin_send_size(r);
        snap.sin_send_state[r].resize(bc);
        snap.sin_send_op[r].resize(bc);
        if (bc > 0) {
            auto &bs = snap.sin_send_state[r];
            auto &bh = snap.sin_send_op[r];
            layer.for_each_cross_rank_sin_send_range(r, 0, bc, [&bs, &bh, &state, &op](size_t k, size_t i) {
                bs[k] = state[i];
                bh[k] = op[i];
            });
        }
        const size_t dc = layer.cross_rank_sin_recv_size(r);
        snap.sin_recv_state[r].resize(dc);
        snap.sin_recv_op[r].resize(dc);
        if (dc > 0) {
            auto &ds = snap.sin_recv_state[r];
            auto &dh = snap.sin_recv_op[r];
            layer.for_each_cross_rank_sin_recv_range(r,
                                                     0,
                                                     dc,
                                                     [&ds, &dh, &state, &op](size_t k, size_t i, int /*phi*/) {
                                                         ds[k] = state[i];
                                                         dh[k] = op[i];
                                                     });
        }
    }
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
    snapshot_remote_endpoints(state, op, layer, my_rank, R, snap);

    // No-op at single rank; the transfer touches only buffers, so the cos pass below may mutate state/op.
    auto in_flight = begin_cross_rank_derivative_exchange(snap, layer, comm);

    // A = Σ s_old·h_old over all anticommuting indices, endpoints included — hence the subtraction below.
    const double A = cos_acc(layer_idx, state.data(), op.data(), trig.cos_val, trig.sec_val);

    EndpointContrib ep;
    if (my_rank < R) {
        ep = apply_self_slot_derivative_paired(state, op, layer, my_rank, trig);
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
    const int my_rank_int = mpi::rank(comm);
    const auto my_rank = static_cast<size_t>(my_rank_int);

    // Snapshot my_rank's own sin_send values before the cos pass; runs unconditionally (the remote pack
    // skips my_rank) so single-rank works.
    const size_t self_b_count = (my_rank < layer.cross_rank_rank_count()) ? layer.cross_rank_sin_send_size(my_rank) : 0;
    VecD self_b_snapshot;
    self_b_snapshot.resize(self_b_count);
    if (self_b_count > 0) {
        auto &snap = self_b_snapshot;
        layer.for_each_cross_rank_sin_send_range(my_rank, 0, self_b_count, [&snap, &op](size_t k, size_t i) {
            snap[k] = op[i];
        });
    }

    // Pack + start the exchange before the cos scan so partner values are pre-cos and the transfer overlaps.
    auto in_flight = begin_cross_rank_evolution_exchange(op, layer, comm);
    cos_scale(layer_idx, op_data, cos_val);
    finish_cross_rank_evolution_exchange(op, layer, sin_val, in_flight);

    // Self-slot sin_recv entries: op[i] is already cos-scaled, so only the sine term is added.
    if (self_b_count > 0) {
        const size_t self_d_count = layer.cross_rank_sin_recv_size(my_rank);
        layer.for_each_cross_rank_sin_recv_range(my_rank,
                                                 0,
                                                 self_d_count,
                                                 [&op, &sin_val, &self_b_snapshot](size_t k, size_t i, int phi_signed) {
                                                     op[i] +=
                                                         sin_val * static_cast<double>(phi_signed) * self_b_snapshot[k];
                                                 });
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
