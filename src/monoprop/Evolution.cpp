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
#include <utility>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop {
namespace {

// Endpoint accumulators for the gradient identity: cos_terms = A_ep (Σ s_old·h_old),
// sin_terms = B (Σ σ·s_old·h_p).
struct EndpointContrib {
    double cos_terms = 0.0;
    double sin_terms = 0.0;
};

struct TrigValues {
    double cos_val;
    double sin_val;
    double sec_val;
    double g_val;   // 2·gen_coeff
    double tan_val; // sin/cos

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
};

auto &acquire_flat_exchange_buffers() {
    struct Scratch {
        FlatExchangeBuffers buffers;
    };
    static thread_local Scratch scratch;
    return scratch.buffers;
}

void resize_flat_exchange_buffers(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers) {
    // Recv size isn't known until counts are exchanged; keep it at 1 element so data() stays non-null.
    const size_t send_alloc = layout.total_count == 0 ? 1 : layout.total_count;
    buffers.send_buffer.resize(send_alloc);
    buffers.recv_buffer.resize(1);
    buffers.recv_counts.clear();
    buffers.recv_displs.clear();
}

auto active_evolution_exchange_layout(const LayerTraversal &layer, const mpi::Comm &comm)
    -> const LayerExchangeLayout * {
    if (mpi::size(comm) == 1) {
        return nullptr;
    }
    // All ranks must participate even at local total_count 0, else MPI_Alltoallv deadlocks.
    return &layer.evolution_exchange_layout();
}

// In-flight cross-rank exchange handle; an empty ticket means nothing is in flight.
struct CrossRankExchangeHandle {
    const LayerExchangeLayout *layout = nullptr;
    FlatExchangeBuffers *buffers = nullptr;
    [[no_unique_address]] mpi::Ticket ticket;
};

// Size the recv buffer from the cached layout and post the non-blocking payload transfer.
inline auto begin_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, const mpi::Comm &comm)
    -> CrossRankExchangeHandle {
    CrossRankExchangeHandle handle;
    handle.layout = &layout;
    handle.buffers = &buffers;
    const auto &recv = mpi::resolve_recv(layout.counts, comm, layout.recv_cache);
    buffers.recv_counts = recv.counts;
    buffers.recv_displs = recv.displs;
    buffers.recv_buffer.resize(recv.total == 0 ? 1 : static_cast<size_t>(recv.total));
    handle.ticket = mpi::post_flat_alltoallv<double>(buffers.send_buffer.data(),
                                                     layout.counts.data(),
                                                     layout.displs.data(),
                                                     buffers.recv_buffer.data(),
                                                     buffers.recv_counts.data(),
                                                     buffers.recv_displs.data(),
                                                     mpi::size(comm),
                                                     comm);
    return handle;
}

inline auto wait_flat_exchange(CrossRankExchangeHandle &handle) -> void {
    handle.ticket.wait();
}

// Pack B entries from the pre-cos snapshots; the live state/op at B-indices are clobbered by the cos pass.
void pack_cross_rank_derivative_payload_impl(const std::vector<VecD> &sin_send_state,
                                             const std::vector<VecD> &sin_send_op,
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
        const size_t base = static_cast<size_t>(layout.displs[rank]);
        const auto &bs = sin_send_state[rank];
        const auto &bh = sin_send_op[rank];
        layer.for_each_cross_rank_sin_send_range(rank, 0, end, [&send_buffer, &base, &bs, &bh](size_t k, size_t /*i*/) {
            send_buffer[base + 2 * k] = bs[k];
            send_buffer[base + 2 * k + 1] = bh[k];
        });
    }
}

// Remote endpoint pass: own pre-cos values come from sin_recv snapshots, partner values from the
// received B-payload; accumulates A_ep/B and overwrites state/op with the rotation.
auto apply_cross_rank_derivative_exchange_impl(VecD &state,
                                               VecD &op,
                                               const LayerTraversal &layer,
                                               const std::vector<VecD> &sin_recv_state,
                                               const std::vector<VecD> &sin_recv_op,
                                               const TrigValues &trig,
                                               const VecD &recv_buffer,
                                               const std::vector<int> &recv_displs,
                                               int my_rank) -> EndpointContrib {
    const size_t num_ranks = layer.cross_rank_rank_count();
    EndpointContrib local{};
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == my_rank) {
            continue;
        }
        const size_t end = layer.cross_rank_sin_recv_size(rank);
        if (end == 0) {
            continue;
        }
        const auto *rv = recv_buffer.data() + recv_displs[rank];
        const auto &ds = sin_recv_state[rank];
        const auto &dh = sin_recv_op[rank];
        layer.for_each_cross_rank_sin_recv_range(
            rank,
            0,
            end,
            [&trig, &ds, &dh, &rv, &local, &op, &state](size_t k, size_t i, int phi_signed) {
                const auto phi = static_cast<double>(phi_signed);
                // INVERSE-rotation write-back (−sin): un-evolves state/op for the next reverse layer.
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

// In-flight cross-rank derivative exchange: pack + Ialltoallv fire up front so the transfer overlaps
// the cos pass + self-slot; finish_cross_rank_derivative_exchange waits and applies the payloads.
struct InFlightCrossRankDerivative {
    CrossRankExchangeHandle handle;
    int my_rank = 0;
    bool active = false;
};

inline auto begin_cross_rank_derivative_exchange(const std::vector<VecD> &sin_send_state,
                                                 const std::vector<VecD> &sin_send_op,
                                                 const LayerTraversal &layer,
                                                 const mpi::Comm &comm) -> InFlightCrossRankDerivative {
    InFlightCrossRankDerivative in_flight;
    // Single-rank (or no peer participating): nothing to exchange — the self slot covers everything.
    if (active_evolution_exchange_layout(layer, comm) == nullptr) {
        return in_flight;
    }
    in_flight.my_rank = mpi::rank(comm);
    in_flight.active = true;

    const auto &layout = layer.derivative_exchange_layout();
    auto &buffers = acquire_flat_exchange_buffers();
    resize_flat_exchange_buffers(layout, buffers);
    // Safe to fire before the cos pass: pack reads pre-cos snapshots and the transfer touches only buffers.
    pack_cross_rank_derivative_payload_impl(sin_send_state,
                                            sin_send_op,
                                            layer,
                                            in_flight.my_rank,
                                            layout,
                                            buffers.send_buffer);
    in_flight.handle = begin_flat_exchange(layout, buffers, comm);
    return in_flight;
}

// Wait for the transfer, then apply partner payloads at the remote D-endpoints (after the cos pass,
// so results are bit-identical to the original blocking order).
inline auto finish_cross_rank_derivative_exchange(VecD &state,
                                                  VecD &op,
                                                  const LayerTraversal &layer,
                                                  const std::vector<VecD> &sin_recv_state,
                                                  const std::vector<VecD> &sin_recv_op,
                                                  const TrigValues &trig,
                                                  InFlightCrossRankDerivative &in_flight) -> EndpointContrib {
    if (!in_flight.active || in_flight.handle.layout == nullptr) {
        return {};
    }
    wait_flat_exchange(in_flight.handle);
    return apply_cross_rank_derivative_exchange_impl(state,
                                                     op,
                                                     layer,
                                                     sin_recv_state,
                                                     sin_recv_op,
                                                     trig,
                                                     in_flight.handle.buffers->recv_buffer,
                                                     in_flight.handle.buffers->recv_displs,
                                                     in_flight.my_rank);
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
        const size_t base = static_cast<size_t>(layout.displs[rank]);
        layer.for_each_cross_rank_sin_send_range(rank, 0, end, [&send_buffer, &base, &op](size_t k, size_t i) {
            send_buffer[base + k] = op[i];
        });
    }
}

void apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              double sin_val,
                                              const VecD &recv_buffer,
                                              const std::vector<int> &recv_displs,
                                              int my_rank) {
    // op[i] is already cos-scaled, so this only ADDS the sine rotation op[i] += sin·φ·partner_old,
    // where recv[k] is the partner's pre-cos B-snapshot.
    const size_t num_ranks = layer.cross_rank_rank_count();
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == my_rank) {
            continue;
        }
        const size_t end = layer.cross_rank_sin_recv_size(rank);
        if (end == 0) {
            continue;
        }
        const auto *rv = recv_buffer.data() + recv_displs[rank];
        layer.for_each_cross_rank_sin_recv_range(rank,
                                                 0,
                                                 end,
                                                 [&op, &sin_val, &rv](size_t k, size_t i, int phi_signed) {
                                                     op[i] += sin_val * static_cast<double>(phi_signed) * rv[k];
                                                 });
    }
}

// In-flight cross-rank evolution exchange: pack + Ialltoallv fire up front so local compute overlaps;
// finish_cross_rank_evolution_exchange applies the received contributions.
struct InFlightCrossRankEvolution {
    CrossRankExchangeHandle handle;
    int my_rank = 0;
    bool active = false;
};

inline auto begin_cross_rank_evolution_exchange(VecD &op, const LayerTraversal &layer, const mpi::Comm &comm)
    -> InFlightCrossRankEvolution {
    InFlightCrossRankEvolution in_flight;
    const auto *layout = active_evolution_exchange_layout(layer, comm);
    if (layout == nullptr) {
        return in_flight;
    }

    in_flight.my_rank = mpi::rank(comm);
    in_flight.active = true;

    auto &buffers = acquire_flat_exchange_buffers();
    resize_flat_exchange_buffers(*layout, buffers);
    pack_cross_rank_evolution_payload_impl(op, layer, in_flight.my_rank, *layout, buffers.send_buffer);
    in_flight.handle = begin_flat_exchange(*layout, buffers, comm);
    return in_flight;
}

inline auto finish_cross_rank_evolution_exchange(VecD &op,
                                                 const LayerTraversal &layer,
                                                 double sin_val,
                                                 InFlightCrossRankEvolution &in_flight) -> void {
    if (!in_flight.active || in_flight.handle.layout == nullptr) {
        return;
    }

    wait_flat_exchange(in_flight.handle);
    apply_cross_rank_evolution_exchange_impl(op,
                                             layer,
                                             sin_val,
                                             in_flight.handle.buffers->recv_buffer,
                                             in_flight.handle.buffers->recv_displs,
                                             in_flight.my_rank);
}

// Snapshot-free self-slot endpoint pass: d-entries k and k+P are the two endpoints of one rotation, so
// reading both (pre-cos recovered from the post-cos slots) before writing either avoids the RAW hazard.
// Rotations are index-disjoint, so the pair loop is parallel-safe.
auto apply_self_slot_derivative_paired(VecD &state,
                                       VecD &op,
                                       const LayerTraversal &layer,
                                       size_t my_rank,
                                       const TrigValues &trig) -> EndpointContrib {
    const size_t self_d_count = layer.cross_rank_sin_recv_size(my_rank);
    if (self_d_count == 0) {
        return {};
    }
    const size_t pairs = self_d_count / 2; // == P; rotation k = (d[k], d[k+P])
    EndpointContrib local{};
    for (size_t k = 0; k < pairs; ++k) {
        const size_t i1 = layer.cross_rank_sin_recv_index_at(my_rank, k);
        const double phi1 = static_cast<double>(layer.cross_rank_sin_recv_phase_at(my_rank, k));
        const size_t i2 = layer.cross_rank_sin_recv_index_at(my_rank, k + pairs);
        const auto phi2 = static_cast<double>(layer.cross_rank_sin_recv_phase_at(my_rank, k + pairs));
        // Recover pre-cos values (read both endpoints before any write).
        const double s1 = state[i1] * trig.sec_val;
        const double h1 = op[i1] * trig.cos_val;
        const double s2 = state[i2] * trig.sec_val;
        const double h2 = op[i2] * trig.cos_val;
        local.cos_terms += (s1 * h1) + (s2 * h2);
        local.sin_terms += (phi1 * s1 * h2) + (phi2 * s2 * h1);
        // INVERSE-rotation write-back (−sin); see apply_cross_rank_derivative_exchange_impl.
        const double ps1 = -trig.sin_val * phi1;
        const double ps2 = -trig.sin_val * phi2;
        op[i1] = (h1 * trig.cos_val) + (ps1 * h2);
        state[i1] = (s1 * trig.cos_val) + (ps1 * s2);
        op[i2] = (h2 * trig.cos_val) + (ps2 * h1);
        state[i2] = (s2 * trig.cos_val) + (ps2 * s1);
    }
    return local;
}

// Per-thread pool for the derivative's pre-cos snapshot buffers, reused across layers to avoid
// malloc/free per layer.
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

// Snapshot pre-cos (state, op) at every REMOTE rank's B/D endpoints before the cos pass clobbers them
// (sin_send is sent, sin_recv applied on receipt). The self slot needs none — it recovers live — so it is cleared.
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

// Reverse-mode gradient contribution of one layer: applies the inverse rotation to (state, op) in place
// and returns this layer's parameter-gradient term.
auto state_operator_derivative_local_impl(VecD &state,
                                          VecD &op,
                                          const MPGraphView &graph,
                                          size_t layer_idx,
                                          double gen_coeff,
                                          double param,
                                          const mpi::Comm &comm,
                                          const detail::LayerCosAccumulate &cos_acc) -> double {
    const TrigValues trig(param, gen_coeff);
    const auto layer = graph.get_layer_traversal(layer_idx);
    const auto my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = layer.cross_rank_rank_count();

    // The cos pass clobbers state/op at every B/D index, so remote endpoints are snapshotted pre-cos.
    auto &snap = derivative_snapshot_scratch();
    snapshot_remote_endpoints(state, op, layer, my_rank, R, snap);
    const auto &sin_send_state = snap.sin_send_state;
    const auto &sin_send_op = snap.sin_send_op;
    const auto &sin_recv_state = snap.sin_recv_state;
    const auto &sin_recv_op = snap.sin_recv_op;

    // Fire the remote exchange now so the transfer overlaps the cos pass + self-slot below (no-op at
    // single rank; the transfer touches only buffers, so concurrent state/op mutation is safe).
    auto in_flight = begin_cross_rank_derivative_exchange(sin_send_state, sin_send_op, layer, comm);

    // Cos pass over all anticommuting indices via the mandatory cos_acc callback: A = Σ s_old·h_old,
    // then state*=cos, op*=sec.
    const double A = cos_acc(layer_idx, state.data(), op.data(), trig.cos_val, trig.sec_val);

    // Endpoint passes overwrite endpoints and accumulate A_ep, B via the snapshot-free paired path.
    EndpointContrib ep;
    if (my_rank < R) {
        ep = apply_self_slot_derivative_paired(state, op, layer, my_rank, trig);
    }
    // Wait for the transfer and apply remote partner payloads (after the cos pass).
    const auto remote =
        finish_cross_rank_derivative_exchange(state, op, layer, sin_recv_state, sin_recv_op, trig, in_flight);
    ep = combine_endpoint_contrib(ep, remote);

    // dE/dθ_j = −g·(tan·(A − A_ep) − B). NOTE: B enters with a PLUS sign; the earlier −g·B negated every
    // nonzero rotation gradient (caught as sign-flipped analytic vs finite-difference gradients).
    return -trig.g_val * (trig.tan_val * (A - ep.cos_terms) - ep.sin_terms);
}

// Recompute-routed reverse-derivative entry point (cos accumulation via the mandatory callback).
auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraphView &graph,
                                     size_t layer_idx,
                                     double gen_coeff,
                                     double param,
                                     const detail::LayerCosAccumulate &cos_acc,
                                     mpi::Comm comm) -> double {
    return state_operator_derivative_local_impl(state, op, graph, layer_idx, gen_coeff, param, comm, cos_acc);
}

auto evolve_step_traversal_impl(VecD &op,
                                const LayerTraversal &layer,
                                double param,
                                size_t layer_idx,
                                const mpi::Comm &comm,
                                const detail::LayerCosScale &cos_scale) -> void {
    const double cos_val = std::cos(2 * param), sin_val = std::sin(2 * param);

    auto *const op_data = op.data();
    const int my_rank_int = mpi::rank(comm);
    const auto my_rank = static_cast<size_t>(my_rank_int);

    // Snapshot self-B (my_rank's B-indices) BEFORE the cos pass; runs unconditionally (remote pack
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

    // Pack + start the exchange BEFORE the cos scan so partner values are pre-cos and the transfer overlaps.
    auto in_flight = begin_cross_rank_evolution_exchange(op, layer, comm);
    // Cos scaling via the mandatory callback; MUST stay between begin_/finish so the transfer overlaps it.
    cos_scale(layer_idx, op_data, cos_val);
    finish_cross_rank_evolution_exchange(op, layer, sin_val, in_flight);

    // Apply self-slot D-entries: op[i] is already cos-scaled, so this only ADDS the sine rotation
    // op[i] += sin·φ·self_b_snapshot[k].
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

// Forward-evolve `op` through one layer of a graph view, via the traversal impl.
auto evolve_step_impl(VecD &op,
                      const MPGraphView &graph,
                      double param,
                      size_t layer_idx,
                      const mpi::Comm &comm,
                      const detail::LayerCosScale &cos_scale) -> void {
    evolve_step_traversal_impl(op, graph.get_layer_traversal(layer_idx), param, layer_idx, comm, cos_scale);
}

auto evolve_step(VecD &op, const Layer &layer, double param, const detail::LayerCosScale &cos_scale, mpi::Comm comm)
    -> void {
    evolve_step_traversal_impl(op, layer.traversal(), param, 0, comm, cos_scale);
}

// Forward-evolve `coeffs` through every layer in order, applying params[i] at layer i.
auto evolve_operator_impl(VecD coeffs,
                          const MPGraphView &graph,
                          const VecD &params,
                          const mpi::Comm &comm,
                          const detail::LayerCosScale &cos_scale) -> VecD {
    for (size_t i = 0; i < graph.layers(); ++i) {
        evolve_step_impl(coeffs, graph, params[i], i, comm, cos_scale);
    }
    return coeffs;
}

// Recompute-routed forward entry point (cos scaling via the mandatory callback).
auto evolve_operator(VecD &&coeffs,
                     const MPGraphView &graph,
                     const VecD &params,
                     const detail::LayerCosScale &cos_scale,
                     mpi::Comm comm) -> VecD {
    return evolve_operator_impl(std::move(coeffs), graph, params, comm, cos_scale);
}

} // namespace monoprop
