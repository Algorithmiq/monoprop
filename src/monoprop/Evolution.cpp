#include "monoprop/Evolution.h"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <utility>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/Exchange.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/profiling/RegionProfiler.h"

namespace monoprop {
namespace {

// Endpoint (rotation) accumulators for the gradient identity g_j = −g·(tan·(A − A_ep) + B).
// Named for the trig factor each rides in that identity (paper symbol in parens):
//   cos_terms = Σ_endpoints s_old·h_old        (pre-cos own product; paper A_ep — the cos/tan term)
//   sin_terms = Σ_endpoints σ·s_old·h_p         (pre-cos own state × partner ham, signed phase σ; paper B)
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
    // Allocate send buffer sized to the layout's total send count. The recv
    // buffer size is not known until we exchange send-counts with peers,
    // so reserve a single element to ensure data() is non-null.
    const size_t send_alloc = layout.total_count == 0 ? 1 : layout.total_count;
    buffers.send_buffer.resize(send_alloc);
    buffers.recv_buffer.resize(1);
    buffers.recv_counts.clear();
    buffers.recv_displs.clear();
}

auto active_evolution_exchange_layout(const LayerTraversal &layer, mpi::Comm comm) -> const LayerExchangeLayout * {
    if (mpi::size(comm) == 1) {
        return nullptr;
    }
    // All ranks must participate in MPI collectives even when this rank's local
    // total_count is 0. Skipping based on local total_count causes a deadlock
    // when another rank has non-zero counts for the same layer (MPI_Alltoallv
    // requires all processes in the communicator to call it).
    return &layer.evolution_exchange_layout();
}

// In-flight cross-rank exchange handle. The mpi::Ticket completes the non-blocking transfer;
// an empty Ticket means nothing is in flight (single-rank / non-MPI build).
struct CrossRankExchangeHandle {
    const LayerExchangeLayout *layout = nullptr;
    FlatExchangeBuffers *buffers = nullptr;
    mpi::Ticket ticket;
};

// Resolve the recv layout (cached per layer via the facade), size the recv buffer, and post the
// payload transfer. All ranks must participate — never skip on zero counts (the facade owns that
// deadlock discipline). The transfer is non-blocking; the returned handle's ticket completes it.
// Buffers are always sized ≥ 1 (see resize_flat_exchange_buffers).
inline auto begin_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, mpi::Comm comm)
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

inline auto wait_flat_exchange(CrossRankExchangeHandle &handle) -> void { handle.ticket.wait(); }

// ─── Cross-rank derivative helpers ──────────────────────────────────────────

// Pack B entries from PRE-COS snapshots (sin_send_state[rank][k], sin_send_op[rank][k]). The live
// state/op at B-indices have been clobbered by the cos pass (endpoints are in cos_data now),
// so the payload must come from the snapshots taken before the cos pass.
void pack_cross_rank_derivative_payload_impl(const std::vector<VecD> &sin_send_state,
                                             const std::vector<VecD> &sin_send_op,
                                             const LayerTraversal &layer,
                                             int my_rank,
                                             const LayerExchangeLayout &layout,
                                             VecD &send_buffer) {
    threading::parallel_for_cross_rank_sin_send_ranges(layer, my_rank, [&](size_t rank, size_t begin, size_t end) {
        const size_t base = static_cast<size_t>(layout.displs[rank]);
        const auto &bs = sin_send_state[rank];
        const auto &bh = sin_send_op[rank];
        layer.for_each_cross_rank_sin_send_range(rank, begin, end, [&](size_t k, size_t /*i*/) {
            send_buffer[base + 2 * k] = bs[k];
            send_buffer[base + 2 * k + 1] = bh[k];
        });
    });
}

// Remote endpoint pass. Own pre-cos (s_old,h_old) come from sin_recv snapshots (cos pass clobbered
// the live values); partner (s_p,h_p) = (rv[2k], rv[2k+1]) is the sender's pre-cos B-payload.
//   A_ep += s_old·h_old;  B += σ·s_old·h_p
//   op[i]    = cos·h_old + sin·σ·h_p   (overwrite)
//   state[i] = cos·s_old + sin·σ·s_p   (overwrite)
auto apply_cross_rank_derivative_exchange_impl(VecD &state,
                                               VecD &op,
                                               const LayerTraversal &layer,
                                               const std::vector<VecD> &sin_recv_state,
                                               const std::vector<VecD> &sin_recv_op,
                                               const TrigValues &trig,
                                               const VecD &recv_buffer,
                                               const std::vector<int> &recv_displs,
                                               int my_rank) -> EndpointContrib {
    return threading::parallel_reduce_cross_rank_sin_recv_ranges(
        layer,
        my_rank,
        EndpointContrib{},
        [&](size_t rank, size_t begin, size_t end, EndpointContrib local) {
            const auto *rv = recv_buffer.data() + recv_displs[rank];
            const auto &ds = sin_recv_state[rank];
            const auto &dh = sin_recv_op[rank];
            layer.for_each_cross_rank_sin_recv_range(rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
                const double phi = static_cast<double>(phi_signed);
                // INVERSE-rotation write-back (−sin): the reverse sweep un-evolves state/op one
                // layer for the next iteration. cos_terms/b below use the recovered pre-cos values
                // directly, so this sign only affects the values fed to the subsequent layer.
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
            return local;
        },
        combine_endpoint_contrib);
}

// In-flight cross-rank DERIVATIVE exchange. Pack + Ialltoallv fire up front (from the pre-cos
// sin_send snapshots), the caller runs the cos pass + self-slot during the transfer, then calls
// finish_cross_rank_derivative_exchange to wait + apply the received partner payloads. Mirrors
// the forward (evolution) overlap so the reverse-sweep network IO is hidden behind compute.
struct InFlightCrossRankDerivative {
    CrossRankExchangeHandle handle;
    int my_rank = 0;
    bool active = false;
};

inline auto begin_cross_rank_derivative_exchange(const std::vector<VecD> &sin_send_state,
                                                 const std::vector<VecD> &sin_send_op,
                                                 const LayerTraversal &layer,
                                                 mpi::Comm comm) -> InFlightCrossRankDerivative {
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
    // Pack reads the PRE-COS sin_send snapshots, so it is safe to fire before the cos pass mutates
    // the live state/ham. The Ialltoallv only touches send/recv buffers, never state/ham.
    pack_cross_rank_derivative_payload_impl(sin_send_state, sin_send_op, layer, in_flight.my_rank, layout,
                                            buffers.send_buffer);
    in_flight.handle = begin_flat_exchange(layout, buffers, comm);
    return in_flight;
}

// Wait for the in-flight transfer, then apply the partner payloads into the live state/op at the
// remote D-endpoints. Runs AFTER the cos pass, so endpoints are overwritten with snapshot-based
// values exactly as in the original blocking order — bit-identical result.
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

// ─── Cross-rank evolution helpers ───────────────────────────────────────────

void pack_cross_rank_evolution_payload_impl(VecD &op,
                                            const LayerTraversal &layer,
                                            int my_rank,
                                            const LayerExchangeLayout &layout,
                                            VecD &send_buffer) {
    // Pack B entries: each entry contributes one scalar.
    threading::parallel_for_cross_rank_sin_send_ranges(layer, my_rank, [&](size_t rank, size_t begin, size_t end) {
        const size_t base = static_cast<size_t>(layout.displs[rank]);
        layer.for_each_cross_rank_sin_send_range(rank, begin, end, [&](size_t k, size_t i) { send_buffer[base + k] = op[i]; });
    });
}

void apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              double sin_val,
                                              const VecD &recv_buffer,
                                              const std::vector<int> &recv_displs,
                                              int my_rank) {
    // Cross-rank D apply: op[i] += sin·φ·B_partner_old[k].
    // cos_data now includes the endpoints, so the cosine pass already scaled op[i] to cos·op_old[i];
    // this pass only ADDS the sine rotation. recv[k] holds the partner's pre-cos B-snapshot (packed
    // before any cos mutation), so op[i] = cos·op_old[i] + sin·φ·partner_old — identical to before.
    threading::parallel_for_cross_rank_sin_recv_ranges(layer, my_rank, [&](size_t rank, size_t begin, size_t end) {
        const auto *rv = recv_buffer.data() + recv_displs[rank];
        layer.for_each_cross_rank_sin_recv_range(rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
            op[i] += sin_val * static_cast<double>(phi_signed) * rv[k];
        });
    });
}

// In-flight cross-rank evolution exchange. Pack + Ialltoallv have already fired by the time
// this struct is returned; the caller is expected to do local compute then call
// finish_cross_rank_evolution_exchange to apply the received contributions.
struct InFlightCrossRankEvolution {
    CrossRankExchangeHandle handle;
    int my_rank = 0;
    bool active = false;
};

inline auto begin_cross_rank_evolution_exchange(VecD &op,
                                                const LayerTraversal &layer,
                                                mpi::Comm comm) -> InFlightCrossRankEvolution {
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

// Snapshot-free self-slot endpoint pass. assemble_partners lays out
// d = [{out,−φ}]++[{in,+φ}] with P==Q, so d-entry k and k+P are the two endpoints of one Givens
// rotation and each other's partner. Reading BOTH endpoints' (recovered pre-cos) values before writing
// EITHER removes the read-after-write hazard that forced the sin_send/sin_recv snapshots — so neither is
// needed. Pre-cos values are recovered from the live post-cos slots; rotations are index-disjoint, so
// the pair loop is parallel-safe.
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
    return threading::parallel_reduce_ranges(
        pairs,
        EndpointContrib{},
        [&](size_t begin, size_t end, EndpointContrib local) {
            for (size_t k = begin; k < end; ++k) {
                const size_t i1 = layer.cross_rank_sin_recv_index_at(my_rank, k);
                const double phi1 = static_cast<double>(layer.cross_rank_sin_recv_phase_at(my_rank, k));
                const size_t i2 = layer.cross_rank_sin_recv_index_at(my_rank, k + pairs);
                const double phi2 = static_cast<double>(layer.cross_rank_sin_recv_phase_at(my_rank, k + pairs));
                // Recover pre-cos values (read both endpoints before any write).
                const double s1 = state[i1] * trig.sec_val;
                const double h1 = op[i1] * trig.cos_val;
                const double s2 = state[i2] * trig.sec_val;
                const double h2 = op[i2] * trig.cos_val;
                // i1's partner is i2 and vice-versa.
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
        },
        combine_endpoint_contrib,
        threading::range_grain_size(pairs, 1));
}

// Per-thread pool for the derivative's pre-cos snapshot buffers, reused across layers so the reverse
// walk does not malloc/free four vector<VecD> per layer (resize keeps capacity; snapshot semantics
// unchanged). thread_local + capture-by-reference is safe: the buffers are filled and read on the
// calling thread and handed to TBB workers by reference (workers never touch their own copy).
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

// Snapshot the pre-cos (state, op) values at every REMOTE rank's B and D endpoints before the cos pass
// clobbers them. sin_send[r] is packed and sent in the cross-rank exchange; sin_recv[r] is this rank's own
// pre-cos value applied when the partner payload arrives. The self slot (r == my_rank) needs no
// snapshot — apply_self_slot_derivative_paired recovers its pre-cos values live from the post-cos slots
// (state[i]=s_old·cos, op[i]=h_old·sec) — so its four slots are cleared. Buffers are pooled per-thread
// (bound by reference so TBB workers read the calling thread's copy); each k writes its own slot.
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
            threading::parallel_for_ranges(bc, [&](size_t begin, size_t end) {
                layer.for_each_cross_rank_sin_send_range(r, begin, end, [&](size_t k, size_t i) {
                    bs[k] = state[i];
                    bh[k] = op[i];
                });
            });
        }
        const size_t dc = layer.cross_rank_sin_recv_size(r);
        snap.sin_recv_state[r].resize(dc);
        snap.sin_recv_op[r].resize(dc);
        if (dc > 0) {
            auto &ds = snap.sin_recv_state[r];
            auto &dh = snap.sin_recv_op[r];
            threading::parallel_for_ranges(dc, [&](size_t begin, size_t end) {
                layer.for_each_cross_rank_sin_recv_range(r, begin, end, [&](size_t k, size_t i, int /*phi*/) {
                    ds[k] = state[i];
                    dh[k] = op[i];
                });
            });
        }
    }
}

} // namespace

// ─── Public API ─────────────────────────────────────────────────────────────

// ─── Derivative & evolution dispatch ────────────────────────────────────────

// Reverse-mode gradient contribution of one layer: applies the layer's inverse rotation to (state, op)
// in place and returns this layer's term of the parameter gradient. Cross-rank endpoints are resolved
// by MPI exchange; pre-cos values the cos pass would clobber are snapshotted as needed (see below).
auto state_operator_derivative_local_impl(VecD &state,
                                          VecD &op,
                                          const MPGraphView &graph,
                                          size_t layer_idx,
                                          double gen_coeff,
                                          double param,
                                          mpi::Comm comm,
                                          const detail::LayerCosAccumulate &cos_acc) -> double {
    const TrigValues trig(param, gen_coeff);
    const auto layer = graph.get_layer_traversal(layer_idx);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = layer.cross_rank_rank_count();

    // The cos pass clobbers state/op at every anticommuting (B/D) index, so the remote endpoints must
    // be snapshotted pre-cos (sin_send sent, sin_recv applied on receipt); the self slot recovers its values
    // live. See snapshot_remote_endpoints.
    auto &snap = derivative_snapshot_scratch();
    snapshot_remote_endpoints(state, op, layer, my_rank, R, snap);
    auto &sin_send_state = snap.sin_send_state;
    auto &sin_send_op = snap.sin_send_op;
    auto &sin_recv_state = snap.sin_recv_state;
    auto &sin_recv_op = snap.sin_recv_op;

    // Fire the remote cross-rank exchange NOW (pack from the pre-cos sin_send snapshots, start the
    // non-blocking Ialltoallv) so the network transfer overlaps the cos pass + self-slot below —
    // mirroring the forward path. No-op at single rank. The transfer touches only send/recv
    // buffers, so the cos pass mutating state/op concurrently is safe.
    auto in_flight = begin_cross_rank_derivative_exchange(sin_send_state, sin_send_op, layer, comm);

    // Cos pass over ALL anticommuting indices: A = Σ s_old·h_old (un-inflated), then state*=cos, op*=sec.
    // The cosine set is always RECOMPUTED (or read from a transient/filtered word list) via the mandatory
    // cos_acc callback for layer_idx — no layer stores its cos bitmap anymore.
    const double A = cos_acc(layer_idx, state.data(), op.data(), trig.cos_val, trig.sec_val);

    // Endpoint passes overwrite endpoints and accumulate A_ep, B via the snapshot-free paired path.
    EndpointContrib ep;
    if (my_rank < R) {
        ep = apply_self_slot_derivative_paired(state, op, layer, my_rank, trig);
    }
    // Wait for the transfer and apply remote partner payloads (runs after the cos pass, so remote
    // D-endpoints are overwritten with snapshot-based values exactly as in the blocking order).
    const auto remote = finish_cross_rank_derivative_exchange(
        state, op, layer, sin_recv_state, sin_recv_op, trig, in_flight);
    ep = combine_endpoint_contrib(ep, remote);

    // Reverse-mode contribution of layer j: dE/dθ_j = g·(−sin·A_ep + cos·B), where A_ep and B are
    // built from the pre-layer endpoint values. Expressed in the post-cos accumulators the code holds:
    //   • cos-only terms (in cos_data but not rotation endpoints) contribute −g·tan·(A − A_ep): the
    //     A−A_ep difference isolates the cos-only products and tan supplies the sin/cos factor.
    //   • each rotation endpoint contributes +g·B (B = Σ φ·s·h_partner). The endpoint's own A and A_ep
    //     products are identical post-cos and cancel in A−A_ep, leaving only the cross (B) term.
    // NOTE: B enters with a PLUS sign (g·B). The earlier −g·B negated every nonzero rotation gradient
    // (the cos-only/tan part was already correct), which is why test_infinite_cutoff
    // analytic gradients came out as the exact negative of the finite-difference reference.
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
    return state_operator_derivative_local_impl(
        state, op, graph, layer_idx, gen_coeff, param, comm, cos_acc);
}

auto evolve_step_traversal_impl(VecD &op,
                                const LayerTraversal &layer,
                                double param,
                                size_t layer_idx,
                                mpi::Comm comm,
                                const detail::LayerCosScale &cos_scale) -> void {
    profiling::ScopedRegion prof_evolve(profiling::Region::Evolve);
    const double cos_val = std::cos(2 * param), sin_val = std::sin(2 * param);

    auto *const op_data = op.data();
    const int my_rank_int = mpi::rank(comm);
    const size_t my_rank = static_cast<size_t>(my_rank_int);

    // Snapshot self-B (cross_rank[my_rank] B-indices) BEFORE the cos pass.
    // This must run unconditionally (not gated on MPI size) so single-rank works.
    // The pack for remote ranks skips my_rank, so this is a separate local snapshot.
    const size_t self_b_count = (my_rank < layer.cross_rank_rank_count())
                                    ? layer.cross_rank_sin_send_size(my_rank)
                                    : 0;
    // NOTE: must NOT be thread_local — it is filled on the calling thread and then read
    // from TBB worker threads in the parallel self-apply below. A thread_local buffer would
    // be empty on the workers (out-of-bounds → crash). A plain local is shared by reference.
    VecD self_b_snapshot;
    self_b_snapshot.resize(self_b_count);
    if (self_b_count > 0) {
        // Parallel gather: each k writes its own snapshot[k] (disjoint) and only reads op[i]. At R==1
        // every rotation partner is self-rank, so self_b_count is the full rotation set — running this
        // serially was an Amdahl anchor that capped the whole apply's scaling. Mirrors the parallel
        // B-snapshot in state_operator_derivative_local_impl.
        auto &snap = self_b_snapshot;
        threading::parallel_for_ranges(self_b_count, [&](size_t begin, size_t end) {
            layer.for_each_cross_rank_sin_send_range(my_rank, begin, end,
                                              [&](size_t k, size_t i) { snap[k] = op[i]; });
        });
    }

    // Pack B and start non-blocking exchange BEFORE the cos scan. B ⊆ cos_data now, so packing
    // first is what guarantees the partner values are pre-cos. Overlaps transfer with cos compute.
    auto in_flight = begin_cross_rank_evolution_exchange(op, layer, comm);
    // Cos scaling via the mandatory callback (recompute fold / transient or filtered word list — no
    // layer stores its cos bitmap anymore). MUST stay between begin_/finish_cross_rank_evolution_exchange
    // so the MPI transfer overlaps it.
    cos_scale(layer_idx, op_data, cos_val);
    finish_cross_rank_evolution_exchange(op, layer, sin_val, in_flight);

    // Apply self-slot D-entries using the pre-cos snapshot.
    // op[d_index(my_rank,k)] is already post-cos (endpoints are in cos_data), so this only ADDS
    // the sine rotation: op[i] += sin·φ_signed·self_b_snapshot[k].
    if (self_b_count > 0) {
        const size_t self_d_count = layer.cross_rank_sin_recv_size(my_rank);
        threading::parallel_for_ranges(
            self_d_count,
            [&](size_t begin, size_t end) {
                layer.for_each_cross_rank_sin_recv_range(my_rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
                    op[i] += sin_val * static_cast<double>(phi_signed) * self_b_snapshot[k];
                });
            });
    }
}

// Forward-evolve `op` through one layer of a graph view, via the traversal impl.
auto evolve_step_impl(VecD &op,
                      const MPGraphView &graph,
                      double param,
                      size_t layer_idx,
                      mpi::Comm comm,
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
                          mpi::Comm comm,
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
