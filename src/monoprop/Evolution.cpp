#include "monoprop/Evolution.h"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <utility>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop {
namespace {

// Endpoint (rotation) accumulators for the gradient identity g_j = −g·(tan·(A − A_ep) + B):
//   a_ep = Σ_endpoints s_old·h_old   (pre-cos own product)
//   b    = Σ_endpoints σ·s_old·h_p   (pre-cos own state × partner ham, signed phase σ)
struct EndpointContrib {
    double a_ep = 0.0;
    double b = 0.0;
};

struct TrigValues {
    double cos_val;
    double sin_val;
    double sec_val;
    double der_cos_val;
    double der_sin_val;
    double g_val;   // 2·gen_coeff
    double tan_val; // sin/cos

    explicit TrigValues(double param, double gen_coeff = 1.0) {
        const double g = 2.0 * gen_coeff;
        cos_val = std::cos(g * param);
        sin_val = std::sin(g * param);
        sec_val = 1.0 / cos_val;
        der_cos_val = -g * sin_val;
        der_sin_val = g * cos_val;
        g_val = g;
        tan_val = sin_val * sec_val;
    }
};

auto combine_endpoint_contrib(const EndpointContrib &a, const EndpointContrib &b) -> EndpointContrib {
    return {.a_ep = a.a_ep + b.a_ep, .b = a.b + b.b};
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

auto active_evolution_exchange_layout(const LayerTraversal &layer, MPI_Comm comm) -> const LayerExchangeLayout * {
    if (mpi::size(comm) == 1) {
        return nullptr;
    }
    // All ranks must participate in MPI collectives even when this rank's local
    // total_count is 0. Skipping based on local total_count causes a deadlock
    // when another rank has non-zero counts for the same layer (MPI_Alltoallv
    // requires all processes in the communicator to call it).
    return &layer.evolution_exchange_layout();
}

void execute_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, MPI_Comm comm);

#ifdef monoprop_ENABLE_MPI
// Fill buffers.recv_counts/recv_displs (and size recv_buffer) for `layout`, reusing the cached
// count-exchange result when the communicator size is unchanged. The send-count pattern of a
// replayed graph is fixed, so this removes one blocking MPI_Alltoall latency round-trip per layer
// per call from BOTH the forward and the reverse exchange — the redundant collective grows with
// rank count, so the saving scales on multi-node clusters. The first call (cold cache) performs the
// real MPI_Alltoall; the result is deterministic, so a benign race would only re-write identical
// data. Returns true on a cache hit (caller may skip the count exchange it would otherwise do).
auto ensure_recv_layout(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, MPI_Comm comm) -> bool {
    const int n = static_cast<int>(layout.counts.size());
    const int comm_size = mpi::size(comm);
    if (layout.cached_comm_size == comm_size && static_cast<int>(layout.cached_recv_counts.size()) == n) {
        buffers.recv_counts = layout.cached_recv_counts;
        buffers.recv_displs = layout.cached_recv_displs;
        buffers.recv_buffer.resize(layout.cached_recv_total == 0 ? 1 : static_cast<size_t>(layout.cached_recv_total));
        return true;
    }

    buffers.recv_counts.resize(n);
    MPI_Alltoall(layout.counts.data(), 1, MPI_INT, buffers.recv_counts.data(), 1, MPI_INT, comm);
    buffers.recv_displs.resize(n);
    int recv_total = 0;
    for (int i = 0; i < n; ++i) {
        buffers.recv_displs[i] = recv_total;
        recv_total += buffers.recv_counts[i];
    }
    buffers.recv_buffer.resize(recv_total == 0 ? 1 : static_cast<size_t>(recv_total));

    layout.cached_recv_counts = buffers.recv_counts;
    layout.cached_recv_displs = buffers.recv_displs;
    layout.cached_recv_total = recv_total;
    layout.cached_comm_size = comm_size;
    return false;
}
#endif

// Blocking pack -> alltoallv -> apply driver over the shared thread-local exchange buffers.
template <typename Pack, typename Apply>
auto with_cross_rank_exchange(const LayerExchangeLayout &layout, MPI_Comm comm, Pack &&pack, Apply &&apply) -> void {
    const int my_rank = mpi::rank(comm);

    auto &buffers = acquire_flat_exchange_buffers();
    resize_flat_exchange_buffers(layout, buffers);
    pack(my_rank, layout, buffers.send_buffer);
    execute_flat_exchange(layout, buffers, comm);
    apply(my_rank, layout, buffers.recv_buffer, buffers.recv_displs);
}

void execute_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, MPI_Comm comm) {
    // All ranks in the communicator must call the count-exchange and the
    // subsequent Alltoallv together. Do not skip even when total_count == 0
    // otherwise the collective can deadlock.
#ifdef monoprop_ENABLE_MPI
    const int n = static_cast<int>(layout.counts.size());

    // Exchange send counts to compute per-rank recv counts/displacements (cached across replays).
    ensure_recv_layout(layout, buffers, comm);

    // Now perform the full variable-length exchange using asymmetric counts.
    MPI_Alltoallv(buffers.send_buffer.data(),
                  layout.counts.data(),
                  layout.displs.data(),
                  MPI_DOUBLE,
                  buffers.recv_buffer.data(),
                  buffers.recv_counts.data(),
                  buffers.recv_displs.data(),
                  MPI_DOUBLE,
                  comm);
#else
    (void)comm;
    // Single-rank fallback: recv layout equals send layout.
    buffers.recv_counts = layout.counts;
    buffers.recv_displs = layout.displs;
    buffers.recv_buffer = buffers.send_buffer;
#endif
}

// Non-blocking exchange support: gate behind MONOPROP_OVERLAP_EXCHANGE env var (default on).
// When enabled, the caller drives a pack -> Ialltoallv -> compute -> wait -> apply cycle so
// local cosine + local-cycle work can run concurrently with network IO.
inline auto overlap_exchange_enabled() -> bool {
    static const bool enabled = [] {
        const char *value = std::getenv("MONOPROP_OVERLAP_EXCHANGE");
        if (value == nullptr || value[0] == '\0') {
            return true; // default on
        }
        return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' || value[0] == 'n' || value[0] == 'N');
    }();
    return enabled;
}

// In-flight cross-rank exchange handle. When request == MPI_REQUEST_NULL, no exchange is
// in flight (either single-rank, no data, or the blocking path was used).
struct CrossRankExchangeHandle {
    const LayerExchangeLayout *layout = nullptr;
    FlatExchangeBuffers *buffers = nullptr;
#ifdef monoprop_ENABLE_MPI
    MPI_Request request = MPI_REQUEST_NULL;
#endif
    bool used_blocking = false;
};

inline auto begin_flat_exchange(const LayerExchangeLayout &layout, FlatExchangeBuffers &buffers, MPI_Comm comm)
    -> CrossRankExchangeHandle {
    CrossRankExchangeHandle handle;
    handle.layout = &layout;
    handle.buffers = &buffers;
    // Do not skip on total_count == 0: all ranks must participate in every
    // MPI collective regardless of local count (deadlock otherwise).
    // Buffers are always sized ≥ 1 (see resize_flat_exchange_buffers).
#ifdef monoprop_ENABLE_MPI
    // First obtain per-rank recv counts/displacements (cached count-exchange across replays).
    const int n = static_cast<int>(layout.counts.size());
    ensure_recv_layout(layout, buffers, comm);

    if (overlap_exchange_enabled()) {
        MPI_Ialltoallv(buffers.send_buffer.data(),
                       layout.counts.data(),
                       layout.displs.data(),
                       MPI_DOUBLE,
                       buffers.recv_buffer.data(),
                       buffers.recv_counts.data(),
                       buffers.recv_displs.data(),
                       MPI_DOUBLE,
                       comm,
                       &handle.request);
        return handle;
    }

    MPI_Alltoallv(buffers.send_buffer.data(),
                  layout.counts.data(),
                  layout.displs.data(),
                  MPI_DOUBLE,
                  buffers.recv_buffer.data(),
                  buffers.recv_counts.data(),
                  buffers.recv_displs.data(),
                  MPI_DOUBLE,
                  comm);
    handle.used_blocking = true;
#else
    (void)comm;
    // Single-rank fallback: recv layout equals send layout.
    buffers.recv_counts = layout.counts;
    buffers.recv_displs = layout.displs;
    buffers.recv_buffer = buffers.send_buffer;
    handle.used_blocking = true;
#endif
    return handle;
}

inline auto wait_flat_exchange(CrossRankExchangeHandle &handle) -> void {
#ifdef monoprop_ENABLE_MPI
    if (handle.request != MPI_REQUEST_NULL) {
        MPI_Wait(&handle.request, MPI_STATUS_IGNORE);
        handle.request = MPI_REQUEST_NULL;
    }
#else
    (void)handle;
#endif
}

// ─── Cross-rank derivative helpers ──────────────────────────────────────────

// Pack B entries from PRE-COS snapshots (own_b_state[rank][k], own_b_op[rank][k]). The live
// state/op at B-indices have been clobbered by the cos pass (endpoints are in cos_data now),
// so the payload must come from the snapshots taken before the cos pass.
void pack_cross_rank_derivative_payload_impl(const std::vector<VecD> &own_b_state,
                                             const std::vector<VecD> &own_b_op,
                                             const LayerTraversal &layer,
                                             int my_rank,
                                             const LayerExchangeLayout &layout,
                                             VecD &send_buffer) {
    threading::parallel_for_cross_rank_b_ranges(layer, my_rank, [&](size_t rank, size_t begin, size_t end) {
        const size_t base = static_cast<size_t>(layout.displs[rank]);
        const auto &bs = own_b_state[rank];
        const auto &bh = own_b_op[rank];
        layer.for_each_cross_rank_b_range(rank, begin, end, [&](size_t k, size_t /*i*/) {
            send_buffer[base + 2 * k] = bs[k];
            send_buffer[base + 2 * k + 1] = bh[k];
        });
    });
}

// Remote endpoint pass. Own pre-cos (s_old,h_old) come from own_d snapshots (cos pass clobbered
// the live values); partner (s_p,h_p) = (rv[2k], rv[2k+1]) is the sender's pre-cos B-payload.
//   A_ep += s_old·h_old;  B += σ·s_old·h_p
//   op[i]    = cos·h_old + sin·σ·h_p   (overwrite)
//   state[i] = cos·s_old + sin·σ·s_p   (overwrite)
auto apply_cross_rank_derivative_exchange_impl(VecD &state,
                                               VecD &op,
                                               const LayerTraversal &layer,
                                               const std::vector<VecD> &own_d_state,
                                               const std::vector<VecD> &own_d_op,
                                               const TrigValues &trig,
                                               const VecD &recv_buffer,
                                               const std::vector<int> &recv_displs,
                                               int my_rank) -> EndpointContrib {
    return threading::parallel_reduce_cross_rank_d_ranges(
        layer,
        my_rank,
        EndpointContrib{},
        [&](size_t rank, size_t begin, size_t end, EndpointContrib local) {
            const auto *rv = recv_buffer.data() + recv_displs[rank];
            const auto &ds = own_d_state[rank];
            const auto &dh = own_d_op[rank];
            layer.for_each_cross_rank_d_range(rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
                const double phi = static_cast<double>(phi_signed);
                // INVERSE-rotation write-back (−sin): the reverse sweep un-evolves state/op one
                // layer for the next iteration. a_ep/b below use the recovered pre-cos values
                // directly, so this sign only affects the values fed to the subsequent layer.
                const double ps = -trig.sin_val * phi;
                const double s_old = ds[k];
                const double h_old = dh[k];
                const double s_p = rv[2 * k];
                const double h_p = rv[2 * k + 1];
                local.a_ep += s_old * h_old;
                local.b += phi * s_old * h_p;
                op[i] = (h_old * trig.cos_val) + (ps * h_p);
                state[i] = (s_old * trig.cos_val) + (ps * s_p);
            });
            return local;
        },
        combine_endpoint_contrib);
}

// In-flight cross-rank DERIVATIVE exchange. Pack + Ialltoallv fire up front (from the pre-cos
// own_b snapshots), the caller runs the cos pass + self-slot during the transfer, then calls
// finish_cross_rank_derivative_exchange to wait + apply the received partner payloads. Mirrors
// the forward (evolution) overlap so the reverse-sweep network IO is hidden behind compute.
struct InFlightCrossRankDerivative {
    CrossRankExchangeHandle handle;
    int my_rank = 0;
    bool active = false;
};

inline auto begin_cross_rank_derivative_exchange(const std::vector<VecD> &own_b_state,
                                                 const std::vector<VecD> &own_b_op,
                                                 const LayerTraversal &layer,
                                                 MPI_Comm comm) -> InFlightCrossRankDerivative {
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
    // Pack reads the PRE-COS own_b snapshots, so it is safe to fire before the cos pass mutates
    // the live state/ham. The Ialltoallv only touches send/recv buffers, never state/ham.
    pack_cross_rank_derivative_payload_impl(own_b_state, own_b_op, layer, in_flight.my_rank, layout,
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
                                                  const std::vector<VecD> &own_d_state,
                                                  const std::vector<VecD> &own_d_op,
                                                  const TrigValues &trig,
                                                  InFlightCrossRankDerivative &in_flight) -> EndpointContrib {
    if (!in_flight.active || in_flight.handle.layout == nullptr) {
        return {};
    }
    wait_flat_exchange(in_flight.handle);
    return apply_cross_rank_derivative_exchange_impl(state,
                                                     op,
                                                     layer,
                                                     own_d_state,
                                                     own_d_op,
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
    threading::parallel_for_cross_rank_b_ranges(layer, my_rank, [&](size_t rank, size_t begin, size_t end) {
        const size_t base = static_cast<size_t>(layout.displs[rank]);
        layer.for_each_cross_rank_b_range(rank, begin, end, [&](size_t k, size_t i) { send_buffer[base + k] = op[i]; });
    });
}

void apply_cross_rank_evolution_exchange_impl(VecD &op,
                                              const LayerTraversal &layer,
                                              const LayerExchangeLayout & /*layout*/,
                                              double cos_val,
                                              double sin_val,
                                              const VecD &recv_buffer,
                                              const std::vector<int> &recv_displs,
                                              int my_rank) {
    // Cross-rank D apply: op[i] += sin·φ·B_partner_old[k].
    // cos_data now includes the endpoints, so the cosine pass already scaled op[i] to cos·op_old[i];
    // this pass only ADDS the sine rotation. recv[k] holds the partner's pre-cos B-snapshot (packed
    // before any cos mutation), so op[i] = cos·op_old[i] + sin·φ·partner_old — identical to before.
    (void)cos_val;
    threading::parallel_for_cross_rank_d_ranges(layer, my_rank, [&](size_t rank, size_t begin, size_t end) {
        const auto *rv = recv_buffer.data() + recv_displs[rank];
        layer.for_each_cross_rank_d_range(rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
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
                                                MPI_Comm comm) -> InFlightCrossRankEvolution {
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
                                                 double cos_val,
                                                 double sin_val,
                                                 InFlightCrossRankEvolution &in_flight) -> void {
    if (!in_flight.active || in_flight.handle.layout == nullptr) {
        return;
    }

    wait_flat_exchange(in_flight.handle);
    apply_cross_rank_evolution_exchange_impl(op,
                                             layer,
                                             *in_flight.handle.layout,
                                             cos_val,
                                             sin_val,
                                             in_flight.handle.buffers->recv_buffer,
                                             in_flight.handle.buffers->recv_displs,
                                             in_flight.my_rank);
}

// Cosine accumulate pass for the reverse derivative. The cosine set is always RECOMPUTED (or read
// from a transient/filtered word list) via the mandatory `cos_acc` callback for layer `layer_idx` —
// no layer stores its cos bitmap anymore.
auto accumulate_cosine_derivative(VecD &state,
                                  VecD &op,
                                  const LayerTraversal & /*layer*/,
                                  double cos_val,
                                  double sec_val,
                                  size_t layer_idx,
                                  const detail::LayerCosAccumulate &cos_acc) -> double {
    return cos_acc(layer_idx, state.data(), op.data(), cos_val, sec_val);
}

// MASKED-path self-slot endpoint pass. Used only when the cross-rank lists are filtered (pared),
// which breaks the d=[{out}]++[{in}] pairing the snapshot-free path needs, so partners (s_p,h_p)
// come from the own_b snapshot. The endpoint's OWN pre-cos (s_old,h_old) are still recovered live
// from its post-cos slot (state[i]=s_old·cos, op[i]=h_old·sec; sec=1/cos) — order-safe since each
// endpoint reads-then-writes its own slot — so no own_d snapshot is needed even here. (The common
// unmasked path uses apply_self_slot_derivative_paired and needs no snapshot at all.)
auto apply_self_slot_derivative(VecD &state,
                                VecD &op,
                                const LayerTraversal &layer,
                                size_t my_rank,
                                const TrigValues &trig,
                                const VecD &own_b_state_self,
                                const VecD &own_b_op_self) -> EndpointContrib {
    const size_t self_d_count = layer.cross_rank_d_size(my_rank);
    if (self_d_count == 0) {
        return {};
    }
    return threading::parallel_reduce_ranges(
        self_d_count,
        EndpointContrib{},
        [&](size_t begin, size_t end, EndpointContrib local) {
            layer.for_each_cross_rank_d_range(my_rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
                const double phi = static_cast<double>(phi_signed);
                // INVERSE-rotation write-back (−sin); see apply_cross_rank_derivative_exchange_impl.
                const double ps = -trig.sin_val * phi;
                const double s_old = state[i] * trig.sec_val;      // recover pre-cos: state[i]=s_old·cos
                const double h_old = op[i] * trig.cos_val; // recover pre-cos: op[i]=h_old·sec
                const double s_p = own_b_state_self[k];
                const double h_p = own_b_op_self[k];
                local.a_ep += s_old * h_old;
                local.b += phi * s_old * h_p;
                op[i] = (h_old * trig.cos_val) + (ps * h_p);
                state[i] = (s_old * trig.cos_val) + (ps * s_p);
            });
            return local;
        },
        combine_endpoint_contrib,
        threading::range_grain_size(self_d_count, 1));
}

// Snapshot-free self-slot endpoint pass (unmasked layers only). The verbatim assemble_partners
// layout gives d = [{out,−φ}]++[{in,+φ}] with P==Q, so d-entry k and d-entry k+P are the two
// endpoints (out_k, in_k) of one Givens rotation and are each other's partner. Processing the pair
// atomically — read BOTH endpoints' (recovered pre-cos) values before writing EITHER — removes the
// read-after-write hazard that forced the own_b/own_d snapshots, so neither is needed. Pre-cos
// values are recovered from the post-cos live slots (state[i]=s_old·cos, op[i]=h_old·sec), exact to
// one round-trip ULP. Rotations are index-disjoint, so the pair loop is parallel-safe.
auto apply_self_slot_derivative_paired(VecD &state,
                                       VecD &op,
                                       const LayerTraversal &layer,
                                       size_t my_rank,
                                       const TrigValues &trig) -> EndpointContrib {
    const size_t self_d_count = layer.cross_rank_d_size(my_rank);
    if (self_d_count == 0) {
        return {};
    }
    const size_t pairs = self_d_count / 2; // == P; rotation k = (d[k], d[k+P])
    return threading::parallel_reduce_ranges(
        pairs,
        EndpointContrib{},
        [&](size_t begin, size_t end, EndpointContrib local) {
            for (size_t k = begin; k < end; ++k) {
                const size_t i1 = layer.cross_rank_d_index_at(my_rank, k);
                const double phi1 = static_cast<double>(layer.cross_rank_d_phase_at(my_rank, k));
                const size_t i2 = layer.cross_rank_d_index_at(my_rank, k + pairs);
                const double phi2 = static_cast<double>(layer.cross_rank_d_phase_at(my_rank, k + pairs));
                // Recover pre-cos values (read both endpoints before any write).
                const double s1 = state[i1] * trig.sec_val;
                const double h1 = op[i1] * trig.cos_val;
                const double s2 = state[i2] * trig.sec_val;
                const double h2 = op[i2] * trig.cos_val;
                // i1's partner is i2 and vice-versa.
                local.a_ep += (s1 * h1) + (s2 * h2);
                local.b += (phi1 * s1 * h2) + (phi2 * s2 * h1);
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

// Per-thread pool for the derivative's pre-cos snapshot buffers. The gradient loop
// calls the derivative once per layer, so allocating these four vector<VecD> (plus
// their inner VecDs) fresh every layer was pure malloc/free churn on the serial
// reverse-walk thread. Pooling them (resize keeps capacity) removes that churn while
// preserving the exact snapshot semantics. thread_local + captured-by-reference is
// safe: the buffers are filled and read on the calling thread and handed to TBB
// workers by reference (workers never touch their own thread_local copy).
struct DerivativeSnapshotScratch {
    std::vector<VecD> own_b_state;
    std::vector<VecD> own_b_op;
    std::vector<VecD> own_d_state;
    std::vector<VecD> own_d_op;
};

auto derivative_snapshot_scratch() -> DerivativeSnapshotScratch & {
    static thread_local DerivativeSnapshotScratch scratch;
    return scratch;
}

} // namespace

// ─── Public API ─────────────────────────────────────────────────────────────

// ─── Derivative & evolution dispatch ────────────────────────────────────────

template <typename GraphType>
auto state_operator_derivative_local_impl(VecD &state,
                                          VecD &op,
                                          const GraphType &graph,
                                          size_t layer_idx,
                                          double gen_coeff,
                                          double param,
                                          MPI_Comm comm,
                                          const detail::LayerCosAccumulate &cos_acc) -> double {
    const TrigValues trig(param, gen_coeff);
    const auto layer = graph.get_layer_traversal(layer_idx);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = layer.cross_rank_rank_count();

    // The cos pass clobbers state/op at every anticommuting (B/D) index, so any endpoint pass that
    // cannot read its pre-cos values live must snapshot them first. Pre-cos snapshots are needed
    // ONLY for:
    //   • own_b[r], r≠my_rank → packed and sent for the MPI cross-rank exchange
    //   • own_d[r], r≠my_rank → own pre-cos values applied in the cross-rank exchange
    //   • own_b[my_rank]      → partner values for the MASKED self-slot fallback (masking breaks the
    //                           d=[{out}]++[{in}] pairing that the snapshot-free path relies on)
    // The common UNMASKED self-slot needs no snapshots: it recovers each endpoint's pre-cos value
    // live from its own post-cos slot (state[i]=s_old·cos, op[i]=h_old·sec) and rotates Givens pairs
    // atomically (apply_self_slot_derivative_paired). Hence own_d[my_rank] is never built, and
    // own_b[my_rank] only when masked. Buffers are pooled per-thread (DerivativeSnapshotScratch;
    // resize keeps capacity) and bound by reference so TBB workers read the calling thread's copy.
    const bool self_slot_paired = layer.cross_rank_unmasked();
    auto &snap = derivative_snapshot_scratch();
    snap.own_b_state.resize(R);
    snap.own_b_op.resize(R);
    snap.own_d_state.resize(R);
    snap.own_d_op.resize(R);
    auto &own_b_state = snap.own_b_state;
    auto &own_b_op = snap.own_b_op;
    auto &own_d_state = snap.own_d_state;
    auto &own_d_op = snap.own_d_op;
    for (size_t r = 0; r < R; ++r) {
        // own_b: needed for the remote exchange pack (r≠my_rank), and for the self slot only when masked.
        if (r != my_rank || !self_slot_paired) {
            const size_t bc = layer.cross_rank_b_size(r);
            own_b_state[r].resize(bc);
            own_b_op[r].resize(bc);
            if (bc > 0) {
                auto &bs = own_b_state[r];
                auto &bh = own_b_op[r];
                threading::parallel_for_ranges(bc, [&](size_t begin, size_t end) {
                    layer.for_each_cross_rank_b_range(r, begin, end, [&](size_t k, size_t i) {
                        bs[k] = state[i];
                        bh[k] = op[i];
                    });
                });
            }
        }
        else {
            own_b_state[r].clear();
            own_b_op[r].clear();
        }
        // own_d: the self-rank D-endpoints are recovered live inside apply_self_slot_derivative
        // (each endpoint reconstructs its own pre-cos value from its own post-cos slot:
        // s_old = state[i]·sec, h_old = op[i]·cos, since cos pass did state*=cos, op*=sec).
        // Only the remote ranks (r != my_rank) still need a pre-cos D snapshot, because the
        // exchange-apply pairs them with partner payloads packed from another rank.
        if (r == my_rank) {
            own_d_state[r].clear();
            own_d_op[r].clear();
            continue;
        }
        const size_t dc = layer.cross_rank_d_size(r);
        own_d_state[r].resize(dc);
        own_d_op[r].resize(dc);
        if (dc > 0) {
            auto &ds = own_d_state[r];
            auto &dh = own_d_op[r];
            threading::parallel_for_ranges(dc, [&](size_t begin, size_t end) {
                layer.for_each_cross_rank_d_range(r, begin, end, [&](size_t k, size_t i, int /*phi*/) {
                    ds[k] = state[i];
                    dh[k] = op[i];
                });
            });
        }
    }

    // Fire the remote cross-rank exchange NOW (pack from the pre-cos own_b snapshots, start the
    // non-blocking Ialltoallv) so the network transfer overlaps the cos pass + self-slot below —
    // mirroring the forward path. No-op at single rank. The transfer touches only send/recv
    // buffers, so the cos pass mutating state/op concurrently is safe.
    auto in_flight = begin_cross_rank_derivative_exchange(own_b_state, own_b_op, layer, comm);

    // Cos pass over ALL anticommuting indices: A = Σ s_old·h_old (un-inflated), then state*=cos, op*=sec.
    const double A =
        accumulate_cosine_derivative(state, op, layer, trig.cos_val, trig.sec_val, layer_idx, cos_acc);

    // Endpoint passes overwrite endpoints and accumulate A_ep, B. The unmasked self slot uses the
    // snapshot-free paired path; the masked path falls back to the own_b snapshot.
    EndpointContrib ep;
    if (my_rank < R) {
        ep = self_slot_paired
                 ? apply_self_slot_derivative_paired(state, op, layer, my_rank, trig)
                 : apply_self_slot_derivative(state, op, layer, my_rank, trig,
                                              own_b_state[my_rank], own_b_op[my_rank]);
    }
    // Wait for the transfer and apply remote partner payloads (runs after the cos pass, so remote
    // D-endpoints are overwritten with snapshot-based values exactly as in the blocking order).
    const auto remote = finish_cross_rank_derivative_exchange(
        state, op, layer, own_d_state, own_d_op, trig, in_flight);
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
    return -trig.g_val * (trig.tan_val * (A - ep.a_ep) - ep.b);
}

// Recompute-routed reverse-derivative overloads (cos accumulation via the mandatory callback).
auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraph &graph,
                                     size_t layer_idx,
                                     double gen_coeff,
                                     double param,
                                     const detail::LayerCosAccumulate &cos_acc,
                                     MPI_Comm comm) -> double {
    return state_operator_derivative_local_impl(
        state, op, graph, layer_idx, gen_coeff, param, comm, cos_acc);
}

auto state_operator_derivative_local(VecD &state,
                                     VecD &op,
                                     const MPGraphView &graph,
                                     size_t layer_idx,
                                     double gen_coeff,
                                     double param,
                                     const detail::LayerCosAccumulate &cos_acc,
                                     MPI_Comm comm) -> double {
    return state_operator_derivative_local_impl(
        state, op, graph, layer_idx, gen_coeff, param, comm, cos_acc);
}

auto evolve_step_traversal_impl(VecD &op,
                                const LayerTraversal &layer,
                                double param,
                                size_t layer_idx,
                                MPI_Comm comm,
                                const detail::LayerCosScale &cos_scale) -> void {
    const double cos_val = std::cos(2 * param), sin_val = std::sin(2 * param);

    auto *const op_data = op.data();
    const int my_rank_int = mpi::rank(comm);
    const size_t my_rank = static_cast<size_t>(my_rank_int);

    // Snapshot self-B (cross_rank[my_rank] B-indices) BEFORE the cos pass.
    // This must run unconditionally (not gated on MPI size) so single-rank works.
    // The pack for remote ranks skips my_rank, so this is a separate local snapshot.
    const size_t self_b_count = (my_rank < layer.cross_rank_rank_count())
                                    ? layer.cross_rank_b_size(my_rank)
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
            layer.for_each_cross_rank_b_range(my_rank, begin, end,
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
    finish_cross_rank_evolution_exchange(op, layer, cos_val, sin_val, in_flight);

    // Apply self-slot D-entries using the pre-cos snapshot.
    // op[d_index(my_rank,k)] is already post-cos (endpoints are in cos_data), so this only ADDS
    // the sine rotation: op[i] += sin·φ_signed·self_b_snapshot[k].
    if (self_b_count > 0) {
        const size_t self_d_count = layer.cross_rank_d_size(my_rank);
        threading::parallel_for_ranges(
            self_d_count,
            [&](size_t begin, size_t end) {
                layer.for_each_cross_rank_d_range(my_rank, begin, end, [&](size_t k, size_t i, int phi_signed) {
                    op[i] += sin_val * static_cast<double>(phi_signed) * self_b_snapshot[k];
                });
            });
    }
}

template <typename GraphType>
auto evolve_step_impl(VecD &op,
                      const GraphType &graph,
                      double param,
                      size_t layer_idx,
                      MPI_Comm comm,
                      const detail::LayerCosScale &cos_scale) -> void {
    evolve_step_traversal_impl(op, graph.get_layer_traversal(layer_idx), param, layer_idx, comm, cos_scale);
}

auto evolve_step(VecD &op,
                 const MPGraph &graph,
                 double param,
                 size_t layer_idx,
                 const detail::LayerCosScale &cos_scale,
                 MPI_Comm comm) -> void {
    evolve_step_impl(op, graph, param, layer_idx, comm, cos_scale);
}

auto evolve_step(VecD &op, const Layer &layer, double param, const detail::LayerCosScale &cos_scale, MPI_Comm comm)
    -> void {
    evolve_step_traversal_impl(op, layer.traversal(), param, 0, comm, cos_scale);
}

template <typename GraphType>
auto evolve_operator_impl(VecD coeffs,
                          const GraphType &graph,
                          const VecD &params,
                          MPI_Comm comm,
                          const detail::LayerCosScale &cos_scale) -> VecD {
    for (size_t i = 0; i < graph.layers(); ++i) {
        evolve_step_impl(coeffs, graph, params[i], i, comm, cos_scale);
    }
    return coeffs;
}

// Recompute-routed forward overloads (cos scaling via the mandatory callback).
auto evolve_operator(VecD &&coeffs,
                     const MPGraph &graph,
                     const VecD &params,
                     const detail::LayerCosScale &cos_scale,
                     MPI_Comm comm) -> VecD {
    return evolve_operator_impl(std::move(coeffs), graph, params, comm, cos_scale);
}

auto evolve_operator(VecD &&coeffs,
                     const MPGraphView &graph,
                     const VecD &params,
                     const detail::LayerCosScale &cos_scale,
                     MPI_Comm comm) -> VecD {
    return evolve_operator_impl(std::move(coeffs), graph, params, comm, cos_scale);
}

} // namespace monoprop
