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

#pragma once

/*
 * Owner-local layer replay and reverse-derivative kernels, shared by the low-level single-store evaluator
 * (Evolution.cpp, MPFunctions.cpp) and the sharded evaluator (detail/sharded/Evaluation.cpp).
 *
 * Library-internal: included only by compiled library sources, never by an installed header, so every
 * instantiation is compiled with the library's own floating-point flags. The kernels perform no MPI and no
 * synchronization. Where a kernel reads a partner owner's endpoint values it takes a `block_of(rank)` callable
 * returning that partner's block for this owner: the low-level path points into a received MPI buffer, the sharded
 * path into the partner's published snapshot. Slots are visited in ascending order, the self slot is skipped,
 * and each block is read at the same positions, so both paths perform the same operations in the same order.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"

namespace monoprop {

/*!
 * \brief A per-layer cosine callback an evaluation path needs was left empty.
 */
class MissingLayerCallback : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/*!
 * \brief A state's rows and values disagree in length, a sparse row names a slot outside the state, or the operator
 *        being contracted is shorter than the state.
 *
 * One type because every case is a caller-supplied length or index that does not fit the rest of the call.
 */
class EvalStateArgumentError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

} // namespace monoprop

namespace monoprop::detail::replay {

//! Endpoint accumulators, in pre-layer units like the sum they are subtracted from.
struct EndpointContrib {
    double cos_terms = 0.0; //!< Sum of s_old * h over the rotation endpoints.
    double sin_terms = 0.0; //!< Sum of phase * s_old * h_partner over the rotation endpoints.
};

//! One layer's trigonometric factors for the reverse pass.
struct TrigValues {
    double cos_val; //!< cos(g * param)
    double sin_val; //!< sin(g * param)
    double sec_val; //!< 1 / cos_val
    double g_val;   //!< 2 * gen_coeff

    /*!
     * \brief Factors of the rotation by `2 * gen_coeff * param`.
     * \param param     The optimizer parameter.
     * \param gen_coeff The generator coefficient.
     */
    explicit TrigValues(double param, double gen_coeff = 1.0) {
        const double g = 2.0 * gen_coeff;
        cos_val = std::cos(g * param);
        sin_val = std::sin(g * param);
        sec_val = 1.0 / cos_val;
        g_val = g;
    }
};

//! The two endpoint sums added componentwise, self first.
inline auto combine_endpoint_contrib(const EndpointContrib &a, const EndpointContrib &b) -> EndpointContrib {
    return {.cos_terms = a.cos_terms + b.cos_terms, .sin_terms = a.sin_terms + b.sin_terms};
}

//! dE/dθ = −g·(sin·(A − ep.cos_terms) − ep.sin_terms); both sums are in pre-layer units.
inline auto layer_derivative(const TrigValues &trig, double A, const EndpointContrib &ep) -> double {
    return -trig.g_val * (trig.sin_val * (A - ep.cos_terms) - ep.sin_terms);
}

/*!
 * \brief layer_derivative() from the raw cosine accumulation (the kernel's return value, before its ×sec).
 *
 * For an evaluator that finishes a layer in a later phase than it accumulates it. The low-level evaluator computes
 * `A = raw * sec` and the derivative inline in one function, where the compiler may contract the product into the
 * subtraction `A - ep.cos_terms`; keeping `A` as its own statement here, from the unrounded `raw`, gives the compiler
 * the same dataflow. Out of line on purpose: the low-level evaluator adds the derivative to the gradient across a
 * translation-unit boundary, so it never contracts with that addition either.
 */
[[gnu::noinline]] inline auto layer_derivative_from_raw(const TrigValues &trig, double raw, const EndpointContrib &ep)
    -> double {
    const double A = raw * trig.sec_val;
    return -trig.g_val * (trig.sin_val * (A - ep.cos_terms) - ep.sin_terms);
}

// --- Forward replay -------------------------------------------------------------------------------------------

/*!
 * \brief Snapshot this owner's own sin_send values before the cosine pass overwrites them.
 * \param op    The coefficients about to be stepped.
 * \param layer The layer.
 * \param out   Resized to the self slot's endpoint count and overwritten.
 */
inline auto snapshot_self_sources(const VecD &op, const LayerTraversal &layer, VecD &out) -> void {
    const auto self_slot = layer.cross_rank_self_slot();
    const size_t count = self_slot.sin_send_count;
    out.resize(count);
    for (size_t k = 0; k < count; ++k) {
        out[k] = op[detail::slot_sin_send_index(self_slot, k)];
    }
}

/*!
 * \brief Pack every partner slot's pre-cosine sin_send values into its block.
 * \param block_of Callable `block_of(rank) -> double *`, the destination of partner `rank`'s block.
 */
template <class BlockOf>
auto pack_evolution_payload(const VecD &op, const LayerTraversal &layer, size_t my_rank, BlockOf &&block_of) -> void {
    layer.for_each_occupied_slot([&](size_t rank, const detail::CrossRankSlotView &slot) {
        if (rank == my_rank) {
            return;
        }
        double *const dst = block_of(rank);
        for (size_t k = 0; k < slot.sin_send_count; ++k) {
            dst[k] = op[detail::slot_sin_send_index(slot, k)];
        }
    });
}

/*!
 * \brief Add every partner's pre-cosine values into this owner's already cos-scaled sin_recv targets.
 * \param block_of Callable `block_of(rank) -> const double *`, partner `rank`'s block for this owner.
 */
template <class BlockOf>
auto apply_evolution_payload(VecD &op, const LayerTraversal &layer, double sin_val, size_t my_rank, BlockOf &&block_of)
    -> void {
    // op[i] is already cos-scaled, so only the sine term is added; rv[k] is the partner's pre-cos value.
    layer.for_each_occupied_slot([&](size_t rank, const detail::CrossRankSlotView &slot) {
        if (rank == my_rank) {
            return;
        }
        const double *const rv = block_of(rank);
        for (size_t k = 0; k < slot.sin_send_count; ++k) {
            const size_t i = detail::slot_sin_recv_index(slot, k);
            op[i] += sin_val * static_cast<double>(detail::slot_sin_recv_phase(slot, k)) * rv[k];
        }
    });
}

//! Add the self slot's snapshotted sources into its already cos-scaled sin_recv targets.
inline auto apply_self_sources(VecD &op, const LayerTraversal &layer, double sin_val, const VecD &self_sources)
    -> void {
    const auto self_slot = layer.cross_rank_self_slot();
    for (size_t k = 0; k < self_slot.sin_send_count; ++k) {
        const size_t i = detail::slot_sin_recv_index(self_slot, k);
        op[i] += sin_val * static_cast<double>(detail::slot_sin_recv_phase(self_slot, k)) * self_sources[k];
    }
}

// --- Reverse derivatives --------------------------------------------------------------------------------------

/*!
 * \brief Pre-cosine snapshot buffers of one owner, reused across layers, indexed by occupied position.
 *
 * Passed whole rather than re-split at each hand-off, which is how a send buffer becomes a receive one.
 */
struct DerivativeSnapshotScratch {
    std::vector<VecD> sin_send_state; //!< Per occupied position: state at the sin_send endpoints.
    std::vector<VecD> sin_send_op;    //!< Per occupied position: op at the sin_send endpoints.
    std::vector<VecD> sin_recv_state; //!< Per occupied position: state at the sin_recv endpoints.
    std::vector<VecD> sin_recv_op;    //!< Per occupied position: op at the sin_recv endpoints.
    VecD self_recv_op;                //!< Self-slot entry op, kept only where a record overwrites op before it is read.
};

// Emptied, not filled: the self slot recovers live, and a previous layer's snapshot must not be read.
inline auto clear_slot_snapshot(DerivativeSnapshotScratch &snap, size_t pos) -> void {
    snap.sin_send_state[pos].clear();
    snap.sin_send_op[pos].clear();
    snap.sin_recv_state[pos].clear();
    snap.sin_recv_op[pos].clear();
}

inline auto fill_slot_snapshot(DerivativeSnapshotScratch &snap,
                               size_t pos,
                               const VecD &state,
                               const VecD &op,
                               const detail::CrossRankSlotView &slot) -> void {
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

/*!
 * \brief Snapshot pre-cosine (state, op) at every partner slot's sin_send/sin_recv endpoints.
 *
 * The self slot needs none -- it recovers live -- so it is cleared.
 */
inline auto snapshot_remote_endpoints(const VecD &state,
                                      const VecD &op,
                                      const LayerTraversal &layer,
                                      size_t my_rank,
                                      DerivativeSnapshotScratch &snap) -> void {
    // Grow-only: shrinking would free the allocations this scratch exists to reuse.
    const size_t occupied = layer.occupied_slot_count();
    snap.sin_send_state.resize(std::max(snap.sin_send_state.size(), occupied));
    snap.sin_send_op.resize(std::max(snap.sin_send_op.size(), occupied));
    snap.sin_recv_state.resize(std::max(snap.sin_recv_state.size(), occupied));
    snap.sin_recv_op.resize(std::max(snap.sin_recv_op.size(), occupied));
    layer.for_each_occupied_slot(
        [&snap, my_rank, &state, &op](size_t pos, size_t r, const detail::CrossRankSlotView &slot) {
            if (r == my_rank) {
                clear_slot_snapshot(snap, pos);
                return;
            }
            fill_slot_snapshot(snap, pos, state, op, slot);
        });
}

/*!
 * \brief Keep the self slot's entry op at its sin_recv endpoints, which a record overwrites before the self pass.
 */
inline auto snapshot_self_recv_op(const VecD &op, const LayerTraversal &layer, DerivativeSnapshotScratch &snap)
    -> void {
    const auto slot = layer.cross_rank_self_slot();
    snap.self_recv_op.resize(slot.sin_send_count);
    for (size_t k = 0; k < slot.sin_send_count; ++k) {
        snap.self_recv_op[k] = op[detail::slot_sin_recv_index(slot, k)];
    }
}

/*!
 * \brief Pack sin_send (state, op) pairs from the pre-cosine snapshots into every partner slot's block.
 * \param block_of Callable `block_of(rank) -> double *`, the destination of partner `rank`'s block.
 */
template <class BlockOf>
auto pack_derivative_payload(const DerivativeSnapshotScratch &snap,
                             const LayerTraversal &layer,
                             size_t my_rank,
                             BlockOf &&block_of) -> void {
    layer.for_each_occupied_slot([&](size_t pos, size_t rank, const detail::CrossRankSlotView &slot) {
        if (rank == my_rank) {
            return;
        }
        double *const dst = block_of(rank);
        const auto &bs = snap.sin_send_state[pos];
        const auto &bh = snap.sin_send_op[pos];
        for (size_t k = 0; k < slot.sin_send_count; ++k) {
            dst[2 * k] = bs[k];
            dst[(2 * k) + 1] = bh[k];
        }
    });
}

/*!
 * \brief Remote endpoint pass: own pre-cosine values from the sin_recv snapshots, partner values from its block.
 *
 * Must run after the cosine pass -- the ordering is floating-point significant.
 *
 * \param block_of Callable `block_of(rank) -> const double *`, partner `rank`'s (state, op) pairs for this owner.
 * \return The endpoint sums over every partner slot, accumulated in ascending slot order from zero.
 */
template <class BlockOf>
auto apply_derivative_payload(VecD &state,
                              VecD &op,
                              const LayerTraversal &layer,
                              const DerivativeSnapshotScratch &snap,
                              const TrigValues &trig,
                              size_t my_rank,
                              BlockOf &&block_of) -> EndpointContrib {
    EndpointContrib local{};
    layer.for_each_occupied_slot([&](size_t pos, size_t rank, const detail::CrossRankSlotView &slot) {
        if (rank == my_rank) {
            return;
        }
        const double *const rv = block_of(rank);
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
            const double h_p = rv[(2 * k) + 1];
            local.cos_terms += s_old * op[i]; // pre-layer, matching what the cos pass added to A
            local.sin_terms += phi * s_old * h_p;
            op[i] = (h_old * trig.cos_val) + (ps * h_p);
            state[i] = (s_old * trig.cos_val) + (ps * s_p);
        }
    });
    return local;
}

/*!
 * \brief Self-slot endpoint pass over whole rotation pairs.
 *
 * sin_recv entries k and k + count/2 are the two endpoints of one rotation, so reading both before writing either
 * avoids the read-after-write hazard. `h_pre` carries the entry op where a record overwrote it, and is null where
 * op still holds it.
 */
inline auto apply_self_slot_derivative_paired(VecD &state,
                                              VecD &op,
                                              const LayerTraversal &layer,
                                              const TrigValues &trig,
                                              const double *h_pre) -> EndpointContrib {
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
        const double s2 = state[i2] * trig.sec_val;
        const double h1 = h_pre != nullptr ? h_pre[k] : op[i1] * trig.cos_val;
        const double h2 = h_pre != nullptr ? h_pre[k + pairs] : op[i2] * trig.cos_val;
        local.cos_terms += (s1 * op[i1]) + (s2 * op[i2]); // pre-layer, as in A
        local.sin_terms += (phi1 * s1 * h2) + (phi2 * s2 * h1);
        // Inverse-rotation write-back (−sin); see apply_derivative_payload.
        const double ps1 = -trig.sin_val * phi1;
        const double ps2 = -trig.sin_val * phi2;
        op[i1] = (h1 * trig.cos_val) + (ps1 * h2);
        state[i1] = (s1 * trig.cos_val) + (ps1 * s2);
        op[i2] = (h2 * trig.cos_val) + (ps2 * h1);
        state[i2] = (s2 * trig.cos_val) + (ps2 * s1);
    }
    return local;
}

/*!
 * \brief The layer's cosine accumulation A in pre-layer units, with the record stood in and put back.
 *
 * A = Σ s_old·h_pre over all anticommuting indices, endpoints included -- hence the later subtraction. Pre-dividing
 * lets the kernel's own ×sec land back on the recorded value; the restore fixes up the rest.
 */
inline auto accumulate_layer(VecD &state,
                             VecD &op,
                             size_t layer_idx,
                             const TrigValues &trig,
                             const detail::CosRecordView &record,
                             const detail::LayerCosAccumulate &cos_acc) -> double {
    detail::predivide_cos_record(op.data(), record, trig.cos_val);
    const double a = cos_acc(layer_idx, state.data(), op.data(), trig.cos_val, trig.sec_val) * trig.sec_val;
    detail::restore_cos_record(op.data(), record);
    return a;
}

/*!
 * \brief As accumulate_layer(), but returns the kernel's raw accumulation, before its ×sec.
 *
 * The sharded evaluator's form; see layer_derivative_from_raw().
 */
inline auto accumulate_layer_raw(VecD &state,
                                 VecD &op,
                                 size_t layer_idx,
                                 const TrigValues &trig,
                                 const detail::CosRecordView &record,
                                 const detail::LayerCosAccumulate &cos_acc) -> double {
    detail::predivide_cos_record(op.data(), record, trig.cos_val);
    const double raw = cos_acc(layer_idx, state.data(), op.data(), trig.cos_val, trig.sec_val);
    detail::restore_cos_record(op.data(), record);
    return raw;
}

// --- Selective pre-layer records ------------------------------------------------------------------------------

//! Record flag: the coefficients the layer below rotates.
inline constexpr uint8_t kRecordRotationsBelow = 1;
//! Record flag: the layer's own cosine set, which a vanishing cosine destroys.
inline constexpr uint8_t kRecordCosineSet = 2;

/*!
 * \brief Flag the layers whose record can change an answer; return whether any needs its cosine set.
 *
 * Layer j records what layer j-1 rotates, so it earns its keep only once the layers already reversed can amplify an
 * error there by a full significand -- a bound read off the parameters, before any coefficient is touched.
 *
 * \param mapped_params The replay angles, in layer order.
 * \param wanted        Overwritten with one flag set per layer.
 */
inline auto plan_cos_records(std::span<const double> mapped_params, std::vector<uint8_t> &wanted) -> bool {
    const size_t layers = mapped_params.size();
    wanted.assign(layers, 0);
    bool any_cosine_set = false;
    double spread = 0.0;
    for (size_t j = layers; j-- > 0;) {
        const double cos_val = std::cos(2 * mapped_params[j]);
        const double own = -std::log2(std::abs(cos_val));
        if (j > 0 && spread + own >= detail::kRecordSpreadBits) {
            wanted[j] |= kRecordRotationsBelow;
        }
        if (std::abs(cos_val) < detail::kVanishingCos) {
            wanted[j] |= kRecordCosineSet;
            any_cosine_set = true;
        }
        spread += own;
    }
    return any_cosine_set;
}

//! One owner's selective pre-layer records: layer slices back to back.
struct CosRecords {
    std::vector<TermIndex> indices; //!< Recorded coefficient indices, layer slices back to back.
    VecD values;                    //!< The pre-layer coefficient at each of them.
    VecZ offset;                    //!< Per layer: start of its slice, plus a tail entry.
};

//! Empty the records for a replay of `layers` layers, keeping their capacity.
inline auto reset_records(CosRecords &records, size_t layers) -> void {
    records.offset.assign(layers + 1, 0);
    records.indices.clear();
    records.values.clear();
}

/*!
 * \brief Reserve `records` for one replay with exactly what record_pre_layer() appends.
 *
 * Per selected layer: the sin_recv entries of the layer below, plus the layer's cosine set as `count` reports it.
 * Reserving once replaces the geometric growth whose freed blocks the allocator may keep resident. Without `count`,
 * only the rotations below are reserved and the cosine sets grow as before. Only capacity depends on this; the
 * contents are unchanged.
 *
 * \param records The owner's records, after reset_records().
 * \param wanted  The per-layer record flags of plan_cos_records().
 * \param graph   The owner's graph; `wanted` has one entry per layer.
 * \param count   Optional, CosCallbacks::count: exactly what the indices callback appends per layer. Called on the
 *                calling thread, so it must be safe to call there.
 * \throws std::length_error if the total overflows `size_t`.
 * \throws std::bad_alloc if the reservation fails.
 */
inline auto reserve_records(CosRecords &records,
                            std::span<const uint8_t> wanted,
                            const MPGraphView &graph,
                            const detail::LayerCosCount &count) -> void {
    size_t total = 0;
    const auto add = [&total](size_t entries) {
        if (entries > std::numeric_limits<size_t>::max() - total) {
            throw std::length_error("reserve_records: the record size overflows size_t");
        }
        total += entries;
    };
    for (size_t layer_idx = 0; layer_idx < wanted.size(); ++layer_idx) {
        const uint8_t want = wanted[layer_idx];
        if ((want & kRecordRotationsBelow) != 0U) {
            const auto below = graph.get_layer_traversal(layer_idx - 1);
            for (size_t r = 0; r < below.cross_rank_rank_count(); ++r) {
                add(below.cross_rank_sin_recv_size(r));
            }
        }
        if ((want & kRecordCosineSet) != 0U && count) {
            add(count(layer_idx));
        }
    }
    records.indices.reserve(total);
    records.values.reserve(total);
}

/*!
 * \brief Open `layer_idx`'s slice and fill it from the not-yet-stepped `op`.
 *
 * Called in layer order, so each slice starts where the previous one ended.
 */
inline auto record_pre_layer(CosRecords &records,
                             std::span<const uint8_t> wanted,
                             const MPGraphView &graph,
                             const detail::LayerCosIndices &cos_inds,
                             size_t layer_idx,
                             const VecD &op) -> void {
    records.offset[layer_idx] = records.values.size();
    const uint8_t want = wanted[layer_idx];
    if (want == 0) {
        return;
    }
    const size_t begin = records.indices.size();
    if ((want & kRecordRotationsBelow) != 0U) {
        const auto below = graph.get_layer_traversal(layer_idx - 1);
        const auto mark = [&records](size_t, size_t i, auto) { records.indices.push_back(static_cast<TermIndex>(i)); };
        for (size_t r = 0; r < below.cross_rank_rank_count(); ++r) {
            below.for_each_cross_rank_sin_recv_range(r, 0, below.cross_rank_sin_recv_size(r), mark);
        }
    }
    if ((want & kRecordCosineSet) != 0U) {
        cos_inds(layer_idx, records.indices);
    }
    const size_t end = records.indices.size();
    records.values.resize(end);
    for (size_t k = begin; k < end; ++k) {
        records.values[k] = op[records.indices[k]];
    }
}

//! Close the last slice after the forward pass over `layers` layers.
inline auto close_records(CosRecords &records, size_t layers) -> void {
    records.offset[layers] = records.values.size();
}

//! What the reverse pass reads back for `layer_idx`.
inline auto layer_record_in(const CosRecords &records, size_t layer_idx) -> detail::CosRecordView {
    const size_t begin = records.offset[layer_idx];
    const size_t count = records.offset[layer_idx + 1] - begin;
    if (count == 0) {
        return {};
    }
    return {.indices = records.indices.data() + begin, .values = records.values.data() + begin, .count = count};
}

} // namespace monoprop::detail::replay
