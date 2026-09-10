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

// Per-partition scratch the layer build reuses from gate to gate. Nothing here carries state between
// gates -- every member is re-initialised by the gate that uses it -- so it is owned by the propagator
// only to keep its capacity, and a copied propagator starts with an empty one. `counters` is the one
// exception: it accumulates over a call, and its owner resets it (MonomialPropagator::run_gate_loop_).

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/TableJoin.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/operator/OperatorIndex.h"

namespace monoprop::detail {

/*! @brief The protocol's per-row state for one gate (Engine.h has the rules), as six bitsets over rows.
 *
 *  The scan sets `rot` (the term's own emission predicate) and `foll` (the pivot bit: which endpoint of
 *  a pair the row is); the probe sets `matched` the moment a record confirms the row; the resolve sets
 *  `received` (the partner's record arrived, i.e. the partner is tracked) and `partner_rot` (that
 *  record's rot bit); round 2 sets `answered` (the partner is tracked but silent, and sent its
 *  coefficient back instead of a record). `matched` and `received` name the same event at two moments:
 *  the probe runs over the whole gate before the resolve does, so a resolve-time rule that read
 *  `matched` would see confirms the resolve has not reached yet, which is why the two are kept apart
 *  (the pair-once rule reads `matched` at the probe and `received` at the resolve, Resolve.h join_self).
 *
 *  Only rows of Anti(G) are ever set or read, and Anti(G) is exactly the set bits of the gate's `nz`
 *  words, so a gate clears its state by zeroing one word per nz word rather than the whole operator's
 *  worth of bits. Rows inserted by the gate's own mints are past the pre-gate size and carry no state.
 */
class RowMarks {
public:
    /*! @brief Sizes the bitsets to `n_rows`, rebinds the six bases, and clears the words `nz` names.
     *
     *  Must run after the fold's pass 1 (which produces `nz`) and before the emit pass sets a bit -- and
     *  it is the only place the vectors can move, which is what makes the bases good for the whole gate.
     */
    auto begin(size_t n_rows, std::span<const EvenParityNzWord> nz) -> void {
        const size_t words = (n_rows + 63) / 64;
        for (auto *bits : arrays_()) {
            if (bits->size() < words) {
                bits->resize(words, 0);
            }
        }
        bind_();
        // The six bases in registers for the whole clear: `arrays_()` would otherwise rebuild its
        // pointer array once per anticommuting word.
        uint64_t *const rot = rot_;
        uint64_t *const foll = foll_;
        uint64_t *const matched = matched_;
        uint64_t *const received = received_;
        uint64_t *const partner_rot = partner_rot_;
        uint64_t *const answered = answered_;
        for (const auto &w : nz) {
            const size_t wi = w.base / 64;
            rot[wi] = 0;
            foll[wi] = 0;
            matched[wi] = 0;
            received[wi] = 0;
            partner_rot[wi] = 0;
            answered[wi] = 0;
        }
    }

    auto set_rot(size_t row) -> void { set_(rot_, row); }
    auto set_foll(size_t row) -> void { set_(foll_, row); }
    //! The whole pivot word at once: `nz` already carries word `word`'s followers, and begin() has just
    //! zeroed it, so the scan sets them in one store instead of one per anticommuting row.
    auto set_foll_word(size_t word, uint64_t bits) -> void { foll_[word] = bits; }
    auto set_matched(size_t row) -> void { set_(matched_, row); }
    auto set_received(size_t row) -> void { set_(received_, row); }
    auto set_partner_rot(size_t row) -> void { set_(partner_rot_, row); }
    auto set_answered(size_t row) -> void { set_(answered_, row); }
    [[nodiscard]] auto rot(size_t row) const -> bool { return get_(rot_, row); }
    [[nodiscard]] auto foll(size_t row) const -> bool { return get_(foll_, row); }
    [[nodiscard]] auto matched(size_t row) const -> bool { return get_(matched_, row); }
    [[nodiscard]] auto received(size_t row) const -> bool { return get_(received_, row); }
    [[nodiscard]] auto partner_rot(size_t row) const -> bool { return get_(partner_rot_, row); }
    [[nodiscard]] auto answered(size_t row) const -> bool { return get_(answered_, row); }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t bytes = 0;
        for (const auto *bits : arrays_()) {
            bytes += bits->capacity() * sizeof(uint64_t);
        }
        return bytes;
    }

private:
    static auto set_(uint64_t *bits, size_t row) -> void { bits[row >> 6U] |= uint64_t{1} << (row & 63U); }
    static auto get_(const uint64_t *bits, size_t row) -> bool { return ((bits[row >> 6U] >> (row & 63U)) & 1U) != 0; }
    auto bind_() -> void {
        rot_ = rot_words_.data();
        foll_ = foll_words_.data();
        matched_ = matched_words_.data();
        received_ = received_words_.data();
        partner_rot_ = partner_rot_words_.data();
        answered_ = answered_words_.data();
    }
    auto arrays_() -> std::array<std::vector<uint64_t> *, 6> {
        return {&rot_words_, &foll_words_, &matched_words_, &received_words_, &partner_rot_words_, &answered_words_};
    }
    auto arrays_() const -> std::array<const std::vector<uint64_t> *, 6> {
        return {&rot_words_, &foll_words_, &matched_words_, &received_words_, &partner_rot_words_, &answered_words_};
    }

    std::vector<uint64_t> rot_words_;
    std::vector<uint64_t> foll_words_;
    std::vector<uint64_t> matched_words_;
    std::vector<uint64_t> received_words_;
    std::vector<uint64_t> partner_rot_words_;
    std::vector<uint64_t> answered_words_;
    // The bitsets' bases, rebound by begin() and valid for the gate it opened: a single load off `this`
    // instead of a load of the owning vector's data pointer, on every mark this gate sets or reads.
    // Null until the first begin(), which every path that touches a mark runs first. A copy of a
    // GateScratch would carry the source's bases, so the propagator's copy constructor default-builds
    // its scratch rather than copying it (MonomialPropagator.inl).
    uint64_t *rot_ = nullptr;
    uint64_t *foll_ = nullptr;
    uint64_t *matched_ = nullptr;
    uint64_t *received_ = nullptr;
    uint64_t *partner_rot_ = nullptr;
    uint64_t *answered_ = nullptr;
};

/*! @brief The keys that missed, in join order: miss j becomes row base + j.
 *
 *  Keys are source^G over globally distinct sources and ^G is injective, so the keys of one gate are
 *  pairwise distinct and every miss is a distinct absent monomial, whichever slot or record it came from.
 */
template <size_t NumModes>
struct MissStage {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    DefaultInitVector<PosT> pos_flat;
    std::vector<size_t> pos_off;
    std::vector<uint16_t> k_of;

    [[nodiscard]] auto size() const -> size_t { return k_of.size(); }
    //! Sizes the per-miss arrays to the join's exact mint upper bound, so push() never reallocates them.
    auto reserve(size_t n) -> void {
        pos_off.reserve(n);
        k_of.reserve(n);
    }
    auto clear() -> void {
        pos_flat.clear();
        pos_off.clear();
        k_of.clear();
    }
    auto push(std::span<const PosT> pos) -> void {
        pos_off.push_back(pos_flat.size());
        k_of.push_back(static_cast<uint16_t>(pos.size()));
        pos_flat.insert(pos_flat.end(), pos.begin(), pos.end());
    }
    [[nodiscard]] auto positions_at(size_t j) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[j], k_of[j]);
    }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        return (pos_flat.capacity() * sizeof(PosT)) + (pos_off.capacity() * sizeof(size_t))
               + (k_of.capacity() * sizeof(uint16_t));
    }
};

/*! @brief The records one exchange delivered, decoded (Resolve.h decode_incoming_records fills it).
 *
 *  Record g belongs to the sender at window index k iff goff[k] <= g < goff[k+1]; within a sender they
 *  are in the sender's stream order.
 */
template <size_t NumModes>
struct IncomingRecords {
    // The operator store's position width, not the wire's: these positions exist to become rows.
    using PosT = typename OperatorIndex<NumModes>::PosT;

    // The slots these records came from: record g belongs to window index k iff goff[k] <= g < goff[k+1].
    mpi::SlotWindow window;
    std::vector<size_t> goff;           // window.count+1 flat offsets: g = goff[k] + q
    DefaultInitVector<int8_t> phase_of; // g -> record phase
    DefaultInitVector<uint8_t> rot_of;  // g -> record rot bit
    DefaultInitVector<double> val_of;   // g -> record value; sized only for the Fused form
    size_t nq_total = 0;

    // The keys as they arrived, flat: record g owns pos_flat[pos_off[g] .. pos_off[g] + k_of[g]).
    DefaultInitVector<PosT> pos_flat;
    DefaultInitVector<size_t> pos_off;
    // g -> positions in the key. A popcount of a 2*NumModes bitset, which QueryWire already requires to
    // fit a uint16_t (its kBits static_assert), so this is the wire's own width.
    DefaultInitVector<uint16_t> k_of;
    // g -> the key's join tag, folded off the decoded positions: the join's key for this record. The
    // 32-bit tag rather than the 64-bit fingerprint it projects, because the join wants nothing else and
    // the wire is unchanged either way (RowKey.h join_tag).
    DefaultInitVector<uint32_t> tag_of;

    //! Record g's ascending positions, as a view into pos_flat.
    [[nodiscard]] auto positions_at(size_t g) const -> std::span<const PosT> {
        return std::span<const PosT>(pos_flat).subspan(pos_off[g], k_of[g]);
    }
    [[nodiscard]] auto value_at(size_t g) const -> double { return val_of.empty() ? 0.0 : val_of[g]; }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        return (goff.capacity() * sizeof(size_t)) + (phase_of.capacity() * sizeof(int8_t))
               + (rot_of.capacity() * sizeof(uint8_t)) + (val_of.capacity() * sizeof(double))
               + (pos_flat.capacity() * sizeof(PosT)) + (pos_off.capacity() * sizeof(size_t))
               + (k_of.capacity() * sizeof(uint16_t)) + (tag_of.capacity() * sizeof(uint32_t));
    }
};

template <size_t NumModes>
struct GateScratch {
    using PosT = typename OperatorIndex<NumModes>::PosT;

    TableJoin<NumModes> join;         // this gate's records probed against the operator's term table
    RowMarks marks;                   // the protocol's per-row bits for this gate's anticommuting rows
    std::vector<EvenParityNzWord> nz; // the fold's nonzero words: Anti(G), read by the emit pass and the join
    std::vector<PosT> partner;        // one partner's positions; sized 2*NumModes, the partner's upper bound
    std::vector<uint16_t> gen;        // the generator's ascending positions, the merge's second input
    // The gate's mints, in join order, and the records the exchange delivered. Both are here because the
    // decode and the join write them through out-parameters, NOT so that their storage outlives the gate:
    // it does not, and release_gate_stages() below is where it goes back.
    MissStage<NumModes> misses;
    IncomingRecords<NumModes> incoming_records;
    /*
     * The send buffers pair_exchange's lifetime rule needs. Peers read a published buffer IN PLACE, and
     * the first moment every peer is proved done with it is the return of this partition's next call, so
     * a gate's records must outlive the gate -- the second reason (with `self_records_hint`) something
     * here crosses a gate boundary.
     *
     * ONE buffer per round rather than a pool alternated per call: sharing would let a round-2 staging
     * inherit the capacity a round-1 gate left in the same slot (reuse_wire clears a slot but keeps its
     * storage), making the response buffer as wide as the query buffer. Two query buffers because a
     * one-call (graph) gate's queries are still being read while the next gate's scan is already writing;
     * the response buffer needs no twin, since only the two-call fused sink stages responses at all.
     */
    std::array<mpi::WindowVec<VecZ>, 2> wire_q;
    mpi::WindowVec<VecZ> wire_r;
    size_t wire_gate = 0; // parity of `wire_q`; bumped once per gate that exchanges
    // The outer descriptor array pair_exchange takes, which the verb copies before its barrier and the
    // caller may reuse on return.
    std::vector<std::span<const size_t>> wire_spans;
    // Slot views of an alltoallv result, so the collective path reaches the same decode surface. Owns
    // nothing, and is only valid while the buffer it was filled from is.
    std::vector<std::span<const size_t>> slot_views;
    // How many records the previous gate of this partition staged for itself: what the next gate's
    // self-slot reserve is sized from (Scan.h). A hint only -- wrong in either direction it costs at most
    // a few pushes their geometric grow -- which is why it may cross gates although nothing else here does.
    size_t self_records_hint = 0;
    ExchangeCounters counters; // COMMPROF's per-call wire volume; reset by the caller, not per gate
    // Widest instant of the gate's per-gate buffers over the call, in bytes. Those buffers die with the
    // gate or are resized under it, so no resting field can name their peak; reset by the caller
    // alongside `counters`.
    size_t buffers_hwm_bytes{0uz};

    /*!
     * @brief This gate's round-1 buffer. The scan takes it, the engine puts it back.
     *
     * `two_rounds` is the sink's `wants_responses`: a sink that answers needs no twin, because its OWN
     * round-2 call is the one the lifetime rule measures the queries against, and that call is inside
     * the same gate. A sink that makes one call per gate does need the twin -- its queries are still
     * published when the next gate's scan starts writing -- so it alternates.
     */
    [[nodiscard]] auto wire_queries(bool two_rounds) -> mpi::WindowVec<VecZ> & {
        return wire_q[two_rounds ? 0U : (wire_gate & 1U)];
    }

    /*! @brief Gives the two per-gate stages their storage back, at the end of the gate that filled them.
     *
     *  They are the one part of this scratch whose capacity must NOT survive its gate. The wire buffers
     *  above have to (a peer may still be reading a published one), and everything else -- the join's hit
     *  slots, `marks`, `nz` -- only ever grows to the operator's own width, so its resting figure already
     *  is its peak. A gate's mints and the records it decoded are proportional to THAT gate's |Anti(G)|:
     *  kept, the widest gate of the call would rest in the partition's footprint for the remainder of it,
     *  which is 27.2 MB of ledger and 23.6 MB of kernel VmHWM on the 9.26 M-term cell at P=16 and
     *  215.6 MiB at the 250 M-term rung. Freed, they cost the gate that follows the allocations the
     *  engine paid when it owned them by value, and the gate stamp still prices them at their widest
     *  instant (Engine.h stamp_gate_buffers_), which is the figure a per-gate slab would have to serve.
     */
    auto release_gate_stages() -> void {
        misses = MissStage<NumModes>{};
        incoming_records = IncomingRecords<NumModes>{};
    }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        return join.memory_bytes() + marks.memory_bytes() + (nz.capacity() * sizeof(EvenParityNzWord))
               + (partner.capacity() * sizeof(PosT)) + (gen.capacity() * sizeof(uint16_t)) + misses.memory_bytes()
               + incoming_records.memory_bytes() + wire_bytes()
               + ((wire_spans.capacity() + slot_views.capacity()) * sizeof(std::span<const size_t>));
    }

    //! The wire buffers, which outlive their gate and so are NOT covered by the gate-buffer stamp.
    [[nodiscard]] auto wire_bytes() const -> size_t {
        return wire_slot_bytes_(wire_q[0]) + wire_slot_bytes_(wire_q[1]) + wire_slot_bytes_(wire_r);
    }

private:
    static auto wire_slot_bytes_(const mpi::WindowVec<VecZ> &wire) -> size_t {
        size_t bytes = wire.capacity() * sizeof(VecZ);
        for (const VecZ &slot : wire) {
            bytes += slot.capacity() * sizeof(size_t);
        }
        return bytes;
    }
};

} // namespace monoprop::detail
