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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"

namespace monoprop::detail {

// The ceiling has to track the TermIndex width, not a fixed 32-bit limit.
auto checked_term_index(size_t value, const char *what) -> TermIndex;

auto checked_packed_phase(int value, const char *what) -> int8_t;

inline constexpr size_t kPackedPhaseWordBits = std::numeric_limits<uint64_t>::digits;

inline auto packed_phase_word_count(size_t count) -> size_t {
    return count == 0 ? 0 : (count + kPackedPhaseWordBits - 1) / kPackedPhaseWordBits;
}

inline auto packed_phase_word_index(size_t idx) -> size_t {
    return idx / kPackedPhaseWordBits;
}

inline auto packed_phase_bit_mask(size_t idx) -> uint64_t {
    return uint64_t{1} << (idx % kPackedPhaseWordBits);
}

inline auto is_binary_phase(int value) -> bool {
    return value == -1 || value == 1;
}

auto make_packed_phase_storage(size_t count, bool use_binary_phases) -> PackedPhaseStorage;

auto packed_phase_storage_bytes(const PackedPhaseStorage &storage) -> size_t;

inline auto packed_phase_at(const PackedPhaseStorage &storage, size_t idx) -> int {
    if (storage.uses_binary_phases) {
        return (storage.phase_words[packed_phase_word_index(idx)] & packed_phase_bit_mask(idx)) != 0 ? -1 : 1;
    }
    return static_cast<int>(storage.phase_values[idx]);
}

// In binary storage only φ<0 sets a bit, because the words are assigned zeroed in
// make_packed_phase_storage.
inline auto store_packed_phase(PackedPhaseStorage &storage, size_t idx, int phase, const char *what) -> void {
    if (!storage.uses_binary_phases) {
        storage.phase_values[idx] = checked_packed_phase(phase, what);
    }
    else if (phase < 0) {
        storage.phase_words[packed_phase_word_index(idx)] |= packed_phase_bit_mask(idx);
    }
}

auto build_packed_cross_rank_storage(const std::vector<CrossRankPartnerData> &data) -> PackedCrossRankStorage;

// Record where my_rank's own slot sits, so the gradient's self-slot reads are O(1). Called once per
// layer at build; separate from the builder because only build_layer_storage_unified knows my_rank.
auto resolve_self_slot(PackedCrossRankStorage &storage, size_t my_rank) -> void;

// One world slot's position in the flat B/D arrays, resolved once.
//
// Resolving is per-SLOT work -- an index into `ranges`, which is the array sized by the flat
// world P. The per-element accessors below take this instead of a slot id so that walking a
// slot's endpoints pays that cost once rather than on every endpoint. It matters more than it
// looks: a recv endpoint reads three fields of the record, so the unhoisted form touched the
// P-sized array three times per term. It is also the precondition for ever storing the slots
// sparsely -- under a sparse layout resolving a slot stops being an array index, and anything
// that resolves per term rather than per slot would become unaffordable.
struct CrossRankSlotView final {
    const TermIndex *sin_send_indices = nullptr; // B, already advanced to this slot's offset
    const PackedPhaseStorage *sin_recv_phases = nullptr;
    size_t phase_offset = 0;
    size_t sin_send_count = 0;
    size_t in_count = 0;
};

namespace slot_detail {
inline auto view_at(const PackedCrossRankStorage &storage, const CrossRankOccupiedSlot &entry, size_t offset)
    -> CrossRankSlotView {
    return CrossRankSlotView{.sin_send_indices = storage.sin_send_indices.data() + offset,
                             .sin_recv_phases = &storage.sin_recv_phases,
                             .phase_offset = offset,
                             .sin_send_count = entry.sin_send_count,
                             .in_count = entry.in_count};
}
} // namespace slot_detail

// Every occupied slot in ascending order, each with its B/D offset -- the running prefix, which is
// exactly what the dense layout stored per slot. func(slot_id, view), or func(occupied_pos, slot_id,
// view) for a caller whose own array is indexed by occupied position rather than by world slot.
//
// This is the shape production code should use. It is O(occupied) for the whole sweep rather than
// O(P), and it never visits a slot with nothing in it: under a dense layout those were visited and
// skipped, so on a large world most of the loop was the skip.
//
// The position is handed out here rather than counted at the call site because a hand-rolled counter
// past the self-slot `return`s of those loops would silently skew every later index.
template <typename Func>
auto for_each_occupied_slot(const PackedCrossRankStorage &storage, Func &&func) -> void {
    size_t offset = 0;
    for (size_t pos = 0; pos < storage.occupied.size(); ++pos) {
        const auto &entry = storage.occupied[pos];
        if constexpr (std::is_invocable_v<Func &, size_t, size_t, const CrossRankSlotView &>) {
            func(pos, static_cast<size_t>(entry.slot), slot_detail::view_at(storage, entry, offset));
        }
        else {
            func(static_cast<size_t>(entry.slot), slot_detail::view_at(storage, entry, offset));
        }
        offset += entry.sin_send_count;
    }
}

// O(1). The self slot is read per rotation pair in the innermost gradient loop, so it cannot pay the
// search or the prefix walk that an arbitrary slot does. An all-zero view when it carries no traffic.
inline auto cross_rank_self_slot(const PackedCrossRankStorage &storage) -> CrossRankSlotView {
    if (storage.self_pos == kNoSelfSlot) {
        return CrossRankSlotView{.sin_recv_phases = &storage.sin_recv_phases};
    }
    return slot_detail::view_at(storage, storage.occupied[storage.self_pos], storage.self_offset);
}

// Arbitrary slot, and O(occupied): the offset is a prefix over preceding entries, so resolving one
// slot in isolation walks them. Diagnostic and test use -- a production loop wants
// for_each_occupied_slot, and the self slot wants cross_rank_self_slot.
inline auto cross_rank_slot(const PackedCrossRankStorage &storage, size_t rank) -> CrossRankSlotView {
    size_t offset = 0;
    for (const auto &entry : storage.occupied) {
        if (entry.slot == rank) {
            return slot_detail::view_at(storage, entry, offset);
        }
        if (entry.slot > rank) {
            break;
        }
        offset += entry.sin_send_count;
    }
    return CrossRankSlotView{.sin_recv_phases = &storage.sin_recv_phases};
}

inline auto slot_sin_send_index(const CrossRankSlotView &slot, size_t idx) -> size_t {
    return static_cast<size_t>(slot.sin_send_indices[idx]);
}

// Invariant B=[in(P)]++[out(Q)], D=[out(Q)]++[in(P)] (P=in_count, Q=sin_send_count-P):
// D[idx] = (idx<Q) ? B[P+idx] : B[idx-Q]. So D is not stored (saves ~half of cross_rank).
inline auto slot_sin_recv_index(const CrossRankSlotView &slot, size_t idx) -> size_t {
    // P <= P+Q is a precondition, enforced in build_packed_cross_rank_storage, and it has to be:
    // this subtraction is unsigned, so an in_count past the end of B would not go negative, it
    // would wrap to ~2^64 and send every idx down the (idx < out_count) arm to read B far past its
    // range. Asserted rather than branched on because it is checked once at build and this runs per
    // endpoint of the innermost apply.
    assert(slot.in_count <= slot.sin_send_count && "in-block cannot exceed the slot's endpoint count");
    const size_t out_count = slot.sin_send_count - slot.in_count; // Q
    const size_t sin_send_local = (idx < out_count) ? (slot.in_count + idx) : (idx - out_count);
    return slot_sin_send_index(slot, sin_send_local);
}

// The D phases run parallel to the B indices -- same count per slot, so the same prefix sum
// addresses both. They are still separate arrays; only the offset into them is shared.
inline auto slot_sin_recv_phase(const CrossRankSlotView &slot, size_t idx) -> int {
    return packed_phase_at(*slot.sin_recv_phases, slot.phase_offset + idx);
}

// Single-endpoint forms, for callers that genuinely touch one endpoint of one slot. A loop
// should resolve the slot once with cross_rank_slot() instead of calling these repeatedly.
inline auto cross_rank_sin_send_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    return slot_sin_send_index(cross_rank_slot(storage, rank), idx);
}

inline auto cross_rank_sin_recv_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    return slot_sin_recv_index(cross_rank_slot(storage, rank), idx);
}

inline auto cross_rank_sin_recv_phase(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> int {
    return slot_sin_recv_phase(cross_rank_slot(storage, rank), idx);
}

auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t;

// The slot-proportional slice of cross_rank_storage_bytes: one record per STORED world slot. Once the
// storage is sparse that is one record per OCCUPIED slot, so this stops tracking the world size and
// starts tracking traffic -- which is what makes the two growth laws separable in a measurement.
auto cross_rank_slot_record_bytes(const PackedCrossRankStorage &storage) -> size_t;

// World slots carrying any traffic. Read against rank_count() to get occupancy: low occupancy means a
// sparse layout pays, high occupancy means only a narrower record would.
auto cross_rank_occupied_slots(const PackedCrossRankStorage &storage) -> size_t;

// Total cross-rank endpoints in this layer -- the length of the B array, so the traffic itself. It
// bounds cross_rank_occupied_slots from above (an occupied slot holds at least one endpoint) and,
// unlike the slot count, does not depend on the world size at all.
auto cross_rank_endpoint_count(const PackedCrossRankStorage &storage) -> size_t;

// Derive a layer's send layout into caller-owned scratch instead of reading a stored one.
//
// counts[r] = scale * (r == my_rank ? 0 : cross_rank.sin_send_size(r)), displs the prefix sum --
// the same rule build_layer_storage_unified used to build the stored copy, so this reproduces it
// exactly rather than approximating it. `out` is resized, not reallocated, when reused across
// layers at a fixed world size.
auto derive_exchange_layout(const PackedCrossRankStorage &cross_rank,
                            size_t my_rank,
                            int scale,
                            LayerExchangeLayout &out,
                            const char *what = "Layer exchange") -> void;

// Local cycles fold into the self-rank slot (my_rank); the exchange layout zeroes counts[my_rank] so
// MPI_Alltoallv skips it (replay does a local copy).
auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore>;
} // namespace monoprop::detail
