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

// A slot's B and D sides are the same endpoint set, and its in-block a boundary within that set.
// Both are invariants of how the layer was built, so a violation is a bug here, not bad input.
class CrossRankSlotLayoutError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

auto build_packed_cross_rank_storage(const std::vector<CrossRankPartnerData> &data) -> PackedCrossRankStorage;

// Record where my_rank's own slot sits, so the gradient's self-slot reads are O(1). Once per layer.
auto resolve_self_slot(PackedCrossRankStorage &storage, size_t my_rank) -> void;

// One world slot's position in the flat B/D arrays, resolved once. The per-element accessors below
// take this rather than a slot id so that walking a slot's endpoints pays the lookup once, not per
// endpoint -- which is what makes resolving a slot affordable now that it is a search, not an index.
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

// Every occupied slot in ascending order with its B/D offset (the running prefix). func(slot_id, view),
// or func(occupied_pos, slot_id, view) for a caller indexing by occupied position -- handed out here
// because a counter at the call site would skew past those loops' self-slot `return`s. What production
// code should use: O(occupied) for the whole sweep, and it never visits an empty slot.
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

// O(1), for the innermost gradient loop; an all-zero view when this rank's slot carries no traffic.
inline auto cross_rank_self_slot(const PackedCrossRankStorage &storage) -> CrossRankSlotView {
    if (storage.self_pos == kNoSelfSlot) {
        return CrossRankSlotView{.sin_recv_phases = &storage.sin_recv_phases};
    }
    return slot_detail::view_at(storage, storage.occupied[storage.self_pos], storage.self_offset);
}

// Arbitrary slot, O(occupied) because the offset is a prefix over preceding entries. Diagnostic and
// test use: a production loop wants for_each_occupied_slot, and the self slot cross_rank_self_slot.
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
    // Unsigned subtraction; build_packed_cross_rank_storage enforces the precondition that keeps it safe.
    assert(slot.in_count <= slot.sin_send_count && "in-block cannot exceed the slot's endpoint count");
    const size_t out_count = slot.sin_send_count - slot.in_count; // Q
    const size_t sin_send_local = (idx < out_count) ? (slot.in_count + idx) : (idx - out_count);
    return slot_sin_send_index(slot, sin_send_local);
}

// The D phases run parallel to the B indices, so the same prefix sum addresses both arrays.
inline auto slot_sin_recv_phase(const CrossRankSlotView &slot, size_t idx) -> int {
    return packed_phase_at(*slot.sin_recv_phases, slot.phase_offset + idx);
}

// Single-endpoint forms; a loop should resolve the slot once with cross_rank_slot() instead.
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

// The slot-proportional slice of cross_rank_storage_bytes: one record per STORED world slot.
auto cross_rank_slot_record_bytes(const PackedCrossRankStorage &storage) -> size_t;

// World slots carrying any traffic; read against rank_count() to get occupancy.
auto cross_rank_occupied_slots(const PackedCrossRankStorage &storage) -> size_t;

// Total cross-rank endpoints in this layer: the B array's length, and an upper bound on occupied slots.
auto cross_rank_endpoint_count(const PackedCrossRankStorage &storage) -> size_t;

// Derive a layer's send layout into caller-owned scratch: counts[r] = scale * (r == my_rank ? 0 :
// cross_rank.sin_send_size(r)), displs the prefix sum. `out` is resized, not reallocated, on reuse.
auto derive_exchange_layout(const PackedCrossRankStorage &cross_rank,
                            size_t my_rank,
                            int scale,
                            LayerExchangeLayout &out,
                            const char *what = "Layer exchange") -> void;

// Local cycles fold into the self-rank slot (my_rank); the exchange layout zeroes counts[my_rank] so
// MPI_Alltoallv skips it (replay does a local copy).
auto build_layer_storage_unified(const std::vector<CrossRankPartnerData> &all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore>;
} // namespace monoprop::detail
