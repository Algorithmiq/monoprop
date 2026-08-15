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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"

namespace monoprop::detail {
// The layer exchange layout and the packed cross-rank storage disagree on the rank count.
class ExchangeLayoutRankMismatch : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

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

inline auto cross_rank_slot(const PackedCrossRankStorage &storage, size_t rank) -> CrossRankSlotView {
    const auto &range = storage.ranges[rank];
    return CrossRankSlotView{.sin_send_indices = storage.sin_send_indices.data() + range.sin_send_offset,
                             .sin_recv_phases = &storage.sin_recv_phases,
                             .phase_offset = range.sin_send_offset,
                             .sin_send_count = range.sin_send_count,
                             .in_count = range.in_count};
}

inline auto slot_sin_send_index(const CrossRankSlotView &slot, size_t idx) -> size_t {
    return static_cast<size_t>(slot.sin_send_indices[idx]);
}

// Invariant B=[in(P)]++[out(Q)], D=[out(Q)]++[in(P)] (P=in_count, Q=sin_send_count-P):
// D[idx] = (idx<Q) ? B[P+idx] : B[idx-Q]. So D is not stored (saves ~half of cross_rank).
inline auto slot_sin_recv_index(const CrossRankSlotView &slot, size_t idx) -> size_t {
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

// The slot-proportional part of cross_rank_storage_bytes: one record per world slot whether or
// not that slot carries traffic. The remainder (indices and phases) scales with terms crossing.
auto cross_rank_slot_record_bytes(const PackedCrossRankStorage &storage) -> size_t;

// World slots carrying any traffic for this layer. Read against rank_count() to get occupancy:
// low occupancy would make a sparse layout pay, high occupancy means only a narrower record does.
auto cross_rank_occupied_slots(const PackedCrossRankStorage &storage) -> size_t;

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

auto layer_exchange_layout_storage_bytes(const LayerExchangeLayout &layout) -> size_t;

// The resolve_recv transpose cache a layer retains. Separate from
// layer_exchange_layout_storage_bytes because that function's result is already carried in a
// shipped metric; folding this in would redefine it.
auto layer_exchange_layout_cache_bytes(const mpi::RecvLayoutCache &cache) -> size_t;

// Local cycles fold into the self-rank slot (my_rank); the exchange layout zeroes counts[my_rank] so
// MPI_Alltoallv skips it (replay does a local copy).
auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore>;
} // namespace monoprop::detail
