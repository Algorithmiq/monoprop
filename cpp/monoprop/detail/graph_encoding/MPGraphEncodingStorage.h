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

inline auto cross_rank_sin_send_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const size_t offset = storage.ranges[rank].sin_send_offset + idx;
    return static_cast<size_t>(storage.sin_send_indices[offset]);
}

// Invariant B=[in(P)]++[out(Q)], D=[out(Q)]++[in(P)] (P=in_count, Q=sin_recv_count-P):
// D[idx] = (idx<Q) ? B[P+idx] : B[idx-Q]. So D is not stored (saves ~half of cross_rank).
inline auto cross_rank_sin_recv_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const auto &range = storage.ranges[rank];
    const size_t in_count = range.in_count;                   // P
    const size_t out_count = range.sin_recv_count - in_count; // Q
    const size_t sin_send_local = (idx < out_count) ? (in_count + idx) : (idx - out_count);
    return cross_rank_sin_send_index(storage, rank, sin_send_local);
}

inline auto cross_rank_sin_recv_phase(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> int {
    return packed_phase_at(storage.sin_recv_phases, storage.ranges[rank].sin_recv_offset + idx);
}

auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t;

auto layer_exchange_layout_storage_bytes(const LayerExchangeLayout &layout) -> size_t;

// Local cycles fold into the self-rank slot (my_rank); the exchange layout zeroes counts[my_rank] so
// MPI_Alltoallv skips it (replay does a local copy).
auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore>;
} // namespace monoprop::detail
