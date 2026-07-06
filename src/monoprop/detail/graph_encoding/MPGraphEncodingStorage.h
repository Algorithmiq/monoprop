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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/Threading.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"

namespace monoprop::detail {

inline auto checked_mpi_int(size_t value, const char *what) -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the MPI int limit {}.", what, value, std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

// Bounds-check a TERM-SPACE index (an index into the operator's term list). Capped by the TermIndex
// width: ~2^32 by default, ~2^64 under -Dmonoprop_WIDE_TERM_INDEX. This is the ceiling that the wide
// build exists to lift, so it must track TermIndex — NOT a fixed 32-bit limit.
inline auto checked_term_index(size_t value, const char *what) -> TermIndex {
    if (value > static_cast<size_t>(std::numeric_limits<TermIndex>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the TermIndex ceiling {}; rebuild with -Dmonoprop_WIDE_TERM_INDEX.",
                        what,
                        value,
                        std::numeric_limits<TermIndex>::max()));
    }
    return static_cast<TermIndex>(value);
}

inline auto checked_packed_phase(int value, const char *what) -> int8_t {
    if (value < static_cast<int>(std::numeric_limits<int8_t>::min())
        || value > static_cast<int>(std::numeric_limits<int8_t>::max())) {
        throw std::overflow_error(std::format("{} {} exceeds the 8-bit phase limit.", what, value));
    }
    return static_cast<int8_t>(value);
}

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

inline auto make_packed_phase_storage(size_t count, bool use_binary_phases) -> PackedPhaseStorage {
    PackedPhaseStorage storage;
    storage.uses_binary_phases = use_binary_phases;
    storage.total_count = count;
    if (use_binary_phases) {
        storage.phase_words.assign(packed_phase_word_count(count), 0);
    }
    else {
        storage.phase_values.resize(count);
    }
    return storage;
}

inline auto set_packed_phase(PackedPhaseStorage &storage, size_t idx, int value, const char *what) -> void {
    if (storage.uses_binary_phases) {
        if (!is_binary_phase(value)) {
            throw std::overflow_error(std::format("{} {} is not representable as a binary packed phase.", what, value));
        }
        if (value < 0) {
            storage.phase_words[packed_phase_word_index(idx)] |= packed_phase_bit_mask(idx);
        }
        else {
            storage.phase_words[packed_phase_word_index(idx)] &= ~packed_phase_bit_mask(idx);
        }
        return;
    }

    storage.phase_values[idx] = checked_packed_phase(value, what);
}

inline auto packed_phase_at(const PackedPhaseStorage &storage, size_t idx) -> int {
    if (storage.uses_binary_phases) {
        return (storage.phase_words[packed_phase_word_index(idx)] & packed_phase_bit_mask(idx)) != 0 ? -1 : 1;
    }
    return static_cast<int>(storage.phase_values[idx]);
}

inline auto packed_phase_storage_bytes(const PackedPhaseStorage &storage) -> size_t {
    return storage.uses_binary_phases ? storage.phase_words.capacity() * sizeof(uint64_t)
                                      : storage.phase_values.capacity() * sizeof(int8_t);
}

inline auto build_packed_cross_rank_storage(std::vector<CrossRankPartnerData> data) -> PackedCrossRankStorage {
    PackedCrossRankStorage storage;
    const size_t num_ranks = data.size();
    storage.ranges.resize(num_ranks);

    // ── Pass 1 (serial, one iteration per rank — cheap): assign per-rank offsets/counts so the
    // fill pass can write distinct slots in parallel, and accumulate the global totals. ──
    size_t total_b = 0;
    size_t total_d = 0;
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        auto &range = storage.ranges[rank];
        range.sin_send_offset = total_b; // size_t; cumulative offset must not narrow (may exceed 2^32)
        range.sin_send_count  = static_cast<TermIndex>(partner.sin_send_indices.size());
        range.sin_recv_offset = total_d; // size_t; cumulative offset must not narrow (may exceed 2^32)
        range.sin_recv_count  = static_cast<TermIndex>(partner.sin_recv_entries.size());
        range.in_count = static_cast<TermIndex>(partner.in_count);
        total_b += partner.sin_send_indices.size();
        total_d += partner.sin_recv_entries.size();
    }

    // ── Pass 2: reduce the binary-phase flag over every element (parallel within each rank; AND is
    // order-independent, so the result is thread-count-independent). B indices are stored as u32 and
    // checked at the store site (checked_term_index throws above the TermIndex ceiling), so no
    // width reduction is needed here. ──
    bool uses_binary_phases = true;
    for (const auto &partner : data) {
        // Only the phase needs scanning — the D index list is derived from B at read time, so it
        // is neither width-checked nor stored.
        const bool non_binary_phase = threading::parallel_reduce_indices<bool>(
            partner.sin_recv_entries.size(), false,
            [&](size_t k, bool &acc) { acc = acc || !is_binary_phase(partner.sin_recv_entries[k].second); },
            [](bool a, bool b) { return a || b; });
        uses_binary_phases = uses_binary_phases && !non_binary_phase;
    }

    storage.sin_send_indices.resize(total_b);

    // NOTE: D indices are not stored — derived from B on read (see cross_rank_sin_recv_index).
    storage.sin_recv_phases = make_packed_phase_storage(total_d, uses_binary_phases);

    // ── Pass 3: fill the flat arrays. Within a rank every slot is distinct, so the index writes
    // are race-free. For binary phases the packed bit-words are SHARED across rank boundaries, so
    // set the (rare) negative-phase bits with an atomic OR — the word is zero-initialised, so a
    // non-negative phase needs no write. Non-binary phases occupy one distinct byte per slot. ──
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        const size_t b_off = storage.ranges[rank].sin_send_offset;
        const size_t d_off = storage.ranges[rank].sin_recv_offset;

        threading::parallel_for_indices(partner.sin_send_indices.size(), [&](size_t k) {
            storage.sin_send_indices[b_off + k] = checked_term_index(partner.sin_send_indices[k], "Cross-rank B index");
        });

        // Single phased D list: phi is already signed (former D- carry -phi, former D+ carry +phi).
        // Only the phase is stored; the D index is derived from B at read time (cross_rank_sin_recv_index).
        threading::parallel_for_indices(partner.sin_recv_entries.size(), [&](size_t k) {
            const auto &[i, phi] = partner.sin_recv_entries[k];
            (void)i;
            const size_t slot = d_off + k;
            if (uses_binary_phases) {
                // Pass 2 already proved every phase is binary; only -1 sets a bit (default is 0).
                if (phi < 0) {
                    __atomic_fetch_or(&storage.sin_recv_phases.phase_words[packed_phase_word_index(slot)],
                                      packed_phase_bit_mask(slot), __ATOMIC_RELAXED);
                }
            }
            else {
                storage.sin_recv_phases.phase_values[slot] = checked_packed_phase(phi, "Cross-rank D phase");
            }
        });
    }

    return storage;
}

inline auto cross_rank_sin_send_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const size_t offset = storage.ranges[rank].sin_send_offset + idx;
    return static_cast<size_t>(storage.sin_send_indices[offset]);
}

// Derive the D index from B. Layout invariant (assemble_partners): B = [in(P)]++[out(Q)] and
// D = [out(Q)]++[in(P)] with P=in_count, Q=sin_recv_count-P. Hence D[idx] = (idx<Q) ? B[P+idx] : B[idx-Q].
// The D index list is therefore not stored (saves one full uint32 array ≈ half of cross_rank).
inline auto cross_rank_sin_recv_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const auto &range = storage.ranges[rank];
    const size_t in_count = range.in_count;          // P
    const size_t out_count = range.sin_recv_count - in_count; // Q
    const size_t sin_send_local = (idx < out_count) ? (in_count + idx) : (idx - out_count);
    return cross_rank_sin_send_index(storage, rank, sin_send_local);
}

inline auto cross_rank_sin_recv_phase(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> int {
    return packed_phase_at(storage.sin_recv_phases, storage.ranges[rank].sin_recv_offset + idx);
}

inline auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t {
    size_t bytes = storage.ranges.capacity() * sizeof(CrossRankPartnerRange)
                 + packed_phase_storage_bytes(storage.sin_recv_phases);
    bytes += storage.sin_send_indices.capacity() * sizeof(TermIndex);
    // D indices are derived from B (not stored), so they contribute nothing.
    return bytes;
}

inline auto layer_exchange_layout_storage_bytes(const LayerExchangeLayout &layout) -> size_t {
    return layout.counts.capacity() * sizeof(int) + layout.displs.capacity() * sizeof(int);
}

// build_layer_exchange_layout_impl: sums sin_send_count * scale per rank.
// PartnerRangeLike must have a sin_send_count field (full-width size_t so checked_mpi_int catches overflow).
template <typename PartnerRangeLike>
inline auto build_layer_exchange_layout_impl(const std::vector<PartnerRangeLike> &ranges, int scale)
    -> LayerExchangeLayout {
    LayerExchangeLayout layout;
    layout.counts.resize(ranges.size());
    layout.displs.resize(ranges.size());
    size_t total = 0;
    for (size_t r = 0; r < ranges.size(); ++r) {
        const size_t count = static_cast<size_t>(scale) * static_cast<size_t>(ranges[r].sin_send_count);
        layout.counts[r] = checked_mpi_int(count, "Layer exchange count");
        layout.displs[r] = checked_mpi_int(total, "Layer exchange displacement");
        total += count;
    }
    layout.total_count = total;
    return layout;
}

inline auto build_layer_exchange_layout(const std::vector<CrossRankPartnerData> &data, int scale)
    -> LayerExchangeLayout {
    // Build temporary range-like objects with sin_send_count for the impl.
    struct BCountOnly { uint32_t sin_send_count; };
    std::vector<BCountOnly> ranges;
    ranges.reserve(data.size());
    for (const auto &partner : data) {
        ranges.push_back({static_cast<uint32_t>(partner.sin_send_indices.size())});
    }
    return build_layer_exchange_layout_impl(ranges, scale);
}

// build_layer_storage_unified: stores C = all anticommuting, with local cycles folded
// into the self-rank partner slot (my_rank).  The exchange layout zeroes counts[my_rank]
// so MPI_Alltoallv never touches the self-rank slot; the replay handles it as a local
// buffer copy.  This matches paper Algorithm 3 (BuildDistributedLayer / ContractLayer).
inline auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners,
                                        size_t my_rank) -> std::shared_ptr<LayerCore> {
    auto storage = std::make_shared<LayerCore>();

    // Build exchange layout excluding self-rank (counts[my_rank] = 0).
    {
        struct BCountOnly { size_t sin_send_count; };
        std::vector<BCountOnly> ranges;
        ranges.reserve(all_partners.size());
        for (size_t r = 0; r < all_partners.size(); ++r) {
            // Self-rank slot: zero MPI count (handled locally by the replay). Full-width count so
            // checked_mpi_int (in build_layer_exchange_layout_impl) throws on overflow instead of wrapping.
            const size_t cnt = (r == my_rank) ? size_t{0} : all_partners[r].sin_send_indices.size();
            ranges.push_back({cnt});
        }
        storage->evolution_exchange_layout = build_layer_exchange_layout_impl(ranges, 1);
        storage->derivative_exchange_layout = build_layer_exchange_layout_impl(ranges, 2);
    }

    // Local cycles are folded into the self-rank cross_rank slot (no PackedLocalCycleStorage).
    storage->cross_rank = build_packed_cross_rank_storage(std::move(all_partners));
    return storage;
}

} // namespace monoprop::detail
