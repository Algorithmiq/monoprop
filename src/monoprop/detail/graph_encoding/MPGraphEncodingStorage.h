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

#include "monoprop/detail/graph_encoding/MPGraphEncodingTypes.h"

namespace monoprop::detail {

// checked_mpi_int and build_layer_exchange_layout live in MPGraphEncodingTypes.h, next to the
// LayerExchangeLayout they guard and build.

// Bounds-check a term-space index. Capped by the TermIndex width (~2^32, or ~2^64 under
// -Dmonoprop_WIDE_TERM_INDEX), so it must track TermIndex, NOT a fixed 32-bit limit.
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

    // Pass 1 (serial, cheap): assign per-rank offsets/counts so the fill pass writes distinct slots in
    // parallel, and accumulate global totals.
    size_t total_b = 0;
    size_t total_d = 0;
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        auto &range = storage.ranges[rank];
        range.sin_send_offset = total_b; // size_t; cumulative offset must not narrow (may exceed 2^32)
        range.sin_send_count = static_cast<TermIndex>(partner.sin_send_indices.size());
        range.sin_recv_offset = total_d; // size_t; cumulative offset must not narrow (may exceed 2^32)
        range.sin_recv_count = static_cast<TermIndex>(partner.sin_recv_entries.size());
        range.in_count = static_cast<TermIndex>(partner.in_count);
        total_b += partner.sin_send_indices.size();
        total_d += partner.sin_recv_entries.size();
    }

    // Pass 2: reduce the binary-phase flag over every element (AND is order-independent ⇒
    // thread-count-independent). B indices are width-checked at the store site, not here.
    bool uses_binary_phases = true;
    for (const auto &partner : data) {
        // Only the phase needs scanning — the D index list is derived from B at read time.
        bool non_binary_phase = false;
        for (const auto &entry : partner.sin_recv_entries) {
            non_binary_phase = non_binary_phase || !is_binary_phase(entry.second);
        }
        uses_binary_phases = uses_binary_phases && !non_binary_phase;
    }

    storage.sin_send_indices.resize(total_b);

    // NOTE: D indices are not stored — derived from B on read (see cross_rank_sin_recv_index).
    storage.sin_recv_phases = make_packed_phase_storage(total_d, uses_binary_phases);

    // Pass 3: fill the flat arrays. Within a rank slots are distinct (race-free). Binary phases set only
    // the rare negative-phase bit in the zero-initialised packed word; non-binary phases get one byte per slot.
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        const size_t b_off = storage.ranges[rank].sin_send_offset;
        const size_t d_off = storage.ranges[rank].sin_recv_offset;

        for (size_t k = 0; k < partner.sin_send_indices.size(); ++k) {
            storage.sin_send_indices[b_off + k] = checked_term_index(partner.sin_send_indices[k], "Cross-rank B index");
        }

        // phi is already signed; only the phase is stored, D index derived from B.
        for (size_t k = 0; k < partner.sin_recv_entries.size(); ++k) {
            const auto &[i, phi] = partner.sin_recv_entries[k];
            (void)i;
            const size_t slot = d_off + k;
            if (uses_binary_phases) {
                // Every phase is binary (Pass 2); only -1 sets a bit. Serial fill (one writer) ⇒ plain OR, no atomics.
                if (phi < 0) {
                    storage.sin_recv_phases.phase_words[packed_phase_word_index(slot)] |= packed_phase_bit_mask(slot);
                }
            }
            else {
                storage.sin_recv_phases.phase_values[slot] = checked_packed_phase(phi, "Cross-rank D phase");
            }
        }
    }

    return storage;
}

inline auto cross_rank_sin_send_index(const PackedCrossRankStorage &storage, size_t rank, size_t idx) -> size_t {
    const size_t offset = storage.ranges[rank].sin_send_offset + idx;
    return static_cast<size_t>(storage.sin_send_indices[offset]);
}

// Derive the D index from B. Invariant B=[in(P)]++[out(Q)], D=[out(Q)]++[in(P)] (P=in_count,
// Q=sin_recv_count-P): D[idx] = (idx<Q) ? B[P+idx] : B[idx-Q]. So D is not stored (saves ~half of cross_rank).
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

inline auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t {
    size_t bytes =
        storage.ranges.capacity() * sizeof(CrossRankPartnerRange) + packed_phase_storage_bytes(storage.sin_recv_phases);
    bytes += storage.sin_send_indices.capacity() * sizeof(TermIndex);
    // D indices are derived from B (not stored), so they contribute nothing.
    return bytes;
}

inline auto layer_exchange_layout_storage_bytes(const LayerExchangeLayout &layout) -> size_t {
    return layout.counts.capacity() * sizeof(int) + layout.displs.capacity() * sizeof(int);
}

// build_layer_storage_unified: local cycles fold into the self-rank slot (my_rank); the exchange layout
// zeroes counts[my_rank] so MPI_Alltoallv skips it (replay does a local copy). Paper Algorithm 3.
inline auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore> {
    auto storage = std::make_shared<LayerCore>();

    // Build exchange layout excluding self-rank (counts[my_rank] = 0).
    {
        std::vector<size_t> send_counts;
        send_counts.reserve(all_partners.size());
        for (size_t r = 0; r < all_partners.size(); ++r) {
            // Self-rank slot: zero MPI count (replay handles it locally). Full-width count so checked_mpi_int catches
            // overflow.
            send_counts.push_back((r == my_rank) ? size_t{0} : all_partners[r].sin_send_indices.size());
        }
        storage->evolution_exchange_layout = build_layer_exchange_layout(send_counts, 1);

        // The derivative layout (2x) is ALLOCATED lazily on first gradient read, but validated here: an
        // overflow must throw during build_graph, not from inside the gradient collective window, where
        // peers are already committed and blocked in mpi::resolve_recv's count round -> distributed hang
        // instead of an error. Discarding the result keeps energy-only runs allocation-free at rest.
        static_cast<void>(build_derivative_exchange_layout(storage->evolution_exchange_layout));
    }

    // Local cycles are folded into the self-rank cross_rank slot (no PackedLocalCycleStorage).
    storage->cross_rank = build_packed_cross_rank_storage(std::move(all_partners));

    // The exchange layouts are indexed by the same rank space the packing loops iterate
    // (cross_rank_rank_count()); assert it here, where both are built, rather than two hops away.
    if (storage->evolution_exchange_layout.counts.size() != storage->cross_rank.rank_count()) {
        throw std::logic_error(std::format("Layer exchange layout covers {} ranks but cross-rank storage has {}.",
                                           storage->evolution_exchange_layout.counts.size(),
                                           storage->cross_rank.rank_count()));
    }
    return storage;
}

} // namespace monoprop::detail
