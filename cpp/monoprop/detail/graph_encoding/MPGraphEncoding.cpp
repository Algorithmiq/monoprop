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

#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"

#include <algorithm>
#include <format>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace monoprop::detail {

auto checked_mpi_int(size_t value, const char *what) -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the MPI int limit {}.", what, value, std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

// The u32 slot id bounds the flat world; a narrowing conversion, so checked rather than cast.
auto checked_world_slot(size_t rank) -> uint32_t {
    if (rank > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::overflow_error(std::format("World slot {} exceeds the {} the occupied-slot record can hold.",
                                              rank,
                                              std::numeric_limits<uint32_t>::max()));
    }
    return static_cast<uint32_t>(rank);
}

auto checked_term_index(size_t value, const char *what) -> TermIndex {
    if (value > static_cast<size_t>(std::numeric_limits<TermIndex>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the TermIndex ceiling {}; rebuild with -Dmonoprop_WIDE_TERM_INDEX.",
                        what,
                        value,
                        std::numeric_limits<TermIndex>::max()));
    }
    return static_cast<TermIndex>(value);
}

auto checked_packed_phase(int value, const char *what) -> int8_t {
    if (value < static_cast<int>(std::numeric_limits<int8_t>::min())
        || value > static_cast<int>(std::numeric_limits<int8_t>::max())) {
        throw std::overflow_error(std::format("{} {} exceeds the 8-bit phase limit.", what, value));
    }
    return static_cast<int8_t>(value);
}

auto make_packed_phase_storage(size_t count, bool use_binary_phases) -> PackedPhaseStorage {
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

auto packed_phase_storage_bytes(const PackedPhaseStorage &storage) -> size_t {
    return storage.uses_binary_phases ? storage.phase_words.capacity() * sizeof(uint64_t)
                                      : storage.phase_values.capacity() * sizeof(int8_t);
}

auto build_packed_cross_rank_storage(const std::vector<CrossRankPartnerData> &data) -> PackedCrossRankStorage {
    PackedCrossRankStorage storage;
    const size_t num_ranks = data.size();
    storage.world_size = num_ranks;

    size_t total_b = 0;
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        const auto &partner = data[rank];
        // The record stores one count and one offset for both B and D, so their equal length is a
        // precondition: a skew would not throw, it would mis-derive Q and read the wrong endpoint.
        if (partner.sin_send_indices.size() != partner.sin_recv_entries.size()) {
            throw CrossRankSlotLayoutError(std::format(
                "Cross-rank slot {} has {} send endpoints against {} recv endpoints; B and D are the same set.",
                rank,
                partner.sin_send_indices.size(),
                partner.sin_recv_entries.size()));
        }
        // P (the in-block) is a boundary WITHIN B. Q = P+Q - P is computed in unsigned width by
        // slot_sin_recv_index, so a P past the end yields not a negative Q but one near 2^64, which
        // every index compares less than -- and every D read then addresses B off the end.
        if (partner.in_count > partner.sin_send_indices.size()) {
            throw CrossRankSlotLayoutError(
                std::format("Cross-rank slot {} declares an in-block of {} inside {} endpoints; the in-block is a "
                            "boundary within the endpoint list, not an addition to it.",
                            rank,
                            partner.in_count,
                            partner.sin_send_indices.size()));
        }
        // A slot with no traffic gets no record; ascending rank order leaves `occupied` sorted.
        if (partner.sin_send_indices.empty()) {
            continue;
        }
        // Checked, not cast: readers rebuild offsets from the STORED counts, so a truncated slot
        // would shift every later slot's window.
        storage.occupied.push_back(
            {.slot = checked_world_slot(rank),
             .sin_send_count = checked_term_index(partner.sin_send_indices.size(), "Cross-rank slot endpoint count"),
             .in_count = checked_term_index(partner.in_count, "Cross-rank slot in-block size")});
        total_b += partner.sin_send_indices.size();
    }
    storage.occupied.shrink_to_fit(); // push_back overshoots, and this array is the thing being shrunk
    const size_t total_d = total_b;

    bool uses_binary_phases = true;
    for (const auto &partner : data) {
        bool non_binary_phase = false;
        for (const auto &[recv_index, phase] : partner.sin_recv_entries) {
            non_binary_phase = non_binary_phase || !is_binary_phase(phase);
        }
        uses_binary_phases = uses_binary_phases && !non_binary_phase;
    }

    storage.sin_send_indices.resize(total_b);
    storage.sin_recv_phases = make_packed_phase_storage(total_d, uses_binary_phases);

    // Same ascending order the offsets accumulate in, so readers reconstruct this exact prefix.
    size_t offset = 0;
    for (const auto &entry : storage.occupied) {
        const auto &partner = data[entry.slot];
        const size_t b_off = offset;
        const size_t d_off = b_off; // equal counts, so equal prefix sums

        for (size_t k = 0; k < partner.sin_send_indices.size(); ++k) {
            storage.sin_send_indices[b_off + k] = checked_term_index(partner.sin_send_indices[k], "Cross-rank B index");
        }

        for (size_t k = 0; k < partner.sin_recv_entries.size(); ++k) {
            const auto &[i, phi] = partner.sin_recv_entries[k];
            (void)i;
            store_packed_phase(storage.sin_recv_phases, d_off + k, phi, "Cross-rank D phase");
        }
        offset += entry.sin_send_count;
    }

    return storage;
}

auto resolve_self_slot(PackedCrossRankStorage &storage, size_t my_rank) -> void {
    storage.self_pos = kNoSelfSlot;
    storage.self_offset = 0;
    size_t offset = 0;
    for (size_t pos = 0; pos < storage.occupied.size(); ++pos) {
        const auto &entry = storage.occupied[pos];
        if (entry.slot == my_rank) {
            storage.self_pos = pos;
            storage.self_offset = offset;
            return;
        }
        offset += entry.sin_send_count;
    }
}

auto cross_rank_storage_bytes(const PackedCrossRankStorage &storage) -> size_t {
    size_t bytes = cross_rank_slot_record_bytes(storage) + packed_phase_storage_bytes(storage.sin_recv_phases);
    bytes += storage.sin_send_indices.capacity() * sizeof(TermIndex);
    return bytes;
}

auto cross_rank_slot_record_bytes(const PackedCrossRankStorage &storage) -> size_t {
    return storage.occupied.capacity() * sizeof(CrossRankOccupiedSlot);
}

auto cross_rank_occupied_slots(const PackedCrossRankStorage &storage) -> size_t {
    // No predicate and no scan any more: an entry exists only if the slot carries traffic.
    return storage.occupied.size();
}

auto cross_rank_endpoint_count(const PackedCrossRankStorage &storage) -> size_t {
    size_t count = 0;
    for (const auto &entry : storage.occupied) {
        count += entry.sin_send_count;
    }
    return count;
}

namespace {
// Formats the label only on the throwing path: this runs per posted exchange, not once at build.
auto checked_exchange_int(size_t value, const char *what, const char *field) -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return checked_mpi_int(value, std::format("{} {}", what, field).c_str());
    }
    return static_cast<int>(value);
}
} // namespace

auto derive_exchange_layout(const PackedCrossRankStorage &cross_rank,
                            size_t my_rank,
                            int scale,
                            LayerExchangeLayout &out,
                            const char *what) -> void {
    const size_t num_ranks = cross_rank.rank_count();
    // assign(), not resize(): `out` is reused across layers and an empty slot must read zero.
    out.counts.assign(num_ranks, 0);
    out.displs.resize(num_ranks);

    // Scatter over the slots that carry traffic: a dense probe would binary-search every possible
    // partner, at O(P log occupied) per exchange, to fill an array that is mostly zeros.
    for_each_occupied_slot(cross_rank, [my_rank, &out, scale, what](size_t slot, const CrossRankSlotView &view) {
        if (slot == my_rank) {
            return; // excluded from the transfer and handled locally, as the stored layout did
        }
        out.counts[slot] = checked_exchange_int(static_cast<size_t>(scale) * view.sin_send_count, what, "count");
    });

    // The prefix sum stays dense: MPI_Alltoallv wants a valid displacement for every rank.
    size_t total = 0;
    for (size_t r = 0; r < num_ranks; ++r) {
        out.displs[r] = checked_exchange_int(total, what, "displacement");
        total += static_cast<size_t>(out.counts[r]);
    }
    out.total_count = total;
}

auto build_layer_storage_unified(const std::vector<CrossRankPartnerData> &all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore> {
    auto storage = std::make_shared<LayerCore>();

    storage->cross_rank = build_packed_cross_rank_storage(all_partners);
    resolve_self_slot(storage->cross_rank, my_rank);

    {
        // Eager validation, result discarded: an int overflow must throw from build_graph, not from
        // inside an exchange whose peers are already committed, where it is a hang. Scale 2 alone
        // suffices -- its counts are exactly 2x scale 1's, so scale 1 cannot overflow on its own.
        LayerExchangeLayout scratch;
        derive_exchange_layout(storage->cross_rank, my_rank, 2, scratch, "Layer derivative exchange");
    }

    return storage;
}

} // namespace monoprop::detail
