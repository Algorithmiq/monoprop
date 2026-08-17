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

// The slot id is stored as uint32 to keep the occupied record at 12 B. That bounds the flat world at
// ~4.3e9 participants, which is not a limit anyone will meet, but it is a narrowing conversion and so
// it is checked rather than cast.
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
        // B and D are the two endpoints of the same rotation set, so they must be the same
        // length. The record stores one count and one offset for both; checking here is what
        // makes that a precondition instead of a convention. Unchecked, a skew would not throw
        // -- cross_rank_sin_recv_index would mis-derive Q and silently read the wrong endpoint,
        // and Evolution's self-slot snapshot would run off the end of its B-sized buffer.
        if (partner.sin_send_indices.size() != partner.sin_recv_entries.size()) {
            throw std::logic_error(std::format(
                "Cross-rank slot {} has {} send endpoints against {} recv endpoints; B and D are the same set.",
                rank,
                partner.sin_send_indices.size(),
                partner.sin_recv_entries.size()));
        }
        // P (the in-block) is a boundary WITHIN B, so it cannot point past B's end. Checked for the
        // same reason as the skew above, and more urgently: Q = P+Q - P is computed in unsigned
        // width by slot_sin_recv_index, so a P past the end does not produce a negative Q that some
        // signed comparison would reject, it produces a Q near 2^64 that every index compares less
        // than -- and every D read then addresses B at in_count + idx, off the end of the array.
        if (partner.in_count > partner.sin_send_indices.size()) {
            throw std::logic_error(
                std::format("Cross-rank slot {} declares an in-block of {} inside {} endpoints; the in-block is a "
                            "boundary within the endpoint list, not an addition to it.",
                            rank,
                            partner.in_count,
                            partner.sin_send_indices.size()));
        }
        // The whole point: a slot with no traffic gets no record. Ascending rank order makes `occupied`
        // sorted by construction, which is what lets readers binary-search it and lets the offset be a
        // running prefix rather than a stored field.
        if (partner.sin_send_indices.empty()) {
            continue;
        }
        storage.occupied.push_back({.slot = checked_world_slot(rank),
                                    .sin_send_count = static_cast<TermIndex>(partner.sin_send_indices.size()),
                                    .in_count = static_cast<TermIndex>(partner.in_count)});
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

    // Fill in the same ascending order the offsets were accumulated in, so the running prefix here is
    // the one for_each_occupied_slot will reconstruct on every later read.
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

auto derive_exchange_layout(const PackedCrossRankStorage &cross_rank,
                            size_t my_rank,
                            int scale,
                            LayerExchangeLayout &out,
                            const char *what) -> void {
    const std::string count_label = std::format("{} count", what);
    const std::string displacement_label = std::format("{} displacement", what);

    const size_t num_ranks = cross_rank.rank_count();
    // assign() over resize(): every slot that carries nothing must read zero, and `out` is reused
    // across layers, so last layer's counts would otherwise survive into this one's.
    out.counts.assign(num_ranks, 0);
    out.displs.resize(num_ranks);

    // Scatter over the slots that carry traffic instead of interrogating every possible partner.
    // sin_send_size(r) is a binary search once the storage is sparse, so the dense probe this
    // replaces would cost O(P log occupied) per layer per exchange -- to fill an array that is
    // ~82% zeros at P=512 by construction. Occupancy only falls as P grows, so the gap widens.
    for_each_occupied_slot(cross_rank, [&](size_t slot, const CrossRankSlotView &view) {
        if (slot == my_rank) {
            return; // excluded from the transfer and handled locally, as the stored layout did
        }
        out.counts[slot] =
            checked_mpi_int(static_cast<size_t>(scale) * view.sin_send_count, count_label.c_str());
    });

    // The prefix sum stays dense: MPI_Alltoallv wants a displacement for every rank, and an empty
    // slot still needs a valid (repeated) one.
    size_t total = 0;
    for (size_t r = 0; r < num_ranks; ++r) {
        out.displs[r] = checked_mpi_int(total, displacement_label.c_str());
        total += static_cast<size_t>(out.counts[r]);
    }
    out.total_count = total;
}

auto build_layer_storage_unified(std::vector<CrossRankPartnerData> all_partners, size_t my_rank)
    -> std::shared_ptr<LayerCore> {
    auto storage = std::make_shared<LayerCore>();
    const size_t num_ranks = all_partners.size();

    storage->cross_rank = build_packed_cross_rank_storage(std::move(all_partners));
    resolve_self_slot(storage->cross_rank, my_rank);

    // Both are indexed by the same rank space. Checked here because everything downstream now
    // derives the layout from cross_rank, so this is the one place the two can still disagree.
    if (num_ranks != storage->cross_rank.rank_count()) {
        throw ExchangeLayoutRankMismatch(
            std::format("Layer exchange layout covers {} ranks but cross-rank storage has {}.",
                        num_ranks,
                        storage->cross_rank.rank_count()));
    }

    {
        // Derive both scales once at build time and throw the result away. This is purely eager
        // validation: an overflow of MPI's int has to throw from build_graph, not from inside the
        // exchange, where peers are already committed to a transfer of that size -- there it is a
        // distributed hang rather than an error. Scale 2 is checked as well as 1 because the
        // derivative round overflows first and a gradient may run long after the graph was built.
        //
        // The vectors are not kept. They are a prefix sum of what cross_rank already holds, and
        // retaining them per layer per partition is the O(P^2) term this change removes.
        LayerExchangeLayout scratch;
        derive_exchange_layout(storage->cross_rank, my_rank, 1, scratch);
        derive_exchange_layout(storage->cross_rank, my_rank, 2, scratch, "Layer derivative exchange");
    }

    return storage;
}

} // namespace monoprop::detail
