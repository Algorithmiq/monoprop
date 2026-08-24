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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop {

// counts/displs for one alltoallv, dense int[P] as MPI requires. A TRANSIENT: materialized into
// per-thread scratch for the exchange being posted, never retained per layer. It describes the recv
// side too, the count matrix being symmetric.
struct LayerExchangeLayout final {
    std::vector<int> counts;
    std::vector<int> displs;
    size_t total_count = 0;
};

} // namespace monoprop

namespace monoprop::detail {

auto checked_mpi_int(size_t value, const char *what) -> int;

} // namespace monoprop::detail

namespace monoprop {

// Materialized cosine (anticommuting) index set: ascending (block_base, 64-bit mask) blocks. Only for sets
// that must be stored (pruned layers) or carried transiently; the fold recompute is the primary replay path.
struct CosMask final {
    std::vector<std::pair<size_t, uint64_t>> blocks;
    size_t total_count = 0;                                     // number of set bits
    auto span_count() const -> size_t { return blocks.size(); } // number of 64-bit blocks
};

// Coalesces absolute indices (or whole word-aligned blocks) into a CosMask. Indices/blocks must arrive in
// ascending order.
struct CosineWordBuilder final {
    CosMask list;
    size_t cur_base = std::numeric_limits<size_t>::max();
    uint64_t cur_bits = 0;
    auto flush() -> void {
        if (cur_bits != 0) {
            list.blocks.emplace_back(cur_base, cur_bits);
            cur_bits = 0;
        }
        cur_base = std::numeric_limits<size_t>::max();
    }
    auto push_index(size_t idx) -> void {
        if (const size_t base = (idx >> 6) << 6; base != cur_base) {
            flush();
            cur_base = base;
        }
        cur_bits |= (uint64_t{1} << (idx & 63U));
        ++list.total_count;
    }
    auto push_word(size_t block_base, uint64_t bits) -> void { // block_base % 64 == 0
        if (bits == 0) {
            return;
        }
        flush();
        list.blocks.emplace_back(block_base, bits);
        list.total_count += static_cast<size_t>(std::popcount(bits));
    }
    auto finish() -> CosMask {
        flush();
        return std::move(list);
    }
};

struct PackedPhaseStorage final {
    bool uses_binary_phases = false;
    size_t total_count = 0;
    std::vector<uint64_t> phase_words;
    std::vector<int8_t> phase_values;

    auto empty() const -> bool { return total_count == 0; }
};

// naming legend for the cross-rank structs below. sin_send (B) = local indices whose coefficient this rank
// sends; sin_recv (D) = local targets to add into — the off-diagonal sin(θ) endpoints of each Givens
// rotation. P = in-block size, Q = out-block size; B = [in]++[out] and D = [out]++[in], so D is derived from B.

// Build-time input for one partner rank: sin_recv_entries are (local target index, signed phase) pairs.
struct CrossRankPartnerData {
    // Default-init: every element is overwritten before any read.
    DefaultInitVector<size_t> sin_send_indices;
    DefaultInitVector<std::pair<size_t, int>> sin_recv_entries;
    // Size of the in-block (P); a boundary within sin_send_indices, so P <= sin_send_indices.size().
    size_t in_count = 0;
};

// One world slot that carries traffic. The dense array this replaces held a record per POSSIBLE
// partner, so it was sized by the flat world P and the graph carried P-squared records however little
// was ever sent; what is stored here is bounded by the traffic instead. Neither the D range (it IS the
// B range, permuted; enforced in build_packed_cross_rank_storage) nor the B/D offset (the running
// prefix over stored entries) is a field: a size_t offset would pad the record from 12 B to 24.
struct CrossRankOccupiedSlot final {
    uint32_t slot = 0; // flat world slot id -- what the dense array encoded by position
    // TermIndex-wide so one slot in one layer can exceed 2^32 endpoints.
    TermIndex sin_send_count = 0;
    TermIndex in_count = 0;
};
// Stated as a width-independent rule so that it holds, and is checked, on both TermIndex widths.
inline constexpr size_t kOccupiedSlotIdField = std::max(sizeof(uint32_t), alignof(TermIndex));
static_assert(sizeof(CrossRankOccupiedSlot) == kOccupiedSlotIdField + 2 * sizeof(TermIndex),
              "the occupied-slot record is the graph's P coefficient: it must carry no padding beyond the "
              "alignment slot its u32 id already occupies.");
static_assert(alignof(CrossRankOccupiedSlot) == alignof(TermIndex),
              "the record aligns to its widest member; if that stops holding the size rule above is "
              "measuring something else.");

// Sentinel for self_pos: this rank's own slot carries no traffic in this layer.
inline constexpr size_t kNoSelfSlot = static_cast<size_t>(-1);

struct PackedCrossRankStorage final {
    // Ascending by slot, unique, every entry carrying traffic.
    std::vector<CrossRankOccupiedSlot> occupied;
    std::vector<TermIndex> sin_send_indices;
    PackedPhaseStorage sin_recv_phases; // one phased entry per D index, sign baked in

    // P. Not recoverable from the array length any more, and MPI_Alltoallv still wants dense counts.
    size_t world_size = 0;

    // The self slot is read in the innermost gradient loop, so it is resolved once at build for O(1).
    size_t self_pos = kNoSelfSlot;
    size_t self_offset = 0;

    auto rank_count() const -> size_t { return world_size; }

    // O(log occupied); a sweep over every partner should use for_each_occupied_slot instead.
    auto find(size_t rank) const -> const CrossRankOccupiedSlot * {
        const auto it = std::ranges::lower_bound(occupied, rank, {}, [](const CrossRankOccupiedSlot &e) {
            return static_cast<size_t>(e.slot);
        });
        return (it == occupied.end() || it->slot != rank) ? nullptr : std::to_address(it);
    }

    auto sin_send_size(size_t rank) const -> size_t {
        const auto *e = find(rank);
        return e == nullptr ? 0 : e->sin_send_count;
    }
    // Same count as the send side: B and D are the same endpoint set, permuted.
    auto sin_recv_size(size_t rank) const -> size_t { return sin_send_size(rank); }
    auto in_count(size_t rank) const -> size_t {
        const auto *e = find(rank);
        return e == nullptr ? 0 : e->in_count;
    }
};

struct LayerCore final {
    PackedCrossRankStorage cross_rank;

    // NOTHING about the exchange is retained here: counts[r] is cross_rank.sin_send_size(r) except at
    // my_rank, displs is its prefix sum, and the recv layout is that same array (the count matrix is
    // symmetric). detail::derive_exchange_layout rebuilds it into per-thread scratch when needed.

    // Per-layer recompute metadata: generator_words = this layer's generator G as backing words;
    // scaled_count = fold truncation bound = operator size after this layer's partner inserts.
    std::vector<uint64_t> generator_words;
    uint64_t scaled_count = 0;

    // Rotation angle = parameters[param_index] * gen_coeff.
    size_t param_index = 0;
    double gen_coeff = 0.0;
    // Shared by all layers of one multi-term gate; absolute across build_graph calls (parameter_mapping).
    size_t gate_index = 0;
};

} // namespace monoprop
