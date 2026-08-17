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
#include <optional>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop {

// counts/displs for one alltoallv. Both are dense int[P] because MPI requires that at the call
// site, but this is a TRANSIENT: it is materialized into per-thread scratch for the exchange
// being posted, never retained per layer. A retained one costs P ints x2 x layers x partitions,
// which is O(P^2) across a job for something derivable in a prefix sum.
//
// It describes the recv side too: the count matrix is symmetric, so the transpose of a send
// pattern is that send pattern. There is no RecvLayoutCache anywhere any more -- not here, and
// not on LayerCore, which is where one briefly lived.
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
    // Size of the in-block (P).
    size_t in_count = 0;
};

// One world slot that carries traffic.
//
// Slots with none are not stored at all. The dense array this replaces reserved a record for every
// POSSIBLE partner, so its size was the flat world P and, summed over the P participants that each
// hold one, the graph carried P-squared records regardless of how much was ever sent. What is stored
// instead is bounded by the traffic, which does not depend on P.
//
// The D range is not stored: it IS the B range. B and D hold the same endpoint set in two orders, so
// their counts are equal by construction (Engine.h resizes both to P+Q) and the two prefix sums are
// therefore identical. The equality is enforced in build_packed_cross_rank_storage, not assumed.
//
// The B/D offset is not a field either. It is the running prefix over stored entries in ascending slot
// order, and empty slots contributed zero to the dense prefix, so the derived value equals the stored
// one exactly. Deriving it is what keeps this record at 12 B: a size_t offset would pad it to 24.
struct CrossRankOccupiedSlot final {
    uint32_t slot = 0; // flat world slot id -- what the dense array encoded by position
    // TermIndex-wide so one slot in one layer can exceed 2^32 endpoints.
    TermIndex sin_send_count = 0;
    TermIndex in_count = 0;
};
static_assert(sizeof(TermIndex) != sizeof(uint32_t) || sizeof(CrossRankOccupiedSlot) == 12,
              "narrow build: the occupied-slot record is the graph's P coefficient; keep it padding-free.");

// Sentinel for self_pos: this rank's own slot carries no traffic in this layer.
inline constexpr size_t kNoSelfSlot = static_cast<size_t>(-1);

struct PackedCrossRankStorage final {
    // Ascending by slot, unique, every entry carrying traffic.
    std::vector<CrossRankOccupiedSlot> occupied;
    std::vector<TermIndex> sin_send_indices;
    PackedPhaseStorage sin_recv_phases; // one phased entry per D index, sign baked in

    // P. No longer recoverable from the array length -- that is the point -- but the exchange still
    // needs it, since MPI_Alltoallv wants dense counts and displacements.
    size_t world_size = 0;

    // The self slot is read in the innermost gradient loop, so it gets O(1) access instead of the
    // search the general case pays. Resolved once at build; see resolve_self_slot.
    size_t self_pos = kNoSelfSlot;
    size_t self_offset = 0;

    auto rank_count() const -> size_t { return world_size; }

    // O(log occupied). Callers walking every partner should use for_each_occupied_slot instead, which
    // is O(occupied) for the whole sweep and carries the offset with it.
    auto find(size_t rank) const -> const CrossRankOccupiedSlot * {
        const auto it = std::lower_bound(occupied.begin(),
                                         occupied.end(),
                                         rank,
                                         [](const CrossRankOccupiedSlot &e, size_t r) { return e.slot < r; });
        return (it == occupied.end() || it->slot != rank) ? nullptr : &*it;
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

    // The evolution layout is NOT stored at all: counts[r] is
    // (r == my_rank ? 0 : cross_rank.sin_send_size(r)), displs is its prefix sum, and the total
    // is the last displacement -- so the whole 2*P-int array was a second copy of what `ranges`
    // already says, retained per layer per partition. detail::derive_exchange_layout rebuilds it
    // into per-thread scratch for the transfer being posted.

    // NOTHING about the exchange is retained here -- no send layout, no transpose, no identity
    // for one. The recv layout equals the send layout (the count matrix is symmetric; see
    // Evolution.cpp's derive_layer_exchange), so the transpose that used to be cached per layer
    // at 8 B per world slot is not merely derivable, it is the same array. With it goes the
    // rank-uniform generation id that existed only to make reusing that cache safe, and the
    // hazard it managed: there is no longer a collective on any cache-miss path to split ranks on.

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
