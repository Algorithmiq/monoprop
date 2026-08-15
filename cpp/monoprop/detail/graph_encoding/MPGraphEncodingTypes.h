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

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/RecvLayout.h"

namespace monoprop {

struct LayerExchangeLayout final {
    std::vector<int> counts;
    std::vector<int> displs;
    size_t total_count = 0;

    // Cached recv counts/displs (see mpi::resolve_recv); mutable — filled through const handles at eval time.
    mutable mpi::RecvLayoutCache recv_cache;
};

} // namespace monoprop

namespace monoprop::detail {

auto checked_mpi_int(size_t value, const char *what) -> int;

// Per-rank MPI counts = send_counts[r] * scale, with prefix-sum displacements. send_counts is full-width
// (size_t) so checked_mpi_int catches the narrowing to MPI's int.
auto build_layer_exchange_layout(const std::vector<size_t> &send_counts, int scale, const char *what = "Layer exchange")
    -> LayerExchangeLayout;

// The derivative layout is the evolution layout at 2x (each rotation endpoint carries both the op and
// state payload).
auto build_derivative_exchange_layout(const LayerExchangeLayout &evolution) -> LayerExchangeLayout;

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

// One record per world slot, occupied or not, retained for every layer -- so on a partitioned run
// this is sizeof(record) x P x layers x partitions per rank, i.e. O(P^2) across the job. That is
// why it holds only what cannot be recovered: B and D are the two endpoints of the same rotation
// set, so their counts are equal and their prefix sums therefore identical, and storing the D pair
// separately cost 16 bytes a slot to say twice what the B pair already said.
// build_packed_cross_rank_storage enforces the equality rather than trusting it.
struct CrossRankPartnerRange final {
    size_t sin_send_offset = 0; // into sin_send_indices AND sin_recv_phases; cumulative, so may exceed 2^32
    // == the D count; TermIndex-wide so one rank/layer can exceed 2^32.
    TermIndex sin_send_count = 0;
    TermIndex in_count = 0;
};

// Pins the saving: 8 + 4 + 4 narrow, 8 + 8 + 8 wide, with no tail padding either way. A new field
// here is paid for once per world slot per layer per partition, so it should be a deliberate act.
static_assert(sizeof(CrossRankPartnerRange) == sizeof(size_t) + 2 * sizeof(TermIndex),
              "CrossRankPartnerRange is the per-world-slot record; keep it free of padding.");

struct PackedCrossRankStorage final {
    std::vector<CrossRankPartnerRange> ranges; // size == the flat world P, not the MPI rank count
    std::vector<TermIndex> sin_send_indices;
    PackedPhaseStorage sin_recv_phases; // one phased entry per D index, sign baked in

    auto rank_count() const -> size_t { return ranges.size(); }
    auto sin_send_size(size_t rank) const -> size_t { return ranges[rank].sin_send_count; }
    // D holds the same endpoints as B in the other order, so it has the same length.
    auto sin_recv_size(size_t rank) const -> size_t { return ranges[rank].sin_send_count; }
    auto in_count(size_t rank) const -> size_t { return ranges[rank].in_count; }
};

struct LayerCore final {
    PackedCrossRankStorage cross_rank;
    LayerExchangeLayout evolution_exchange_layout;

    auto derivative_exchange_layout() const -> const LayerExchangeLayout &;

    // Bytes held by the lazily built derivative layout, 0 while it has never been asked for.
    // Deliberately NOT implemented as a call to derivative_exchange_layout(): that would
    // allocate the very thing being measured, turning an accounting read into a 2*P-int
    // allocation on every layer and making the instrument report its own footprint.
    auto derivative_exchange_layout_bytes() const -> size_t;

    // A copied core must not inherit the source's cache: it is eval-time state, not data.
    auto reset_derivative_exchange_layout() -> void { derivative_exchange_layout_cache_.reset(); }

    // Per-layer recompute metadata: generator_words = this layer's generator G as backing words;
    // scaled_count = fold truncation bound = operator size after this layer's partner inserts.
    std::vector<uint64_t> generator_words;
    uint64_t scaled_count = 0;

    // Rotation angle = parameters[param_index] * gen_coeff.
    size_t param_index = 0;
    double gen_coeff = 0.0;
    // Shared by all layers of one multi-term gate; absolute across build_graph calls (parameter_mapping).
    size_t gate_index = 0;

private:
    mutable std::optional<LayerExchangeLayout> derivative_exchange_layout_cache_;
};

} // namespace monoprop
