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
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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

inline auto checked_mpi_int(size_t value, const char *what) -> int {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::format("{} {} exceeds the MPI int limit {}.", what, value, std::numeric_limits<int>::max()));
    }
    return static_cast<int>(value);
}

// Per-rank MPI counts = send_counts[r] * scale, with prefix-sum displacements. send_counts is full-width
// (size_t) so checked_mpi_int catches the narrowing to MPI's int.
inline auto build_layer_exchange_layout(const std::vector<size_t> &send_counts,
                                        int scale,
                                        const char *what = "Layer exchange") -> LayerExchangeLayout {
    const std::string count_label = std::format("{} count", what);
    const std::string displacement_label = std::format("{} displacement", what);

    LayerExchangeLayout layout;
    layout.counts.resize(send_counts.size());
    layout.displs.resize(send_counts.size());
    size_t total = 0;
    for (size_t r = 0; r < send_counts.size(); ++r) {
        const size_t count = static_cast<size_t>(scale) * send_counts[r];
        layout.counts[r] = checked_mpi_int(count, count_label.c_str());
        layout.displs[r] = checked_mpi_int(total, displacement_label.c_str());
        total += count;
    }
    layout.total_count = total;
    return layout;
}

// The derivative layout is the evolution layout at 2x (each rotation endpoint carries both the op and
// state payload).
inline auto build_derivative_exchange_layout(const LayerExchangeLayout &evolution) -> LayerExchangeLayout {
    std::vector<size_t> send_counts;
    send_counts.reserve(evolution.counts.size());
    for (const int count : evolution.counts) {
        send_counts.push_back(static_cast<size_t>(count));
    }
    return build_layer_exchange_layout(send_counts, 2, "Layer derivative exchange");
}

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

struct CrossRankPartnerRange final {
    size_t sin_send_offset = 0; // into sin_send_indices; cumulative across ranks, so size_t (may exceed 2^32)
    TermIndex sin_send_count =
        0;                      // == sin_recv_count (both endpoints); TermIndex-wide so one rank/layer can exceed 2^32
    size_t sin_recv_offset = 0; // into sin_recv_phases; cumulative across ranks, so size_t (see sin_send_offset)
    TermIndex sin_recv_count = 0;
    TermIndex in_count = 0;
};

struct PackedCrossRankStorage final {
    std::vector<CrossRankPartnerRange> ranges; // size == R
    std::vector<TermIndex> sin_send_indices;
    PackedPhaseStorage sin_recv_phases; // one phased entry per D index, sign baked in

    auto rank_count() const -> size_t { return ranges.size(); }
    auto sin_send_size(size_t rank) const -> size_t { return ranges[rank].sin_send_count; }
    auto sin_recv_size(size_t rank) const -> size_t { return ranges[rank].sin_recv_count; }
    auto in_count(size_t rank) const -> size_t { return ranges[rank].in_count; }
};

struct LayerCore final {
    PackedCrossRankStorage cross_rank;
    LayerExchangeLayout evolution_exchange_layout;

    auto derivative_exchange_layout() const -> const LayerExchangeLayout &;

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

// Derived lazily (gradient path only), but the 2x overflow check is not deferred with it:
// build_layer_storage_unified validates it eagerly.
inline auto LayerCore::derivative_exchange_layout() const -> const LayerExchangeLayout & {
    if (!derivative_exchange_layout_cache_) {
        derivative_exchange_layout_cache_ = detail::build_derivative_exchange_layout(evolution_exchange_layout);
    }
    return *derivative_exchange_layout_cache_;
}

} // namespace monoprop
