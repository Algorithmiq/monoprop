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

    // Cached per-layer recv counts/displs (see mpi::resolve_recv): the send pattern is fixed for a
    // replayed graph, so they are identical every eval. Filled lazily, reused while comm size matches;
    // `mutable` because reached through const traversal handles at eval time.
    mutable mpi::RecvLayoutCache recv_cache;
};

} // namespace monoprop

namespace monoprop {

// Materialized cosine (anticommuting) index set: ascending (block_base, 64-bit mask) blocks. Only for
// sets that must be stored (pruned layers) or carried transiently; the inverted-index fold recompute is
// the primary replay path.
struct CosMask final {
    std::vector<std::pair<size_t, uint64_t>> blocks;
    size_t total_count = 0; // number of set bits
    auto span_count() const -> size_t { return blocks.size(); } // WORD count (parallel split unit)
};

// Coalesces ascending absolute indices (or whole word-aligned blocks) into a CosMask. Indices/blocks
// MUST arrive in ascending order.
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

// NAMING LEGEND for the cross-rank structs below: `sin_send` = paper's send recipe B^{(r')}, `sin_recv`
// = apply recipe D^{(r')} — the off-diagonal sin(θ) endpoints of each Givens rotation. B/D/P/Q are the
// paper's symbols; invariant "b = [in(P)]++[out(Q)], d = [out(Q)]++[in(P)]".

/// Build-time input for one partner rank's per-layer cross-rank data.
/// sin_send_indices: local indices whose op[i] we send (in(P) sources then out(Q) sources).
/// sin_recv_entries: (local_target_idx, signed phi) pairs — the single phased D list (former D- then D+).
struct CrossRankPartnerData {
    // default-init storage: assemble_partners overwrites every element in parallel, so the serial
    // resize() zero-fill was pure waste.
    DefaultInitVector<size_t> sin_send_indices;
    DefaultInitVector<std::pair<size_t, int>> sin_recv_entries;
    // Size of the in-block (P). Layout invariant b=[in(P)]++[out(Q)], d=[out(Q)]++[in(P)]: D indices are
    // derived from B (not stored); D PHASES differ per endpoint and ARE stored. See cross_rank_sin_recv_index.
    size_t in_count = 0;
};

struct CrossRankPartnerRange final {
    size_t sin_send_offset = 0; // into sin_send_indices; cumulative across ranks, so size_t (a layer's total may exceed
                                // 2^32 even when each rank's term count does not)
    TermIndex sin_send_count =
        0;                      // == sin_recv_count (paper invariant); TermIndex-wide so one rank/layer can exceed 2^32
    size_t sin_recv_offset = 0; // into sin_recv_phases; cumulative across ranks, so size_t (see sin_send_offset)
    // Single phased D list (former D- then D+); the signed phase carries everything, no boundary stored.
    TermIndex sin_recv_count = 0;
    // Size of the in-block P within B. D index k = (k<Q) ? B[P+k] : B[k-Q], Q=sin_recv_count-P (derive D from B).
    TermIndex in_count = 0;
};

struct PackedCrossRankStorage final {
    std::vector<CrossRankPartnerRange> ranges; // size == R
    std::vector<TermIndex> sin_send_indices;   // D indices are derived from B on read, not stored
    PackedPhaseStorage sin_recv_phases;        // one phased entry per D index, sign baked in

    auto rank_count() const -> size_t { return ranges.size(); }
    auto sin_send_size(size_t rank) const -> size_t { return ranges[rank].sin_send_count; }
    auto sin_recv_size(size_t rank) const -> size_t { return ranges[rank].sin_recv_count; }
    // P = in-entries = rotations on this rank; sin_recv_size counts both endpoints (double-counts self-rank).
    auto in_count(size_t rank) const -> size_t { return ranges[rank].in_count; }
};

struct LayerCore final {
    PackedCrossRankStorage cross_rank;
    LayerExchangeLayout evolution_exchange_layout;

    // Derivative exchange layout = 2x the evolution layout (each rotation endpoint carries both the op
    // and state payload). Built lazily on the first derivative read (gradient path only), so energy-only
    // runs never allocate it. Definition in MPGraphEncodingStorage.h (needs the checked_mpi_int guard).
    auto derivative_exchange_layout() const -> const LayerExchangeLayout &;

    // Per-layer recompute metadata: lets the cosine-recompute path rebuild this layer's cosine set on the
    // fly (XOR-fold of the generator's inverted-index columns) instead of storing it.
    //   generator_words: this layer's generator G as W=kWords<NumModes> backing words.
    //   scaled_count:    fold truncation bound = operator size AFTER this layer's partner inserts.
    std::vector<uint64_t> generator_words;
    uint64_t scaled_count = 0;

    // Gate info owned by this layer: param_index into the variational parameter vector and generator
    // coefficient g (rotation angle = parameters[param_index] * gen_coeff). Read by evaluation.
    size_t param_index = 0;
    double gen_coeff = 0.0;
    // Index of the ingested gate this layer came from (shared by layers from one multi-term gate;
    // absolute across build_graph calls). Enables per-gate parameter_mapping relabelling.
    size_t gate_index = 0;

private:
    // Lazily-materialized 2x-scaled evolution layout; see derivative_exchange_layout(). mutable because
    // it is filled through const traversal handles at eval time (mirrors LayerExchangeLayout::recv_cache).
    mutable std::optional<LayerExchangeLayout> derivative_exchange_layout_cache_;
};

} // namespace monoprop
