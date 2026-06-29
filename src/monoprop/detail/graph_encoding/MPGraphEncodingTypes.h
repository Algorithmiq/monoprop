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
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop {

struct LayerExchangeLayout final {
    std::vector<int> counts;
    std::vector<int> displs;
    size_t total_count = 0;

    // Cached result of the per-layer send-count exchange (MPI_Alltoall of `counts`). The send
    // pattern is FIXED for a replayed graph, so the recv counts/displs an optimizer would otherwise
    // recompute on every one of its thousands of evaluations are identical each call. Filled lazily
    // on the first exchange and reused while `cached_comm_size` matches. A monoprop graph is bound to a
    // single communicator for its lifetime, so comm size is the relevant invalidation signal; the
    // fields are eval-time-only (default-empty, never touched by the build path). `mutable` because
    // the layout is reached through const traversal handles during evaluation.
    mutable std::vector<int> cached_recv_counts;
    mutable std::vector<int> cached_recv_displs;
    mutable int cached_recv_total = 0;
    mutable int cached_comm_size = -1;
};

} // namespace monoprop

namespace monoprop {

// Materialized cosine (anticommuting) index set: ascending (block_base, 64-bit mask) blocks,
// block_base = absolute operator index of the word's bit 0. The sidecar fold (scale_cos_fold) is
// still the primary path; this type is only for sets that must be stored (pruned pare layers) or
// carried transiently (in-build contraction, combined cos, graph_data export).
struct CosineWordList final {
    std::vector<std::pair<size_t, uint64_t>> blocks;
    size_t total_count = 0; // number of set bits
    auto empty() const -> bool { return blocks.empty(); }
    auto span_count() const -> size_t { return blocks.size(); } // WORD count (parallel split unit)
    auto reset() -> void { blocks.clear(); total_count = 0; }
    auto shrink_to_fit() -> void { blocks.shrink_to_fit(); }
};

// Coalesces ascending absolute indices (or whole word-aligned blocks) into a CosineWordList.
// Mirrors the build scan's two emit modes: whole-word stores (primary, word-aligned) and per-index
// appends (orbital, not word-aligned). Indices/blocks MUST arrive in ascending order.
struct CosineWordBuilder final {
    CosineWordList list;
    size_t cur_base = std::numeric_limits<size_t>::max();
    uint64_t cur_bits = 0;
    auto flush() -> void {
        if (cur_bits != 0) { list.blocks.emplace_back(cur_base, cur_bits); cur_bits = 0; }
        cur_base = std::numeric_limits<size_t>::max();
    }
    auto push_index(size_t idx) -> void {
        const size_t base = (idx >> 6) << 6;
        if (base != cur_base) { flush(); cur_base = base; }
        cur_bits |= (uint64_t{1} << (idx & 63U));
        ++list.total_count;
    }
    auto push_word(size_t block_base, uint64_t bits) -> void { // block_base % 64 == 0
        if (bits == 0) { return; }
        flush();
        list.blocks.emplace_back(block_base, bits);
        list.total_count += static_cast<size_t>(std::popcount(bits));
    }
    auto finish() -> CosineWordList { flush(); return std::move(list); }
};

struct PackedPhaseStorage final {
    bool uses_binary_phases = false;
    size_t total_count = 0;
    std::vector<uint64_t> phase_words;
    std::vector<int8_t> phase_values;

    auto size() const -> size_t { return total_count; }
    auto empty() const -> bool { return total_count == 0; }
};

/// Build-time input for one partner rank's per-layer cross-rank data.
/// `b_indices`: local indices whose op[i] we send to this partner (in paper order:
///   first the "in" block's source idx, then the "out" block's source idx).
/// `d`: (local_target_idx, phi_signed) pairs forming the single phased D list. Former D-
///   entries (sign already negated to -phi) come first, former D+ entries (+phi) second.
///   No boundary is stored — the signed phase carries everything downstream consumers need.
struct CrossRankPartnerData {
    // default-init storage: assemble_partners resizes then overwrites EVERY element in parallel, so
    // the serial resize() zero-fill was pure waste (and the Amdahl anchor that capped this phase ~2.3×).
    DefaultInitVector<size_t> b_indices;
    DefaultInitVector<std::pair<size_t, int>> d;
    // Size of the in-block (P). Layout invariant: b = [in(P)]++[out(Q)], d = [out(Q)]++[in(P)], so the
    // D index list is a permutation of B and is NOT stored — it is derived from B via in_count (see
    // cross_rank_d_index). The D PHASES are not derivable (in/out phases differ) and ARE stored.
    size_t in_count = 0;
    bool empty() const { return b_indices.empty() && d.empty(); }
};

struct CrossRankPartnerRange final {
    TermIndex b_offset = 0; // into b_indices
    TermIndex b_count  = 0; // == d_count (paper invariant); TermIndex-wide so one rank/layer can exceed 2^32
    TermIndex d_offset = 0; // into d_phases (the D index list is derived from B, not stored)
    // Single phased D list: former D- entries (sign baked as -phi) come first, former D+ entries
    // (+phi) second, but no consumer needs the boundary — the signed phase carries everything.
    TermIndex d_count = 0;
    // Size of the in-block within B (P). B = [in(P)]++[out(Q)], D = [out(Q)]++[in(P)] with Q=d_count-P,
    // so D index k = (k<Q) ? B[P+k] : B[k-Q]. Lets us store B only and derive D (see cross_rank_d_index).
    TermIndex in_count = 0;
};

struct PackedCrossRankStorage final {
    std::vector<CrossRankPartnerRange> ranges;     // size == R
    std::vector<TermIndex> b_indices;              // D indices are derived from B on read, not stored
    PackedPhaseStorage    d_phases;                // one phased entry per D index, sign baked in

    auto rank_count() const -> size_t { return ranges.size(); }
    auto b_size(size_t rank)  const -> size_t { return ranges[rank].b_count; }
    auto d_size(size_t rank)  const -> size_t { return ranges[rank].d_count; }
    // P = number of in-entries = number of rotations on this rank (each rotation has one in/target).
    // d_size = in_count + out_count counts BOTH endpoints, so it double-counts self-rank rotations.
    auto in_count(size_t rank) const -> size_t { return ranges[rank].in_count; }
    auto empty() const -> bool { return d_phases.empty() && b_indices.empty(); }
};

struct LayerCore final {
    PackedCrossRankStorage cross_rank;
    LayerExchangeLayout evolution_exchange_layout;
    LayerExchangeLayout derivative_exchange_layout;  // precomputed 2x of evolution_exchange_layout

    // ── Per-layer recompute metadata (NumModes-agnostic) ─────────────────────────────────────────
    // What the future cosine-recompute path needs to reconstruct this layer's cosine set on the fly
    // from the operator's even-parity sidecar (an XOR-fold of the generator's sidecar columns),
    // instead of storing it. These ride WITH the layer (in its shared LayerCore), so they travel
    // correctly through every graph transform (slice/union/consume/Schrödinger-prepend) for free.
    //   - generator_words: this layer's generator G serialized as W = kWords<NumModes> backing words.
    //   - cos_count: the fold truncation bound — the operator size BEFORE this layer's partner
    //     inserts, i.e. the number of operator indices the cosine set may cover.
    std::vector<uint64_t> generator_words;
    uint64_t cos_count = 0;
};

} // namespace monoprop
