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
#include "monoprop/detail/mpi/RecvLayout.h"

namespace monoprop {

struct LayerExchangeLayout final {
    std::vector<int> counts;
    std::vector<int> displs;
    size_t total_count = 0;

    // Cached result of the per-layer send-count exchange (see mpi::resolve_recv). The send pattern is
    // FIXED for a replayed graph, so the recv counts/displs an optimizer would otherwise recompute on
    // every one of its thousands of evaluations are identical each call; the cache is filled lazily on
    // the first exchange and reused while the communicator size matches (a monoprop graph is bound to
    // one communicator for its lifetime). Eval-time-only (default-empty, never touched by the build
    // path); `mutable` because the layout is reached through const traversal handles during evaluation.
    mutable mpi::RecvLayoutCache recv_cache;
};

} // namespace monoprop

namespace monoprop {

// Materialized cosine (anticommuting) index set: ascending (block_base, 64-bit mask) blocks,
// block_base = absolute operator index of the word's bit 0. The on-the-fly inverted index fold recompute
// (scale_cos_lazy) is the primary replay path; this type is only for sets that must be stored (pruned
// pare layers) or carried transiently (in-build contraction, combined cos, graph_data export).
struct CosMask final {
    std::vector<std::pair<size_t, uint64_t>> blocks;
    size_t total_count = 0; // number of set bits
    auto empty() const -> bool { return blocks.empty(); }
    auto span_count() const -> size_t { return blocks.size(); } // WORD count (parallel split unit)
    auto reset() -> void { blocks.clear(); total_count = 0; }
    auto shrink_to_fit() -> void { blocks.shrink_to_fit(); }
};

// Coalesces ascending absolute indices (or whole word-aligned blocks) into a CosMask.
// Mirrors the build scan's two emit modes: whole-word stores (primary, word-aligned) and per-index
// appends (orbital, not word-aligned). Indices/blocks MUST arrive in ascending order.
struct CosineWordBuilder final {
    CosMask list;
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
    auto finish() -> CosMask { flush(); return std::move(list); }
};

struct PackedPhaseStorage final {
    bool uses_binary_phases = false;
    size_t total_count = 0;
    std::vector<uint64_t> phase_words;
    std::vector<int8_t> phase_values;

    auto size() const -> size_t { return total_count; }
    auto empty() const -> bool { return total_count == 0; }
};

// NAMING LEGEND for the cross-rank structs below: `sin_send` == the paper's send recipe B^{(r')},
// `sin_recv` == the paper's apply recipe D^{(r')}. They are the off-diagonal, sin(θ)-coupled endpoints
// of each Givens rotation (the diagonal is the cos-scaled part). The uppercase B/D/P/Q in the comments
// are the paper's symbols; the invariant "b = [in(P)]++[out(Q)], d = [out(Q)]++[in(P)]" is preserved.

/// Build-time input for one partner rank's per-layer cross-rank data.
/// `sin_send_indices`: local indices whose op[i] we send to this partner (in paper order:
///   first the "in" block's source idx, then the "out" block's source idx).
/// `sin_recv_entries`: (local_target_idx, phi_signed) pairs forming the single phased D list. Former D-
///   entries (sign already negated to -phi) come first, former D+ entries (+phi) second.
///   No boundary is stored — the signed phase carries everything downstream consumers need.
struct CrossRankPartnerData {
    // default-init storage: assemble_partners resizes then overwrites EVERY element in parallel, so
    // the serial resize() zero-fill was pure waste (and the Amdahl anchor that capped this phase ~2.3×).
    DefaultInitVector<size_t> sin_send_indices;
    DefaultInitVector<std::pair<size_t, int>> sin_recv_entries;
    // Size of the in-block (P). Layout invariant: b = [in(P)]++[out(Q)], d = [out(Q)]++[in(P)], so the
    // D index list is a permutation of B and is NOT stored — it is derived from B via in_count (see
    // cross_rank_sin_recv_index). The D PHASES are not derivable (in/out phases differ) and ARE stored.
    size_t in_count = 0;
    bool empty() const { return sin_send_indices.empty() && sin_recv_entries.empty(); }
};

struct CrossRankPartnerRange final {
    size_t    sin_send_offset = 0; // into sin_send_indices; cumulative across ranks, so size_t (a layer's total may exceed 2^32 even when each rank's term count does not)
    TermIndex sin_send_count  = 0; // == sin_recv_count (paper invariant); TermIndex-wide so one rank/layer can exceed 2^32
    size_t    sin_recv_offset = 0; // into sin_recv_phases; cumulative across ranks, so size_t (see sin_send_offset)
    // Single phased D list: former D- entries (sign baked as -phi) come first, former D+ entries
    // (+phi) second, but no consumer needs the boundary — the signed phase carries everything.
    TermIndex sin_recv_count = 0;
    // Size of the in-block within B (P). B = [in(P)]++[out(Q)], D = [out(Q)]++[in(P)] with Q=sin_recv_count-P,
    // so D index k = (k<Q) ? B[P+k] : B[k-Q]. Lets us store B only and derive D (see cross_rank_sin_recv_index).
    TermIndex in_count = 0;
};

struct PackedCrossRankStorage final {
    std::vector<CrossRankPartnerRange> ranges;     // size == R
    std::vector<TermIndex> sin_send_indices;              // D indices are derived from B on read, not stored
    PackedPhaseStorage    sin_recv_phases;                // one phased entry per D index, sign baked in

    auto rank_count() const -> size_t { return ranges.size(); }
    auto sin_send_size(size_t rank)  const -> size_t { return ranges[rank].sin_send_count; }
    auto sin_recv_size(size_t rank)  const -> size_t { return ranges[rank].sin_recv_count; }
    // P = number of in-entries = number of rotations on this rank (each rotation has one in/target).
    // sin_recv_size = in_count + out_count counts BOTH endpoints, so it double-counts self-rank rotations.
    auto in_count(size_t rank) const -> size_t { return ranges[rank].in_count; }
    auto empty() const -> bool { return sin_recv_phases.empty() && sin_send_indices.empty(); }
};

struct LayerCore final {
    PackedCrossRankStorage cross_rank;
    LayerExchangeLayout evolution_exchange_layout;
    LayerExchangeLayout derivative_exchange_layout;  // precomputed 2x of evolution_exchange_layout

    // ── Per-layer recompute metadata (NumModes-agnostic) ─────────────────────────────────────────
    // Lets the cosine-recompute path rebuild this layer's cosine set on the fly (an XOR-fold of the
    // generator's inverted index columns) instead of storing it. Held in the shared LayerCore so it survives
    // every graph transform (slice/union/consume/prepend) for free.
    //   - generator_words: this layer's generator G as W = kWords<NumModes> backing words.
    //   - scaled_count: fold truncation bound = the operator size AFTER this layer's partner inserts, so
    //     the recompute reaches the freshly-inserted rotation endpoints the cosine set also covers.
    std::vector<uint64_t> generator_words;
    uint64_t scaled_count = 0;

    // Gate information owned by this layer: the index into the variational parameter
    // vector that drives this layer's rotation, and the generator coefficient g so the
    // rotation angle is parameters[param_index] * gen_coeff. Populated when the layer is
    // appended during graph building; read by evaluation instead of threading the
    // parameter_mapping / gen_coeffs arrays through every call.
    size_t param_index = 0;
    double gen_coeff = 0.0;
    // Index of the ingested gate this layer came from; layers expanded from the same
    // multi-term gate share it. Absolute across build_graph calls (offset by the gate
    // count already in the graph). Enables per-gate parameter_mapping relabelling.
    size_t gate_index = 0;
};

} // namespace monoprop
