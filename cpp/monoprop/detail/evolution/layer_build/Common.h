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
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::detail {

// Sentinel for "no index resolved yet" in the resolve slots. Deliberately equal to
// OperatorIndex::kNotFound, so one `>= size` bound check covers a miss and an unresolved slot alike.
inline constexpr size_t kMissingIndex = std::numeric_limits<size_t>::max();

// A miss that max-sins truncation declined to create: no index, no row, no rotation. Above every row
// index, like kMissingIndex, so the same `>= size` bound check also rejects it.
inline constexpr size_t kDroppedIndex = kMissingIndex - 1;

/* Max-sins truncation (build_graph only). Each row carries the fewest sine factors on any path that
 * reached it. A rotation's partner gets its source's count plus `inc`; one that exists before the gate
 * keeps the min of the two (both endpoints merge), and one that does not is created only if that count is
 * at most `max_sins`. So only creation is ever refused, and a pair is always rotated or not at all. Counts
 * are read and written in place: within one gate each row belongs to at most one rotation. */
//
// In Schrodinger a fully paired row is part of the reference state, which the engine materializes
// lazily (it scores against the state whenever it appears), so `paired_is_initial` treats a fully paired
// partner as present from the start: always created, at count 0, and merged into its source like a hit.
struct SinTrack {
    std::vector<uint8_t> *counts; // the partition's per-row counts, sized to the operator
    uint8_t max_sins;             // at most 254, so a parent count plus inc always fits a byte
    uint8_t inc;                  // 1, or 0 for a call whose sine branches are free
    bool paired_is_initial;       // Schrodinger: see above

    [[nodiscard]] auto child(uint8_t parent) const -> uint8_t { return static_cast<uint8_t>(parent + inc); }
    //! An uncounted call (inc == 0) never declines, even below counts left by a lowered limit.
    [[nodiscard]] auto admits(uint8_t parent) const -> bool { return inc == 0 || parent + inc <= max_sins; }
    //! Merges one endpoint: `row` keeps the fewer of its own count and `partner`'s plus inc.
    auto merge_into(size_t row, uint8_t partner) const -> void {
        uint8_t &c = (*counts)[row];
        c = std::min(c, child(partner));
    }
};

// Marks matched followers without a per-gate O(n) memset: one counter bump clears every mark. Reused
// across gates.
struct MatchedEpochSet {
    // One 2-byte stamp per term: the counter is never serialised, never exchanged, only compared to cur_.
    using Stamp = uint16_t;

    std::vector<Stamp> epoch_;
    Stamp cur_ = 0;

    // Wraps once per 65535 gates; without the fill a stale stamp on a row reused after a truncation aliases.
    auto begin_gate(size_t n) -> void {
        if (cur_ == std::numeric_limits<Stamp>::max()) {
            std::fill(epoch_.begin(), epoch_.end(), Stamp{0});
            cur_ = 0;
        }
        ++cur_;
        if (epoch_.size() < n) {
            epoch_.resize(n, Stamp{0});
        }
    }
    auto mark(size_t i) -> void { epoch_[i] = cur_; }
    [[nodiscard]] auto is_marked(size_t i) const -> bool { return epoch_[i] == cur_; }
    [[nodiscard]] auto memory_bytes() const -> size_t { return epoch_.capacity() * sizeof(Stamp); }
};

// A trivial aggregate on purpose — not std::pair — so DefaultInitVector can skip the zero-fill and lower
// the gather to memmove.
struct PhasedEntry {
    size_t idx; // local target index for in_entries, local source index for out_entries
    int phase;
};

// Uniform per-rank rotation accumulator, drained into the LayerCore's sin_send/sin_recv lists by
// GraphSink::finalize. Self slot: in:=(tgt,φ), out:=(src,φ); cross-rank: in=resolver, out=querier side.
struct PartnerAcc {
    // Default-init storage: every resize-then-overwrite path must fully overwrite [base, base+n) before reading.
    DefaultInitVector<PhasedEntry> in_entries;
    DefaultInitVector<PhasedEntry> out_entries;
};

// Fused contraction (ContractImmediately — the default forward path at all rank counts): one rotation
// (source S, target T, phase φ) applied directly to op_coeffs, bypassing the LayerCore, after cos-scaling S and T:
//   op[S] += -sin·φ·op_pre[T]      op[T] += +sin·φ·op_pre[S]
struct RotationRec {
    size_t src = 0;
    size_t tgt = 0;
    double v_src = 0.0; // op_pre[src] — signed coeff captured at scan emit
    double v_tgt = 0.0; // op_pre[tgt] — resolve-time (hits) / post-extension (inserts) coeff
    int32_t phase = 0;  // ±1 rotation phase (A::emit_phase)
};

// One cross-rank half-rotation (R>1): each rank applies only the add to the slot it owns. The resolver
// owns target T: {T, v_src, +φ}; the querier owns source S: {S, v_tgt, −φ}. Applied like the self-rank
// sin_recv apply: op[local_idx] += sin·phase_signed·v_partner (v_partner off the wire, pre-cos).
struct HalfRotationRec {
    size_t local_idx = 0;     // slot this rank owns: T (resolver) or S (querier)
    double v_partner = 0.0;   // partner's pre-cos coeff: v_src (resolver) / v_tgt (querier)
    int32_t phase_signed = 0; // +φ (resolver) / −φ (querier), pre-signed so the apply never negates
    // Resolver miss halves write a slot inserted this gate (after the fused cos sweep), so the apply folds
    // the gate's cos in (c = cos·c + sin) instead of a plain add. False for hit/querier halves: pre-gate terms.
    bool is_insert = false;
};

// Sink threaded through build_layer's fused branch; a non-null FusedContract* selects the fused path.
// Self-routed rotations (both endpoints local) are full RotationRecs, split into hit and insert lists so
// the apply can fill insert v_tgt (readable only after extend_coeffs) without scanning for a sentinel.
struct FusedContract {
    std::vector<RotationRec> hits;
    std::vector<RotationRec> inserts;
    std::vector<HalfRotationRec> cross_half; // R>1: one half per cross-rank query (resolver +φ, querier −φ)
};

// bit_cast, not a conversion, so v_src arrives over the wire bit-identical.
static_assert(sizeof(size_t) == sizeof(double), "fused query value word assumes 64-bit VecZ element");
inline auto encode_value(double v) -> size_t {
    return std::bit_cast<size_t>(v);
}
inline auto decode_value(size_t word) -> double {
    return std::bit_cast<double>(word);
}

} // namespace monoprop::detail
