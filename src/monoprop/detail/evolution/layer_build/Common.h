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
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::detail {

// ─── MatchedEpochSet ───────────────────────────────────────────────────────────
// Marks matched followers without the per-gate O(n) allocate+memset a std::vector<uint8_t>(n, 0)
// would pay: slot i is marked iff epoch_[i] == cur_, so starting a gate is one counter bump (all
// marks clear in O(1)) and only the NEW tail is written when the operator grew. Owned by the
// propagator, reused across gates. ≤1 writer per slot per gate (distinct leaders → distinct found
// indices via injective ⊕G), so no atomics.
struct MatchedEpochSet {
    std::vector<uint32_t> epoch_;
    uint32_t cur_ = 0;

    // Start a gate over slots [0, n). u32 wrap resets the array — once per 2^32-1 gates.
    auto begin_gate(size_t n) -> void {
        if (cur_ == std::numeric_limits<uint32_t>::max()) {
            std::fill(epoch_.begin(), epoch_.end(), 0);
            cur_ = 0;
        }
        ++cur_;
        if (epoch_.size() < n) {
            epoch_.resize(n, 0);
        }
    }
    auto mark(size_t i) -> void { epoch_[i] = cur_; }
    [[nodiscard]] auto is_marked(size_t i) const -> bool { return epoch_[i] == cur_; }
};

// One rotation participant: a local operator index plus its ±1 phase. A trivial aggregate on purpose
// (see PartnerAcc) — no std::pair, so DefaultInitVector can skip the zero-fill and memmove the gather.
struct PhasedEntry {
    size_t idx; // local target index for in_entries, local source index for out_entries
    int phase;
};

// ─── PartnerAcc ────────────────────────────────────────────────────────────────
// Uniform per-rank rotation accumulator, drained at finish() into the LayerCore's two per-rank
// participant arrays that contraction consumes: B (the partner index list) and D (the (index,
// signed-phase) list). The self slot is just the partner with in:=(tgt,φ), out:=(src,φ); cross-rank
// has in=resolver side, out=querier side. Every rank assembles B/D from in/out the same way — the
// exact layout (B=[in]++[out], D=[out]++[in]) lives at assemble_partners, the SINGLE copy, so the
// two descriptions can't drift.
struct PartnerAcc {
    // Default-init storage: PhasedEntry is a trivial aggregate, so resize-then-overwrite paths skip
    // the serial zero-fill (better parallel scaling) AND the parallel gather's std::copy lowers to
    // memmove. Load-bearing: every such path fully overwrites [base, base+n) before any read (see
    // resolve_incoming_queries / insert_deferred_self_misses / append_parts_in_order), so the skipped
    // init is never observed. push_back/emplace are unaffected.
    DefaultInitVector<PhasedEntry> in_entries;  // (local_target_idx, phase)
    DefaultInitVector<PhasedEntry> out_entries; // (local_source_idx, phase)
};

// ─── Fused contraction (ContractImmediately — the default forward path at all rank counts) ──
// One rotation applied DIRECTLY to op_coeffs, bypassing the transient LayerCore + evolve_step's
// self-B snapshot gather and B/D index decode. Each rotation (source S, partner target T, phase φ):
//   op[S] += -sin·φ·op_pre[T]      op[T] += +sin·φ·op_pre[S]
// applied AFTER cos-scaling S and T. v_src/v_tgt are the PRE-cos source/target coefficients.
// Each op slot is touched by exactly one add (pivot leader/follower split + ⊕G-injectivity), so the
// parallel apply is order-free and thread-count invariant — the same invariant the non-fused
// evolve_step's atomics-free parallel D-apply relies on.
struct RotationRec {
    size_t src = 0;     // rotation source op index (op_pre[src] = v_src)
    size_t tgt = 0;     // rotation partner op index (op_pre[tgt] = v_tgt)
    double v_src = 0.0; // op_pre[src] — signed coeff captured at scan emit
    double v_tgt = 0.0; // op_pre[tgt] — resolve-time (hits) / post-extension (inserts) coeff
    int32_t phase = 0;  // ±1 hermitian·interleave phase
};

// One CROSS-RANK half-rotation (R>1): the rotation's two endpoints live on different ranks, so each
// rank applies only the ADD to the slot it OWNS. Resolver rank B (owns target T): {T, v_src, +φ};
// querier rank A (owns source S): {S, v_tgt, −φ}. Applied identically to the self-rank D-apply:
//   op[local_idx] += sin·phase_signed·v_partner
// v_partner is the PARTNER's pre-cos coefficient shipped over the wire (v_src on the query stream,
// v_tgt on the response stream, or — for a Schrödinger cross-rank MISS — via a post-extension exchange).
struct HalfRotationRec {
    size_t local_idx = 0;     // slot THIS rank owns: T (resolver) or S (querier)
    double v_partner = 0.0;   // partner's PRE-cos coeff: v_src (resolver) / v_tgt (querier)
    int32_t phase_signed = 0; // +φ (resolver) or −φ (querier), pre-signed to match Evolution.cpp:392
    // Resolver MISS halves write a slot INSERTED this gate (ip ≥ the pre-insert operator size). Fresh
    // inserts are born AFTER the scan's fused cos sweep, so the apply must fold the gate's cos into the
    // slot itself (c = cos·c + sin term) instead of the plain add that pre-scaled slots get. False for
    // hit halves and all querier halves (their local slot is a pre-gate term the sweep covered).
    bool is_insert = false;
};

// Sink threaded through build_layer's fused branch. Self-routed rotations (both endpoints local) are
// full RotationRecs: HIT records (partner already in the operator, v_tgt captured at resolve) and
// INSERT records (partner freshly inserted this layer, v_tgt filled after extend_coeffs) in SEPARATE
// lists so the apply can fill INSERT v_tgt without scanning for a sentinel. R>1 cross-rank rotations are
// HalfRotationRecs (one slot per rank) in `cross_half`: one half per cross-rank query — the resolver's +φ
// half on the target it owns, and the querier's −φ half on the source it owns. The querier's v_partner is
// the target coeff the resolver ships back per query: the evolved coeff for a HIT, and — for a freshly
// inserted MISS — the fresh term's picture coeff, which the resolver computes on the spot (0 in Heisenberg;
// is_paired ? hf_phase : 0 in Schrödinger, a pure ±1/0 function of the majorana identical to get_state's
// scoring), so no second exchange is needed. Empty at R==1. When a non-null FusedContract* reaches
// build_layer, the fused path is taken.
struct FusedContract {
    std::vector<RotationRec> hits;
    std::vector<RotationRec> inserts;
    std::vector<HalfRotationRec> cross_half; // R>1: one half per cross-rank query (resolver +φ, querier −φ)
};

// ─── Query serialization ──────────────────────────────────────────────────────
// Queries are exchanged as flat VecZ buffers: every kQueryWords elements = one query — the W
// monomial words plus one trailing phase word (±1). The querier's source index is NOT in the
// payload: the resolver returns the partner by position and the querier holds the source in its
// parallel src list (src_idx_r[r][q]).
template <size_t NumModes>
inline constexpr size_t kQueryWords = mpi_detail::kWords<NumModes> + 1;

// Phase ↔ word codec for the trailing phase word. Only ±1 is ever stored; the unsigned-int
// intermediate normalizes the sign bit into a fixed 32-bit pattern so the ±1 round-trip is exact
// no matter how wide VecZ's element is. encode/decode are inverses — edit them as a pair.
inline auto encode_phase(int phase) -> size_t {
    return static_cast<size_t>(static_cast<unsigned int>(phase));
}
inline auto decode_phase(size_t word) -> int {
    return static_cast<int>(static_cast<unsigned int>(word));
}

template <size_t NumModes>
inline auto query_push(VecZ &buf, const MajoranaSet<NumModes> &maj, int phase) -> void {
    mpi_detail::append_majorana_words<NumModes>(maj, buf);
    buf.push_back(encode_phase(phase));
}

template <size_t NumModes>
inline auto query_read(const VecZ &buf, size_t q, MajoranaSet<NumModes> &maj_out, int &phase_out) -> void {
    const size_t base = q * kQueryWords<NumModes>;
    maj_out = mpi_detail::read_majorana_from_words<NumModes>(buf, base);
    phase_out = decode_phase(buf[base + mpi_detail::kWords<NumModes>]);
}

// Read ONLY the trailing phase word of query q — the phase is the last word of every fixed-width
// query record, so recovering it needs no majorana reconstruction (used where the partner M' is not
// needed, only its phase; see process_query_responses).
template <size_t NumModes>
inline auto query_phase(const VecZ &buf, size_t q) -> int {
    return decode_phase(buf[q * kQueryWords<NumModes> + mpi_detail::kWords<NumModes>]);
}

} // namespace monoprop::detail
