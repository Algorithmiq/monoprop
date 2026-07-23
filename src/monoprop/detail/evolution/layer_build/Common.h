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
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::detail {

// Marks matched followers without a per-gate O(n) memset: slot i is marked iff epoch_[i] == cur_, so
// starting a gate is one counter bump (all marks clear in O(1)). Reused across gates; ≤1 writer per slot
// (distinct leaders → distinct found via injective ⊕G), so no atomics.
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

// Uniform per-rank rotation accumulator, drained at finish() into the LayerCore's B (partner index list)
// and D ((index, signed-phase) list). Self slot = partner with in:=(tgt,φ), out:=(src,φ); cross-rank has
// in=resolver side, out=querier side. The exact B/D layout lives at assemble_partners (the SINGLE copy).
struct PartnerAcc {
    // Default-init storage: resize-then-overwrite paths skip the zero-fill and the gather lowers to
    // memmove. Load-bearing: every such path fully overwrites [base, base+n) before any read, so the
    // skipped init is never observed.
    DefaultInitVector<PhasedEntry> in_entries;  // (local_target_idx, phase)
    DefaultInitVector<PhasedEntry> out_entries; // (local_source_idx, phase)
};

// Fused contraction (ContractImmediately — the default forward path at all rank counts): one rotation
// applied DIRECTLY to op_coeffs, bypassing the transient LayerCore. Each rotation (source S, target T,
// phase φ), after cos-scaling S and T (v_src/v_tgt are the PRE-cos coeffs):
//   op[S] += -sin·φ·op_pre[T]      op[T] += +sin·φ·op_pre[S]
// Each op slot is touched by exactly one add (pivot split + ⊕G-injectivity), so the parallel apply is
// order-free and thread-count invariant.
struct RotationRec {
    size_t src = 0;     // rotation source op index (op_pre[src] = v_src)
    size_t tgt = 0;     // rotation partner op index (op_pre[tgt] = v_tgt)
    double v_src = 0.0; // op_pre[src] — signed coeff captured at scan emit
    double v_tgt = 0.0; // op_pre[tgt] — resolve-time (hits) / post-extension (inserts) coeff
    int32_t phase = 0;  // ±1 hermitian·interleave phase
};

// One CROSS-RANK half-rotation (R>1): each rank applies only the ADD to the slot it OWNS. Resolver B
// (owns target T): {T, v_src, +φ}; querier A (owns source S): {S, v_tgt, −φ}. Applied like the self-rank
// D-apply: op[local_idx] += sin·phase_signed·v_partner, v_partner the partner's pre-cos coeff off the wire.
struct HalfRotationRec {
    size_t local_idx = 0;     // slot THIS rank owns: T (resolver) or S (querier)
    double v_partner = 0.0;   // partner's PRE-cos coeff: v_src (resolver) / v_tgt (querier)
    int32_t phase_signed = 0; // +φ (resolver) or −φ (querier), pre-signed to match Evolution.cpp:392
    // Resolver MISS halves write a slot INSERTED this gate (born AFTER the fused cos sweep), so the apply
    // folds the gate's cos into the slot itself (c = cos·c + sin) instead of the plain add. False for hit
    // and querier halves (their slot is a pre-gate term the sweep covered).
    bool is_insert = false;
};

// Sink threaded through build_layer's fused branch. Self-routed rotations (both endpoints local) are full
// RotationRecs, split into HIT and INSERT lists so the apply can fill INSERT v_tgt (available only after
// extend_coeffs) without scanning for a sentinel. R>1 cross-rank rotations are HalfRotationRecs in
// `cross_half` (one half per query: resolver +φ, querier −φ), empty at R==1. A non-null FusedContract*
// reaching build_layer takes the fused path.
struct FusedContract {
    std::vector<RotationRec> hits;
    std::vector<RotationRec> inserts;
    std::vector<HalfRotationRec> cross_half; // R>1: one half per cross-rank query (resolver +φ, querier −φ)
};

// Queries are exchanged as flat VecZ buffers: every kQueryWords elements = one query (W monomial words +
// one trailing ±1 phase word). The source index is NOT in the payload — the resolver returns the partner
// by position and the querier holds the source in its parallel src list (src_idx_r[r][q]).
template <size_t NumModes>
inline constexpr size_t kQueryWords = mpi_detail::kWords<NumModes> + 1;

// Fused query+value record width (R>1): the plain query record plus ONE trailing word holding the source's
// pre-cos coeff (v_src, bit-cast from double), so query + value ride a SINGLE alltoallv instead of two. The
// 64-bit round-trip is exact ⇒ byte-identical to the two-stream exchange.
template <size_t NumModes>
inline constexpr size_t kQueryWordsFused = kQueryWords<NumModes> + 1;

// Phase ↔ word codec for the trailing phase word. The unsigned-int intermediate normalizes the ±1 sign
// bit into a fixed 32-bit pattern so the round-trip is exact for any VecZ element width. Edit as a pair.
inline auto encode_phase(int phase) -> size_t {
    return static_cast<size_t>(static_cast<unsigned int>(phase));
}
inline auto decode_phase(size_t word) -> int {
    return static_cast<int>(static_cast<unsigned int>(word));
}

// Value ↔ word codec for the fused record's trailing coefficient word. A lossless bit_cast between the
// 64-bit VecZ element and double (static_assert guards the size match) ⇒ v_src arrives bit-identical.
static_assert(sizeof(size_t) == sizeof(double), "fused query value word assumes 64-bit VecZ element");
inline auto encode_value(double v) -> size_t {
    return std::bit_cast<size_t>(v);
}
inline auto decode_value(size_t word) -> double {
    return std::bit_cast<double>(word);
}

template <size_t NumModes>
inline auto query_push(VecZ &buf, const Monomial<NumModes> &maj, int phase) -> void {
    mpi_detail::append_majorana_words<NumModes>(maj, buf);
    buf.push_back(encode_phase(phase));
}

// The maj + phase words occupy the SAME leading offsets in both the plain and fused record, so readers
// differ only in the per-record stride QW (defaulted to the plain width, leaving existing calls unchanged).
template <size_t NumModes, size_t QW = kQueryWords<NumModes>>
inline auto query_read(const VecZ &buf, size_t q, Monomial<NumModes> &maj_out, int &phase_out) -> void {
    const size_t base = q * QW;
    maj_out = mpi_detail::read_majorana_from_words<NumModes>(buf, base);
    phase_out = decode_phase(buf[base + mpi_detail::kWords<NumModes>]);
}

// Read ONLY the trailing phase word of query q — no majorana reconstruction (used where only the phase is
// needed, not the partner M'; see process_responses).
template <size_t NumModes, size_t QW = kQueryWords<NumModes>>
inline auto query_phase(const VecZ &buf, size_t q) -> int {
    return decode_phase(buf[q * QW + mpi_detail::kWords<NumModes>]);
}

// Read the value word of a FUSED record (v_src the querier attached at emit). The word sits right after
// the phase word, i.e. at offset kWords+1 within each kQueryWordsFused-wide record.
template <size_t NumModes>
inline auto query_value(const VecZ &buf, size_t q) -> double {
    return decode_value(buf[q * kQueryWordsFused<NumModes> + mpi_detail::kWords<NumModes> + 1]);
}

// Interleave a rank's plain query records (`q`) with its parallel value stream (`v`, one double per query)
// into the fused send buffer `out`, reused across gates (clear + capacity-preserving reserve). Requires
// v.size() == q.size()/kQueryWords.
template <size_t NumModes>
inline auto build_fused_query_value(const VecZ &q, const std::vector<double> &v, VecZ &out) -> void {
    constexpr size_t W = kQueryWords<NumModes>;
    const size_t nq = q.empty() ? 0 : q.size() / W;
    out.clear();
    out.reserve(nq * kQueryWordsFused<NumModes>);
    for (size_t i = 0; i < nq; ++i) {
        out.insert(out.end(),
                   q.begin() + static_cast<std::ptrdiff_t>(i * W),
                   q.begin() + static_cast<std::ptrdiff_t>((i + 1) * W));
        out.push_back(encode_value(v[i]));
    }
}

} // namespace monoprop::detail
