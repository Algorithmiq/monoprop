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

// Marks matched followers without a per-gate O(n) memset: one counter bump clears every mark. Reused
// across gates.
struct MatchedEpochSet {
    std::vector<uint32_t> epoch_;
    uint32_t cur_ = 0;

    // u32 wrap resets the array — once per 2^32-1 gates.
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

// Queries ride flat VecZ buffers: query_words(nw) elements per query (nw monomial words + one ±1 phase
// word), where nw is the monomial word count.
// The source index is not in the payload — the resolver answers by position; the querier holds src_idx_r[r][q].
//
// Functions of the word count rather than kQueryWords<NumModes> constants: the width is data now. Every
// caller either has a monomial to ask (`mono.num_words()`) or the operator (`op.num_bits()`).
inline constexpr auto query_words(size_t num_words) -> size_t {
    return num_words + 1;
}

// Fused query+value record width (R>1): the plain query record plus one trailing word holding the source's
// pre-cos coeff (v_src, bit-cast from double), so query + value ride a single alltoallv instead of two.
inline constexpr auto query_words_fused(size_t num_words) -> size_t {
    return query_words(num_words) + 1;
}

// The unsigned-int intermediate normalizes the ±1 sign bit into a fixed 32-bit pattern so the round-trip
// is exact for any VecZ element width. Edit encode/decode as a pair.
inline auto encode_phase(int phase) -> size_t {
    return static_cast<size_t>(static_cast<unsigned int>(phase));
}
inline auto decode_phase(size_t word) -> int {
    return static_cast<int>(static_cast<unsigned int>(word));
}

// bit_cast, not a conversion, so v_src arrives over the wire bit-identical.
static_assert(sizeof(size_t) == sizeof(double), "fused query value word assumes 64-bit VecZ element");
inline auto encode_value(double v) -> size_t {
    return std::bit_cast<size_t>(v);
}
inline auto decode_value(size_t word) -> double {
    return std::bit_cast<double>(word);
}

inline auto query_push(VecZ &buf, const Bitset &mono, int phase) -> void {
    mpi_detail::append_monomial_words(mono, buf);
    buf.push_back(encode_phase(phase));
}

// The mono + phase words occupy the same leading offsets in the plain and fused record, so readers differ
// only in the per-record stride: query_words for a plain record, query_words_fused for a fused one.
//
// `stride` is an ordinary argument rather than a template parameter (Stage 2c of the
// NumModes-NTTP-removal plan), and the word count now comes from mono_out, which as the destination
// already carries the record's width.
inline auto query_read(const VecZ &buf, size_t q, size_t stride, Bitset &mono_out, int &phase_out) -> void {
    const size_t base = q * stride;
    mpi_detail::read_monomial_from_words(buf, base, mono_out);
    phase_out = decode_phase(buf[base + mono_out.num_words()]);
}

// No monomial reconstruction: process_responses needs only the phase.
inline auto query_phase(const VecZ &buf, size_t q, size_t num_words) -> int {
    return decode_phase(buf[q * query_words(num_words) + num_words]);
}

inline auto query_value(const VecZ &buf, size_t q, size_t num_words) -> double {
    return decode_value(buf[q * query_words_fused(num_words) + num_words + 1]);
}

// Requires v.size() == q.size()/query_words(num_words): exactly one value per query record.
inline auto build_fused_query_value(const VecZ &q, const std::vector<double> &v, VecZ &out, size_t num_words) -> void {
    const size_t W = query_words(num_words);
    const size_t nq = q.empty() ? 0 : q.size() / W;
    out.clear();
    out.reserve(nq * query_words_fused(num_words));
    for (size_t i = 0; i < nq; ++i) {
        out.insert(out.end(),
                   q.begin() + static_cast<std::ptrdiff_t>(i * W),
                   q.begin() + static_cast<std::ptrdiff_t>((i + 1) * W));
        out.push_back(encode_value(v[i]));
    }
}

} // namespace monoprop::detail
