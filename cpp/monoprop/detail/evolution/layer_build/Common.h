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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

namespace monoprop::detail {

// Sentinel for "no index resolved yet" in the resolve slots. Deliberately equal to
// OperatorIndex::kNotFound, so one `>= size` bound check covers a miss and an unresolved slot alike.
inline constexpr size_t kMissingIndex = std::numeric_limits<size_t>::max();

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
            std::ranges::fill(epoch_, Stamp{0});
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

// Queries ride flat VecZ buffers: one header word holding the record count, then query_words(nw) elements
// per query (nw payload words + one ±1 phase word), then -- support form only -- a tail of dense escape
// monomials for the queries no fixed-stride sparse record can hold.
// The source index is not in the payload — the resolver answers by position; the querier holds src_idx_r[r][q].
//
// Functions of the word count rather than width-derived constants: every caller either has a monomial
// to ask (`mono.num_words()`) or the operator (`op.num_bits()`).
constexpr auto query_words(size_t num_words) -> size_t {
    return num_words + 1;
}

// The record count leads the buffer rather than being divided out of its size, for two reasons: a
// support-form buffer carries a tail after its records, so size/stride is not the count; and the resolver
// has nothing but the received buffer to derive it from, where the querier still has its parallel source
// array. One word per (rank, pass) buffer.
//
// A stream nothing was pushed to may be an empty buffer rather than a zero header -- the scan allocates
// per rank and only some ranks are queried -- so both must read as zero records.
inline constexpr size_t kQueryHeaderWords = 1;

[[nodiscard]] inline auto query_buffer() -> VecZ {
    return VecZ(kQueryHeaderWords, 0);
}
[[nodiscard]] inline auto query_record_count(const VecZ &buf) -> size_t {
    return buf.size() < kQueryHeaderWords ? 0 : buf[0];
}
[[nodiscard]] constexpr auto query_record_offset(size_t q, size_t stride) -> size_t {
    return kQueryHeaderWords + (q * stride);
}

// Fused query+value record width (R>1): the plain query record plus one trailing word holding the source's
// pre-cos coeff (v_src, bit-cast from double), so query + value ride a single alltoallv instead of two.
constexpr auto query_words_fused(size_t num_words) -> size_t {
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

// Appends one record and bumps the header, so the count is always the buffer's own rather than
// something a reader has to derive from size and stride.
inline auto query_push(VecZ &buf, const Bitset &mono, int phase) -> void {
    assert(buf.size() >= kQueryHeaderWords && "a query buffer must be created with query_buffer()");
    mpi_detail::append_monomial_words(mono, buf);
    buf.push_back(encode_phase(phase));
    ++buf[0];
}

// The mono + phase words occupy the same leading offsets in the plain and fused record, so readers differ
// only in the per-record stride: query_words for a plain record, query_words_fused for a fused one.
//
// The word count comes from mono_out, which as the destination already carries the record's width.
inline auto query_read(const VecZ &buf, size_t q, size_t stride, Bitset &mono_out, int &phase_out) -> void {
    const size_t base = query_record_offset(q, stride);
    mpi_detail::read_monomial_from_words(buf, base, mono_out);
    phase_out = decode_phase(buf[base + mono_out.num_words()]);
}

// Owned storage for a batch of query keys in whichever form a store keys its rows by, plus the record
// reader that fills it. Both resolve paths -- the self-resolve batch in Engine.h and the incoming probe in
// Resolve.h -- want exactly this, and both used to hand-roll it: allocate once, keep the storage across
// layers, overwrite every element before reading it, and hand the whole run to find_batch contiguously.
//
// Grow-only and never cleared by ensure()/read within one object's lifetime: an element is overwritten
// whole before any read, so a per-batch rebuild bought nothing and cost a construction per query. Whether
// that lifetime spans layers depends on the call site, not the class. The thread_local batch in Resolve.h
// keeps its storage across layers on purpose, so its resting footprint is the largest layer's worth until
// the thread exits (peak RSS is unchanged -- the peak was always reached *during* a layer). Engine.h's
// keys_ does not: it is a plain member of a LayerBuildEngine built fresh per build_layer call, so it
// starts default-constructed and pays ensure()'s construction cost every layer -- required, not a missed
// optimization, because retain()'s handles index into storage that must not survive past the layer that
// produced them.
//
// configure() must be called before any use and re-called if the extent changes -- a thread servicing two
// propagators of different widths must not reuse elements sized for the other.
class DenseQueryKeys {
public:
    using key_type = Bitset;

    // num_bits is the monomial storage width: query_read memcpys the destination's full word count, so the
    // destination is what fixes the record width.
    auto configure(size_t num_bits) -> void {
        if (extent_ != num_bits) {
            keys_.clear();
            retained_words_.clear();
            retained_count_ = 0;
            extent_ = num_bits;
            view_ = Bitset(extent_);
        }
    }
    auto ensure(size_t n) -> void {
        if (keys_.size() < n) {
            // resize only constructs the new tail; elements an earlier layer built keep their storage.
            keys_.resize(n, Bitset(extent_));
        }
    }
    // Slots are refilled for every batch, so anything a caller must still read afterwards has to be
    // retained first (see retain below). Nothing per-batch to reset on this side.
    auto begin_batch() -> void {}
    [[nodiscard]] auto read_record(const VecZ &buf, size_t q, size_t stride, size_t slot) -> int {
        int phase = 0;
        query_read(buf, q, stride, keys_[slot], phase);
        return phase;
    }
    [[nodiscard]] auto data() const -> const key_type * { return keys_.data(); }
    [[nodiscard]] auto operator[](size_t slot) const -> const key_type & { return keys_[slot]; }

    // Copies slot's key into storage that lives as long as this batch, and returns its handle. The
    // deferred self-miss list is what needs it: it is read after the batch has been refilled several times
    // over, once both resolve passes are done.
    //
    // The retained keys are a flat word arena at the batch's own width rather than a second MonomialList:
    // a Bitset is sized for the widest inline width whatever its own is, so a vector of them carried 72
    // bytes per key where a 128-bit monomial needs 16, and one is retained per term the layer inserts.
    // The support-form batch below already keeps its retained rows in an arena, for the same reason.
    [[nodiscard]] auto retain(size_t slot) -> size_t {
        const size_t words = Bitset::words_for(extent_);
        const auto *src = keys_[slot].data();
        retained_words_.insert(retained_words_.end(), src, src + words);
        return retained_count_++;
    }
    // Returns a reference to a scratch monomial refilled per call, so at most one retained key may be
    // read at a time. Both readers satisfy that: insert_absent_terms writes slot k's row and is done with
    // the key before asking for k+1, and its bulk_insert hashes one key per slot. The support form's
    // retained() has the same one-at-a-time contract -- its view points into an arena that must not have
    // grown since.
    [[nodiscard]] auto retained(size_t handle) const -> const key_type & {
        const size_t words = Bitset::words_for(extent_);
        std::memcpy(view_.data(), retained_words_.data() + (handle * words), words * sizeof(Bitset::word_type));
        return view_;
    }

private:
    MonomialList keys_ = {};
    // Handle h occupies words [h * words_for(extent_), (h+1) * words_for(extent_)).
    DefaultInitVector<Bitset::word_type> retained_words_ = {};
    size_t retained_count_ = 0;
    mutable Bitset view_{0};
    size_t extent_ = 0;
};

// The payload width of one query record, in VecZ words -- the quantity every stride, alltoallv count and
// record offset in Engine.h derives from. Both backends are queried with dense monomials, so the width is
// the store's own: a support-form row still has to be materialized before it goes on the wire.
[[nodiscard]] inline auto query_payload_words_for(const auto &store) -> size_t {
    return Bitset::words_for(store.num_bits());
}

// No monomial reconstruction: process_responses needs only the phase.
inline auto query_phase(const VecZ &buf, size_t q, size_t num_words) -> int {
    return decode_phase(buf[query_record_offset(q, query_words(num_words)) + num_words]);
}

inline auto query_value(const VecZ &buf, size_t q, size_t num_words) -> double {
    return decode_value(buf[query_record_offset(q, query_words_fused(num_words)) + num_words + 1]);
}

// Requires v.size() == query_record_count(q): exactly one value per query record.
inline auto build_fused_query_value(const VecZ &q, const std::vector<double> &v, VecZ &out, size_t num_words) -> void {
    out.clear();
    if (q.size() < kQueryHeaderWords) {
        // No stream at all rather than an empty one: the self entry is cleared once resolved inline, and it
        // must stay empty so the alltoallv sends nothing to self.
        return;
    }
    const size_t W = query_words(num_words);
    const size_t nq = query_record_count(q);
    out.reserve(kQueryHeaderWords + (nq * query_words_fused(num_words)));
    out.push_back(nq);
    for (size_t i = 0; i < nq; ++i) {
        out.insert(out.end(),
                   q.begin() + static_cast<std::ptrdiff_t>(query_record_offset(i, W)),
                   q.begin() + static_cast<std::ptrdiff_t>(query_record_offset(i + 1, W)));
        out.push_back(encode_value(v[i]));
    }
}

} // namespace monoprop::detail
