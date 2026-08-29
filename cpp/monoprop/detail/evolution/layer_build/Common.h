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
// Where the escape tail starts: right after the last record. Both sides derive it from the header and the
// stride, so an escape's own index into the tail is independent of either -- which is what lets the fused
// re-layout below move the records without touching the tail.
[[nodiscard]] inline auto query_tail_offset(const VecZ &buf, size_t stride) -> size_t {
    return query_record_offset(query_record_count(buf), stride);
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

// The two buffers a scan pushes a query into. A record cannot be appended once the tail has started, so
// the tail accumulates separately and is concatenated when the scan is done -- an escape's index names a
// position within the tail, so the concatenation moves nothing it refers to.
struct QueryOut {
    VecZ &records;
    VecZ &escapes;
};

// The scan's single exit: fold each stream's escapes in behind its records.
inline auto append_escape_tail(VecZ &records, VecZ &escapes) -> void {
    if (escapes.empty()) {
        return;
    }
    records.insert(records.end(), escapes.begin(), escapes.end());
    escapes.clear();
}

// Appends one record and bumps the header. Every push must precede the escape tail, which the scan
// guarantees by collecting escapes in a buffer of their own and concatenating once the scan is done.
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

// A query record in support form. Deliberately the same *shape* as the dense one -- a fixed-stride
// payload followed by the phase word -- so every stride computation, alltoallv count and reader offset
// above holds with `num_words` reinterpreted as the payload word count. Only the payload differs:
// `sparse_lane_words` words of four uint16 mode lanes each plus one codes word, against the monomial's
// full word count. That is what the format is for: a 1024-mode monomial is 32 words, where a 12-slot row
// is 3 lane words plus 1.
//
// The stride needs a capacity, and it must be one every rank derives identically without communication --
// SparseRowStore::scratch_slots_for(cutoff mode bound, widest generator), the same value the scan's
// scratch row uses, since a query carries exactly such a product. Ranks already owe each other this kind
// of agreement for the hash width (see find_rank).
//
// A row past that capacity has no sparse record: the caller must fall back to the dense one, and the
// push below asserts rather than truncating.
constexpr auto sparse_lane_words(size_t capacity) -> size_t {
    return (capacity + 3) / 4;
}
constexpr auto sparse_payload_words(size_t capacity) -> size_t {
    return sparse_lane_words(capacity) + 1;
}

inline auto sparse_query_push(VecZ &buf, const SparseRow &row, size_t capacity, int phase) -> void {
    assert(buf.size() >= kQueryHeaderWords && "a query buffer must be created with query_buffer()");
    const size_t lane_words = sparse_lane_words(capacity);
    const size_t n = row.num_slots();
    assert(n <= capacity && "sparse_query_push row exceeds the record capacity");
    const size_t base = buf.size();
    // Zero-filled, so the lanes past the row's own are deterministic; the reader ignores them, taking the
    // slot count off the codes word.
    buf.resize(base + lane_words + 2, 0);
    for (size_t j = 0; j < n; ++j) {
        buf[base + (j / 4)] |= static_cast<size_t>(row.modes[j]) << (16 * (j % 4));
    }
    buf[base + lane_words] = static_cast<size_t>(row.codes);
    buf[base + lane_words + 1] = encode_phase(phase);
    ++buf[0];
}

// The record left behind by a query no sparse record can hold, and it is not a corner case to be sized
// away: a query is M ⊕ G and a fully paired product escapes the cutoff, so nothing bounds its support.
//
// The record keeps its place and its stride -- which is what leaves every offset, alltoallv count and
// compaction in the engine as plain arithmetic -- and says where to find the monomial instead: lane 0
// carries the store's own overflow marker and the codes slot carries the escape's index into the buffer's
// tail. Both conventions are SparseRowStore's for a spilled row, deliberately: one representation of "too
// wide for a codes word", not two.
//
// `escapes` accumulates the tail separately during the scan, because a record cannot be appended after
// the tail has started; the scan concatenates the two once it is done. An escape's index is its position
// in that tail, so it survives the concatenation and the fused re-layout alike.
inline auto sparse_query_push_escape(VecZ &buf, VecZ &escapes, const Bitset &mono, size_t capacity, int phase) -> void {
    assert(buf.size() >= kQueryHeaderWords && "a query buffer must be created with query_buffer()");
    const size_t lane_words = sparse_lane_words(capacity);
    const size_t base = buf.size();
    buf.resize(base + lane_words + 2, 0);
    buf[base] = static_cast<size_t>(SparseRowStore::kOverflowLane);
    buf[base + lane_words] = escapes.size() / mono.num_words();
    buf[base + lane_words + 1] = encode_phase(phase);
    ++buf[0];
    mpi_detail::append_monomial_words(mono, escapes);
}

// Reading a record's shape needs the lane word count, which the reader has from the capacity; taking the
// record base as an argument keeps this usable from both the plain and the fused stride.
[[nodiscard]] inline auto sparse_record_is_escape(const VecZ &buf, size_t base) -> bool {
    return buf[base] == static_cast<size_t>(SparseRowStore::kOverflowLane);
}

// lanes_out must hold `capacity` lanes. Writes only the row's own, for the same reason
// read_monomial_from_words overwrites whole words: what a previous record left beyond them cannot be read.
inline auto sparse_query_read(const VecZ &buf,
                              size_t q,
                              size_t stride,
                              size_t capacity,
                              RowMode *lanes_out,
                              RowCodes &codes_out,
                              int &phase_out) -> void {
    const size_t base = query_record_offset(q, stride);
    const size_t lane_words = sparse_lane_words(capacity);
    assert(!sparse_record_is_escape(buf, base) && "an escaped record has no row to read");
    codes_out = static_cast<RowCodes>(buf[base + lane_words]);
    const size_t n = row_slot_count(codes_out);
    for (size_t j = 0; j < n; ++j) {
        lanes_out[j] = static_cast<RowMode>((buf[base + (j / 4)] >> (16 * (j % 4))) & 0xFFFFU);
    }
    phase_out = decode_phase(buf[base + lane_words + 1]);
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
    // destination is what fixes the record width. capacity is the support form's row capacity, which a
    // dense record has no use for -- both batches take both so the call sites need no branch.
    auto configure(size_t num_bits, size_t /*capacity*/) -> void {
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

// The support-form counterpart: one lane array for the whole batch plus a parallel array of keys viewing
// into it, since find_batch wants the keys contiguous and a SparseRow is only a pointer and a word.
//
// The escaped records are why the key is a SparseRowKey rather than a SparseRow: one of those queries has
// no row, so its key points at a monomial materialized out of the buffer's tail. That storage is a deque
// on purpose -- push_back must not invalidate a key handed out for an earlier slot, and a vector's would.
class SparseQueryKeys {
public:
    using key_type = SparseRowKey;

    // capacity is the row capacity in slots -- the same value that fixes the record stride, since a record
    // holds exactly that many lanes. num_bits sizes the escape monomials.
    auto configure(size_t num_bits, size_t capacity) -> void {
        if (capacity_ != capacity || num_bits_ != num_bits) {
            lanes_.clear();
            keys_.clear();
            escapes_.clear();
            retained_lanes_.clear();
            retained_bases_.clear();
            retained_escapes_.clear();
            retained_.clear();
            capacity_ = capacity;
            num_bits_ = num_bits;
        }
    }
    auto ensure(size_t n) -> void {
        if (keys_.size() >= n) {
            return;
        }
        lanes_.resize(n * capacity_);
        keys_.resize(n);
        // Every view is rebuilt, not just the new tail: the resize above may have moved lanes_, which
        // would leave the existing views pointing into freed storage.
        for (size_t i = 0; i < n; ++i) {
            keys_[i] = SparseRowKey{.row = SparseRow{&lanes_[i * capacity_], 0}};
        }
    }
    // Hygiene rather than correctness: a key always points at the entry read for it, so stale entries are
    // unreachable -- they would just accumulate for the whole resolve. Dropped per batch because that is
    // the granularity at which slots are refilled anyway.
    auto begin_batch() -> void { escapes_.clear(); }
    [[nodiscard]] auto read_record(const VecZ &buf, size_t q, size_t stride, size_t slot) -> int {
        const size_t base = query_record_offset(q, stride);
        const size_t lane_words = sparse_lane_words(capacity_);
        if (sparse_record_is_escape(buf, base)) {
            // The tail entry this record named. Its offset needs the record count and the stride, both of
            // which the buffer and the caller already carry.
            const size_t tail = query_tail_offset(buf, stride);
            const size_t words = Bitset::words_for(num_bits_);
            escapes_.emplace_back(num_bits_);
            mpi_detail::read_monomial_from_words(buf, tail + (buf[base + lane_words] * words), escapes_.back());
            keys_[slot].spilled = &escapes_.back();
            return decode_phase(buf[base + lane_words + 1]);
        }
        int phase = 0;
        keys_[slot].spilled = nullptr;
        // The lane pointer stays the one ensure() set: a record's lanes are read into this slot's own run.
        sparse_query_read(buf, q, stride, capacity_, &lanes_[slot * capacity_], keys_[slot].row.codes, phase);
        return phase;
    }
    [[nodiscard]] auto data() const -> const key_type * { return keys_.data(); }
    [[nodiscard]] auto operator[](size_t slot) const -> const key_type & { return keys_[slot]; }

    // See DenseQueryKeys::retain. A retained key owns its lanes here too, in a second arena, and an escaped
    // one owns its monomial -- the batch's own escape storage is dropped every batch.
    [[nodiscard]] auto retain(size_t slot) -> size_t {
        const size_t handle = retained_.size();
        // A base is recorded for every handle, escaped or not, so retained_bases_ stays indexable by
        // handle; an escaped key's is simply never read.
        retained_bases_.push_back(retained_lanes_.size());
        if (keys_[slot].is_spilled()) {
            retained_escapes_.push_back(*keys_[slot].spilled);
            // .row left empty: is_spilled() sends every read to .spilled instead.
            retained_.push_back(SparseRowKey{.row = {}, .spilled = &retained_escapes_.back()});
            return handle;
        }
        retained_lanes_.resize(retained_bases_.back() + capacity_);
        const size_t n = keys_[slot].row.num_slots();
        std::copy_n(keys_[slot].row.modes,
                    n,
                    retained_lanes_.begin() + static_cast<std::ptrdiff_t>(retained_bases_.back()));
        // The lane array grows, so a key cannot hold a pointer into it; retained() rebuilds the view.
        retained_.push_back(SparseRowKey{.row = SparseRow{nullptr, keys_[slot].row.codes}});
        return handle;
    }
    [[nodiscard]] auto retained(size_t handle) const -> key_type {
        const key_type &key = retained_[handle];
        if (key.is_spilled()) {
            return key;
        }
        return SparseRowKey{.row = SparseRow{&retained_lanes_[retained_bases_[handle]], key.row.codes}};
    }

private:
    DefaultInitVector<RowMode> lanes_ = {};
    std::vector<SparseRowKey> keys_ = {};
    std::deque<Bitset> escapes_ = {};
    // Retained keys, whose storage must outlive the batch's own (see retain). retained_bases_ is parallel
    // to retained_ but indexed only for the non-escaped ones -- an escaped key's entry is unread.
    DefaultInitVector<RowMode> retained_lanes_ = {};
    std::vector<size_t> retained_bases_ = {};
    std::deque<Bitset> retained_escapes_ = {};
    std::vector<SparseRowKey> retained_ = {};
    size_t capacity_ = 0;
    size_t num_bits_ = 0;
};

// A query key as a dense monomial, for the handful of places that need one -- the Schrodinger fresh-insert
// scoring, which has no codes form. Returns a reference when the key already is one and a value otherwise,
// so callers bind with `const auto &` to extend the temporary, exactly as materialize_row documents.
[[nodiscard]] inline auto key_monomial(const Bitset &key, size_t /*num_bits*/) -> const Bitset & {
    return key;
}
[[nodiscard]] inline auto key_monomial(const SparseRowKey &key, size_t num_bits) -> Bitset {
    if (key.is_spilled()) {
        return *key.spilled;
    }
    return sparse_row_to_bitset(key.row, num_bits);
}

// The payload width of one query record for a store, in VecZ words -- the quantity every stride, alltoallv
// count and record offset in Engine.h derives from. An overload per store rather than one accessor on
// MPOperator, because it is a property of the wire format the store is queried through, not of the store.
//
// Dense rows put the monomial's own words on the wire. The support form will put lane words plus the codes
// word: 5 words for a 12-slot row against 33 for a 1024-mode monomial, but slightly *wider* just above the
// store's own crossover -- 5 against 4 at 96 modes, break-even near 128 modes. That band is why the record
// form is tied to the store rather than chosen per layer: a runtime record form would double the engine's
// template instantiations again, to save a word in a narrow range.
[[nodiscard]] inline auto query_payload_words_for(const OperatorIndex &store, size_t /*capacity*/) -> size_t {
    return Bitset::words_for(store.num_bits());
}
[[nodiscard]] inline auto query_payload_words_for(const SparseRowStore & /*store*/, size_t capacity) -> size_t {
    return sparse_payload_words(capacity);
}

// Which key batch a store's query records arrive in. Explicit specializations rather than a member
// typedef on the stores: the record codec lives here, and a store must not depend on the wire format it
// is queried through.
//
// Each store is queried in the form it keys its rows by, so a resolve never converts: the dense store
// receives monomials, the support form receives rows (and, for the queries no row can hold, the escape
// monomials its tail carries).
template <typename Store>
struct QueryKeysFor;
template <>
struct QueryKeysFor<OperatorIndex> {
    using type = DenseQueryKeys;
};
template <>
struct QueryKeysFor<SparseRowStore> {
    using type = SparseQueryKeys;
};

// No monomial reconstruction: process_responses needs only the phase.
inline auto query_phase(const VecZ &buf, size_t q, size_t num_words) -> int {
    return decode_phase(buf[query_record_offset(q, query_words(num_words)) + num_words]);
}

inline auto query_value(const VecZ &buf, size_t q, size_t num_words) -> double {
    return decode_value(buf[query_record_offset(q, query_words_fused(num_words)) + num_words + 1]);
}

// Requires v.size() == query_record_count(q): exactly one value per query record.
//
// The escape tail rides along unchanged. It can, because an escape's index names its position *within the
// tail* rather than an offset into the buffer -- so widening every record by a value word moves the tail
// without renumbering anything in it.
inline auto build_fused_query_value(const VecZ &q, const std::vector<double> &v, VecZ &out, size_t num_words) -> void {
    out.clear();
    if (q.size() < kQueryHeaderWords) {
        // No stream at all rather than an empty one: the self entry is cleared once resolved inline, and it
        // must stay empty so the alltoallv sends nothing to self.
        return;
    }
    const size_t W = query_words(num_words);
    const size_t nq = query_record_count(q);
    const size_t tail = query_tail_offset(q, W);
    out.reserve(kQueryHeaderWords + (nq * query_words_fused(num_words)) + (q.size() - tail));
    out.push_back(nq);
    for (size_t i = 0; i < nq; ++i) {
        out.insert(out.end(),
                   q.begin() + static_cast<std::ptrdiff_t>(query_record_offset(i, W)),
                   q.begin() + static_cast<std::ptrdiff_t>(query_record_offset(i + 1, W)));
        out.push_back(encode_value(v[i]));
    }
    out.insert(out.end(), q.begin() + static_cast<std::ptrdiff_t>(tail), q.end());
}

} // namespace monoprop::detail
