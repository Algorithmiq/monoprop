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
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

/// Thrown when the term count would exceed the TermIndex range (rebuild with -Dmonoprop_WIDE_TERM_INDEX).
class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Operator-term store: entropy-packed position-list rows plus a keyless open-addressing hash
 * index over those rows, in one self-contained object.
 *
 * Rows are ascending set-bit positions, HEADERLESS wherever PosT has a codepoint to spare above the
 * valid positions [0, 2N): a sentinel terminator ends a short row and a sentinel at slot 0 marks an
 * overflow row, so stride_ == inline_width_ with no leading popcount slot (see kHeaderless — only
 * 2*NumModes == 256 has to keep the header). stride_ is fixed for the container's life so row offsets
 * stay stable. inline_width_ is a construction invariant: any width is correct — over-long rows spill
 * losslessly to an overflow map — so callers pass the cutoff that bounds the common case.
 *
 * Rows live in a SEGMENTED arena: a vector of fixed-size, page-scale chunks, never one growing buffer.
 * A single buffer had to grow geometrically (1.5×), because an exact fit would realloc-and-copy the
 * whole operator on every Trotter layer — and geometric growth left up to 1/3 of the row bytes as dead
 * capacity (measured at 29–32% on the hubbard and pauli models). Chunking bounds that dead capacity at
 * one chunk per store instead of a fraction of the store, with no realloc, no copy spike (shrinking a
 * single buffer would be worse still: the old and new buffers are live at once, and the benchmarks
 * measure PEAK memory), and stable row addresses as a bonus. Fixed stride is preserved WITHIN a chunk,
 * so a row is one chunk-table load away and find_batch can still prefetch it.
 *
 * The hand-rolled index exists for one capability boost::unordered_flat_set cannot expose: find_batch,
 * a group-prefetch pipelined lookup that overlaps DRAM misses, which the latency-bound resolve phases
 * need. Its slots are split Swiss/F14 style into a TermIndex array plus a 1-byte tag array.
 *
 * Single-writer: one shard owns its store on one thread (parallelism is cross-shard, up in ShardGroup),
 * so no method locks. Non-copyable/non-movable and heap-owned via unique_ptr; clone() is the single
 * deep-copy (called only on an idle store).
 *
 * Index contract: a row's key is immutable once indexed. The table stores an 8-bit tag rather than a
 * reconstructible hash, so a rehash re-derives each entry's hash from its row (see row_hash); set()ing
 * a *different* key into an already-indexed row therefore leaves the table inconsistent. Production
 * writes rows only in the grow → assign → bulk_insert order, which never violates this.
 */
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = Monomial<NumModes>;
    using key_type = Monomial<NumModes>;
    using mapped_type = size_t;

    // Position element: u8 when 2N<=256 (byte-identical to the original packed layout), widening
    // only for larger mode counts so positions never truncate.
    using PosT = std::
        conditional_t<(2 * NumModes <= 256), uint8_t, std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;

    // Default inline width when no cutoff-derived bound is supplied (e.g. Schrödinger state rows).
    // Kept at the historical value so default-constructed stores are byte-identical.
    static constexpr size_t kDefaultInlinePositions = 11;
    // Ceiling on the caller-requested inline width. A weight-w Pauli needs 2w positions; 32 covers the
    // common case inline at the supported Pauli cutoffs (2*cutoff <= 32 for cutoff <= 16).
    static constexpr size_t kMaxInlinePositions = 32;
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max();

    static_assert(2 * NumModes - 1 <= std::numeric_limits<PosT>::max(),
                  "OperatorIndex PosT too narrow for 2*NumModes positions");
    static_assert(kMaxInlinePositions < std::numeric_limits<PosT>::max(),
                  "kOverflowMarker sentinel must not collide with a valid popcount");

    // Row header elision. Positions occupy codepoints [0, 2N), so PosT usually has spare codepoints
    // above them that can act as sentinels: a TERMINATOR marking "the row ends here" replaces the
    // leading popcount slot outright, and an OVERFLOW codepoint at slot 0 replaces kOverflowMarker.
    // That is stride = inline_width_ rather than 1 + inline_width_ -- one byte off every row.
    //
    // The exception is 2 * NumModes == 256 with PosT = uint8_t (NumModes == 128 exactly): every
    // codepoint is a valid position, so there is no sentinel to spare and such widths keep the header.
    // Narrowing PosT's selection boundary instead would DOUBLE the row width at NumModes == 128, a far
    // bigger regression than this saves, so the layout is chosen per width with `if constexpr`.
    //
    // 2 * NumModes is even and so is the codepoint count, so the spare count is never exactly 1: a
    // headerless width always has BOTH sentinels, and the "signal overflow with a full row" fallback
    // that a single spare codepoint would force is unreachable. A row holding exactly inline_width_
    // positions carries no terminator and is recognised by the scan hitting the width limit.
    static constexpr size_t kPosCodepoints = static_cast<size_t>(std::numeric_limits<PosT>::max()) + 1;
    static constexpr size_t kSparePosCodepoints = kPosCodepoints - 2 * NumModes;
    static_assert(kSparePosCodepoints != 1, "a single spare position codepoint should be impossible");
    static constexpr bool kHeaderless = kSparePosCodepoints >= 2;
    static constexpr PosT kTerminator = static_cast<PosT>(2 * NumModes);
    static constexpr PosT kOverflowPos = static_cast<PosT>(kHeaderless ? 2 * NumModes + 1 : 0);

    // Valid term indices are < kIndexCeiling (check_index_fits throws at the ceiling), so the
    // all-ones TermIndex is free to mark an empty slot.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    static constexpr TermIndex kEmptySlot = std::numeric_limits<TermIndex>::max();
    // find_batch's "absent" result; same value as detail::kMissingIndex (not included here — the
    // operator store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

    // Arena geometry. Rows per chunk is the smallest power of two spanning at least one 4 KiB page, so
    // chunk bytes land in [4 KiB, 8 KiB) for every stride and the arena's worst-case dead capacity (one
    // chunk minus one row) is ~one page PER STORE. That absolute bound is the point: the default
    // parallelism is one shard -- hence one store -- per physical core, so a chunk sized as a fraction
    // of a big store would multiply its slack by the core count on the many-shard configurations.
    static constexpr size_t kChunkTargetBytes = 4096;
    static constexpr size_t kMinChunkRows = 64;
    static constexpr size_t kMaxChunkRows = 4096;
    static constexpr auto chunk_rows_for_(size_t stride) -> size_t {
        const size_t row_bytes = stride * sizeof(PosT);
        const size_t rows = std::bit_ceil((kChunkTargetBytes + row_bytes - 1) / row_bytes);
        return std::clamp(rows, kMinChunkRows, kMaxChunkRows);
    }

    // Stride for a given inline width: the positions, plus a leading popcount slot only for the widths
    // that have no spare sentinel codepoint (see kHeaderless).
    static constexpr auto stride_for_(size_t inline_width) -> size_t { return inline_width + (kHeaderless ? 0 : 1); }

    // The inline width (hence stride) is a construction invariant: any width is correct, since
    // over-long rows spill to overflow losslessly.
    explicit OperatorIndex(size_t inline_width = kDefaultInlinePositions)
        : inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)),
          stride_(stride_for_(inline_width_)),
          chunk_log_(static_cast<size_t>(std::countr_zero(chunk_rows_for_(stride_)))),
          chunk_mask_(chunk_rows_for_(stride_) - 1) {}
    OperatorIndex(const OperatorIndex &) = delete;
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    OperatorIndex(OperatorIndex &&) = delete;
    OperatorIndex &operator=(OperatorIndex &&) = delete;

    // Single named deep-copy: entries are re-inserted into the clone's table (not copied verbatim).
    // Called only on an idle store, so it needs no synchronization.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(inline_width_);
        // Copy only the LIVE rows: the clone gets a right-sized arena rather than the source's tail
        // slack. Chunk geometry is derived from inline_width_, so both sides chunk identically.
        out->reserve_rows(size_);
        out->size_ = size_;
        for (size_t c = 0; c < out->chunks_.size(); ++c) {
            const size_t first = c << chunk_log_;
            const size_t rows_here = std::min(chunk_mask_ + 1, size_ - first);
            std::copy_n(chunks_[c].get(), rows_here * stride_, out->chunks_[c].get());
        }
        out->overflow_ = overflow_;
        out->reserve_index(table_.count);
        // The clone's rows/overflow are already in place, so it can re-derive each key's hash from its
        // own rows -- the split table stores only an 8-bit tag, not a reconstructible 32-bit hash.
        for (const TermIndex e : table_.idx) {
            if (e != kEmptySlot) {
                out->insert_slot_(e, out->row_hash(static_cast<size_t>(e)));
            }
        }
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    // reserve grows ROW capacity and right-sizes the index together; grow_rows_geometric grows rows
    // alone per layer, and bulk_insert right-sizes the index by its element count.
    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    // Grow the row store by `n` rows, returning the pre-growth size (the caller's insert base). Growth
    // is CHUNK-granular: append however many fixed-size chunks the new rows need. No geometric
    // over-allocation is required (nothing is ever reallocated or copied), so the only dead capacity is
    // the tail of the last chunk. The name is kept for its call sites; "geometric" describes the growth
    // policy this arena replaced.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        reserve_rows(base + n);
        // Chunks are allocated with make_unique_for_overwrite: default-init, NOT zeroed. Every freshly
        // grown row is overwritten by set() before any read, so a zero-fill would be wasted bandwidth.
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &maj) -> void { set(grow_rows_geometric(1), maj); }

    // Write row i from `maj` (grown-but-uninitialized or a prior value). Never pre-reads the row's own
    // header/terminator (freshly grown rows are indeterminate); a stale overflow entry at i, if any, is
    // dropped — cheap when the overflow map is empty, which is the common case.
    auto set(size_t i, const value_type &maj) -> void {
        const size_t c = maj.count();
        PosT *row = row_ptr(i);
        if (c > inline_width_) {
            row[0] = kHeaderless ? kOverflowPos : kOverflowMarker;
            overflow_[i] = maj;
            return;
        }
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
        PosT *out = positions_of(row);
        if constexpr (!kHeaderless) {
            row[0] = static_cast<PosT>(c);
        }
        for (size_t b = maj.find_first(); b < maj.size(); b = maj.find_next(b)) {
            *out++ = static_cast<PosT>(b);
        }
        if constexpr (kHeaderless) {
            // A row holding exactly inline_width_ positions is full: there is no slot for a terminator
            // and none is needed, since the scan stops at the width limit.
            if (c < inline_width_) {
                *out = kTerminator;
            }
        }
    }

    [[nodiscard]] auto row(size_t i) const -> value_type {
        const PosT *row = row_ptr(i);
        const size_t c = inline_count(row);
        if (c == kOverflowRow) {
            return overflow_.at(i);
        }
        value_type maj;
        const PosT *pos = positions_of(row);
        for (size_t j = 0; j < c; ++j) {
            maj.set(pos[j]);
        }
        return maj;
    }
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        const PosT *row = row_ptr(i);
        const size_t c = inline_count(row);
        if (c == kOverflowRow) {
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        const PosT *pos = positions_of(row);
        for (size_t j = 0; j < c; ++j) {
            fn(static_cast<size_t>(pos[j]));
        }
    }
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        if (const size_t c = inline_count(row_ptr(i)); c != kOverflowRow) {
            return c;
        }
        return overflow_.at(i).count();
    }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = chunks_.size() * (chunk_mask_ + 1) * stride_ * sizeof(PosT);
        total += chunks_.capacity() * sizeof(typename decltype(chunks_)::value_type);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    // Returns the dense row index for `key`, or nullopt if absent. Usage: `if (auto i = find(k)) ...`.
    auto find(const key_type &key) const -> std::optional<size_t> {
        const uint32_t h = fold_hash(key);
        if (table_.count == 0) {
            return std::nullopt;
        }
        const uint8_t t = tag_of(h);
        size_t s = spread(h) & table_.mask;
        for (;; s = (s + 1) & table_.mask) {
            const TermIndex e = table_.idx[s];
            if (e == kEmptySlot) {
                return std::nullopt;
            }
            if (table_.tags[s] == t && row_eq_key(static_cast<size_t>(e), key)) {
                return static_cast<size_t>(e);
            }
        }
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kMissingIndex. Same result as n
    // find() calls, but overlaps DRAM misses via a per-group hash/probe/confirm pipeline. A tag
    // collision falls back to an exact find. MUST NOT run concurrently with inserts.
    auto find_batch(const key_type *keys, size_t n, size_t *out) const -> void {
        static constexpr size_t G = 16; // keys prefetched together per pipeline pass
        std::array<uint8_t, G> tt;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                const uint32_t h = fold_hash(keys[base + j]);
                tt[j] = tag_of(h);
                sp[j] = spread(h);
                // Both halves of the split slot are on the probe's critical path.
                __builtin_prefetch(&table_.idx[sp[j] & table_.mask], 0, 0);
                __builtin_prefetch(&table_.tags[sp[j] & table_.mask], 0, 0);
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kEmptySlot;
                if (table_.count == 0) {
                    continue;
                }
                size_t s = sp[j] & table_.mask;
                for (;; s = (s + 1) & table_.mask) {
                    const TermIndex e = table_.idx[s];
                    if (e == kEmptySlot) {
                        break;
                    }
                    if (table_.tags[s] == tt[j]) {
                        cand[j] = e;
                        break;
                    }
                }
                if (cand[j] != kEmptySlot) {
                    __builtin_prefetch(row_ptr(static_cast<size_t>(cand[j])), 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                if (cand[j] != kEmptySlot && row_eq_key(static_cast<size_t>(cand[j]), keys[base + j])) {
                    out[base + j] = static_cast<size_t>(cand[j]);
                }
                else if (cand[j] != kEmptySlot) {
                    // tag collision: the first tag match wasn't the key — resolve exactly.
                    const auto v = find(keys[base + j]);
                    out[base + j] = v ? *v : kNotFound;
                }
                else {
                    out[base + j] = kNotFound;
                }
            }
        }
    }

    // Insert-or-no-op. Row at `value` MUST already be written (the confirm reads dense rows).
    auto emplace(const key_type &key, mapped_type value) -> void {
        check_index_fits(value);
        const uint32_t h = fold_hash(key);
        rehash_if_needed();
        const uint8_t t = tag_of(h);
        size_t s = spread(h) & table_.mask;
        while (table_.idx[s] != kEmptySlot) {
            if (table_.tags[s] == t && row_eq_key(static_cast<size_t>(table_.idx[s]), key)) {
                return; // key already present — no-op (matches the former set semantics)
            }
            s = (s + 1) & table_.mask;
        }
        table_.idx[s] = static_cast<TermIndex>(value);
        table_.tags[s] = t;
        ++table_.count;
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows MUST already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        // Right-size the table ONCE up front, to the exact post-insert entry count. Without this the
        // table doubles its way up from kMinSlots (~17 rehashes over a 10^6-term build) and every
        // rehash now re-derives each key's hash from its row -- a cost this reserve removes entirely:
        // the loop below provably never rehashes.
        reserve_index(table_.count + n);
        for (size_t k = 0; k < n; ++k) {
            insert_slot_(static_cast<TermIndex>(base + k), fold_hash(key_at(k)));
        }
    }
    // Visits every indexed (row, index) pair in table order.
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (const TermIndex e : table_.idx) {
            if (e != kEmptySlot) {
                fn(row(static_cast<size_t>(e)), static_cast<size_t>(e));
            }
        }
    }
    /// Diagnostic: the part of @ref memory_bytes that is allocated but unused row capacity. With the
    /// segmented arena that is the tail of the last chunk plus the chunk table's spare slots, so it is
    /// bounded by ~one page per store rather than by a fraction of the row bytes.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        const size_t rows_slack = (capacity() - size_) * stride_ * sizeof(PosT);
        const size_t table_slack =
            (chunks_.capacity() - chunks_.size()) * sizeof(typename decltype(chunks_)::value_type);
        return rows_slack + table_slack;
    }

    auto index_estimated_memory_bytes() const -> size_t {
        return sizeof(OperatorIndex) + table_.idx.capacity() * sizeof(TermIndex)
               + table_.tags.capacity() * sizeof(uint8_t);
    }

private:
    static uint32_t fold_hash(const key_type &q) noexcept {
        const size_t full = MonomialHash<NumModes>{}(q);
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }
    // Avalanche the cached 32-bit fold into a full-width hash (splitmix64 finalizer): the stored h
    // is only an equality pre-filter, so it must be re-mixed before its low bits drive table bucketing.
    static size_t spread(uint32_t h) noexcept {
        uint64_t x = static_cast<uint64_t>(h) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }
    // 8-bit equality pre-filter derived from the 32-bit fold. Bucketing goes through spread(), which
    // avalanches every input bit, so the low byte is statistically independent of the bucket index; a
    // tag match that is not a key match costs one extra row_eq_key at a rate of 1/256.
    static constexpr auto tag_of(uint32_t h) noexcept -> uint8_t { return static_cast<uint8_t>(h); }

    /// Re-derive the fold hash of the key stored in row @p i. The table keeps only an 8-bit tag, so a
    /// rehash cannot re-place entries from the table alone; the term index IS a pointer to the key, and
    /// this is how a rehash recovers the full hash. Costs one row materialization per entry moved.
    [[nodiscard]] auto row_hash(size_t i) const -> uint32_t { return fold_hash(row(i)); }

    // One open-addressing table: power-of-2 slot count, linear probing, max load factor 0.7
    // (the group-prefetch win erodes at high load — longer probe chains add un-prefetched reads).
    //
    // SPLIT (Swiss/F14) layout: a TermIndex array plus a parallel 1-byte tag array, 5 B/slot in the
    // default build against the 8 B an interleaved {TermIndex, uint32_t} slot cost (padding included).
    // The saving is a flat 37.5% at any load factor, and the tag array is far denser per cache line
    // than interleaved slots. `idx[s] == kEmptySlot` alone marks an empty slot, so `tags[s]` is read
    // only after the index says the slot is occupied.
    struct Table {
        std::vector<TermIndex> idx = std::vector<TermIndex>(kMinSlots, kEmptySlot);
        std::vector<uint8_t> tags = std::vector<uint8_t>(kMinSlots, uint8_t{0});
        size_t mask = kMinSlots - 1;
        size_t count = 0;

        [[nodiscard]] auto slot_count() const -> size_t { return idx.size(); }
    };
    static constexpr size_t kMinSlots = 16;
    // Slot count for `n` entries at ≤0.7 load.
    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, n * 10 / 7 + 1)); }

    auto rehash_if_needed() -> void {
        if ((table_.count + 1) * 10 >= table_.slot_count() * 7) {
            rehash_to(table_.slot_count() * 2);
        }
    }
    // Grow to >= new_cap slots (rounded up to a power of two) and re-place every entry. Unlike the
    // former 8-byte layout this cannot re-probe from a stored hash, so each surviving entry's hash is
    // re-derived from its row via row_hash(). bulk_insert/reserve_index right-size the table up front
    // so the dominant build path pays this at most once per size class.
    auto rehash_to(size_t new_cap) -> void {
        new_cap = std::bit_ceil(std::max<size_t>(new_cap, kMinSlots));
        if (new_cap <= table_.slot_count()) {
            return;
        }
        const std::vector<TermIndex> old = std::move(table_.idx);
        table_.idx.assign(new_cap, kEmptySlot);
        table_.tags.assign(new_cap, uint8_t{0});
        table_.mask = new_cap - 1;
        for (const TermIndex e : old) {
            if (e == kEmptySlot) {
                continue;
            }
            const uint32_t h = row_hash(static_cast<size_t>(e));
            size_t s = spread(h) & table_.mask;
            while (table_.idx[s] != kEmptySlot) {
                s = (s + 1) & table_.mask;
            }
            table_.idx[s] = e;
            table_.tags[s] = tag_of(h);
        }
    }

    // inline_count()'s "this row spilled to the overflow side-map" result.
    static constexpr size_t kOverflowRow = std::numeric_limits<size_t>::max();

    // The two primitives that hide the header/headerless split from every row reader: where a row's
    // positions start, and how many of them there are.
    [[nodiscard]] static constexpr auto positions_of(const PosT *row) -> const PosT * {
        return row + (kHeaderless ? 0 : 1);
    }
    [[nodiscard]] static constexpr auto positions_of(PosT *row) -> PosT * { return row + (kHeaderless ? 0 : 1); }
    [[nodiscard]] auto inline_count(const PosT *row) const -> size_t {
        if constexpr (kHeaderless) {
            if (row[0] == kOverflowPos) {
                return kOverflowRow;
            }
            // Scan to the terminator, or to the width limit for an exactly-full row. Bounded by
            // inline_width_ elements, all within the row's own (already-touched) cache line(s).
            size_t c = 0;
            while (c < inline_width_ && row[c] != kTerminator) {
                ++c;
            }
            return c;
        }
        else {
            return row[0] == kOverflowMarker ? kOverflowRow : static_cast<size_t>(row[0]);
        }
    }

    // Single point of row addressing: every reader and writer goes through these. Stride is fixed
    // within a chunk, so the returned span [p, p + stride_) is the whole row -- which is what lets
    // find_batch prefetch a candidate row from its index alone. One extra load, from a chunk table
    // small enough to stay resident (a 1.17M-row store at stride 7 tables 1143 chunks = 9 KiB).
    [[nodiscard]] auto row_ptr(size_t i) const -> const PosT * {
        return chunks_[i >> chunk_log_].get() + (i & chunk_mask_) * stride_;
    }
    [[nodiscard]] auto row_ptr(size_t i) -> PosT * {
        return chunks_[i >> chunk_log_].get() + (i & chunk_mask_) * stride_;
    }

    [[nodiscard]] auto capacity() const -> size_t { return chunks_.size() << chunk_log_; }
    auto reserve_rows(size_t n) -> void {
        const size_t need = (n + chunk_mask_) >> chunk_log_;
        chunks_.reserve(need);
        while (chunks_.size() < need) {
            chunks_.push_back(std::make_unique_for_overwrite<PosT[]>((chunk_mask_ + 1) * stride_));
        }
    }
    auto reserve_index(size_t n) -> void { rehash_to(slots_for_(n + 1)); }

    // Insert (idx, h) into the table with NO dup probe — callers on this path insert provably distinct
    // keys (⊕G-injective miss batches, clone re-insertion). Grows the table first if needed.
    auto insert_slot_(TermIndex idx, uint32_t h) -> void {
        rehash_if_needed();
        size_t s = spread(h) & table_.mask;
        while (table_.idx[s] != kEmptySlot) {
            s = (s + 1) & table_.mask;
        }
        table_.idx[s] = idx;
        table_.tags[s] = tag_of(h);
        ++table_.count;
    }

    // Compare row i against key q without materializing the row (the find confirm). Reads the
    // popcount byte first, so a false h prefilter match usually costs one byte compare.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &q) const -> bool {
        const PosT *row = row_ptr(i);
        if constexpr (kHeaderless) {
            if (row[0] == kOverflowPos) {
                return overflow_.at(i) == q;
            }
            // SINGLE pass, deliberately not inline_count() + a position walk: this is the confirm on
            // every index hit, and counting first then re-walking measured ~10% slower on find_batch.
            // Testing positions as they are scanned costs the same reads as the header layout's
            // "read popcount, then walk", and rejects a wrong row on its first mismatching position.
            size_t c = 0;
            for (; c < inline_width_; ++c) {
                const PosT p = row[c];
                if (p == kTerminator) {
                    break;
                }
                if (!q.test(p)) {
                    return false;
                }
            }
            return q.count() == c;
        }
        else {
            if (row[0] == kOverflowMarker) {
                return overflow_.at(i) == q;
            }
            const size_t c = static_cast<size_t>(row[0]);
            if (q.count() != c) {
                return false;
            }
            for (size_t j = 0; j < c; ++j) {
                if (!q.test(row[1 + j])) {
                    return false;
                }
            }
            return true;
        }
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                                          "-Dmonoprop_WIDE_TERM_INDEX (this shard's term count exceeded ~2^32).");
        }
    }

    // Segmented row arena. Chunks are allocated for-overwrite (default-init, no zero-fill): set()
    // overwrites each row's header and positions before any read, and never pre-reads the
    // (indeterminate) header. Never reallocated, so row addresses are stable for the store's life.
    std::vector<std::unique_ptr<PosT[]>> chunks_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    size_t chunk_log_ = static_cast<size_t>(std::countr_zero(chunk_rows_for_(1 + kMaxInlinePositions)));
    size_t chunk_mask_ = chunk_rows_for_(1 + kMaxInlinePositions) - 1;
    // Lossless side-map for rows whose popcount exceeds inline_width_. Single-writer: never accessed
    // concurrently (parallelism is cross-shard), so it needs no lock.
    std::unordered_map<size_t, value_type> overflow_ = {};
    Table table_ = {};
};

} // namespace monoprop::detail
