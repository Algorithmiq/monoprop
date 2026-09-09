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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/ChunkedArray.h"

namespace monoprop::detail {

class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/*! @brief Operator-term store: entropy-packed position-list rows in pooled chunks, plus a keyless
 *  open-addressing hash index over those rows.
 *
 *  Row layout: slot 0 = popcount c (or kOverflowMarker if c > inline_width_), slots 1..c = ascending
 *  set-bit positions. inline_width_ is a free parameter -- any width is correct, over-long rows spill
 *  losslessly to overflow. Rows live in ChunkedRowArray chunks, so growth appends a chunk and never
 *  moves a row; row indices are handed out consecutively by grow_rows_geometric and never change.
 *
 *  Single-writer: one partition, one thread; parallelism is cross-partition.
 */
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = Monomial<NumModes>;
    using key_type = Monomial<NumModes>;
    using mapped_type = size_t;

    using PosT = std::
        conditional_t<(2 * NumModes <= 256), uint8_t, std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;

    static constexpr size_t kDefaultInlinePositions = 11;
    /*! @brief Rows per chunk of the row store, chosen from the height the store is built at.
     *
     *  Both bounds are powers of two and multiples of 64, so every chunk length is: the 64 rows an
     *  inverted-index word names always sit in one chunk. A row never straddles a chunk either, so a
     *  span over one row stays contiguous.
     */
    static constexpr size_t kMinRowsPerChunk = size_t{1} << 12;
    static constexpr size_t kMaxRowsPerChunk = size_t{1} << 18;

    /*! @brief The chunk length for a store of @a rows rows: a quarter of it, rounded down to a power
     *  of two and clamped to [kMinRowsPerChunk, kMaxRowsPerChunk].
     *
     *  The store overshoots by at most one chunk, so a quarter bounds its slack under a quarter of
     *  itself once it clears four minimum chunks, and under one 4096-row chunk below that. Rounding
     *  down costs only chunk count, which is pooled. It reaches kMaxRowsPerChunk at 1M rows and stops,
     *  so the tail is never more than 2^18 rows however large the operator grows.
     */
    static auto chunk_rows_for_rows(size_t rows) noexcept -> size_t {
        return std::clamp(std::bit_floor(std::max(rows / 4, size_t{1})), kMinRowsPerChunk, kMaxRowsPerChunk);
    }
    // A weight-w Pauli needs 2w positions; 32 covers the common case inline at the supported Pauli
    // cutoffs (2*cutoff <= 32 for cutoff <= 16).
    static constexpr size_t kMaxInlinePositions = 32;
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max();
    //! Header of a row held in the wide tier; its inline slots carry the tier slot instead of positions.
    static constexpr PosT kWideMarker = static_cast<PosT>(std::numeric_limits<PosT>::max() - 1);
    //! A wide row's slot is a uint32 written over the inline positions, so the narrow row needs 4 of them.
    static constexpr size_t kMinInlineForWideTier = sizeof(uint32_t);
    //! Wide rows above this share of the store and the inline width was guessed too narrow: restride.
    static constexpr size_t kRestridePercent = 3;

    static_assert((2 * NumModes) - 1 <= std::numeric_limits<PosT>::max(),
                  "OperatorIndex PosT too narrow for 2*NumModes positions");
    static_assert(kMaxInlinePositions < std::numeric_limits<PosT>::max() - 1,
                  "the marker sentinels must not collide with a valid popcount");

    // Valid term indices are < kIndexCeiling (check_append_fits refuses the append that would reach
    // it, check_index_fits the index itself), so the all-ones TermIndex is free to mark an empty slot.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    static constexpr TermIndex kEmptySlot = std::numeric_limits<TermIndex>::max();
    // find_batch's "absent" result; same value as detail::kMissingIndex (not included here — the
    // operator store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

    /*! @brief An empty store. Its chunk length is settled by the first reserve() or growth, from the
     *  height asked for there, and rises with the store thereafter.
     *
     *  @param forced_rows_per_chunk 0 to size the chunks from that height (production), or a fixed
     *  length: a power of two and a multiple of 64. A test knob, so chunk boundaries can be driven
     *  with a few hundred rows instead of a million.
     *  @param wide_width the structural bound: rows over @a inline_width but no wider than this get a
     *  fixed-stride second tier instead of the side-map. 0, or anything below @a inline_width, means
     *  no tier.
     */
    explicit OperatorIndex(size_t inline_width = kDefaultInlinePositions,
                           size_t forced_rows_per_chunk = 0,
                           size_t wide_width = 0)
        : inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)),
          stride_(1 + inline_width_),
          wide_width_(std::clamp<size_t>(wide_width, inline_width_, kMaxInlinePositions)),
          forced_rows_per_chunk_(forced_rows_per_chunk) {
        assert(forced_rows_per_chunk == 0
               || (std::has_single_bit(forced_rows_per_chunk) && forced_rows_per_chunk % 64 == 0));
    }
    //! Rows wider than the inline width but within the structural bound: the second tier's population.
    [[nodiscard]] auto wide_size() const -> size_t { return wide_size_; }
    [[nodiscard]] auto inline_width() const -> size_t { return inline_width_; }
    [[nodiscard]] auto wide_width() const -> size_t { return wide_width_; }
    //! How many times this store has re-laid its rows at a wider inline width.
    [[nodiscard]] auto restrides() const -> size_t { return restrides_; }

    /*! @brief Whether the inline width was guessed too narrow to be worth keeping.
     *
     *  The inline saving is paid by every row and the tier only by the rows in it, so a few per cent
     *  of wide rows is the price of a much narrower row for the rest; past kRestridePercent it is not.
     *  Only ever true while a tier exists, so a store restrides at most once.
     */
    [[nodiscard]] auto should_restride() const -> bool {
        return wide_tier_live_() && wide_size_ * 100 > kRestridePercent * size_;
    }

    /*! @brief Widens the structural bound, so rows a raised cutoff admits get a tier, not the map.
     *
     *  Draining the current tier first leaves the rows laid out at the old bound and a fresh tier
     *  spanning old bound to new, which is the shape a store built at the new cutoff would have taken.
     *  Rows already in the side-map stay there: they are stored losslessly.
     */
    auto raise_bound(size_t new_wide_width) -> void {
        if (new_wide_width <= wide_width_) {
            return;
        }
        restride_to_bound();
        wide_width_ = std::clamp<size_t>(new_wide_width, inline_width_, kMaxInlinePositions);
    }

    /*! @brief Re-lays every row at the structural bound, emptying the wide tier. Between gates only.
     *
     *  Only the byte layout changes: row indices, the hash index, the overflow map and every TermIndex
     *  the rest of the engine holds are untouched, so nothing has to be rebuilt. A caller holding a
     *  span into a row must not span this call -- restriding moves every row.
     */
    auto restride_to_bound() -> void {
        if (!wide_tier_live_()) {
            return;
        }
        const size_t new_stride = 1 + wide_width_;
        auto new_pool = std::make_unique<ChunkPool>(rows_.rows_per_chunk() * new_stride * sizeof(PosT));
        ChunkedRowArray<PosT> new_rows;
        new_rows.attach(*new_pool, rows_.rows_per_chunk(), new_stride);
        new_rows.grow(size_);
        for (size_t i = 0; i < size_; ++i) {
            PosT *const dst = new_rows.at(i);
            const StoredRow src = stored_(rows_.at(i));
            if (src.pos == nullptr) {
                dst[0] = kOverflowMarker; // stays in the side-map: it is over the bound, not under it
                continue;
            }
            dst[0] = static_cast<PosT>(src.count);
            std::copy_n(src.pos, src.count, dst + 1);
        }
        rows_ = std::move(new_rows);
        row_pool_ = std::move(new_pool);
        inline_width_ = wide_width_;
        stride_ = new_stride;
        wide_ = {};
        wide_pool_.reset();
        wide_size_ = 0;
        ++restrides_;
    }
    OperatorIndex(const OperatorIndex &) = delete;
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    OperatorIndex(OperatorIndex &&) = delete;
    OperatorIndex &operator=(OperatorIndex &&) = delete;

    // Called only on an idle store, so it needs no synchronization.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(inline_width_, forced_rows_per_chunk_, wide_width_);
        if (wide_.attached()) {
            out->attach_wide_();
            out->wide_ = wide_.clone_into(*out->wide_pool_);
            out->wide_size_ = wide_size_;
        }
        out->restrides_ = restrides_;
        if (rows_.attached()) {
            // The clone takes this store's settled length, not one re-derived from its size: the two
            // must agree row for row, and attach_(0) would round a small store down differently.
            out->attach_(rows_.rows_per_chunk());
            out->rows_ = rows_.clone_into(*out->row_pool_); // a deep copy; the two share no chunk
        }
        out->size_ = size_;
        out->overflow_ = overflow_;
        out->reserve_index(table_.count);
        for (const Slot &e : table_.slots) {
            if (e.idx != kEmptySlot) {
                out->insert_slot_(e.idx, e.h);
            }
        }
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    // Rows that exceeded inline_width_ and spilled; observable so a test can compare the two insert paths.
    [[nodiscard]] auto overflow_size() const -> size_t { return overflow_.size(); }

    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    /*! @brief Grows by @a n rows and returns the pre-growth size, the caller's insert base.
     *
     *  Indices are consecutive from that base and are never reassigned: the store appends whole chunks
     *  and never moves a row, so there is no reallocation to amortise and the only slack is one
     *  chunk's tail. Refused at the TermIndex ceiling before anything grows.
     *
     *  Freshly grown rows are default-initialized, not zeroed: every one is overwritten by its set()
     *  before any read.
     */
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        check_append_fits(base, n);
        ensure_capacity_(base + n);
        rows_.grow(base + n);
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so the row header is never pre-read
    // (freshly grown headers are indeterminate); a stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        const size_t c = mono.count();
        PosT *row = rows_.at(i);
        if (c > inline_width_) {
            PosT *const wide = spill_(row, c);
            if (wide == nullptr) {
                overflow_[i] = mono;
                return;
            }
            PosT *w = wide + 1;
            for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
                *w++ = static_cast<PosT>(b);
            }
            drop_stale_overflow_(i);
            return;
        }
        drop_stale_overflow_(i);
        row[0] = static_cast<PosT>(c);
        PosT *out = row + 1;
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            *out++ = static_cast<PosT>(b);
        }
    }

    // set() from the row's own form: a row is an ascending position list. Same postcondition as set(),
    // including the dropped stale overflow entry.
    //
    // Precondition: `pos` strictly ascending, every entry < 2*NumModes. A violation is silent in release
    // -- an unsorted row simply never matches, and an out-of-range one decodes to a different term.
    auto set_positions(size_t i, std::span<const PosT> pos) -> void {
        const size_t count = pos.size();
        assert((count == 0 || static_cast<size_t>(pos[count - 1]) < 2 * NumModes) && "row position out of range");
        PosT *row = rows_.at(i);
        if (count > inline_width_) {
            if (PosT *const wide = spill_(row, count); wide != nullptr) {
                std::copy_n(pos.data(), count, wide + 1);
                drop_stale_overflow_(i);
                return;
            }
            // Over the structural bound, so the side-map takes it. Only this path has no position
            // array to hand it, so it is the one place that builds the dense form.
            value_type mono;
            for (size_t j = 0; j < count; ++j) {
                mono.set(pos[j]);
            }
            overflow_[i] = mono;
            return;
        }
        drop_stale_overflow_(i);
        row[0] = static_cast<PosT>(count);
        std::copy_n(pos.data(), count, row + 1);
    }

    [[nodiscard]] auto row(size_t i) const -> value_type {
        const StoredRow r = stored_(rows_.at(i));
        if (r.pos == nullptr) {
            return overflow_.at(i);
        }
        value_type mono;
        for (size_t j = 0; j < r.count; ++j) {
            mono.set(r.pos[j]);
        }
        return mono;
    }
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        const StoredRow r = stored_(rows_.at(i));
        if (r.pos == nullptr) {
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        for (size_t j = 0; j < r.count; ++j) {
            fn(static_cast<size_t>(r.pos[j]));
        }
    }
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        const StoredRow r = stored_(rows_.at(i));
        return r.pos == nullptr ? overflow_.at(i).count() : r.count;
    }
    /*! @brief The row's stored ascending positions, empty for a spilled row. Invalidated by any insert. */
    struct RowPositions {
        std::span<const PosT> pos;
        //! A spilled row has no position array at all, which an empty inline row still does.
        [[nodiscard]] auto inlined() const -> bool { return pos.data() != nullptr; }
    };
    [[nodiscard]] auto row_positions(size_t i) const -> RowPositions {
        const StoredRow r = stored_(rows_.at(i));
        if (r.pos == nullptr) {
            return {};
        }
        return {std::span<const PosT>(r.pos, r.count)};
    }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = rows_.bytes() + wide_.bytes();
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    auto find(const key_type &key) const -> std::optional<size_t> {
        const uint32_t h = fold_hash(key);
        if (table_.count == 0) {
            return std::nullopt;
        }
        size_t s = spread(h) & table_.mask;
        for (;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return std::nullopt;
            }
            if (e.h == h && row_eq_key(static_cast<size_t>(e.idx), key)) {
                return static_cast<size_t>(e.idx);
            }
        }
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kNotFound. Same result as n
    // find() calls, but overlaps dram misses via a per-group hash/probe/confirm pipeline. An h
    // collision falls back to an exact find. must not run concurrently with inserts.
    auto find_batch(const key_type *keys, size_t n, size_t *out) const -> void {
        static constexpr size_t G = 16; // keys prefetched together per pipeline pass
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = fold_hash(keys[base + j]);
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&table_.slots[sp[j] & table_.mask], 0, 0);
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kEmptySlot;
                if (table_.count == 0) {
                    continue;
                }
                cand[j] = probe_hash_match_(hh[j], sp[j] & table_.mask);
                if (cand[j] != kEmptySlot) {
                    __builtin_prefetch(rows_.at(static_cast<size_t>(cand[j])), 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                if (cand[j] != kEmptySlot && row_eq_key(static_cast<size_t>(cand[j]), keys[base + j])) {
                    out[base + j] = static_cast<size_t>(cand[j]);
                }
                else if (cand[j] != kEmptySlot) {
                    const auto v = find(keys[base + j]);
                    out[base + j] = v ? *v : kNotFound;
                }
                else {
                    out[base + j] = kNotFound;
                }
            }
        }
    }

    // find_batch over ascending position lists: query q is pos_flat[pos_off[q] .. pos_off[q] + k_of[q]).
    // Identical results to find_batch on the monomials those positions describe.
    auto find_batch_positions(std::span<const PosT> pos_flat,
                              std::span<const size_t> pos_off,
                              std::span<const uint32_t> k_of,
                              std::span<size_t> out,
                              std::span<uint32_t> hash_out = {}) const -> void {
        const size_t n = pos_off.size();
        static constexpr size_t G = 16;
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = fold_hash_positions(pos_flat.subspan(pos_off[base + j], k_of[base + j]));
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&table_.slots[sp[j] & table_.mask], 0, 0);
            }
            if (!hash_out.empty()) {
                std::copy_n(hh.begin(), g, hash_out.begin() + static_cast<std::ptrdiff_t>(base));
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kEmptySlot;
                if (table_.count == 0) {
                    continue;
                }
                cand[j] = probe_hash_match_(hh[j], sp[j] & table_.mask);
                if (cand[j] != kEmptySlot) {
                    __builtin_prefetch(rows_.at(static_cast<size_t>(cand[j])), 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                const size_t q = base + j;
                const std::span<const PosT> qpos = pos_flat.subspan(pos_off[q], k_of[q]);
                if (cand[j] == kEmptySlot) {
                    out[q] = kNotFound;
                }
                else if (row_eq_positions(static_cast<size_t>(cand[j]), qpos)) {
                    out[q] = static_cast<size_t>(cand[j]);
                }
                else {
                    // A 32-bit collision: rare enough to walk the chain from the top rather than resume it.
                    out[q] = find_positions_(hh[j], qpos);
                }
            }
        }
    }

    // fold_hash of the monomial `pos` describes, through the same fold, so it is equal by construction.
    [[nodiscard]] static auto fold_hash_positions(std::span<const PosT> pos) noexcept -> uint32_t {
        key_type mono;
        for (size_t j = 0; j < pos.size(); ++j) {
            mono.set(pos[j]);
        }
        return fold_hash(mono);
    }

    // Insert-or-no-op. Row at `value` must already be written (the confirm reads dense rows).
    auto emplace(const key_type &key, mapped_type value) -> void {
        check_index_fits(value);
        const uint32_t h = fold_hash(key);
        table_.rehash_if_needed();
        size_t s = spread(h) & table_.mask;
        while (table_.slots[s].idx != kEmptySlot) {
            if (table_.slots[s].h == h && row_eq_key(static_cast<size_t>(table_.slots[s].idx), key)) {
                return;
            }
            s = (s + 1) & table_.mask;
        }
        table_.slots[s] = Slot{static_cast<TermIndex>(value), h};
        ++table_.count;
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows must already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        if (n == 0) {
            return;
        }
        bulk_insert_hashed(n, base, [&key_at](size_t k) { return fold_hash(key_at(k)); });
    }
    // bulk_insert with the hashes already in hand: same precondition (n distinct rows, already written,
    // at consecutive indices) and the same slot assignment. `hashes[k]` must be fold_hash of the key of
    // row base+k -- a wrong one leaves the row unfindable, which surfaces later as a duplicate insert.
    template <typename HashFn>
    auto bulk_insert_hashed(size_t n, mapped_type base, HashFn &&hash_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        static constexpr size_t G = 16;
        std::array<uint32_t, G> hh;
        for (size_t b = 0; b < n; b += G) {
            const size_t g = std::min(G, n - b);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = hash_at(b + j);
                __builtin_prefetch(&table_.slots[spread(hh[j]) & table_.mask], /*rw=*/1, /*locality=*/0);
            }
            for (size_t j = 0; j < g; ++j) {
                insert_slot_(static_cast<TermIndex>(base + b + j), hh[j]);
            }
        }
    }
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (const Slot &e : table_.slots) {
            if (e.idx != kEmptySlot) {
                fn(row(static_cast<size_t>(e.idx)), static_cast<size_t>(e.idx));
            }
        }
    }
    // Diagnostic: the part of memory_bytes() that is unused capacity -- the tail of the last chunk.
    [[nodiscard]] auto slack_bytes() const -> size_t { return rows_.slack_bytes(); }

    /*! @brief What this store's pool has MAPPED, free chunks included.
     *
     *  memory_bytes() prices the chunks the array holds; a pool maps whole arenas and keeps one as
     *  long as a single chunk in it is live, so the two differ by whatever a partly used arena has not
     *  handed out. Not a subset of any other field.
     */
    [[nodiscard]] auto pool_mapped_bytes() const -> size_t { return pool_sum_(&ChunkPool::mapped_bytes); }
    //! Of pool_mapped_bytes(): the arenas' chunks that are not currently handed out.
    [[nodiscard]] auto pool_free_chunk_bytes() const -> size_t {
        return pool_mapped_bytes() - pool_sum_(&ChunkPool::live_bytes);
    }

    auto index_estimated_memory_bytes() const -> size_t {
        return sizeof(OperatorIndex) + (table_.slots.capacity() * sizeof(Slot));
    }

private:
    //! @a metric summed over the pools this store owns; an unmade pool contributes nothing.
    [[nodiscard]] auto pool_sum_(size_t (ChunkPool::*metric)() const noexcept) const -> size_t {
        size_t total = 0;
        for (const ChunkPool *pool : {row_pool_.get(), wide_pool_.get()}) {
            if (pool != nullptr) {
                total += (pool->*metric)();
            }
        }
        return total;
    }

    /*! @brief A row's stored positions, whichever tier holds them.
     *
     *  One unsigned compare carries the common case: a header at or below the inline width is the
     *  row's own popcount and its positions follow it. Both sentinels are above kMaxInlinePositions,
     *  so the branch is never a table lookup. @a pos == nullptr means the row is over the structural
     *  bound and lives in the side-map.
     */
    struct StoredRow {
        const PosT *pos;
        size_t count;
    };
    [[nodiscard]] auto stored_(const PosT *src) const noexcept -> StoredRow {
        const size_t c = src[0];
        if (c <= inline_width_) [[likely]] {
            return {src + 1, c};
        }
        if (c == kWideMarker) {
            const PosT *const w = wide_.at(wide_slot_(src));
            return {w + 1, static_cast<size_t>(w[0])};
        }
        return {nullptr, 0};
    }
    //! A wide row's tier slot, written unaligned over the narrow row's inline positions.
    [[nodiscard]] static auto wide_slot_(const PosT *src) noexcept -> size_t {
        uint32_t slot = 0;
        std::memcpy(&slot, src + 1, sizeof(slot));
        return slot;
    }
    //! The tier is worth having only while it is narrower than the bound and the slot fits inline.
    [[nodiscard]] auto wide_tier_live_() const noexcept -> bool {
        return wide_width_ > inline_width_ && inline_width_ >= kMinInlineForWideTier;
    }
    auto attach_wide_() -> void {
        if (!wide_.attached()) {
            wide_pool_ = std::make_unique<ChunkPool>(kWideRowsPerChunk * (1 + wide_width_) * sizeof(PosT));
            wide_.attach(*wide_pool_, kWideRowsPerChunk, 1 + wide_width_);
        }
    }

    /*! @brief Marks @a row as wide and returns its tier row, or nullptr if it belongs in the side-map.
     *
     *  Slots are appended, never reclaimed: a row is written once in every engine path, and a store
     *  whose tier grows past kRestridePercent is re-laid wide anyway, which frees the tier entirely.
     */
    auto spill_(PosT *row, size_t count) -> PosT * {
        if (count > wide_width_ || !wide_tier_live_()) {
            row[0] = kOverflowMarker;
            return nullptr;
        }
        attach_wide_();
        const auto slot = static_cast<uint32_t>(wide_.size());
        wide_.grow(wide_.size() + 1);
        ++wide_size_;
        row[0] = kWideMarker;
        std::memcpy(row + 1, &slot, sizeof(slot));
        PosT *const out = wide_.at(slot);
        out[0] = static_cast<PosT>(count);
        return out;
    }

    //! Drops a side-map entry left by a previous value at @a i; the map is empty on every hot path.
    auto drop_stale_overflow_(size_t i) -> void {
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
    }

    struct Slot {
        TermIndex idx = kEmptySlot;
        uint32_t h = 0;
    };

    // First slot on h's probe chain whose stored hash matches, or kEmptySlot if the chain ends first.
    // Matches on h alone and leaves the dense-row comparison to the caller — that deferral is what lets
    // find_batch prefetch the row between probe and confirm, so do not fold row_eq_key in here (find()
    // deliberately keeps its own confirming variant). `start` must already be masked; the table must not
    // be mutated concurrently.
    [[gnu::always_inline]] auto probe_hash_match_(uint32_t h, size_t start) const -> TermIndex {
        for (size_t s = start;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return kEmptySlot;
            }
            if (e.h == h) {
                return e.idx;
            }
        }
    }

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

    // One open-addressing table: power-of-2 slot count, linear probing, max load factor 0.7
    // (the group-prefetch win erodes at high load — longer probe chains add un-prefetched reads).
    struct Table {
        std::vector<Slot> slots = std::vector<Slot>(kMinSlots, Slot{});
        size_t mask = kMinSlots - 1;
        size_t count = 0;

        auto rehash_if_needed() -> void {
            if ((count + 1) * 10 >= slots.size() * 7) {
                rehash_to(slots.size() * 2);
            }
        }
        auto rehash_to(size_t new_cap) -> void {
            new_cap = std::bit_ceil(std::max<size_t>(new_cap, kMinSlots));
            if (new_cap <= slots.size()) {
                return;
            }
            std::vector<Slot> old = std::move(slots);
            slots.assign(new_cap, Slot{});
            mask = new_cap - 1;
            for (const Slot &e : old) {
                if (e.idx == kEmptySlot) {
                    continue;
                }
                size_t s = spread(e.h) & mask;
                while (slots[s].idx != kEmptySlot) {
                    s = (s + 1) & mask;
                }
                slots[s] = e;
            }
        }
    };
    static constexpr size_t kMinSlots = 16;
    // Slot count for `n` entries at ≤0.7 load.
    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, (n * 10 / 7) + 1)); }

    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity(); }
    auto reserve_rows(size_t n) -> void {
        ensure_capacity_(n);
        rows_.reserve(n);
    }

    /*! @brief Makes the pool and binds the row array at @a rows_per_chunk. @pre Not yet attached. */
    auto attach_(size_t rows_per_chunk) -> void {
        assert(!rows_.attached() && "attach_ binds an unbound store");
        row_pool_ = std::make_unique<ChunkPool>(rows_per_chunk * stride_ * sizeof(PosT));
        rows_.attach(*row_pool_, rows_per_chunk, stride_);
    }

    /*! @brief Re-lays the live rows into chunks of @a new_rows_per_chunk and drops the old pool.
     *
     *  Only the chunk geometry moves: row indices and the overflow map are untouched, so every
     *  TermIndex the rest of the engine holds stays valid. @a new_rows_per_chunk is a multiple of the
     *  current length (both are powers of two and it only ever grows), so one old chunk lands whole
     *  inside one new chunk and each move is a single memcpy.
     */
    auto rechunk_(size_t new_rows_per_chunk) -> void {
        const size_t old_rows_per_chunk = rows_.rows_per_chunk();
        assert(new_rows_per_chunk > old_rows_per_chunk && new_rows_per_chunk % old_rows_per_chunk == 0);
        auto new_pool = std::make_unique<ChunkPool>(new_rows_per_chunk * stride_ * sizeof(PosT));
        ChunkedRowArray<PosT> new_rows;
        new_rows.attach(*new_pool, new_rows_per_chunk, stride_);
        new_rows.grow(size_);
        for (size_t first = 0; first < size_; first += old_rows_per_chunk) {
            const size_t n = std::min(old_rows_per_chunk, size_ - first);
            std::memcpy(new_rows.at(first), rows_.at(first), n * stride_ * sizeof(PosT));
        }
        // The array releases its chunks to its own pool before that pool is replaced.
        rows_ = std::move(new_rows);
        row_pool_ = std::move(new_pool);
    }

    /*! @brief Keeps the chunk length in step with the height the store is heading for.
     *
     *  The store cannot be told its final height -- the propagator reserves the *initial* operator's
     *  size, a handful of terms even for a run that ends at millions -- so the length is re-derived on
     *  every growth. It only ever rises and stops at kMaxRowsPerChunk, so a store crossing 2^20 rows
     *  migrates six times, while one that stays small keeps chunks proportional to it. A forced length
     *  never moves.
     */
    auto ensure_capacity_(size_t rows) -> void {
        const size_t want = forced_rows_per_chunk_ != 0 ? forced_rows_per_chunk_ : chunk_rows_for_rows(rows);
        if (!rows_.attached()) {
            attach_(want);
        }
        else if (want > rows_.rows_per_chunk()) {
            rechunk_(want);
        }
    }
    auto reserve_index(size_t n) -> void { table_.rehash_to(slots_for_(n + 1)); }

    // Insert (idx, h) into the table with no duplicate probe — callers on this path insert provably distinct
    // keys (⊕G-injective miss batches, clone re-insertion).
    auto insert_slot_(TermIndex idx, uint32_t h) -> void {
        table_.rehash_if_needed();
        size_t s = spread(h) & table_.mask;
        while (table_.slots[s].idx != kEmptySlot) {
            s = (s + 1) & table_.mask;
        }
        table_.slots[s] = Slot{idx, h};
        ++table_.count;
    }

    // Compare row i against key q without materializing the row (the find confirm). Reads the
    // popcount byte first, so a false h prefilter match usually costs one byte compare.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &q) const -> bool {
        const StoredRow r = stored_(rows_.at(i));
        if (r.pos == nullptr) [[unlikely]] {
            return overflow_.at(i) == q;
        }
        if (q.count() != r.count) {
            return false;
        }
        for (size_t j = 0; j < r.count; ++j) {
            if (!q.test(r.pos[j])) {
                return false;
            }
        }
        return true;
    }

    // Compare row i against an ascending position list; a spilled row falls back to a dense compare.
    [[nodiscard]] auto row_eq_positions(size_t i, std::span<const PosT> q) const -> bool {
        const StoredRow r = stored_(rows_.at(i));
        if (r.pos == nullptr) [[unlikely]] {
            key_type mono;
            for (size_t j = 0; j < q.size(); ++j) {
                mono.set(q[j]);
            }
            return overflow_.at(i) == mono;
        }
        if (q.size() != r.count) {
            return false;
        }
        return std::equal(q.begin(), q.end(), r.pos);
    }

    // find()'s chain walk for a position-list key, hash already folded; only the collision arm reaches it.
    [[nodiscard]] auto find_positions_(uint32_t h, std::span<const PosT> q) const -> size_t {
        if (table_.count == 0) {
            return kNotFound;
        }
        for (size_t s = spread(h) & table_.mask;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return kNotFound;
            }
            if (e.h == h && row_eq_positions(static_cast<size_t>(e.idx), q)) {
                return static_cast<size_t>(e.idx);
            }
        }
    }

    // Refused before anything grows: an append that would pass the ceiling cannot be unwound, and
    // size_ must never reach a count whose last index is unrepresentable. Written as a subtraction
    // because base + n is the sum that would wrap.
    static auto check_append_fits(size_t base, size_t n) -> void {
        if (n > kIndexCeiling - base) {
            throw TermIndexCeilingReached(
                std::format("OperatorIndex: appending {} terms to a partition holding {} would pass the "
                            "2^32 TermIndex ceiling; raise the partition or rank count to split it further.",
                            n,
                            base));
        }
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: this partition's term count reached the 2^32 TermIndex "
                                          "ceiling; raise the partition or rank count to split it further.");
        }
    }

    // Declared before the array: members are destroyed in reverse declaration order, so the pool
    // outlives the store whose chunks it owns.
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    size_t wide_width_ = kMaxInlinePositions; //!< the structural bound: the second tier's fixed width
    size_t forced_rows_per_chunk_ = 0;        //!< 0 == size the chunks from the first height asked for
    std::unique_ptr<ChunkPool> row_pool_;
    std::unique_ptr<ChunkPool> wide_pool_;

    ChunkedRowArray<PosT> rows_ = {};
    size_t size_ = 0;
    /*! @brief Rows wider than the inline width but no wider than the structural bound, at a fixed
     *  stride of 1 + wide_width_. A few thousand rows per chunk: the tier is a few per cent of the
     *  store by construction, and past kRestridePercent the store re-lays itself and drops it.
     */
    ChunkedRowArray<PosT> wide_ = {};
    static constexpr size_t kWideRowsPerChunk = size_t{1} << 12;
    size_t wide_size_ = 0;
    size_t restrides_ = 0;
    // Lossless side-map for rows over the structural bound: impossible while the cutoff holds, and
    // kept so that a raised one is still stored losslessly until the next restride.
    std::unordered_map<size_t, value_type> overflow_ = {};
    Table table_ = {};
};

} // namespace monoprop::detail
