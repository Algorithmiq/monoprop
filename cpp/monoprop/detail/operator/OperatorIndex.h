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
// For the per-row join key only: routing owns the fingerprint, and a second definition of it here would
// be a second thing to keep in step with BucketJoin's tag.
#include "monoprop/detail/mpi/Routing.h"

namespace monoprop::detail {

class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Operator-term store: entropy-packed position-list rows and nothing else. Row layout: slot 0 = popcount c
// (or kOverflowMarker if c > inline_width_), slots 1..c = ascending set-bit positions; stride_ is fixed
// for the container's life so row offsets stay stable. inline_width_ is a free parameter -- any width is
// correct, over-long rows spill losslessly to overflow. Single-writer: one partition, one thread;
// parallelism is cross-partition.
//
// There is deliberately no key -> row index here. Propagation never looks a term up by value against the
// whole store: a partner M ^ G, if tracked, anticommutes with G and so is inside the gate's own
// anticommuting set, which layer_build/BucketJoin.h joins per gate. The few out-of-gate lookups build a
// transient TermLookup over the rows they need.
//
// What IS kept per row is its 4-byte join key -- the top 32 bits of the mixed routing fingerprint, which
// is exactly BucketJoin's compare tag. Every gate stages its whole anticommuting set into that join, and
// folding the key there meant reading each row's positions once per gate; holding it costs 4 B/term once
// and turns pass 1 into a sequential read of keys_. It is NOT the 64-bit fingerprint: that is
// GF(2)-linear, the mixed key is not, so a partner's key still comes from the partner's own positions.
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = Monomial<NumModes>;
    using key_type = Monomial<NumModes>;
    using mapped_type = size_t;

    using PosT = std::
        conditional_t<(2 * NumModes <= 256), uint8_t, std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;

    static constexpr size_t kDefaultInlinePositions = 11;
    /*! @brief Rows per chunk of the row and key stores, chosen from the height the store is built at.
     *
     *  Both bounds are powers of two and multiples of 64, so every chunk length is: the 64 rows an
     *  inverted-index word names always sit in one chunk and a caller can resolve the chunk once per
     *  word (row_block()). A row never straddles a chunk either, so a span over one row stays
     *  contiguous and RowAccess.h is unaffected.
     */
    static constexpr size_t kMinRowsPerChunk = size_t{1} << 12;
    static constexpr size_t kMaxRowsPerChunk = size_t{1} << 18;

    /*! @brief The chunk length for a store of @a rows rows: a quarter of it, rounded down to a power
     *  of two and clamped to [kMinRowsPerChunk, kMaxRowsPerChunk].
     *
     *  The store overshoots by at most one chunk, so a quarter bounds its slack under a quarter of
     *  itself once it clears four minimum chunks, and under one 4096-row chunk below that. One fixed
     *  2^18-row chunk was 98 B/term at 45 000 terms -- 4.25 MiB of tail on a 0.8 MiB store -- while
     *  rounding down costs only chunk count, which is pooled. It reaches kMaxRowsPerChunk at 1M rows
     *  and stops, so the tail is never more than 2^18 rows however large the operator grows.
     */
    static auto chunk_rows_for_rows(size_t rows) noexcept -> size_t {
        return std::clamp(std::bit_floor(std::max(rows / 4, size_t{1})), kMinRowsPerChunk, kMaxRowsPerChunk);
    }
    // A weight-w Pauli needs 2w positions; 32 covers the common case inline at the supported Pauli
    // cutoffs (2*cutoff <= 32 for cutoff <= 16).
    static constexpr size_t kMaxInlinePositions = 32;
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max();
    //! Header of a row held in the wide tier; its inline bytes carry the tier slot instead of positions.
    static constexpr PosT kWideMarker = static_cast<PosT>(std::numeric_limits<PosT>::max() - 1);
    //! A wide row's slot is a uint32 written over the inline positions, so the narrow row needs 4 of them.
    static constexpr size_t kMinInlineForWideTier = sizeof(uint32_t);
    //! Wide rows above this share of the store and the inline width was guessed too narrow: restride.
    static constexpr size_t kRestridePercent = 3;

    static_assert((2 * NumModes) - 1 <= std::numeric_limits<PosT>::max(),
                  "OperatorIndex PosT too narrow for 2*NumModes positions");
    static_assert(kMaxInlinePositions < std::numeric_limits<PosT>::max() - 1,
                  "the marker sentinels must not collide with a valid popcount");

    // Valid term indices are < kIndexCeiling (check_index_fits throws at the ceiling), so the all-ones
    // TermIndex is free as a sentinel wherever a row index is optional.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    // "Absent" result of a lookup; same value as detail::kMissingIndex (not included here — the operator
    // store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

    //! Row i's join key. Maintained by every writer, so a gate stages its join without reading a row.
    [[nodiscard]] auto key(size_t i) const -> uint32_t { return keys_[i]; }

    /*! @brief The row and key chunks holding the 64 rows starting at @a first_row, resolved once.
     *
     *  Row storage is chunked, so a per-row read costs a chunk lookup on top of the row itself. Every
     *  caller that walks an inverted-index word already has 64 consecutive rows in hand, and a chunk
     *  holds a whole number of those windows, so the lookup is hoisted out of the row loop: the block's
     *  row is `rows + (i & mask) * stride` and its key is `keys[i & mask]`.
     *
     *  @pre first_row is a multiple of 64 and below capacity.
     */
    struct RowBlock {
        const PosT *rows;
        const uint32_t *keys;
        size_t stride;
        size_t mask; //!< row index & mask == the row's offset inside its chunk
    };
    [[nodiscard]] auto row_block(size_t first_row) const -> RowBlock {
        assert(first_row % 64 == 0 && "a row block starts at an inverted-index word boundary");
        return RowBlock{rows_.chunk_base(first_row), keys_.chunk_base(first_row), stride_, rows_.row_mask()};
    }
    //! Row @a i of @a block, which must be the block of a window containing i.
    [[nodiscard]] static auto block_row(const RowBlock &block, size_t i) noexcept -> const PosT * {
        return block.rows + ((i & block.mask) * block.stride);
    }
    //! Row @a i's join key, from @a block rather than from the store's own chunk lookup.
    [[nodiscard]] static auto block_key(const RowBlock &block, size_t i) noexcept -> uint32_t {
        return block.keys[i & block.mask];
    }

    //! The compare tag of a 64-bit fingerprint: what a row's key is, and what BucketJoin tags a query with.
    [[nodiscard]] static auto join_tag(uint64_t fp) noexcept -> uint32_t {
        return static_cast<uint32_t>(routing::mix64(fp) >> 32U);
    }

    //! The join key of a term given as ascending positions, and of a dense one; the same map either way.
    [[nodiscard]] static auto key_of_positions(const PosT *pos, size_t k) noexcept -> uint32_t {
        return join_tag(routing::fingerprint_positions(labels(), pos, k));
    }
    [[nodiscard]] static auto key_of(const value_type &mono) noexcept -> uint32_t {
        return join_tag(routing::linear_hash<2 * NumModes>(mono));
    }

    /*! @brief An empty store. Its chunk length is settled by the first reserve() or growth, from the
     *  height asked for there -- a store is built once and grown to a known size, so that first call
     *  is the best estimate available and the length is fixed for the store's life thereafter.
     *
     *  @param forced_rows_per_chunk 0 to size the chunks from that height (production), or a fixed
     *  length. A power of two and a multiple of 64. The parameter is a test knob: it drives the chunk
     *  boundaries with a few hundred rows instead of a million.
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
     *  The inline saving is paid by every row and the tier only by the rows in it, so a few per cent of
     *  wide rows is the price of a much narrower row for the rest; past kRestridePercent it is not. Only
     *  ever true while a tier exists, so a store restrides at most once.
     */
    [[nodiscard]] auto should_restride() const -> bool {
        return wide_tier_live_() && wide_size_ * 100 > kRestridePercent * size_;
    }

    /*! @brief Widens the structural bound, so rows the new cutoff admits get a tier instead of the map.
     *
     *  A cutoff raised between calls lets through rows the old bound forbade, which would otherwise all
     *  land in the side-map at some eighty bytes each. Draining the current tier first leaves the rows
     *  laid out at the old bound and a fresh tier spanning old bound to new, which is exactly the shape
     *  a store built at the new cutoff would have taken. Rows already in the side-map stay there: they
     *  are stored losslessly, and update_cutoff does not re-truncate existing terms either.
     */
    auto raise_bound(size_t new_wide_width) -> void {
        if (new_wide_width <= wide_width_) {
            return;
        }
        restride_to_bound();
        wide_width_ = std::clamp<size_t>(new_wide_width, inline_width_, kMaxInlinePositions);
    }

    /*! @brief Re-lays every row at the structural bound, emptying the wide tier. Between calls only.
     *
     *  Only the byte layout of the row store changes: row indices, join keys, the overflow map and every
     *  TermIndex the rest of the engine holds are untouched, so no index has to be rebuilt.
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
            // Deep copies into the clone's own pools: the two stores share no chunk.
            out->rows_ = rows_.clone_into(*out->row_pool_);
            out->keys_ = keys_.clone_into(*out->key_pool_);
        }
        out->size_ = size_;
        out->overflow_ = overflow_;
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    // Rows that exceeded inline_width_ and spilled; observable so a test can compare the two insert paths.
    [[nodiscard]] auto overflow_size() const -> size_t { return overflow_.size(); }

    auto reserve(size_t n) -> void { reserve_rows(n); }
    /*! @brief Grows by @a n rows and returns the pre-growth size, the caller's insert base.
     *
     *  Growth is exact now: the store appends whole chunks and never moves a row, so there is no
     *  reallocation to amortise and none of the 1.5× overshoot the name still records -- what used to
     *  be up to half the operator held as spare capacity, plus the old buffer alongside the new one at
     *  the instant of the copy, is now at most one chunk's tail. Throws at the TermIndex ceiling: every
     *  structure indexed by row (inverted index, graph endpoints) is TermIndex-wide.
     *
     *  Freshly grown rows are default-initialized, not zeroed: every one is overwritten by its set()
     *  before any read, so a tail zero-fill would be wasted bandwidth.
     */
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        if (n != 0) {
            check_index_fits(base + n - 1);
        }
        ensure_capacity_(base + n);
        rows_.grow(base + n);
        keys_.grow(base + n);
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so the row header is never pre-read
    // (freshly grown headers are indeterminate); a stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        const size_t c = mono.count();
        keys_[i] = key_of(mono);
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
        keys_[i] = key_of_positions(pos.data(), count);
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

    [[nodiscard]] auto row(size_t i) const -> value_type { return row_from(rows_.at(i), i); }
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
    /*! @brief The stored positions of a row already resolved to its narrow slot (block_row()).
     *  Empty, as row_positions(), for a row that is over the bound and lives in the side-map.
     */
    [[nodiscard]] auto positions_at(const PosT *src) const noexcept -> RowPositions {
        const StoredRow r = stored_(src);
        if (r.pos == nullptr) {
            return {};
        }
        return {std::span<const PosT>(r.pos, r.count)};
    }

    // The rows themselves, both tiers. The join keys are reported separately (row_keys_bytes) so the
    // breakdown can price them on their own; neither figure includes the other.
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = rows_.bytes() + wide_.bytes();
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }
    [[nodiscard]] auto row_keys_bytes() const -> size_t { return keys_.bytes(); }

    // Every row in index order, chunk by chunk so the chunk lookup happens once per chunk and not once
    // per row.
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        // size_ > 0 implies the array is attached, so its settled length is the one to step by.
        const size_t per_chunk = size_ == 0 ? 1 : rows_.rows_per_chunk();
        for (size_t first = 0; first < size_; first += per_chunk) {
            const PosT *const base = rows_.chunk_base(first);
            const size_t last = std::min(first + per_chunk, size_);
            for (size_t i = first; i < last; ++i) {
                fn(row_from(base + ((i - first) * stride_), i), i);
            }
        }
    }

    // Compare row i against an ascending position list without materializing the row (the popcount byte
    // first, so a mismatch usually costs one compare); a spilled row falls back to a dense compare. This
    // is the confirm behind every fingerprint match (BucketJoin::run).
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
    // Diagnostic: the part of memory_bytes() that is unused capacity -- the tail of the last chunk.
    [[nodiscard]] auto slack_bytes() const -> size_t { return rows_.slack_bytes(); }

private:
    /*! @brief A row's stored positions, whichever tier holds them.
     *
     *  One compare carries the common case: a header at or below the inline width is the row's own
     *  popcount and its positions follow it. Both sentinels are above kMaxInlinePositions, so the
     *  branch is a single unsigned compare against the width and never a table lookup or a hash.
     *  `pos == nullptr` means the row is over the structural bound and lives in the side-map.
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

    //! The dense form of a row already resolved to its storage, for the chunk-by-chunk walk.
    [[nodiscard]] auto row_from(const PosT *src_row, size_t i) const -> value_type {
        const StoredRow r = stored_(src_row);
        if (r.pos == nullptr) {
            return overflow_.at(i);
        }
        value_type mono;
        for (size_t j = 0; j < r.count; ++j) {
            mono.set(r.pos[j]);
        }
        return mono;
    }

    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity(); }
    auto reserve_rows(size_t n) -> void {
        ensure_capacity_(n);
        rows_.reserve(n);
        keys_.reserve(n);
    }

    /*! @brief Makes the pools and binds both arrays at @a rows_per_chunk. @pre Not yet attached. */
    auto attach_(size_t rows_per_chunk) -> void {
        assert(!rows_.attached() && "attach_ binds an unbound store");
        row_pool_ = std::make_unique<ChunkPool>(rows_per_chunk * stride_ * sizeof(PosT));
        key_pool_ = std::make_unique<ChunkPool>(rows_per_chunk * sizeof(uint32_t));
        rows_.attach(*row_pool_, rows_per_chunk, stride_);
        keys_.attach(*key_pool_, rows_per_chunk);
    }

    /*! @brief Re-lays the live rows into chunks of @a new_rows_per_chunk and drops the old pools.
     *
     *  Only the chunk geometry moves: row indices, keys and the overflow map are untouched, so every
     *  TermIndex the rest of the engine holds stays valid. @a new_rows_per_chunk is a multiple of the
     *  current length (both are powers of two and it only ever grows), so one old chunk lands whole
     *  inside one new chunk and each move is a single memcpy.
     */
    auto rechunk_(size_t new_rows_per_chunk) -> void {
        const size_t old_rows_per_chunk = rows_.rows_per_chunk();
        assert(new_rows_per_chunk > old_rows_per_chunk && new_rows_per_chunk % old_rows_per_chunk == 0);
        auto new_row_pool = std::make_unique<ChunkPool>(new_rows_per_chunk * stride_ * sizeof(PosT));
        auto new_key_pool = std::make_unique<ChunkPool>(new_rows_per_chunk * sizeof(uint32_t));
        ChunkedRowArray<PosT> new_rows;
        ChunkedArray<uint32_t> new_keys;
        new_rows.attach(*new_row_pool, new_rows_per_chunk, stride_);
        new_keys.attach(*new_key_pool, new_rows_per_chunk);
        new_rows.grow(size_);
        new_keys.grow(size_);
        for (size_t first = 0; first < size_; first += old_rows_per_chunk) {
            const size_t n = std::min(old_rows_per_chunk, size_ - first);
            std::memcpy(new_rows.at(first), rows_.at(first), n * stride_ * sizeof(PosT));
            std::memcpy(&new_keys[first], &keys_[first], n * sizeof(uint32_t));
        }
        // Each array releases its chunks to its own pool before that pool is replaced.
        rows_ = std::move(new_rows);
        keys_ = std::move(new_keys);
        row_pool_ = std::move(new_row_pool);
        key_pool_ = std::move(new_key_pool);
    }

    /*! @brief Keeps the chunk length in step with the height the store is heading for.
     *
     *  The store cannot be told its final height -- the propagator reserves the *initial* operator's
     *  size, which is a handful of terms even for a run that ends at millions -- so the length is
     *  re-derived on every growth instead. It only ever rises and stops at kMaxRowsPerChunk, so a
     *  store crossing 2^20 rows migrates six times and copies about 2M rows in total, once, while a
     *  store that stays small keeps chunks proportional to it. A forced length never moves.
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

    // The label table, bound through a local static so the per-term path does not re-enter routing's guard.
    static auto labels() noexcept -> const uint64_t * {
        static const uint64_t *const table = routing::linear_basis<2 * NumModes>().data();
        return table;
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                                          "-Dmonoprop_WIDE_TERM_INDEX (this partition's term count exceeded ~2^32).");
        }
    }

    // Declared before the arrays: members are destroyed in reverse declaration order, so the pools
    // outlive the stores whose chunks they own. One pool each, because the two size classes differ.
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    size_t wide_width_ = kMaxInlinePositions; //!< the structural bound: the second tier's fixed width
    size_t forced_rows_per_chunk_ = 0;        //!< 0 == size the chunks from the first height asked for
    std::unique_ptr<ChunkPool> row_pool_;
    std::unique_ptr<ChunkPool> key_pool_;
    std::unique_ptr<ChunkPool> wide_pool_;

    ChunkedRowArray<PosT> rows_ = {};
    // One join key per row, parallel to rows_. Default-init like rows_: a grown row's key is
    // indeterminate until its set() writes both.
    ChunkedArray<uint32_t> keys_ = {};
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
};

} // namespace monoprop::detail
