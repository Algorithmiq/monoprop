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
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/ChunkedArray.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop::detail {

//! Fold words per block: an 8 KB block, L1-resident (bench knee). Every fold walks its word range in
//! blocks of this size starting at word 0, so a block boundary is always a multiple of it.
inline constexpr size_t kColumnBlockWords = 1024;

// Lazy transposed operator storage: one bit-vector per column (bit position), bit r set iff term r touches
// that column. XOR-combining a generator G's columns yields |M ∩ G| mod 2 per term M -- the anticommutation
// bit for an even generator; odd generators add a per-row parity(|M|) correction, so both parities are
// served. Columns are stored in two tiers, bit-identical to all-dense: dense (density ≥
// 1/kPromoteDensityInv) full-height uint64 vectors; sparse an ascending set-row list scatter-expanded at
// scan time. Dense columns are held in pooled chunks (ChunkedArray.h) rather than one vector each: at a
// billion terms a column is 125 MB, and a vector's doubling reserve plus the doubled copy it holds while
// reallocating cost more than the columns' own slack ever did. Promotion is one-way (the operator is
// append-only).
template <size_t NumModes>
struct InvertedIndex {
    static constexpr size_t kNumColumns = Monomial<NumModes>::size();
    static constexpr size_t kPromoteDensityInv = 64;
    // Buckets of the diagnostic density histogram below. Log-spaced, one bucket per halving.
    static constexpr size_t kDensityBuckets = 8;
    // Words per chunk of a dense column: 128 KiB, and exactly eight fold blocks. A column grown a chunk
    // at a time never reallocates, so it holds neither a growth overshoot nor a doubled copy of itself
    // mid-growth -- at 91 full-height columns and a billion terms those two together were 3 GiB and the
    // whole run-to-run spread of the index's footprint.
    static constexpr size_t kWordsPerChunk = size_t{1} << 14;

    /*! @brief One column of the transpose: the rows that touch this bit position, in one of two tiers. */
    struct Column {
        //! Full-height bit-vector, chunked; used iff is_dense. Words past words() are slack, never read.
        ChunkedArray<uint64_t> words;
        // Ascending set-row indices (used iff !is_dense). must stay ascending: combine_columns_block
        // lower_bounds these to a word range, and every fill path appends in row order.
        std::vector<TermIndex> set_rows;
        bool is_dense = false; //!< which of the two tiers holds this column
    };

    /*! @brief An empty index whose dense columns take @a words_per_chunk words per chunk.
     *
     *  @pre A power of two and a multiple of kColumnBlockWords, so no fold block ever straddles a chunk
     *  boundary and dense_column_block() can hand the fold a bare pointer.
     *
     *  The parameter is a test knob: production always takes kWordsPerChunk, and a test drives the chunk
     *  boundaries with a few hundred rows instead of a billion.
     */
    explicit InvertedIndex(size_t words_per_chunk = kWordsPerChunk)
        : pool_(std::make_unique<ChunkPool>(words_per_chunk * sizeof(uint64_t))),
          words_per_chunk_(words_per_chunk) {
        assert(std::has_single_bit(words_per_chunk) && words_per_chunk % kColumnBlockWords == 0);
        for (auto &col : cols) {
            col.words.attach(*pool_, words_per_chunk);
        }
    }

    //! Deep copy: the clone takes its chunks from its own pool, so the two indices share no storage.
    InvertedIndex(const InvertedIndex &other) : InvertedIndex(other.words_per_chunk_) {
        row_count = other.row_count;
        row_parity_ = other.row_parity_;
        for (size_t c = 0; c < kNumColumns; ++c) {
            cols[c].is_dense = other.cols[c].is_dense;
            cols[c].set_rows = other.cols[c].set_rows;
            cols[c].words = other.cols[c].words.clone_into(*pool_);
        }
    }

    // Defaulted: the pool is held by pointer, so it keeps its address and the moved-to columns' bare
    // pointers into it stay good.
    InvertedIndex(InvertedIndex &&) noexcept = default;

    /*! @brief Move assignment, which cannot be defaulted: members are assigned in declaration order, so
     *  a defaulted one would destroy this index's pool while its own columns still held chunks from it.
     */
    auto operator=(InvertedIndex &&other) noexcept -> InvertedIndex & {
        if (this != &other) {
            for (auto &col : cols) {
                col.words.reset(); // hand the chunks back while the pool that owns them is still alive
            }
            pool_ = std::move(other.pool_);
            words_per_chunk_ = other.words_per_chunk_;
            cols = std::move(other.cols);
            row_count = other.row_count;
            row_parity_ = std::move(other.row_parity_);
        }
        return *this;
    }

    auto operator=(const InvertedIndex &other) -> InvertedIndex & {
        if (this != &other) {
            *this = InvertedIndex(other);
        }
        return *this;
    }

    // Declared before `cols`: members are destroyed in reverse declaration order, so the pool outlives
    // the columns whose chunks it owns.
    std::unique_ptr<ChunkPool> pool_; //!< the columns' chunk arenas; one size class, one owner
    size_t words_per_chunk_ = kWordsPerChunk;

    std::array<Column, kNumColumns> cols{};
    size_t row_count = 0;

    // Parity of |M| per row, packed 1 bit/row: bit r = popcount(row r) & 1. Built on first use and only
    // by odd-|G| generators, so even-parity workloads never allocate it.
    mutable std::vector<uint64_t> row_parity_; // empty == not built

    auto row_parity_words() const -> const uint64_t * {
        if (row_parity_.empty() && row_count != 0) {
            build_row_parity_();
        }
        return row_parity_.data();
    }

    // Callers gate on row_count != 0, so the bitmap is never sized to zero words here.
    auto build_row_parity_() const -> void {
        const size_t nwords = (row_count + 63) / 64;
        row_parity_.assign(nwords, 0);
        for (const auto &col : cols) {
            if (col.is_dense) {
                for (size_t w = 0; w < nwords && w < col.words.size(); ++w) {
                    row_parity_[w] ^= col.words[w];
                }
            }
            else {
                for (TermIndex r : col.set_rows) {
                    row_parity_[r >> 6] ^= (uint64_t{1} << (r & 63));
                }
            }
        }
    }

    auto rows() const -> size_t { return row_count; }
    auto words() const -> size_t { return (row_count + 63) / 64; }

    auto column_is_dense(size_t c) const -> bool { return cols[c].is_dense; }
    /*! @brief Dense column @a c over fold words [@a bb, @a be), as a plain pointer to word @a bb.
     *
     *  Valid for be-bb words and no further. A fold block starts at a multiple of kColumnBlockWords and
     *  a chunk holds a whole number of blocks, so the requested range never crosses a chunk boundary --
     *  which is what keeps the fold's memcpy and XOR loops on contiguous memory. Ask per block; there is
     *  no pointer to the whole column any more.
     */
    auto dense_column_block(size_t c, size_t bb, [[maybe_unused]] size_t be) const -> const uint64_t * {
        assert(be >= bb && be - bb <= cols[c].words.elems_left_in_chunk(bb)
               && "a fold block must not straddle a chunk");
        return cols[c].words.contiguous_at(bb);
    }
    auto sparse_column_rows(size_t c) const -> const std::vector<TermIndex> & { return cols[c].set_rows; }

    auto promote_to_dense(size_t c) -> void {
        Column &col = cols[c];
        col.words.reset(); // a promoted column has no dense storage yet; grow it chunk by chunk
        col.words.grow_zeroed(words());
        for (TermIndex r : col.set_rows) {
            col.words[r >> 6] |= uint64_t{1} << (r & 63U);
        }
        col.set_rows.clear();
        col.set_rows.shrink_to_fit();
        col.is_dense = true;
    }

    // Scatter the set bits of new rows [base, base+n) of `op` into the tiered columns: dense bits to the
    // word array, sparse rows appended in row order. row_count must already cover [0, base+n).
    template <typename Rows>
    auto fill_rows(const Rows &op, size_t base, size_t n) -> void {
        if (n == 0) {
            return;
        }
        const size_t new_total_rows = base + n;
        const size_t required_words = (new_total_rows + 63) / 64;
        for (auto &col : cols) {
            if (col.is_dense && col.words.size() < required_words) {
                col.words.grow_zeroed(required_words);
            }
        }
        for (size_t row_idx = base; row_idx < new_total_rows; ++row_idx) {
            const size_t w = row_idx >> 6U;
            const uint64_t row_bit = uint64_t{1} << (row_idx & 63U);
            for_each_row_position<NumModes>(op, row_idx, [this, w, row_bit, row_idx](size_t bit) {
                Column &col = cols[bit];
                if (col.is_dense) {
                    col.words[w] |= row_bit;
                }
                else {
                    col.set_rows.push_back(static_cast<TermIndex>(row_idx));
                }
            });
        }
        for (size_t c = 0; c < kNumColumns; ++c) {
            Column &col = cols[c];
            if (!col.is_dense && col.set_rows.size() * kPromoteDensityInv >= row_count) {
                promote_to_dense(c);
            }
            assert(col.is_dense || std::ranges::is_sorted(col.set_rows));
        }
    }

    template <typename Rows>
    auto rebuild(const Rows &op) -> void {
        const size_t size = op.size();
        for (auto &col : cols) {
            col.words.reset();
            col.set_rows.clear();
            col.set_rows.shrink_to_fit();
            col.is_dense = false;
        }
        row_parity_.clear();
        row_count = size;
        if (size == 0) {
            return;
        }
        const size_t required_words = (size + 63) / 64;

        // Count per-column set bits first and decide tiers from the final density, so the fill never has
        // to promote.
        using Counts = std::array<size_t, kNumColumns>;
        Counts counts{};
        for (size_t row_idx = 0; row_idx < size; ++row_idx) {
            for_each_row_position<NumModes>(op, row_idx, [&counts](size_t bit) { ++counts[bit]; });
        }
        for (size_t c = 0; c < kNumColumns; ++c) {
            const size_t count = counts[c];
            Column &col = cols[c];
            if (count * kPromoteDensityInv >= size) {
                col.is_dense = true;
                col.words.grow_zeroed(required_words);
            }
            else if (count != 0) {
                col.set_rows.reserve(count);
            }
        }

        fill_rows(op, 0, size);
    }

    // Bulk-append the contiguous new terms op[base .. base+n), which must equal rows [row_count ..
    // row_count+n).
    template <typename Rows>
    auto append_rows(const Rows &op, size_t base, size_t n) -> void {
        if (n == 0) {
            return;
        }
        row_count = base + n;
        fill_rows(op, base, n);
        if (!row_parity_.empty()) {
            row_parity_.resize((row_count + 63) / 64, 0);
            for (size_t j = 0; j < n; ++j) {
                const size_t r = base + j;
                if (row_popcount<NumModes>(op, r) & 1U) {
                    row_parity_[r >> 6] |= (uint64_t{1} << (r & 63));
                }
            }
        }
    }

    auto memory_bytes() const -> size_t {
        size_t total = 0;
        for (const auto &col : cols) {
            total += col.words.bytes();
            total += col.set_rows.capacity() * sizeof(TermIndex);
        }
        total += row_parity_.capacity() * sizeof(uint64_t);
        return total;
    }

    // Diagnostic tier split of memory_bytes(): {dense_bytes, sparse_bytes, dense_columns}.
    auto tier_memory_bytes() const -> std::array<size_t, 3> {
        std::array<size_t, 3> out{0, 0, 0};
        for (const auto &col : cols) {
            out[0] += col.words.bytes();
            out[1] += col.set_rows.capacity() * sizeof(TermIndex);
            out[2] += static_cast<size_t>(col.is_dense);
        }
        return out;
    }

    // Diagnostic: how the columns are spread over occupancy, as counts of columns per log-spaced density
    // bucket. Bucket i covers density in [2^-(9-i), 2^-(8-i)) -- bucket 0 everything below 1/512, bucket 7
    // everything from 1/8 up -- so bucket 4 begins exactly at the 1/kPromoteDensityInv promotion threshold
    // and buckets 4..7 are the tiers that pay a full-height bitmap. out[0] counts every column, out[1] the
    // dense tier alone; the sparse tier is their elementwise difference.
    //
    // A dense column no longer keeps its set-row list, so its occupancy is recovered by popcount -- one
    // pass over the dense words, which is why this is a ledger call and not something a hot path may use.
    auto density_histogram() const -> std::array<std::array<size_t, kDensityBuckets>, 2> {
        std::array<std::array<size_t, kDensityBuckets>, 2> out{};
        if (row_count == 0) {
            return out;
        }
        for (const auto &col : cols) {
            size_t set_rows = col.set_rows.size();
            if (col.is_dense) {
                set_rows = 0;
                for (size_t w = 0; w < col.words.size(); ++w) {
                    set_rows += static_cast<size_t>(std::popcount(col.words[w]));
                }
            }
            const size_t bucket = density_bucket_(set_rows, row_count);
            ++out[0][bucket];
            if (col.is_dense) {
                ++out[1][bucket];
            }
        }
        return out;
    }

    // Integer-only, so no column lands on the wrong side of an edge through a rounded division: the density
    // reaches 1/inv exactly when set_rows * inv >= rows.
    static auto density_bucket_(size_t set_rows, size_t rows) -> size_t {
        size_t bucket = 0;
        for (size_t inv = 512; inv >= 8; inv /= 2) {
            if (set_rows * inv < rows) {
                break;
            }
            ++bucket;
        }
        return bucket;
    }
};

// Reusable fold blocks (thread_local: each partition master owns its copy). Two independent scratches
// because the build scan needs the generator fold and a sparse pivot column expanded simultaneously.
inline auto column_block_scratch() -> std::vector<uint64_t> & {
    static thread_local std::vector<uint64_t> blk;
    if (blk.size() < kColumnBlockWords) {
        blk.assign(kColumnBlockWords, 0);
    }
    return blk;
}
inline auto pivot_column_block_scratch() -> std::vector<uint64_t> & {
    static thread_local std::vector<uint64_t> blk;
    if (blk.size() < kColumnBlockWords) {
        blk.assign(kColumnBlockWords, 0);
    }
    return blk;
}

// XOR a generator's inverted-index columns for fold words [bb, be) into blk[0 .. be-bb): dense columns
// XOR their words directly, sparse columns lower_bound to the block's row range. XOR associativity means
// any block decomposition reproduces the full-width fold bit-for-bit.
template <size_t NumModes>
[[gnu::always_inline]] inline auto combine_columns_block(const InvertedIndex<NumModes> &sc,
                                                         std::span<const size_t> cols,
                                                         uint64_t *blk,
                                                         size_t bb,
                                                         size_t be) -> void {
    const size_t nb = be - bb;
    // Seed the scratch from the first dense column (memcpy) when there is one: XOR is commutative, so
    // this is bit-identical to memset + XOR-all while saving one pass over the block.
    size_t dense_init = cols.size();
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        if (sc.column_is_dense(cols[ci])) {
            dense_init = ci;
            break;
        }
    }
    if (dense_init < cols.size()) {
        std::memcpy(blk, sc.dense_column_block(cols[dense_init], bb, be), nb * sizeof(uint64_t));
    }
    else {
        std::memset(blk, 0, nb * sizeof(uint64_t));
    }
    const size_t lo = bb * 64;
    const size_t hi = be * 64;
    const auto below = [](TermIndex row, size_t bound) { return static_cast<size_t>(row) < bound; };
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        if (ci == dense_init) {
            continue;
        }
        const size_t c = cols[ci];
        if (sc.column_is_dense(c)) {
            // Block-local: the column is chunked, so the pointer is good for this block and no further.
            const uint64_t *d = sc.dense_column_block(c, bb, be);
            for (size_t wi = 0; wi < nb; ++wi) {
                blk[wi] ^= d[wi];
            }
        }
        else {
            const auto &rows = sc.sparse_column_rows(c);
            auto it = std::ranges::lower_bound(rows, lo, below);
            const auto en = std::ranges::lower_bound(rows, hi, below);
            for (; it != en; ++it) {
                blk[(*it >> 6) - bb] ^= (uint64_t{1} << (*it & 63U));
            }
        }
    }
}

} // namespace monoprop::detail
