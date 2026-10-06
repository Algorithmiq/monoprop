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
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop::detail {

// Lazy transposed operator storage: one bit-vector per column (bit position), bit r set iff term r touches
// that column. XOR-combining a generator G's columns yields |M ∩ G| mod 2 per term M -- the anticommutation
// bit for an even generator; odd generators add a per-row parity(|M|) correction, so both parities are
// served. Columns are stored in two tiers, bit-identical to all-dense: dense (density ≥
// 1/kPromoteDensityInv) full-height uint64 vectors; sparse an ascending set-row list scatter-expanded at
// scan time. Promotion is one-way (the operator is append-only).
template <size_t NumModes>
struct InvertedIndex {
    static constexpr size_t kNumColumns = Monomial<NumModes>::size();
    static constexpr size_t kPromoteDensityInv = 64;

    struct Column {
        std::vector<uint64_t> words; // full-height bit-vector; used iff is_dense
        // Ascending set-row indices (used iff !is_dense). must stay ascending: combine_columns_block
        // lower_bounds these to a word range, and every fill path appends in row order.
        std::vector<TermIndex> set_rows;
        bool is_dense = false;
    };

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
    auto dense_column_data(size_t c) const -> const uint64_t * { return cols[c].words.data(); }
    auto sparse_column_rows(size_t c) const -> const std::vector<TermIndex> & { return cols[c].set_rows; }

    auto promote_to_dense(size_t c) -> void {
        Column &col = cols[c];
        col.words.assign(words(), 0);
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
                col.words.resize(required_words, 0);
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
            col.words.clear();
            col.words.shrink_to_fit();
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
                col.words.assign(required_words, 0);
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
            total += col.words.capacity() * sizeof(uint64_t);
            total += col.set_rows.capacity() * sizeof(TermIndex);
        }
        total += row_parity_.capacity() * sizeof(uint64_t);
        return total;
    }

    // Diagnostic tier split of memory_bytes(): {dense_bytes, sparse_bytes, dense_columns}.
    auto tier_memory_bytes() const -> std::array<size_t, 3> {
        std::array<size_t, 3> out{0, 0, 0};
        for (const auto &col : cols) {
            out[0] += col.words.capacity() * sizeof(uint64_t);
            out[1] += col.set_rows.capacity() * sizeof(TermIndex);
            out[2] += static_cast<size_t>(col.is_dense);
        }
        return out;
    }
};

inline constexpr size_t kColumnBlockWords = 1024; // 8 KB block ≈ L1-resident (bench knee)

static_assert(kColumnBlockWords % 64 == 0, "the zero-block summary holds one bit per block word");

// Per-thread fold block, overwritten on every use.
inline auto column_block_scratch() -> std::vector<uint64_t> & {
    static thread_local std::vector<uint64_t> blk;
    if (blk.size() < kColumnBlockWords) {
        blk.assign(kColumnBlockWords, 0);
    }
    return blk;
}

// Block that is all-zero between uses, with one summary bit per touched word. Users must re-zero what
// they touch.
struct ZeroBlockScratch {
    static constexpr size_t kSummaryWords = kColumnBlockWords / 64;
    std::vector<uint64_t> words = std::vector<uint64_t>(kColumnBlockWords, 0);
    std::array<uint64_t, kSummaryWords> summary{};
};

// Per-thread zero blocks for the fold and the sparse pivot, which are expanded together.
inline auto fold_zero_scratch() -> ZeroBlockScratch & {
    static thread_local ZeroBlockScratch scratch;
    return scratch;
}
inline auto pivot_zero_scratch() -> ZeroBlockScratch & {
    static thread_local ZeroBlockScratch scratch;
    return scratch;
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
        std::memcpy(blk, sc.dense_column_data(cols[dense_init]) + bb, nb * sizeof(uint64_t));
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
            const uint64_t *d = sc.dense_column_data(c);
            for (size_t wi = bb; wi < be; ++wi) {
                blk[wi - bb] ^= d[wi];
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

// Fold kernel per block; all give identical output. Block and Sparse are for tests.
enum class FoldPath : std::uint8_t {
    Auto,   // posting-driven when eligible and sparse enough
    Block,  // always word-sweeping
    Sparse, // posting-driven whenever eligible
};

// Postings per block word above which Auto sweeps words instead. Affects speed only.
inline constexpr double kSparseFoldPostingsPerWord = 0.5;

// Per-thread posting cursors. Not re-entrant: a visitor must not start another fold.
inline auto fold_cursor_scratch() -> std::vector<size_t> & {
    static thread_local std::vector<size_t> cursor;
    return cursor;
}

// Word-sweeping block fold. `dense_seed` indexes the first dense column, or is cols.size().
template <size_t NumModes, typename WordOp>
[[gnu::always_inline]] inline auto fold_block_by_words(const InvertedIndex<NumModes> &sc,
                                                       std::span<const size_t> cols,
                                                       size_t dense_seed,
                                                       const size_t *begin,
                                                       const size_t *end,
                                                       uint64_t *blk,
                                                       size_t bb,
                                                       size_t be,
                                                       size_t last_word,
                                                       uint64_t last_word_mask,
                                                       const uint64_t *row_parity,
                                                       WordOp &on_word) -> void {
    const size_t nb = be - bb;
    if (dense_seed < cols.size()) {
        std::memcpy(blk, sc.dense_column_data(cols[dense_seed]) + bb, nb * sizeof(uint64_t));
    }
    else {
        std::memset(blk, 0, nb * sizeof(uint64_t));
    }
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        if (ci == dense_seed) {
            continue;
        }
        const size_t c = cols[ci];
        if (sc.column_is_dense(c)) {
            const uint64_t *d = sc.dense_column_data(c);
            for (size_t wi = bb; wi < be; ++wi) {
                blk[wi - bb] ^= d[wi];
            }
        }
        else {
            const TermIndex *rows = sc.sparse_column_rows(c).data();
            for (size_t p = begin[ci]; p < end[ci]; ++p) {
                blk[(rows[p] >> 6) - bb] ^= uint64_t{1} << (rows[p] & 63U);
            }
        }
    }
    for (size_t wi = bb; wi < be; ++wi) {
        uint64_t bits = blk[wi - bb];
        if (row_parity != nullptr) {
            bits ^= row_parity[wi];
        }
        if (wi == last_word) {
            bits &= last_word_mask;
        }
        if (bits != 0) {
            on_word(wi, bits);
        }
    }
}

// Posting-driven block fold (all columns sparse, no parity): visits touched words and re-zeroes them.
template <size_t NumModes, typename WordOp>
[[gnu::always_inline]] inline auto fold_block_by_postings(const InvertedIndex<NumModes> &sc,
                                                          std::span<const size_t> cols,
                                                          const size_t *begin,
                                                          const size_t *end,
                                                          ZeroBlockScratch &zs,
                                                          size_t bb,
                                                          size_t be,
                                                          size_t last_word,
                                                          uint64_t last_word_mask,
                                                          WordOp &on_word) -> void {
    uint64_t *const words = zs.words.data();
    uint64_t *const summary = zs.summary.data();
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const TermIndex *rows = sc.sparse_column_rows(cols[ci]).data();
        for (size_t p = begin[ci]; p < end[ci]; ++p) {
            const size_t wo = (static_cast<size_t>(rows[p]) >> 6) - bb;
            words[wo] ^= uint64_t{1} << (rows[p] & 63U);
            summary[wo >> 6] |= uint64_t{1} << (wo & 63U);
        }
    }
    // Keep the zero invariant if the visitor throws.
    struct Wipe {
        ZeroBlockScratch *zs;
        ~Wipe() {
            if (zs != nullptr) {
                std::ranges::fill(zs->words, 0);
                zs->summary.fill(0);
            }
        }
    } wipe{&zs};
    const size_t summary_words = (be - bb + 63) / 64;
    for (size_t s = 0; s < summary_words; ++s) {
        uint64_t touched = summary[s];
        summary[s] = 0;
        for (; touched != 0; touched &= touched - 1) {
            const size_t wo = s * 64 + static_cast<size_t>(std::countr_zero(touched));
            uint64_t bits = words[wo];
            words[wo] = 0;
            const size_t wi = bb + wo;
            if (wi == last_word) {
                bits &= last_word_mask;
            }
            if (bits != 0) { // postings can cancel
                on_word(wi, bits);
            }
        }
    }
    wipe.zs = nullptr;
}

// Calls on_word(wi, bits) for each nonzero word of the masked fold of `cols` over [wlo, whi), ascending,
// XORing row_parity if non-null, and on_block(bb, be) after each block. Output is the same for every
// `path`.
template <size_t NumModes, typename WordOp, typename BlockOp>
inline auto for_each_fold_word(const InvertedIndex<NumModes> &sc,
                               std::span<const size_t> cols,
                               size_t wlo,
                               size_t whi,
                               size_t last_word,
                               uint64_t last_word_mask,
                               const uint64_t *row_parity,
                               WordOp &&on_word,
                               BlockOp &&on_block,
                               FoldPath path = FoldPath::Auto) -> void {
    if (wlo >= whi) {
        return;
    }
    const size_t ncols = cols.size();
    size_t dense_seed = ncols;
    for (size_t ci = 0; ci < ncols; ++ci) {
        if (sc.column_is_dense(cols[ci])) {
            dense_seed = ci;
            break;
        }
    }
    const bool postings_eligible = path != FoldPath::Block && dense_seed == ncols && row_parity == nullptr;

    // Each block's range starts where the previous one ended.
    std::vector<size_t> &cursor = fold_cursor_scratch();
    cursor.assign(2 * ncols, 0);
    size_t *const begin = cursor.data();
    size_t *const end = cursor.data() + ncols;
    const auto below = [](TermIndex row, size_t bound) { return static_cast<size_t>(row) < bound; };
    for (size_t ci = 0; ci < ncols; ++ci) {
        if (!sc.column_is_dense(cols[ci])) {
            const auto &rows = sc.sparse_column_rows(cols[ci]);
            begin[ci] = static_cast<size_t>(std::lower_bound(rows.begin(), rows.end(), wlo * 64, below) - rows.begin());
        }
    }

    uint64_t *const blk = column_block_scratch().data();
    ZeroBlockScratch &zs = fold_zero_scratch();
    for (size_t bb = wlo; bb < whi; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, whi);
        size_t postings = 0;
        for (size_t ci = 0; ci < ncols; ++ci) {
            if (!sc.column_is_dense(cols[ci])) {
                const auto &rows = sc.sparse_column_rows(cols[ci]);
                const auto first = rows.begin() + static_cast<std::ptrdiff_t>(begin[ci]);
                end[ci] = static_cast<size_t>(std::lower_bound(first, rows.end(), be * 64, below) - rows.begin());
                postings += end[ci] - begin[ci];
            }
        }
        const bool by_postings =
            postings_eligible
            && (path == FoldPath::Sparse
                || static_cast<double>(postings) <= kSparseFoldPostingsPerWord * static_cast<double>(be - bb));
        if (by_postings) {
            fold_block_by_postings<NumModes>(sc, cols, begin, end, zs, bb, be, last_word, last_word_mask, on_word);
        }
        else {
            fold_block_by_words<NumModes>(sc,
                                          cols,
                                          dense_seed,
                                          begin,
                                          end,
                                          blk,
                                          bb,
                                          be,
                                          last_word,
                                          last_word_mask,
                                          row_parity,
                                          on_word);
        }
        on_block(bb, be);
        std::copy(end, end + ncols, begin);
    }
}

} // namespace monoprop::detail
