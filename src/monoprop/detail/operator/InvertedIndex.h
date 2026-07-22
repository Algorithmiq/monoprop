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
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

/**
 * @brief Lazy transposed operator storage — an inverted index over Majorana columns.
 *
 * Stores the transpose of the row-major operator: one bit-vector per column (mode), bit r set iff
 * term r touches that mode. Its purpose is the anticommutation scan: XOR-combining a generator G's
 * columns yields, per term M, |M ∩ G| mod 2 — the anticommutation bit for an EVEN generator; ODD
 * generators add a per-row parity(|M|) correction, so the structure serves BOTH parities.
 *
 * Columns are chemistry-sparse (~few percent set), so they are stored in two tiers, bit-identical to
 * all-dense: DENSE (density ≥ 1/kPromoteDensityInv) full-height uint64 vectors folded by the hot word
 * loop; SPARSE (below that) an ASCENDING set-row list scatter-expanded at scan time. Promotion is
 * one-way (the operator is append-only).
 */
template <size_t NumModes>
struct InvertedIndex {
    static constexpr size_t kNumColumns = Monomial<NumModes>::size();
    // Promote a column to DENSE at set density ≥ 1/kPromoteDensityInv. The threshold is the FOLD
    // crossover (1/64), not the storage one (1/32): chemistry-density columns (~2-3%) then store dense
    // and fold in the parallel scan pass. Tiering is storage only — bit-identical to storing all dense.
    static constexpr size_t kPromoteDensityInv = 64;

    struct Column {
        std::vector<uint64_t> words{}; // full-height bit-vector; used iff is_dense
        // Ascending set-row indices (used iff !is_dense); mutable so ensure_sorted_columns() can
        // canonicalize order through a const inverted index.
        mutable std::vector<TermIndex> set_rows{};
        bool is_dense = false;
    };

    std::array<Column, kNumColumns> cols{};
    size_t row_count = 0;

    // Lazily-built parity of |M| per row, packed 1 bit/row. Empty until the first odd-parity generator
    // requests it (even-parity workloads never allocate it); mutable because it is a lazy derived cache.
    mutable std::vector<uint64_t> row_parity_{}; // empty == not built

    // Build the parity bitmap from the dense/sparse columns (called once, lazily). parity(|M|) is the
    // XOR over all mode columns of row M (equivalently popcount(row) & 1).
    auto ensure_row_parity() const -> void {
        if (!row_parity_.empty() || row_count == 0)
            return;
        const size_t words = (row_count + 63) / 64;
        row_parity_.assign(words, 0);
        for (const auto &col : cols) {
            if (col.is_dense) {
                for (size_t w = 0; w < words && w < col.words.size(); ++w)
                    row_parity_[w] ^= col.words[w];
            }
            else {
                for (TermIndex r : col.set_rows)
                    row_parity_[r >> 6] ^= (uint64_t{1} << (r & 63));
            }
        }
    }
    // Base pointer of the row_parity bitmap (for the hot fold loop). Valid only after ensure_row_parity().
    auto row_parity_word_ptr() const -> const uint64_t * { return row_parity_.data(); }

    auto rows() const -> size_t { return row_count; }
    auto words() const -> size_t { return (row_count + 63) / 64; }

    auto column_is_dense(size_t c) const -> bool { return cols[c].is_dense; }
    auto dense_column_data(size_t c) const -> const uint64_t * { return cols[c].words.data(); }
    auto sparse_column_rows(size_t c) const -> const std::vector<TermIndex> & { return cols[c].set_rows; }

    // The fold-recompute eval path lower_bounds each sparse column to a word range's row prefix, so it
    // needs ascending sparse rows. Every fill path already appends ascending, so this is normally an
    // O(n) verify — kept self-healing so a future fill change cannot feed the recompute unsorted rows.
    auto ensure_sorted_columns() const -> void {
        for (auto &col : cols) {
            if (!col.is_dense && !std::ranges::is_sorted(col.set_rows)) {
                std::sort(col.set_rows.begin(), col.set_rows.end());
            }
        }
    }

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

    // Scatter the set bits of new rows [base, base+n) of `op` into the tiered columns: dense bits to
    // the word array, sparse rows appended ascending (see ensure_sorted_columns).
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
        fill_rows_range_serial_(op, base, new_total_rows);
        for (size_t c = 0; c < kNumColumns; ++c) {
            Column &col = cols[c];
            if (!col.is_dense && col.set_rows.size() * kPromoteDensityInv >= row_count) {
                promote_to_dense(c);
            }
        }
    }

    // The fill kernel over absolute rows [lo, hi).
    template <typename Rows>
    auto fill_rows_range_serial_(const Rows &op, size_t lo, size_t hi) -> void {
        for (size_t row_idx = lo; row_idx < hi; ++row_idx) {
            const size_t w = row_idx >> 6U;
            const uint64_t row_bit = uint64_t{1} << (row_idx & 63U);
            for_each_row_position<NumModes>(op, row_idx, [this, &w, &row_bit, &row_idx](size_t bit) {
                Column &col = cols[bit];
                if (col.is_dense) {
                    col.words[w] |= row_bit;
                }
                else {
                    col.set_rows.push_back(static_cast<TermIndex>(row_idx));
                }
            });
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

        // Pass 1: per-column set-bit counts → decide tiers from the FINAL density, so the fill never has
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

        // Pass 2: scatter the bits into the (now tier-decided) columns.
        fill_rows(op, 0, size);
    }

    auto append_row(const Monomial<NumModes> &maj) -> void {
        const size_t row_idx = row_count;
        ++row_count;
        const size_t required_words = (row_count + 63) / 64;
        // Crossing into a new 64-row word: every dense column needs the new (zero) word so the fold's
        // [0, words()) range stays in bounds, even columns with no bit in this row.
        if (row_idx % 64 == 0) {
            for (auto &col : cols) {
                if (col.is_dense) {
                    col.words.resize(required_words, 0);
                }
            }
        }
        const uint64_t row_bit = uint64_t{1} << (row_idx % 64);
        const size_t w = row_idx / 64;
        for (size_t bit = maj.find_first(); bit < maj.size(); bit = maj.find_next(bit)) {
            Column &col = cols[bit];
            if (col.is_dense) {
                col.words[w] |= row_bit;
            }
            else {
                col.set_rows.push_back(static_cast<TermIndex>(row_idx));
                if (col.set_rows.size() * kPromoteDensityInv >= row_count) {
                    promote_to_dense(bit);
                }
            }
        }
        if (!row_parity_.empty()) {
            const size_t r = row_count - 1;
            if ((r >> 6) >= row_parity_.size())
                row_parity_.push_back(0);
            if (maj.count() & 1u)
                row_parity_[r >> 6] |= (uint64_t{1} << (r & 63));
        }
    }

    // Atomics-free bulk append of the contiguous new terms op[base .. base+n) (which must equal
    // rows [row_count .. row_count+n)). See fill_rows for the parallelization scheme.
    template <typename Rows>
    auto append_rows_from_op_disjoint(const Rows &op, size_t base, size_t n) -> void {
        if (n == 0) {
            return;
        }
        row_count = base + n;
        fill_rows(op, base, n);
        if (!row_parity_.empty()) {
            row_parity_.resize((row_count + 63) / 64, 0);
            for (size_t j = 0; j < n; ++j) {
                const size_t r = base + j;
                if (row_popcount<NumModes>(op, r) & 1u) {
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
};

// XOR a generator's inverted-index columns for fold words [bb, be) into blk[0 .. be-bb): dense columns
// XOR their words directly, sparse columns lower_bound to the block's row range. XOR associativity
// means any block decomposition reproduces the full-width fold bit-for-bit. THE fold-combine kernel,
// shared by the build scan, the replay cache (make_fold_cache) and the replay recompute.
inline constexpr size_t kColumnBlockWords = 1024; // 8 KB block ≈ L1-resident (bench knee)

// Per-thread reusable fold blocks (thread_local: each worker owns its copy). Two independent scratches
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
            auto it = std::lower_bound(rows.begin(), rows.end(), lo, below);
            const auto en = std::lower_bound(rows.begin(), rows.end(), hi, below);
            for (; it != en; ++it) {
                blk[(*it >> 6) - bb] ^= (uint64_t{1} << (*it & 63U));
            }
        }
    }
}

} // namespace monoprop::detail
