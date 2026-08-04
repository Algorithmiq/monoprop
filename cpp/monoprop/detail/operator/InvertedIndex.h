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
        std::vector<uint64_t> words{}; // full-height bit-vector; used iff is_dense
        // Ascending set-row indices (used iff !is_dense). must stay ascending: combine_columns_block
        // lower_bounds these to a word range, and every fill path appends in row order.
        std::vector<TermIndex> set_rows{};
        bool is_dense = false;
    };

    std::array<Column, kNumColumns> cols{};
    size_t row_count = 0;

    // Parity of |M| per row, packed 1 bit/row: bit r = popcount(row r) & 1. Built on first use and only
    // by odd-|G| generators, so even-parity workloads never allocate it.
    mutable std::vector<uint64_t> row_parity_{}; // empty == not built

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
