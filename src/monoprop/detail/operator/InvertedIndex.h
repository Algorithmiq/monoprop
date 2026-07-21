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
 * The main operator is stored row-major as K Majorana terms over 2N columns. This inverted index
 * stores the transpose: one bit-vector per Majorana column (mode), with bit r set iff term r
 * contains that mode — i.e. column c's postings are the set of terms touching mode c.
 *
 * Its purpose is the anticommutation scan: XOR-combining a generator G's selected columns yields,
 * per term M, the parity |M ∩ G| mod 2, which for an EVEN generator is exactly the anticommutation
 * bit because (|M| |G| − |M ∩ G|) mod 2 = |M ∩ G| mod 2 when |G| is even. ODD generators run the
 * same kernel plus a per-row parity(|M|) correction (ensure_row_parity / the g_odd path), so this
 * structure serves BOTH parities — see combine_columns_block.
 *
 * Each term is a k ≤ cutoff subset of 2N modes (k≈6 usually for chemistry), so every
 * column is sparse along the row axis (set fraction = per-mode frequency, a few
 * percent). Storing all columns as full-height bit-vectors is ~97% zeros. To cut
 * that without slowing the hot word-combine, columns are stored in two tiers:
 *
 *   - DENSE  (set density ≥ 1/kPromoteDensityInv): full-height uint64 bit-vector,
 *     XOR-combined by the register-accumulating word loop (even_parity_scan_pass1).
 *     This is where the bits AND the combine cost live, so it is byte-identical to
 *     the old structure — no scan regression.
 *   - SPARSE (below that, incl. empty): an ASCENDING set-row list (~4 B/set-bit),
 *     scatter-expanded into a dense scratch column at scan time. Every fill path
 *     appends rows in ascending order (the parallel fill by construction — see
 *     fill_rows), which the block-restricted cos recompute relies on.
 *
 * A column promotes SPARSE→DENSE (one-way; the operator is append-only) once its
 * density crosses 1/kPromoteDensityInv, the point where the row-list would cost
 * more than the dense bit-vector.
 */
template <size_t NumModes>
struct InvertedIndex {
    static constexpr size_t kNumColumns = Monomial<NumModes>::size();
    // Promote a column to DENSE at set density ≥ 1/kPromoteDensityInv. Two crossovers matter:
    //   - STORAGE (1/32): a uint32 row-list (4 B/set-bit) and a dense bit-vector (rows/8 B) cost the same.
    //   - FOLD (1/64): below it a sparse column is scatter-expanded per block in combine_columns_block
    //     (lower_bound + O(set-bits-in-block)); above it a dense column streams through the XOR loop.
    // The threshold is the FOLD crossover, not the storage one: chemistry-density columns (popcount≈6
    // over 2N≈256 ⇒ ~2–3% density) then store dense and fold in the already-parallel scan pass. Tiering
    // is storage only — the result is bit-identical to storing every column dense.
    static constexpr size_t kPromoteDensityInv = 64;

    struct Column {
        std::vector<uint64_t> words{}; // full-height bit-vector; used iff is_dense
        // Set-row indices (ascending; used iff !is_dense). mutable so the logically-const lazy
        // normalization ensure_sorted_columns() can canonicalize order through a const inverted index.
        mutable std::vector<TermIndex> set_rows{};
        bool is_dense = false;
    };

    std::array<Column, kNumColumns> cols{};
    size_t row_count = 0;

    // Lazily-built parity of |M| per row, packed 1 bit/row (word w, bit b == parity of row 64*w+b).
    // Empty until the first odd-parity generator requests it; even-parity workloads never allocate it.
    // mutable: a derived cache over the unchanged rows, populated lazily through a const inverted index.
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

    // ── Column fold accessors ─────────────────────────────────────────────────
    auto column_is_dense(size_t c) const -> bool { return cols[c].is_dense; }
    auto dense_column_data(size_t c) const -> const uint64_t * { return cols[c].words.data(); }
    auto sparse_column_rows(size_t c) const -> const std::vector<TermIndex> & { return cols[c].set_rows; }

    // The fold-RECOMPUTE eval path (CosineRecompute.h, used above the fold-cache memory budget) walks the
    // fold by disjoint WORD ranges across threads and lower_bounds each sparse column to a range's row
    // prefix, so it needs the sparse rows in ascending order. Every fill path appends rows ascending —
    // the serial kernel trivially, the row-block-parallel fill by chunk-order counting-sort layout — so
    // this is normally an O(n) verify that finds them sorted and does nothing. Kept explicit and
    // self-healing so a future fill change cannot silently feed the recompute unsorted rows. Idempotent.
    auto ensure_sorted_columns() const -> void {
        for (auto &col : cols) {
            if (!col.is_dense && !std::is_sorted(col.set_rows.begin(), col.set_rows.end())) {
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

    // Scatter the set bits of new rows [base, base+n) of `op` into the tiered columns: dense bits go
    // straight to the word array, sparse rows append ascending (the fold-recompute path depends on
    // ascending sparse rows — see ensure_sorted_columns).
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

    // The fill kernel over absolute rows [lo, hi): dense bits go straight to the word array,
    // sparse rows append ascending.
    template <typename Rows>
    auto fill_rows_range_serial_(const Rows &op, size_t lo, size_t hi) -> void {
        for (size_t row_idx = lo; row_idx < hi; ++row_idx) {
            const size_t w = row_idx >> 6U;
            const uint64_t row_bit = uint64_t{1} << (row_idx & 63U);
            for_each_row_position<NumModes>(op, row_idx, [&](size_t bit) {
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
        Counts counts{}; // value-initialized → all zeros
        for (size_t row_idx = 0; row_idx < size; ++row_idx) {
            for_each_row_position<NumModes>(op, row_idx, [&](size_t bit) { ++counts[bit]; });
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
        // [0, words()) range stays in bounds, even columns that get no bit in this row.
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

// ─── Block-restricted generator-column fold ────────────────────────────────────
// XOR a generator's inverted index columns for fold words [bb, be) into blk[0 .. be-bb): dense columns
// XOR their words directly; sparse (ascending) columns lower_bound to the block's row range and
// scatter only those rows. XOR is associative/commutative, so any block decomposition reproduces
// the full-width per-word fold bit-for-bit. This is THE fold-combine implementation: the build
// scan (even_parity_scan_pass1), the replay cache (make_fold_cache) and the replay recompute
// (*_cos_fold_recompute) all run it over their own block ranges.
inline constexpr size_t kColumnBlockWords = 1024; // 8 KB block ≈ L1-resident (bench knee)

// Per-thread reusable fold blocks (sized once to kColumnBlockWords). thread_local: parallel workers
// each fill and consume their own copy. Two independent scratches because the build scan needs
// the generator fold and a sparse pivot column expanded simultaneously.
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
    // Initialize the scratch from the first dense column (memcpy) when there is one — XOR is
    // commutative, so seeding with any one column and folding the rest is bit-identical to
    // memset + XOR-all while saving one full pass over the block.
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
