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
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// ---- LEB128, the posting codec's variable-length integer -----------------------------------------
// Gaps between ASCENDING postings are ≥ 1 and overwhelmingly small, so 7 payload bits + a
// continuation bit encodes almost all of them in one byte. The three functions must agree exactly:
// leb128_size is what the tier decision and the reserve are computed from, before a byte is written.

/// Bytes @ref leb128_encode will emit for @p value.
constexpr auto leb128_size(size_t value) -> size_t {
    size_t bytes = 0;
    do {
        ++bytes;
        value >>= 7;
    } while (value != 0);
    return bytes;
}

/// Append @p value to @p out, low 7 bits first, high bit set on every non-final byte.
inline auto leb128_encode(std::vector<uint8_t> &out, size_t value) -> void {
    while (value >= 0x80U) {
        out.push_back(static_cast<uint8_t>(value) | 0x80U);
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value));
}

/// Decode one value, advancing @p cursor past it.
[[gnu::always_inline]] inline auto leb128_decode(const uint8_t *&cursor) -> size_t {
    size_t value = 0;
    unsigned shift = 0;
    uint8_t byte = 0;
    do {
        byte = *cursor++;
        value |= static_cast<size_t>(byte & 0x7FU) << shift;
        shift += 7;
    } while ((byte & 0x80U) != 0);
    return value;
}

/**
 * @brief Lazy transposed operator storage — an inverted index over Majorana columns.
 *
 * Stores the transpose of the row-major operator: one bit-vector per column (mode), bit r set iff
 * term r touches that mode. Its purpose is the anticommutation scan: XOR-combining a generator G's
 * columns yields, per term M, |M ∩ G| mod 2 — the anticommutation bit for an EVEN generator; ODD
 * generators add a per-row parity(|M|) correction, so the structure serves BOTH parities.
 *
 * Columns are sparse (~few percent set), so they are stored in two tiers, bit-identical to
 * all-dense: DENSE full-height uint64 vectors folded by the hot word loop; SPARSE an ASCENDING,
 * delta+LEB128 coded posting list scatter-expanded at scan time. Promotion is one-way (the operator
 * is append-only).
 *
 * The tier is chosen by an ACTUAL BYTE COMPARISON, not a density proxy: a column stays sparse only
 * while its encoded postings are smaller than its full-height bitmap would be. Density is a poor
 * proxy for that — measured on the two bench models, blanket delta-coding every column saves 55% on
 * hubbard but only 5% on pauli, while the per-column cheaper-of-the-two choice saves 58% and 48%
 * respectively. Which columns win differs by model, so only the comparison itself gets it right.
 */
template <size_t NumModes>
struct InvertedIndex {
    static constexpr size_t kNumColumns = Monomial<NumModes>::size();

    // Postings per skip block. Bounds the forward decode of a range query to kPostingsPerBlock-1
    // gaps while costing one header per block; 128 keeps the skip table ~1% of the posting bytes.
    static constexpr size_t kPostingsPerBlock = 128;

    // Bias of the tier comparison TOWARD the bitmap. Dense columns fold with the hot word loop while
    // postings must be decoded, so the byte-optimal tier is not necessarily the time-optimal one.
    // 1.0 = pure byte-optimal; >1 pushes columns dense sooner, buying fold speed with memory. THE
    // time/memory knob of this structure — the tradeoff is a constant, not a rewrite.
    static constexpr double kDenseBias = 1.0;

    /// A posting block's header: its first row stored ABSOLUTELY (so the block decodes without its
    /// predecessors, which is what makes the range query O(log blocks)) and where its gaps start.
    struct SkipEntry {
        TermIndex first_row;
        uint32_t byte_offset;
    };
    static constexpr size_t kBlockHeaderBytes = sizeof(SkipEntry);

    struct Column {
        std::vector<uint64_t> words{}; // full-height bit-vector; used iff is_dense
        // Delta+LEB128 coded postings (used iff !is_dense): one `skips` header per block of
        // kPostingsPerBlock postings, and in `gaps` the LEB128 gap from the previous posting for
        // every non-block-leading one. Rows MUST stay strictly ascending — every gap is then ≥ 1 and
        // the stream is uniquely decodable, and combine_columns_block's range query needs the order.
        // append_sparse_row is the single insertion point and asserts it.
        std::vector<uint8_t> gaps{};
        std::vector<SkipEntry> skips{};
        TermIndex last_row = 0; // last posting appended; the incremental encoder's gap base
        size_t count = 0;       // postings encoded (0 once promoted to dense)
        bool is_dense = false;
    };

    std::array<Column, kNumColumns> cols{};
    size_t row_count = 0;

    // Lazily-built parity of |M| per row, packed 1 bit/row. Empty until the first odd-parity generator
    // requests it (even-parity workloads never allocate it); mutable because it is a lazy derived cache.
    mutable std::vector<uint64_t> row_parity_{}; // empty == not built

    // Base pointer of the per-row parity bitmap, built once on first use: bit r = popcount(row r) & 1
    // (the XOR over all mode columns of row r). Only odd-|G| generators call it; even-parity workloads
    // never allocate it. Const because the bitmap is a lazy derived cache.
    auto row_parity_words() const -> const uint64_t * {
        if (row_parity_.empty() && row_count != 0) {
            const size_t nwords = (row_count + 63) / 64;
            row_parity_.assign(nwords, 0);
            for (size_t c = 0; c < kNumColumns; ++c) {
                const Column &col = cols[c];
                if (col.is_dense) {
                    for (size_t w = 0; w < nwords && w < col.words.size(); ++w)
                        row_parity_[w] ^= col.words[w];
                }
                else {
                    for_each_sparse_row(c, [this](TermIndex r) { row_parity_[r >> 6] ^= (uint64_t{1} << (r & 63)); });
                }
            }
        }
        return row_parity_.data();
    }

    auto rows() const -> size_t { return row_count; }
    auto words() const -> size_t { return (row_count + 63) / 64; }

    auto column_is_dense(size_t c) const -> bool { return cols[c].is_dense; }
    auto dense_column_data(size_t c) const -> const uint64_t * { return cols[c].words.data(); }

    // ---- SPARSE-tier access ---------------------------------------------------------------------
    // The sparse tier is reached ONLY through these three entry points, never through its container:
    // the fold kernel, the build scan and the tests all go via them, so the posting representation is
    // an implementation detail of this struct.

    /// Number of postings (set rows) held in column @p c's SPARSE tier; 0 for a dense column.
    auto sparse_column_count(size_t c) const -> size_t { return cols[c].count; }

    /// Bytes column @p c's SPARSE tier actually occupies, container slack included. THE sizing hook:
    /// memory_bytes, tier_memory_bytes and the diagnostics all route through it, so the accounting
    /// cannot drift from the representation.
    auto sparse_column_bytes(size_t c) const -> size_t {
        return cols[c].gaps.capacity() + cols[c].skips.capacity() * sizeof(SkipEntry);
    }

    /// Encoded size of column @p c's postings, EXACT (no container slack). This is the quantity the
    /// tier rule compares against the bitmap, and what @ref rebuild predicts before allocating.
    auto sparse_encoded_bytes(size_t c) const -> size_t {
        return cols[c].skips.size() * kBlockHeaderBytes + cols[c].gaps.size();
    }

    /// Bytes a full-height bitmap column costs at the current row count.
    auto dense_bytes() const -> size_t { return words() * sizeof(uint64_t); }

    /// THE tier rule. A column is worth keeping as postings only while they encode strictly smaller
    /// than the bitmap (scaled by @ref kDenseBias). Self-tuning per column and per model, and
    /// monotone in the row count, so applying it after every append keeps promotion one-way.
    auto sparse_beats_dense(size_t encoded_bytes) const -> bool {
        return static_cast<double>(encoded_bytes) * kDenseBias < static_cast<double>(dense_bytes());
    }

    /// Invoke @p fn(TermIndex) for each sparse posting of column @p c with lo <= row < hi, ASCENDING.
    /// The range query is what makes the blocked fold possible: seek to the block that can hold @p lo
    /// in O(log blocks), then forward-decode. Callers walk disjoint ranges, and XOR is associative, so
    /// the per-block results recombine bit-identically with a full-width walk.
    template <typename Fn>
    auto for_each_sparse_row_in_range(size_t c, size_t lo, size_t hi, Fn &&fn) const -> void {
        const Column &col = cols[c];
        if (col.count == 0 || lo >= hi) {
            return;
        }
        // First block whose leader is ≥ lo; unless it starts exactly at lo, the first row in range
        // lives inside its predecessor, so step back one.
        const auto after =
            std::lower_bound(col.skips.begin(), col.skips.end(), lo, [](const SkipEntry &e, size_t bound) {
                return static_cast<size_t>(e.first_row) < bound;
            });
        size_t block = static_cast<size_t>(after - col.skips.begin());
        if (block != 0 && (block == col.skips.size() || static_cast<size_t>(col.skips[block].first_row) > lo)) {
            --block;
        }
        size_t posting = block * kPostingsPerBlock;
        size_t row = col.skips[block].first_row;
        const uint8_t *cursor = col.gaps.data() + col.skips[block].byte_offset;
        for (;;) {
            if (row >= hi) {
                return;
            }
            if (row >= lo) {
                fn(static_cast<TermIndex>(row));
            }
            if (++posting >= col.count) {
                return;
            }
            if (posting % kPostingsPerBlock == 0) {
                const SkipEntry &e = col.skips[posting / kPostingsPerBlock];
                row = e.first_row;
                cursor = col.gaps.data() + e.byte_offset;
            }
            else {
                row += leb128_decode(cursor);
            }
        }
    }

    /// Invoke @p fn(TermIndex) for every sparse posting of column @p c, ASCENDING. The whole-column
    /// walk needs no seek, so it decodes straight through rather than going via the range query.
    template <typename Fn>
    auto for_each_sparse_row(size_t c, Fn &&fn) const -> void {
        const Column &col = cols[c];
        if (col.count == 0) {
            return;
        }
        size_t row = col.skips[0].first_row;
        const uint8_t *cursor = col.gaps.data();
        for (size_t posting = 0;;) {
            fn(static_cast<TermIndex>(row));
            if (++posting >= col.count) {
                return;
            }
            if (posting % kPostingsPerBlock == 0) {
                const SkipEntry &e = col.skips[posting / kPostingsPerBlock];
                row = e.first_row;
                cursor = col.gaps.data() + e.byte_offset;
            }
            else {
                row += leb128_decode(cursor);
            }
        }
    }

    /// Append @p row to column @p col's posting stream. THE single insertion point, so the strictly
    /// ascending invariant the range query rests on is enforced in exactly one place.
    static auto append_sparse_row(Column &col, TermIndex row) -> void {
        // Strictly ascending is what combine_columns_block's range query rests on, and what makes every
        // gap ≥ 1; every fill path appends in row order. Checked for block leaders too, since only the
        // gap-coded postings would otherwise be covered.
        assert(col.count == 0 || row > col.last_row);
        if (col.count % kPostingsPerBlock == 0) {
            // Block leader: absolute row in the header, no gap byte — matching what the tier rule and
            // rebuild's pass 1 predict. Offsets index `gaps`, which the tier rule holds below
            // dense_bytes() = row_count/8, so uint32 suffices wherever row indices fit TermIndex.
            assert(col.gaps.size() <= std::numeric_limits<uint32_t>::max());
            col.skips.push_back(SkipEntry{row, static_cast<uint32_t>(col.gaps.size())});
        }
        else {
            leb128_encode(col.gaps, static_cast<size_t>(row) - static_cast<size_t>(col.last_row));
        }
        col.last_row = row;
        ++col.count;
    }

    /// Diagnostic/testing: column @p c's set rows as an ascending vector, whichever tier holds it.
    /// Allocates — never call it on a hot path.
    auto column_rows(size_t c) const -> std::vector<TermIndex> {
        std::vector<TermIndex> rows;
        const Column &col = cols[c];
        if (col.is_dense) {
            for (size_t w = 0; w < col.words.size(); ++w) {
                for (uint64_t x = col.words[w]; x != 0; x &= x - 1) {
                    rows.push_back(static_cast<TermIndex>(w * 64 + static_cast<size_t>(std::countr_zero(x))));
                }
            }
        }
        else {
            rows.reserve(sparse_column_count(c));
            for_each_sparse_row(c, [&rows](TermIndex r) { rows.push_back(r); });
        }
        return rows;
    }

    /// Expand column @p c's postings into a full-height bitmap and release the encoded form. Lossless
    /// by construction: it replays the same decode the fold uses.
    auto promote_to_dense(size_t c) -> void {
        Column &col = cols[c];
        std::vector<uint64_t> dense(words(), 0);
        for_each_sparse_row(c, [&dense](TermIndex r) { dense[r >> 6] |= uint64_t{1} << (r & 63U); });
        col.words = std::move(dense);
        col.gaps.clear();
        col.gaps.shrink_to_fit();
        col.skips.clear();
        col.skips.shrink_to_fit();
        col.last_row = 0;
        col.count = 0; // the sparse walkers key off count; a dense column holds no postings
        col.is_dense = true;
    }

    // Scatter the set bits of new rows [base, base+n) of `op` into the tiered columns: dense bits to the
    // word array, sparse rows encoded in row order. Sparse columns whose postings no longer encode
    // smaller than a bitmap are promoted afterwards. row_count must already cover [0, base+n).
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
                    append_sparse_row(col, static_cast<TermIndex>(row_idx));
                }
            });
        }
        // Re-apply the tier rule now that both the postings and dense_bytes() have grown. The postings
        // are appended in row order, so append_sparse_row's assert already guards the ascending
        // invariant the fold's range query depends on.
        for (size_t c = 0; c < kNumColumns; ++c) {
            const Column &col = cols[c];
            if (!col.is_dense && col.count != 0 && !sparse_beats_dense(sparse_encoded_bytes(c))) {
                promote_to_dense(c);
            }
        }
    }

    template <typename Rows>
    auto rebuild(const Rows &op) -> void {
        const size_t size = op.size();
        for (auto &col : cols) {
            col = Column{};
        }
        row_parity_.clear();
        row_count = size;
        if (size == 0) {
            return;
        }
        const size_t required_words = (size + 63) / 64;

        // Pass 1: per column, count the postings AND accumulate the EXACT gap-stream byte count the
        // encoder would emit, so the tier is decided from the true encoded size at the FINAL row count
        // — before a byte is allocated, and with no promotion left for the fill to do.
        using Counts = std::array<size_t, kNumColumns>;
        Counts counts{};
        Counts gap_bytes{};
        std::array<TermIndex, kNumColumns> last_row{};
        for (size_t row_idx = 0; row_idx < size; ++row_idx) {
            for_each_row_position<NumModes>(op, row_idx, [&counts, &gap_bytes, &last_row, row_idx](size_t bit) {
                if (counts[bit] % kPostingsPerBlock != 0) { // block leaders are absolute, so no gap
                    gap_bytes[bit] += leb128_size(row_idx - static_cast<size_t>(last_row[bit]));
                }
                last_row[bit] = static_cast<TermIndex>(row_idx);
                ++counts[bit];
            });
        }
        for (size_t c = 0; c < kNumColumns; ++c) {
            const size_t count = counts[c];
            if (count == 0) {
                continue;
            }
            Column &col = cols[c];
            const size_t blocks = (count + kPostingsPerBlock - 1) / kPostingsPerBlock;
            if (!sparse_beats_dense(blocks * kBlockHeaderBytes + gap_bytes[c])) {
                col.is_dense = true;
                col.words.assign(required_words, 0);
            }
            else {
                // Exact reserve: pass 1 predicted what the encoder emits, so the streams never
                // reallocate and carry no geometric-growth slack over the encoded size.
                col.gaps.reserve(gap_bytes[c]);
                col.skips.reserve(blocks);
            }
        }

        // Pass 2: scatter the bits into the (now tier-decided) columns.
        fill_rows(op, 0, size);
    }

    // Bulk-append the contiguous new terms op[base .. base+n) (which must equal rows [row_count ..
    // row_count+n)), extending the lazy parity bitmap if it was already built.
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
                if (row_popcount<NumModes>(op, r) & 1u) {
                    row_parity_[r >> 6] |= (uint64_t{1} << (r & 63));
                }
            }
        }
    }

    auto memory_bytes() const -> size_t {
        size_t total = 0;
        for (size_t c = 0; c < kNumColumns; ++c) {
            total += cols[c].words.capacity() * sizeof(uint64_t);
            total += sparse_column_bytes(c);
        }
        total += row_parity_.capacity() * sizeof(uint64_t);
        return total;
    }

    /// Diagnostic: what every column would cost delta+LEB128 coded, what an oracle picking the cheaper
    /// of {full bitmap, coded postings} per column would cost, and how many columns the coded form
    /// wins. Returns {delta_bytes_all_columns, oracle_bytes, columns_delta_wins}.
    ///
    /// It PREDATES the codec — it was the sizing study that chose the per-column byte comparison over
    /// a density threshold. Now that the comparison is the tier rule, it is a VERIFICATION of it:
    /// oracle_bytes must equal the realized tier bytes and columns_delta_wins must be 0, since no
    /// column can be sparse unless coding it already won. Retire it once that is confirmed on the
    /// bench models (it also costs an O(rows) expansion per column).
    auto delta_coded_bytes() const -> std::array<size_t, 3> {
        std::array<size_t, 3> out{0, 0, 0};
        for (size_t c = 0; c < kNumColumns; ++c) {
            const Column &col = cols[c];
            const std::vector<TermIndex> rows = column_rows(c);
            size_t delta = 0;
            size_t prev = 0;
            for (size_t i = 0; i < rows.size(); ++i) {
                if (i % kPostingsPerBlock == 0) {
                    delta += kBlockHeaderBytes; // block leader is absolute, so no gap byte
                }
                else {
                    // Gaps are ≥ 1 because the postings are strictly ascending.
                    delta += leb128_size(static_cast<size_t>(rows[i]) - prev);
                }
                prev = rows[i];
            }
            const size_t here = col.is_dense ? dense_bytes() : sparse_encoded_bytes(c);
            out[0] += delta;
            out[1] += std::min(delta, here);
            out[2] += static_cast<size_t>(delta < here);
        }
        return out;
    }

    /// Diagnostic tier split of @ref memory_bytes: {dense_bytes, sparse_bytes, dense_columns}.
    /// The two tiers have different cost models (a full-height bitmap vs coded postings), so both
    /// sizing the tier rule and checking how it landed on a real model need the split, not the total.
    auto tier_memory_bytes() const -> std::array<size_t, 3> {
        std::array<size_t, 3> out{0, 0, 0};
        for (size_t c = 0; c < kNumColumns; ++c) {
            out[0] += cols[c].words.capacity() * sizeof(uint64_t);
            out[1] += sparse_column_bytes(c);
            out[2] += static_cast<size_t>(cols[c].is_dense);
        }
        return out;
    }
};

// XOR a generator's inverted-index columns for fold words [bb, be) into blk[0 .. be-bb): dense columns
// XOR their words directly, sparse columns lower_bound to the block's row range. XOR associativity
// means any block decomposition reproduces the full-width fold bit-for-bit. THE fold-combine kernel,
// shared by the build scan, the replay cache (make_fold_cache) and the replay recompute.
inline constexpr size_t kColumnBlockWords = 1024; // 8 KB block ≈ L1-resident (bench knee)

// Reusable fold blocks (thread_local: each shard master owns its copy). Two independent scratches
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
            sc.for_each_sparse_row_in_range(c, lo, hi, [blk, bb](TermIndex r) {
                blk[(r >> 6) - bb] ^= (uint64_t{1} << (r & 63U));
            });
        }
    }
}

} // namespace monoprop::detail
