#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "monoprop/TypeAliases.h"

namespace monoprop {

/**
 * @brief Lazy transposed operator storage for even-generator overlap scans.
 *
 * The main operator is stored row-major as K Majorana terms over 2N columns.
 * This sidecar stores the transpose: one bit-vector per Majorana column, with
 * bit r set iff term r contains that mode. For an even generator G, the
 * anticommuting mask is exactly the XOR of the selected columns because
 *   (|M| |G| - |M ∩ G|) mod 2 = |M ∩ G| mod 2
 * when |G| is even.
 *
 * Each term is a k ≤ cutoff subset of 2N modes (k≈6 usually for chemistry), so every
 * column is sparse along the row axis (set fraction = per-mode frequency, a few
 * percent). Storing all columns as full-height bit-vectors is ~97% zeros. To cut
 * that without slowing the hot word-fold, columns are stored in two tiers:
 *
 *   - DENSE  (set density ≥ 1/kPromoteDensityInv): full-height uint64 bit-vector,
 *     folded by the register-accumulating word loop (even_parity_scan_pass1).
 *     This is where the bits AND the fold cost live, so it is byte-identical to
 *     the old structure — no scan regression.
 *   - SPARSE (below that, incl. empty): an UNORDERED uint32 set-row list
 *     (~4 B/set-bit). Consumed only by scatter into a dense scratch column at
 *     scan time (see prepare_fold), so it never needs to be sorted.
 *
 * A column promotes SPARSE→DENSE (one-way; the operator is append-only) once its
 * density crosses 1/kPromoteDensityInv, the point where the row-list would cost
 * more than the dense bit-vector.
 */
template <size_t NumModes>
struct EvenParityMajoranaScanSidecar {
    static constexpr size_t kNumColumns = MajoranaSet<NumModes>::size();
    // Promote a column to DENSE at set density ≥ 1/kPromoteDensityInv. At 1/32 a uint32 row-list
    // (4 B/bit) and a dense bit-vector (rows/8 B) cost the same; above it dense is smaller and the
    // sparse-merge fold would also start losing to the dense word-fold (crossover ≈ 1/64).
    static constexpr size_t kPromoteDensityInv = 32;

    struct Column {
        std::vector<uint64_t> words{};    // full-height bit-vector; used iff is_dense
        std::vector<TermIndex> set_rows{}; // unordered set-row indices; used iff !is_dense
        bool is_dense = false;
    };

    std::array<Column, kNumColumns> cols{};
    size_t row_count = 0;

    // Lazily-built parity of |M| per row, packed 1 bit/row (word w, bit b == parity of row 64*w+b).
    // Empty until the first odd-parity generator requests it; even-parity workloads never allocate it.
    std::vector<uint64_t> row_parity_{}; // empty == not built

    // Build the parity bitmap from the dense/sparse columns (called once, lazily). parity(|M|) is the
    // XOR over all mode columns of row M (equivalently popcount(row) & 1).
    void ensure_row_parity() {
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
    const uint64_t *row_parity_word_ptr() const { return row_parity_.data(); }

    auto rows() const -> size_t { return row_count; }
    auto words() const -> size_t { return (row_count + 63) / 64; }

    // ── Fold/diagnostic accessors ─────────────────────────────────────────────
    auto column_is_dense(size_t c) const -> bool { return cols[c].is_dense; }
    auto dense_column_data(size_t c) const -> const uint64_t * { return cols[c].words.data(); }
    auto sparse_column_rows(size_t c) const -> const std::vector<TermIndex> & { return cols[c].set_rows; }
    auto column_set_bits(size_t c) const -> size_t {
        const auto &col = cols[c];
        if (!col.is_dense) {
            return col.set_rows.size();
        }
        size_t bits = 0;
        for (uint64_t w : col.words) {
            bits += static_cast<size_t>(std::popcount(w));
        }
        return bits;
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

    // Scatter the set bits of new rows [base, base+n) of `op` into the tiered columns, in parallel.
    // Decomposes by 64-row WORD blocks: a task owns word `w` of every column, so dense word writes are
    // sole-writer (no atomics). Sparse columns can be written by any task, so each task collects its
    // (column → rows) into a thread-local buffer; these are concatenated afterwards (order-free —
    // scatter consumers don't need sorted rows). Dense columns must already be sized to words().
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
        // Small batches (the per-generator deferred-insert case, called thousands of times) take a
        // serial path: the parallel machinery's fixed per-call cost (thread-local buffer construction +
        // the all-column merge sweep) dwarfs the actual O(n·k) work and otherwise regresses the append.
        // Dense writes go straight to the word array; sparse rows append straight to the column list (no
        // race when serial). Only large bulk fills (rebuild) amortize the parallel path below.
        constexpr size_t kSerialFillRows = 1U << 15U;
        if (n < kSerialFillRows) {
            for (size_t row_idx = base; row_idx < new_total_rows; ++row_idx) {
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
            for (size_t c = 0; c < kNumColumns; ++c) {
                Column &col = cols[c];
                if (!col.is_dense && col.set_rows.size() * kPromoteDensityInv >= row_count) {
                    promote_to_dense(c);
                }
            }
            return;
        }

        const size_t first_word = base / 64;
        const size_t last_word = (new_total_rows - 1) / 64;
        const size_t word_span = last_word - first_word + 1;
        const size_t grain = std::max<size_t>(1, word_span / 64);

        using Partials = std::array<std::vector<TermIndex>, kNumColumns>;
        tbb::enumerable_thread_specific<Partials> tls;
        tbb::parallel_for(tbb::blocked_range<size_t>(first_word, last_word + 1, grain),
                          [&](const tbb::blocked_range<size_t> &range) {
                              Partials &part = tls.local();
                              for (size_t w = range.begin(); w < range.end(); ++w) {
                                  const size_t row_lo = std::max(base, w * 64);
                                  const size_t row_hi = std::min(new_total_rows, (w + 1) * 64);
                                  for (size_t row_idx = row_lo; row_idx < row_hi; ++row_idx) {
                                      const uint64_t row_bit = uint64_t{1} << (row_idx % 64);
                                      for_each_row_position<NumModes>(op, row_idx, [&](size_t bit) {
                                          Column &col = cols[bit];
                                          if (col.is_dense) {
                                              col.words[w] |= row_bit; // sole writer of (bit, w)
                                          }
                                          else {
                                              part[bit].push_back(static_cast<TermIndex>(row_idx));
                                          }
                                      });
                                  }
                              }
                          });

        // Merge thread-local sparse contributions and promote any column that crossed the density line.
        for (size_t c = 0; c < kNumColumns; ++c) {
            Column &col = cols[c];
            if (col.is_dense) {
                continue;
            }
            size_t add = 0;
            for (auto &part : tls) {
                add += part[c].size();
            }
            if (add == 0) {
                continue;
            }
            col.set_rows.reserve(col.set_rows.size() + add);
            for (auto &part : tls) {
                col.set_rows.insert(col.set_rows.end(), part[c].begin(), part[c].end());
            }
            if (col.set_rows.size() * kPromoteDensityInv >= row_count) {
                promote_to_dense(c);
            }
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

        // Pass 1: per-column set-bit counts (parallel reduce) → decide tiers from the FINAL density,
        // so the fill never has to promote.
        using Counts = std::array<size_t, kNumColumns>;
        tbb::enumerable_thread_specific<Counts> tls_counts([] { return Counts{}; });
        const size_t grain = std::max<size_t>(256, size / 64);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size, grain), [&](const tbb::blocked_range<size_t> &range) {
            Counts &cnt = tls_counts.local();
            for (size_t row_idx = range.begin(); row_idx < range.end(); ++row_idx) {
                for_each_row_position<NumModes>(op, row_idx, [&](size_t bit) { ++cnt[bit]; });
            }
        });
        for (size_t c = 0; c < kNumColumns; ++c) {
            size_t count = 0;
            for (const auto &cnt : tls_counts) {
                count += cnt[c];
            }
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

    auto append_row(const MajoranaSet<NumModes> &maj) -> void {
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

    auto reserve_rows(size_t total_rows) -> void {
        const size_t required_words = (total_rows + 63) / 64;
        for (auto &col : cols) {
            if (col.is_dense && col.words.capacity() < required_words) {
                col.words.reserve(required_words);
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

} // namespace monoprop
