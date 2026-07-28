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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/operator/InvertedIndex.h"

// Internals of the even-parity scan inverted index: the tiered column store, the lazily-built
// per-row parity(|M|) bitmap, and the fill order. Rows are read through the backend-agnostic
// for_each_row_position accessor, so a plain std::vector<Monomial<N>> stands in for the store.

using namespace monoprop;
using namespace monoprop::detail;

namespace {
constexpr size_t N = 32; // 2N = 64 majorana columns
using Sc = InvertedIndex<N>;
using MSet = Monomial<N>;
MSet bs(const VecZ &r) {
    return indices_to_bitset<N>(r);
}
// indices_to_bitset maps mode m to bit 2N-1-m; columns are indexed by raw bit position.
constexpr size_t col_of(size_t mode) {
    return 2 * N - 1 - mode;
}

// A column's set rows, whichever tier holds them — so an assertion can compare two indices that
// picked different tiers for the same membership.
auto rows_of(const Sc &sc, size_t c) -> std::vector<size_t> {
    std::vector<size_t> out;
    if (!sc.column_is_dense(c)) {
        for (const auto row : sc.sparse_column_rows(c)) {
            out.push_back(static_cast<size_t>(row));
        }
        return out;
    }
    const uint64_t *words = sc.dense_column_data(c);
    for (size_t r = 0; r < sc.rows(); ++r) {
        if (((words[r >> 6] >> (r & 63U)) & 1U) != 0U) {
            out.push_back(r);
        }
    }
    return out;
}
} // namespace

// row_parity_words() packs popcount(|M|)&1 over the current rows; the oracle is each row's popcount.
BOOST_AUTO_TEST_CASE(inverted_index_row_parity_matches_popcount) {
    const std::vector<MSet> op{
        bs({0, 1}),
        bs({0, 1, 2}),
        bs({5}),
        bs({4, 5, 6, 7}),
    };
    Sc sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == op.size());

    const uint64_t *parity = sc.row_parity_words();
    bool all_match = true;
    for (size_t i = 0; i < op.size(); ++i) {
        const bool bit = ((parity[i >> 6] >> (i & 63U)) & 1U) != 0;
        if (bit != static_cast<bool>(op[i].count() & 1U)) {
            all_match = false;
        }
    }
    BOOST_TEST(all_match);
    // row_parity_words is idempotent (a lazy cache): a second call must not change the bitmap.
    BOOST_TEST(((sc.row_parity_words()[0] >> 1U) & 1U) == 1U); // row 1 is odd
}

// rebuild decides tiers from the final per-column counts: dense once density >= 1/kPromoteDensityInv.
BOOST_AUTO_TEST_CASE(inverted_index_promotes_column_at_density_crossover) {
    constexpr size_t kR = 128; // threshold = ceil(128/64) = 2 set rows to go dense
    std::vector<MSet> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        VecZ pos;
        if (i < 10) {
            pos.push_back(0); // mode 0 set in 10 rows -> 10*64 >= 128 -> DENSE
            if (i == 0) {
                pos.push_back(1); // mode 1 set in exactly 1 row -> 1*64 < 128 -> SPARSE
            }
        }
        else {
            pos.push_back(2); // mode 2 set in 118 rows -> DENSE (keeps every row non-empty)
        }
        op.push_back(indices_to_bitset<N>(pos));
    }
    Sc sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == kR);
    BOOST_TEST(sc.column_is_dense(col_of(0)));  // 10/128 >= 1/64
    BOOST_TEST(!sc.column_is_dense(col_of(1))); // 1/128  <  1/64
    // The lone sparse hit is recorded losslessly.
    BOOST_TEST(sc.sparse_column_rows(col_of(1)).size() == 1u);
    BOOST_TEST(sc.sparse_column_rows(col_of(1))[0] == 0u);
}

// rebuild fills columns in row order, so sparse row-lists come out ascending — the invariant
// combine_columns_block's lower_bound relies on.
BOOST_AUTO_TEST_CASE(inverted_index_fill_yields_ascending_sparse_rows) {
    constexpr size_t M = 64; // 2M = 128 columns
    using ScW = InvertedIndex<M>;
    constexpr size_t kR = 16'385; // large operator, many sparse columns
    std::vector<Monomial<M>> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        // One mode per row over 128 columns: 128 hits each, and 128*64 < 16385, so all stay sparse.
        op.push_back(indices_to_bitset<M>({i % 128}));
    }
    ScW sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == kR);

    bool all_sorted = true;
    bool saw_nonempty_sparse = false;
    for (size_t c = 0; c < 128; ++c) {
        if (sc.column_is_dense(c)) {
            continue;
        }
        const auto &rows = sc.sparse_column_rows(c);
        if (!rows.empty()) {
            saw_nonempty_sparse = true;
        }
        if (!std::ranges::is_sorted(rows)) {
            all_sorted = false;
        }
    }
    BOOST_TEST(saw_nonempty_sparse); // the fill actually populated sparse columns
    BOOST_TEST(all_sorted);
}

// append_rows is the incremental growth path (MPOperator appends as terms are inserted); rebuild is the
// from-scratch one. On the same final row set the two must agree on membership. Tier choice legitimately
// differs -- rebuild sees the final per-column counts up front, while the append path promotes on
// crossing the threshold -- so the comparison is on set rows, which are tier-independent.
BOOST_AUTO_TEST_CASE(inverted_index_append_rows_matches_rebuild) {
    constexpr size_t kR = 200;
    std::vector<MSet> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        VecZ pos{i % 5}; // modes 0..4: 40 rows each, dense either way
        if (i == 7) {
            pos.push_back(9); // one hit in 200 rows: stays sparse
        }
        if (i >= 64 && i % 3 == 0) {
            // Absent from the seed, then crosses the density threshold during the appends.
            pos.push_back(13);
        }
        op.push_back(bs(pos));
    }

    Sc full;
    full.rebuild(op);

    Sc inc;
    inc.rebuild(std::vector<MSet>(op.begin(), op.begin() + 64));
    // Force the lazy parity bitmap to exist before the append, which is the branch that extends it.
    static_cast<void>(inc.row_parity_words());
    inc.append_rows(op, 64, 100);
    inc.append_rows(op, 164, kR - 164);

    BOOST_REQUIRE_EQUAL(inc.rows(), full.rows());
    for (size_t c = 0; c < Sc::kNumColumns; ++c) {
        BOOST_TEST_CONTEXT("column " << c) {
            BOOST_TEST(rows_of(inc, c) == rows_of(full, c), boost::test_tools::per_element());
        }
    }

    // The bitmap was extended over the appended rows rather than left at its pre-append width.
    const uint64_t *parity = inc.row_parity_words();
    bool parity_matches = true;
    for (size_t i = 0; i < kR; ++i) {
        const bool bit = ((parity[i >> 6] >> (i & 63U)) & 1U) != 0;
        if (bit != static_cast<bool>(op[i].count() & 1U)) {
            parity_matches = false;
        }
    }
    BOOST_TEST(parity_matches);
}

// combine_columns_block is the fold-combine kernel every scan and recompute path shares, and only
// rebuild() was exercised before. It XORs the given columns' row bitmaps over a word range: XOR
// associativity means any block decomposition reproduces the full-width fold bit-for-bit, and the
// dense-column memcpy seed must equal memset + XOR-all.
BOOST_AUTO_TEST_CASE(combine_columns_block_folds_dense_and_sparse_identically) {
    constexpr size_t kR = 300; // 5 row words, so a block split has something to split
    std::vector<MSet> op;
    std::vector<bool> in_a(kR, false), in_b(kR, false), in_sparse(kR, false);
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        VecZ pos;
        if (i % 3 == 0) {
            pos.push_back(0); // 100/300: dense
            in_a[i] = true;
        }
        if (i % 7 == 0) {
            pos.push_back(1); // 43/300: dense
            in_b[i] = true;
        }
        if (i == 5 || i == 200) {
            pos.push_back(2); // 2/300 < 1/64: sparse, and spans two blocks
            in_sparse[i] = true;
        }
        if (pos.empty()) {
            pos.push_back(3); // keep every row non-empty
        }
        op.push_back(bs(pos));
    }
    Sc sc;
    sc.rebuild(op);
    BOOST_REQUIRE(sc.column_is_dense(col_of(0)));
    BOOST_REQUIRE(sc.column_is_dense(col_of(1)));
    BOOST_REQUIRE(!sc.column_is_dense(col_of(2)));

    const size_t words = (kR + 63) / 64;
    const std::vector<size_t> cols{col_of(0), col_of(1), col_of(2)};

    // Oracle built from the authored membership, not from the index's own storage.
    std::vector<uint64_t> expected(words, 0);
    for (size_t r = 0; r < kR; ++r) {
        if (in_a[r] != (in_b[r] != in_sparse[r])) { // three-way XOR
            expected[r >> 6] |= uint64_t{1} << (r & 63U);
        }
    }

    std::vector<uint64_t> whole(words, 0xdeadbeefULL); // pre-dirtied: the kernel seeds, never accumulates
    combine_columns_block<N>(sc, cols, whole.data(), 0, words);
    BOOST_TEST(whole == expected, boost::test_tools::per_element());

    // One word per call must reproduce the same fold word for word.
    std::vector<uint64_t> pieced(words, 0xdeadbeefULL);
    for (size_t w = 0; w < words; ++w) {
        combine_columns_block<N>(sc, cols, pieced.data() + w, w, w + 1);
    }
    BOOST_TEST(pieced == expected, boost::test_tools::per_element());

    // Sparse-only column list: no dense column to memcpy from, so this is the memset seed path.
    const size_t sparse_col = col_of(2);
    std::vector<uint64_t> sparse_fold(words, 0xdeadbeefULL);
    combine_columns_block<N>(sc, std::span<const size_t>(&sparse_col, 1), sparse_fold.data(), 0, words);
    std::vector<uint64_t> sparse_expected(words, 0);
    sparse_expected[5 >> 6] |= uint64_t{1} << (5 & 63U);
    sparse_expected[200 >> 6] |= uint64_t{1} << (200 & 63U);
    BOOST_TEST(sparse_fold == sparse_expected, boost::test_tools::per_element());
}
