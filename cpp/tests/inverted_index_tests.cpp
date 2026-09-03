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
#include <bit>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/RowAccess.h"

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
    // A word at a time: a dense column is chunked, so it is read a fold block at a time and there is no
    // pointer to the whole of it.
    for (size_t w = 0; w * 64 < sc.rows(); ++w) {
        uint64_t bits = *sc.dense_column_block(c, w, w + 1);
        while (bits != 0U) {
            const size_t r = (w * 64) + static_cast<size_t>(std::countr_zero(bits));
            if (r < sc.rows()) {
                out.push_back(r);
            }
            bits &= bits - 1;
        }
    }
    return out;
}
} // namespace

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

// combine_columns_block is the fold-combine kernel every scan and recompute path shares. It XORs the
// given columns' row bitmaps over a word range: XOR associativity means any block decomposition
// reproduces the full-width fold bit-for-bit, and the dense-column memcpy seed must equal
// memset + XOR-all.
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

// Chunking the dense columns must be invisible: the fold is their only reader, it walks a column in
// blocks of kColumnBlockWords, and a chunk holds a whole number of those blocks -- so the same words
// come back whatever the chunk size. These cases pin that down at the smallest legal chunk
// (kColumnBlockWords words, a boundary every 65536 rows) against the production chunk, which holds this
// operator in a single piece and is therefore the unchunked reference.
namespace {

// Enough rows to spill past three chunk boundaries at kColumnBlockWords words per chunk. Cheap: a
// Monomial<32> is eight bytes.
constexpr size_t kChunkTestRows = 200'000;
constexpr size_t kSmallChunkWords = kColumnBlockWords;

// Three tiers' worth of column in one operator: two dense (one dense enough to win the fold's memcpy
// seed), one sparse with set rows in every chunk, so both fold paths cross a boundary.
auto build_chunk_test_op() -> std::vector<MSet> {
    std::vector<MSet> op;
    op.reserve(kChunkTestRows);
    for (size_t i = 0; i < kChunkTestRows; ++i) {
        VecZ pos;
        if (i % 3 == 0) {
            pos.push_back(0);
        }
        if (i % 7 == 0) {
            pos.push_back(1);
        }
        if (i % 40'009 == 5) { // a handful of rows, one per chunk
            pos.push_back(2);
        }
        if (pos.empty()) {
            pos.push_back(3); // keep every row non-empty
        }
        op.push_back(bs(pos));
    }
    return op;
}

} // namespace

// The fold over a multi-chunk column must equal the fold over the same column held in one piece, word
// for word -- the memcpy seed included, that being the one read which is not a plain loop.
BOOST_AUTO_TEST_CASE(dense_columns_fold_identically_across_chunk_boundaries) {
    const auto op = build_chunk_test_op();
    Sc chunked(kSmallChunkWords);
    Sc whole(Sc::kWordsPerChunk);
    chunked.rebuild(op);
    whole.rebuild(op);

    const size_t words = (kChunkTestRows + 63) / 64;
    BOOST_REQUIRE_GT(words, 3 * kSmallChunkWords); // the small-chunk index really is multi-chunk
    BOOST_REQUIRE(chunked.column_is_dense(col_of(0)));
    BOOST_REQUIRE(chunked.column_is_dense(col_of(1)));
    BOOST_REQUIRE(!chunked.column_is_dense(col_of(2)));
    BOOST_REQUIRE_EQUAL(chunked.column_is_dense(col_of(0)), whole.column_is_dense(col_of(0)));

    const std::vector<size_t> cols{col_of(0), col_of(1), col_of(2)};
    std::vector<uint64_t> a(words, 0xdeadbeefULL);
    std::vector<uint64_t> b(words, 0xfeedfaceULL);
    for (size_t bb = 0; bb < words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, words);
        combine_columns_block<N>(chunked, cols, a.data() + bb, bb, be);
        combine_columns_block<N>(whole, cols, b.data() + bb, bb, be);
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());

    // And against the authored membership, so a fold that is wrong the same way in both still fails.
    std::vector<uint64_t> expected(words, 0);
    for (size_t r = 0; r < kChunkTestRows; ++r) {
        const bool in_a = r % 3 == 0;
        const bool in_b = r % 7 == 0;
        const bool in_sparse = r % 40'009 == 5;
        if (in_a != (in_b != in_sparse)) { // three-way XOR
            expected[r >> 6] |= uint64_t{1} << (r & 63U);
        }
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), expected.begin(), expected.end());
}

// make_fold_cache used to combine a whole column in one call; a chunked column has no pointer that
// wide, so it now walks the same blocks as every other fold. XOR is associative, so the words must come
// out identical -- and so must the coefficients a fold scales, which is what the engine observes.
BOOST_AUTO_TEST_CASE(make_fold_cache_is_blocked_and_bit_identical_across_chunk_sizes) {
    const auto op = build_chunk_test_op();
    Sc chunked(kSmallChunkWords);
    Sc whole(Sc::kWordsPerChunk);
    chunked.rebuild(op);
    whole.rebuild(op);

    // An even and an odd generator: the odd one also engages the row-parity correction.
    const std::vector<MSet> gens{bs({0, 1}), bs({0, 1, 2})};
    for (const auto &gen : gens) {
        const auto ca = make_fold_cache<N>(chunked, gen, kChunkTestRows, Basis::Majorana);
        const auto cb = make_fold_cache<N>(whole, gen, kChunkTestRows, Basis::Majorana);
        BOOST_REQUIRE_EQUAL(ca.combined.size(), cb.combined.size());
        BOOST_REQUIRE_GT(ca.combined.size(), kSmallChunkWords);
        BOOST_CHECK_EQUAL_COLLECTIONS(ca.combined.begin(), ca.combined.end(), cb.combined.begin(), cb.combined.end());

        // Distinct, non-degenerate coefficients, so one index scaled twice or not at all shows up.
        std::vector<double> xa(kChunkTestRows);
        for (size_t i = 0; i < kChunkTestRows; ++i) {
            xa[i] = 1.0 + (static_cast<double>(i) * 1e-6);
        }
        std::vector<double> xb = xa;
        const auto ra = make_lazy_fold<N>(chunked, gen, kChunkTestRows, Basis::Majorana);
        const auto rb = make_lazy_fold<N>(whole, gen, kChunkTestRows, Basis::Majorana);
        scale_cos_lazy<N>(chunked, ra, xa.data(), 0.6234);
        scale_cos_lazy<N>(whole, rb, xb.data(), 0.6234);
        // Exactly equal, not within a tolerance: the whole point of the lever is bit-identity.
        BOOST_CHECK_EQUAL_COLLECTIONS(xa.begin(), xa.end(), xb.begin(), xb.end());
    }
}

// A column promoted mid-fill is rebuilt chunk by chunk, and an index grown by appends extends its
// chunks rather than the vector it no longer has. Both paths must land on exactly the bits one rebuild
// over the whole operator does.
BOOST_AUTO_TEST_CASE(appending_and_promotion_reach_the_same_chunked_columns) {
    const auto op = build_chunk_test_op();
    Sc appended(kSmallChunkWords);
    // Three unequal batches; the boundaries fall inside chunks, not on them.
    size_t base = 0;
    for (const size_t n : {size_t{70'000}, size_t{60'000}, kChunkTestRows - 130'000}) {
        appended.append_rows(op, base, n);
        base += n;
    }
    BOOST_REQUIRE_EQUAL(appended.rows(), kChunkTestRows);

    Sc rebuilt(kSmallChunkWords);
    rebuilt.rebuild(op);
    for (size_t c = 0; c < Sc::kNumColumns; ++c) {
        BOOST_TEST_INFO("column " << c);
        const auto got = rows_of(appended, c);
        const auto want = rows_of(rebuilt, c);
        BOOST_CHECK_EQUAL_COLLECTIONS(got.begin(), got.end(), want.begin(), want.end());
    }
}

// The ledger's dense figure is now the pool's page-rounded chunk bytes, so the slack it hides is under
// one chunk per dense column -- where a doubling vector's could be the column over again.
BOOST_AUTO_TEST_CASE(chunked_dense_columns_bound_the_index_slack_by_one_chunk) {
    const auto op = build_chunk_test_op();
    Sc sc(kSmallChunkWords);
    sc.rebuild(op);
    const auto tiers = sc.tier_memory_bytes();
    const size_t dense_columns = tiers[2];
    BOOST_REQUIRE_GT(dense_columns, 0U);

    const size_t live = sc.words() * sizeof(uint64_t) * dense_columns;
    BOOST_CHECK_GE(tiers[0], live);
    BOOST_CHECK_LT(tiers[0] - live, dense_columns * kSmallChunkWords * sizeof(uint64_t));
    BOOST_CHECK_LE(tiers[0] + tiers[1], sc.memory_bytes());
}

// A copied index owns its own chunks: MPOperator deep-copies the index whenever a propagator is copied,
// and a shared chunk would make the two copies aliases of one bitmap.
BOOST_AUTO_TEST_CASE(copying_an_index_deep_copies_its_chunked_columns) {
    const auto op = build_chunk_test_op();
    Sc original(kSmallChunkWords);
    original.rebuild(op);
    Sc copy = original; // NOLINT(performance-unnecessary-copy-initialization)

    BOOST_REQUIRE_EQUAL(copy.rows(), original.rows());
    const size_t c0 = col_of(0);
    BOOST_REQUIRE(copy.column_is_dense(c0));
    BOOST_CHECK(copy.dense_column_block(c0, 0, 1) != original.dense_column_block(c0, 0, 1));
    for (size_t c = 0; c < Sc::kNumColumns; ++c) {
        BOOST_TEST_INFO("column " << c);
        const auto got = rows_of(copy, c);
        const auto want = rows_of(original, c);
        BOOST_CHECK_EQUAL_COLLECTIONS(got.begin(), got.end(), want.begin(), want.end());
    }

    // And moving one through an optional, as MPOperator holds it, must not disturb the storage.
    std::optional<Sc> moved{std::move(copy)};
    BOOST_REQUIRE(moved.has_value());
    const auto got = rows_of(*moved, c0);
    const auto want = rows_of(original, c0);
    BOOST_CHECK_EQUAL_COLLECTIONS(got.begin(), got.end(), want.begin(), want.end());
}
