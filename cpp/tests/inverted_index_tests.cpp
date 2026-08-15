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
#include <random>
#include <span>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/RowAccess.h"

// Internals of the even-parity scan inverted index: the per-(column, chunk) container choice, the
// lazily-built per-row parity(|M|) bitmap, and the fill order. Rows are read through the
// backend-agnostic for_each_row_position accessor, so a plain std::vector<Monomial<N>> stands in for
// the store.

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

// A column's set rows, whichever containers hold them — so an assertion can compare two indices that
// picked different containers for the same membership.
auto rows_of(const Sc &sc, size_t c) -> std::vector<size_t> {
    std::vector<size_t> out;
    sc.for_each_row_in_column(c, [&out](size_t r) { out.push_back(r); });
    return out;
}

// An operator large enough to seal a chunk, authored so that chunk 0 exercises every container. The
// densities are written as fractions of the chunk so the argmin lands the same way at any chunk height,
// which is a build option -- an absolute posting count would silently stop testing what it names:
//
//   mode 0  every 4th row     R/4 postings -> bitmap is R/8 B against R/4 B of 1-byte gaps
//   mode 1  every 10th row    R/10        -> a 1-byte gap each, and that is the smallest
//   mode 2  every 1000th row  R/1000      -> every gap escapes, so 3 B each: the delta stream's worst case
//   mode 3  never             0           -> Empty
//   mode 4  every row         R           -> bitmap, at 1/8 the cost of a gap per row
//
// kMixedRows leaves a partial trailing chunk, which is the tail: always delta-encoded, never sealed.
constexpr size_t kMixedRows = kChunkRows + kChunkRows / 4 + 464;
constexpr size_t kMixedWords = (kMixedRows + 63) / 64;

auto mixed_operator() -> std::vector<MSet> {
    std::vector<MSet> op;
    op.reserve(kMixedRows);
    for (size_t i = 0; i < kMixedRows; ++i) {
        VecZ pos{4};
        if (i % 4 == 0) {
            pos.push_back(0);
        }
        if (i % 10 == 0) {
            pos.push_back(1);
        }
        if (i % 1000 == 0) {
            pos.push_back(2);
        }
        op.push_back(bs(pos));
    }
    return op;
}

// The fold every assertion below is compared against, built from the authored membership rather than
// from the index's own storage.
auto oracle_fold(const std::vector<MSet> &op, const std::vector<size_t> &cols) -> std::vector<uint64_t> {
    std::vector<uint64_t> expected((op.size() + 63) / 64, 0);
    for (size_t r = 0; r < op.size(); ++r) {
        bool bit = false;
        for (const size_t c : cols) {
            bit ^= op[r].test(c);
        }
        if (bit) {
            expected[r >> 6] |= uint64_t{1} << (r & 63U);
        }
    }
    return expected;
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

// The whole tiering policy is "smallest container wins", decided per (column, chunk) at seal. This pins
// each arm of that argmin to a column whose density selects it, so a change to the rule cannot pass
// unnoticed — and it is the guard that makes the differential test below non-hollow.
BOOST_AUTO_TEST_CASE(inverted_index_picks_the_smallest_container_per_chunk) {
    const auto op = mixed_operator();
    Sc sc;
    sc.rebuild(op);
    BOOST_TEST(sc.rows() == kMixedRows);
    BOOST_REQUIRE_EQUAL(sc.sealed_chunks(), 1U);
    BOOST_REQUIRE_EQUAL(sc.chunk_count(), 2U);

    BOOST_TEST((sc.container_tag(col_of(0), 0) == ChunkTag::Bitmap));
    BOOST_TEST((sc.container_tag(col_of(1), 0) == ChunkTag::U8Delta));
    BOOST_TEST((sc.container_tag(col_of(2), 0) == ChunkTag::U8Delta));
    BOOST_TEST((sc.container_tag(col_of(3), 0) == ChunkTag::Empty));
    BOOST_TEST((sc.container_tag(col_of(4), 0) == ChunkTag::Bitmap));

    // The growing chunk obeys the same rule, over the two containers an append can still reach. Mode 0
    // (density 1/4) and mode 4 (every row) cross into a bitmap; mode 1 (density 1/10) stays a delta
    // stream because a byte per posting is still under a bit per row. That boundary is the whole point:
    // a delta-only tail was measured 1.34x worse than the layout this replaces on dense Pauli columns.
    BOOST_TEST((sc.container_tag(col_of(0), 1) == ChunkTag::Bitmap));
    BOOST_TEST((sc.container_tag(col_of(1), 1) == ChunkTag::U8Delta));
    BOOST_TEST((sc.container_tag(col_of(2), 1) == ChunkTag::U8Delta));
    BOOST_TEST((sc.container_tag(col_of(3), 1) == ChunkTag::Empty));
    BOOST_TEST((sc.container_tag(col_of(4), 1) == ChunkTag::Bitmap));

    BOOST_TEST(sc.column_is_empty(col_of(3)));
    BOOST_TEST(!sc.column_is_empty(col_of(2)));
    BOOST_TEST(sc.column_postings(col_of(2)) == (kMixedRows + 999) / 1000);

    // The arena replaced std::vector's doubling precisely so unused capacity stays bounded.
    const auto stats = sc.stats();
    BOOST_TEST(stats.bitmap_chunks == 4U); // modes 0 and 4, once sealed and once in the tail
    BOOST_TEST(stats.arena_slack_bytes <= sc.arena_.size() / 8 + 8);
}

// The tail's bitmap -> delta give-up path. A column goes dense enough to take a tail bitmap, falls
// quiet, then takes one late posting that stretches the bitmap across the whole silence. A delta stream
// never exceeds 3 bytes per posting, so a bitmap past kTailBitmapGiveUpFactor times the posting count is
// provably the worse of the two and must convert back. Nothing else here reaches the transition back --
// and since the late gap is >= 255, this is also the only cover of the escape inside that re-encode.
BOOST_AUTO_TEST_CASE(inverted_index_tail_bitmap_gives_up_on_a_late_sparse_posting) {
    // Dense from row 0, so a byte per posting immediately loses to a bit per row and the tail crosses to
    // a bitmap inside the first word. Then silence, then one posting far enough out that the bitmap it
    // has to stretch to costs more than 4 bytes for each posting it holds.
    constexpr size_t kDense = 9;  // 9 B of gaps against 8 B of bitmap: the crossover
    constexpr size_t kLate = 320; // 6 words = 48 B of bitmap against 10 postings
    static_assert(kLate + 1 < kChunkRows, "nothing here may seal; this is a tail test");

    auto build = [](size_t rows) {
        std::vector<MSet> op;
        op.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            VecZ pos{4}; // keep every row non-empty
            if (i < kDense || i == kLate) {
                pos.push_back(0);
            }
            op.push_back(bs(pos));
        }
        return op;
    };

    // Positive control: with the late posting withheld the column is still a bitmap, so the assertion
    // below reads a transition rather than a column that was a delta stream the whole time.
    Sc quiet;
    quiet.rebuild(build(kLate));
    BOOST_REQUIRE_EQUAL(quiet.sealed_chunks(), 0U);
    BOOST_REQUIRE((quiet.container_tag(col_of(0), 0) == ChunkTag::Bitmap));

    const auto op = build(kLate + 1);
    Sc sc;
    sc.rebuild(op);
    BOOST_REQUIRE_EQUAL(sc.sealed_chunks(), 0U);
    BOOST_TEST((sc.container_tag(col_of(0), 0) == ChunkTag::U8Delta));

    // The re-encode is lossless across the escape: the trailing gap is kLate - kDense = 311.
    std::vector<size_t> expected_rows;
    for (size_t i = 0; i < kDense; ++i) {
        expected_rows.push_back(i);
    }
    expected_rows.push_back(kLate);
    BOOST_TEST(rows_of(sc, col_of(0)) == expected_rows, boost::test_tools::per_element());
    BOOST_TEST(sc.column_postings(col_of(0)) == expected_rows.size());

    // And the fold agrees, which is what the container choice is not allowed to change.
    const std::vector<size_t> cols{col_of(0), col_of(4)};
    const auto expected = oracle_fold(op, cols);
    std::vector<uint64_t> got(expected.size(), 0xdeadbeefULL);
    combine_columns_block<N>(sc, cols, got.data(), 0, expected.size());
    BOOST_TEST(got == expected, boost::test_tools::per_element());
}

// Sealing re-encodes a bitmap tail back to a delta stream. The seal prices every column against what a
// delta stream WOULD cost, not against what the tail happens to hold, so a column that was briefly dense
// and then quiet seals as postings -- taking write_column_'s re-encode arm rather than its memcpy arm.
// The late posting puts a >= 255 gap in that re-encode.
BOOST_AUTO_TEST_CASE(inverted_index_seals_a_bitmap_tail_back_to_a_delta_stream) {
    // Fractions of the chunk, so the argmin and the give-up factor both land the same way at any chunk
    // height: dense over the first 32nd (bitmap), one posting at the half (still under 4 B/posting, so
    // the tail stays a bitmap), and a whole-chunk delta cost of ~1/32 B per row against the bitmap's 1/8.
    constexpr size_t kDense = kChunkRows / 32;
    constexpr size_t kLate = kChunkRows / 2;

    auto build = [](size_t rows) {
        std::vector<MSet> op;
        op.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            VecZ pos{4};
            if (i < kDense || i == kLate) {
                pos.push_back(0);
            }
            op.push_back(bs(pos));
        }
        return op;
    };

    // Positive control: one row short of the seal the tail is a bitmap, so the sealed tag below is the
    // re-encode arm and not a delta tail that was memcpy'd.
    Sc unsealed;
    unsealed.rebuild(build(kChunkRows - 1));
    BOOST_REQUIRE_EQUAL(unsealed.sealed_chunks(), 0U);
    BOOST_REQUIRE((unsealed.container_tag(col_of(0), 0) == ChunkTag::Bitmap));

    const auto op = build(kChunkRows);
    Sc sc;
    sc.rebuild(op);
    BOOST_REQUIRE_EQUAL(sc.sealed_chunks(), 1U);
    BOOST_TEST((sc.container_tag(col_of(0), 0) == ChunkTag::U8Delta));

    std::vector<size_t> expected_rows;
    for (size_t i = 0; i < kDense; ++i) {
        expected_rows.push_back(i);
    }
    expected_rows.push_back(kLate);
    BOOST_TEST(rows_of(sc, col_of(0)) == expected_rows, boost::test_tools::per_element());

    const std::vector<size_t> cols{col_of(0), col_of(4)};
    const auto expected = oracle_fold(op, cols);
    std::vector<uint64_t> got(expected.size(), 0xdeadbeefULL);
    for (size_t bb = 0; bb < expected.size(); bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, expected.size());
        combine_columns_block<N>(sc, cols, got.data() + bb, bb, be);
    }
    BOOST_TEST(got == expected, boost::test_tools::per_element());
}

// Three seals, so chunk_base_ and dir_ are indexed at k > 0 and the first-seal arena projection (which
// needs more than one whole chunk to fire) is taken. Mode 1 is set in the middle chunk only: a directory
// read that ignored k, or a chunk base off by one segment, would report its postings against the wrong
// rows and could not survive the row comparison below.
BOOST_AUTO_TEST_CASE(inverted_index_indexes_the_directory_per_chunk_across_three_seals) {
    constexpr size_t kRows = 3 * kChunkRows + kChunkRows / 3; // three sealed chunks plus a partial tail

    std::vector<MSet> op;
    op.reserve(kRows);
    std::vector<size_t> mid_rows;
    for (size_t i = 0; i < kRows; ++i) {
        VecZ pos{0}; // every row: bitmap in all three sealed chunks and in the tail
        if (i >= kChunkRows && i < 2 * kChunkRows && i % 4 == 0) {
            pos.push_back(1); // middle chunk only, dense enough there to take a bitmap
            mid_rows.push_back(i);
        }
        if (i % 1000 == 0) {
            pos.push_back(2); // every gap escapes: a delta stream in every chunk
        }
        op.push_back(bs(pos));
    }

    Sc sc;
    sc.rebuild(op);
    BOOST_REQUIRE_EQUAL(sc.rows(), kRows);
    BOOST_REQUIRE_EQUAL(sc.sealed_chunks(), 3U);
    BOOST_REQUIRE_EQUAL(sc.chunk_count(), 4U);

    bool tags_as_authored = true;
    for (size_t k = 0; k < 4; ++k) {
        tags_as_authored &= sc.container_tag(col_of(0), k) == ChunkTag::Bitmap;
        tags_as_authored &= sc.container_tag(col_of(1), k) == (k == 1 ? ChunkTag::Bitmap : ChunkTag::Empty);
        tags_as_authored &= sc.container_tag(col_of(2), k) == ChunkTag::U8Delta;
        tags_as_authored &= sc.container_tag(col_of(3), k) == ChunkTag::Empty;
    }
    BOOST_TEST(tags_as_authored);

    BOOST_TEST(rows_of(sc, col_of(1)) == mid_rows, boost::test_tools::per_element());
    BOOST_TEST(rows_of(sc, col_of(0)).size() == kRows);
    BOOST_TEST(sc.column_postings(col_of(2)) == (kRows + 999) / 1000);

    // The one-shot projection must leave the arena no fatter than the geometric path would have.
    BOOST_TEST(sc.stats().arena_slack_bytes <= sc.arena_bytes() / 8 + 8);

    // Fold across all three sealed chunks and the tail, blocked as production blocks it.
    const size_t words = (kRows + 63) / 64;
    const std::vector<size_t> cols{col_of(0), col_of(1), col_of(2), col_of(3)};
    const auto expected = oracle_fold(op, cols);
    std::vector<uint64_t> got(words, 0xdeadbeefULL);
    for (size_t bb = 0; bb < words; bb += kColumnBlockWords) {
        const size_t be = std::min(bb + kColumnBlockWords, words);
        combine_columns_block<N>(sc, cols, got.data() + bb, bb, be);
    }
    BOOST_TEST(got == expected, boost::test_tools::per_element());
}

// Every fill path appends in row order, so a column's rows come out ascending whichever container holds
// them — the invariant the kernel's early break relies on.
BOOST_AUTO_TEST_CASE(inverted_index_yields_ascending_rows_in_every_container) {
    const auto op = mixed_operator();
    Sc sc;
    sc.rebuild(op);

    bool all_sorted = true;
    for (size_t mode = 0; mode < 5; ++mode) {
        const auto rows = rows_of(sc, col_of(mode));
        if (!std::ranges::is_sorted(rows)) {
            all_sorted = false;
        }
    }
    BOOST_TEST(all_sorted);
    BOOST_TEST(rows_of(sc, col_of(2)).front() == 0U);
    BOOST_TEST(rows_of(sc, col_of(2)).back() == ((kMixedRows - 1) / 1000) * 1000);
    BOOST_TEST(rows_of(sc, col_of(4)).size() == kMixedRows);
}

// append_rows is the incremental growth path (MPOperator appends as terms are inserted); rebuild is the
// from-scratch one. On the same final row set the two must agree on membership AND, now that container
// choice is decided at seal from the finished chunk, on storage. The appends deliberately straddle the
// chunk boundary, so one of them seals mid-call.
BOOST_AUTO_TEST_CASE(inverted_index_append_rows_matches_rebuild) {
    const auto op = mixed_operator();

    Sc full;
    full.rebuild(op);

    // Split below the first seal so the append crosses it, whatever the chunk height is.
    constexpr size_t kSplit = kChunkRows / 2;
    Sc inc;
    inc.rebuild(std::vector<MSet>(op.begin(), op.begin() + kSplit));
    // Force the lazy parity bitmap to exist before the append, which is the branch that extends it.
    static_cast<void>(inc.row_parity_words());
    inc.append_rows(op, kSplit, kMixedRows - kSplit); // crosses the seal at kChunkRows
    BOOST_REQUIRE_EQUAL(inc.rows(), full.rows());

    bool rows_match = true;
    bool tags_match = true;
    for (size_t c = 0; c < Sc::kNumColumns; ++c) {
        if (rows_of(inc, c) != rows_of(full, c)) {
            rows_match = false;
        }
        for (size_t k = 0; k < full.chunk_count(); ++k) {
            if (inc.container_tag(c, k) != full.container_tag(c, k)) {
                tags_match = false;
            }
        }
    }
    BOOST_TEST(rows_match);
    BOOST_TEST(tags_match);

    // The bitmap was extended over the appended rows rather than left at its pre-append width.
    const uint64_t *parity = inc.row_parity_words();
    bool parity_matches = true;
    for (size_t i = 0; i < kMixedRows; ++i) {
        const bool bit = ((parity[i >> 6] >> (i & 63U)) & 1U) != 0;
        if (bit != static_cast<bool>(op[i].count() & 1U)) {
            parity_matches = false;
        }
    }
    BOOST_TEST(parity_matches);
}

// combine_columns_block is the fold-combine kernel every scan and recompute path shares. It XORs the
// given columns' row bitmaps over a word range: XOR associativity means any block decomposition
// reproduces the full-width fold bit-for-bit, and every container holds the same row set, so the answer
// must not depend on which containers the index picked.
BOOST_AUTO_TEST_CASE(combine_columns_block_folds_every_container_identically) {
    const auto op = mixed_operator();
    Sc sc;
    sc.rebuild(op);

    // One generator spanning all four container kinds at once, plus the sparsest column alone (the
    // memset seed path: no bitmap to memcpy from).
    const std::vector<std::vector<size_t>> generators{
        {col_of(0), col_of(1), col_of(2), col_of(3)},
        {col_of(2)},
        {col_of(1), col_of(2)},
        {col_of(4)},
    };

    // Block decompositions: whole; word by word; the production chunk-aligned blocking (whose last block
    // is a partial chunk); and a split that straddles the chunk boundary at word kChunkWords.
    std::vector<std::vector<std::pair<size_t, size_t>>> splits;
    splits.push_back({{0, kMixedWords}});
    {
        std::vector<std::pair<size_t, size_t>> by_word;
        for (size_t w = 0; w < kMixedWords; ++w) {
            by_word.emplace_back(w, w + 1);
        }
        splits.push_back(by_word);
    }
    {
        std::vector<std::pair<size_t, size_t>> blocked;
        for (size_t bb = 0; bb < kMixedWords; bb += kColumnBlockWords) {
            blocked.emplace_back(bb, std::min(bb + kColumnBlockWords, kMixedWords));
        }
        splits.push_back(blocked);
    }
    // Boundaries hand-placed to straddle the seal, wherever the chunk height puts it.
    splits.push_back({{0, kChunkWords - 4}, {kChunkWords - 4, kChunkWords + 6}, {kChunkWords + 6, kMixedWords}});
    {
        // Randomised ragged blocks, so no hand-picked boundary can be the only one tested.
        std::mt19937 rng(20260813);
        std::vector<std::pair<size_t, size_t>> ragged;
        size_t bb = 0;
        while (bb < kMixedWords) {
            const size_t be = std::min(bb + 1 + rng() % (kChunkWords + 188), kMixedWords);
            ragged.emplace_back(bb, be);
            bb = be;
        }
        splits.push_back(ragged);
    }

    for (const auto &cols : generators) {
        const auto expected = oracle_fold(op, cols);
        for (const auto &split : splits) {
            // Pre-dirtied: the kernel seeds, never accumulates.
            std::vector<uint64_t> got(kMixedWords, 0xdeadbeefULL);
            for (const auto [bb, be] : split) {
                combine_columns_block<N>(sc, cols, got.data() + bb, bb, be);
            }
            BOOST_TEST(got == expected, boost::test_tools::per_element());
        }
    }
}

// xor_columns_block is combine_columns_block's accumulating form, and `skip` is what lets the scan fold
// a generator whose pivot column it has already expanded on its own. Seeding from the pivot and XORing
// the rest with that entry skipped must equal folding the whole generator in one go.
BOOST_AUTO_TEST_CASE(xor_columns_block_skip_matches_a_whole_fold) {
    const auto op = mixed_operator();
    Sc sc;
    sc.rebuild(op);

    const std::vector<size_t> cols{col_of(0), col_of(1), col_of(2)};
    const auto expected = oracle_fold(op, cols);

    for (size_t pivot_at = 0; pivot_at < cols.size(); ++pivot_at) {
        std::vector<uint64_t> got(kMixedWords, 0xdeadbeefULL);
        for (size_t bb = 0; bb < kMixedWords; bb += kColumnBlockWords) {
            const size_t be = std::min(bb + kColumnBlockWords, kMixedWords);
            const size_t pivot = cols[pivot_at];
            combine_columns_block<N>(sc, std::span<const size_t>(&pivot, 1), got.data() + bb, bb, be);
            xor_columns_block<N>(sc, cols, got.data() + bb, bb, be, pivot_at);
        }
        BOOST_TEST_CONTEXT("pivot at " << pivot_at) {
            BOOST_TEST(got == expected, boost::test_tools::per_element());
        }
    }
}

// A tiny operator that never fills a chunk: everything lives in the tail, which is the state the fold
// sees for most of a layer build.
BOOST_AUTO_TEST_CASE(combine_columns_block_folds_an_unsealed_tail) {
    constexpr size_t kR = 300;
    std::vector<MSet> op;
    op.reserve(kR);
    std::vector<bool> in_a(kR, false), in_b(kR, false), in_sparse(kR, false);
    for (size_t i = 0; i < kR; ++i) {
        VecZ pos;
        if (i % 3 == 0) {
            pos.push_back(0);
            in_a[i] = true;
        }
        if (i % 7 == 0) {
            pos.push_back(1);
            in_b[i] = true;
        }
        if (i == 5 || i == 200) {
            pos.push_back(2);
            in_sparse[i] = true;
        }
        if (pos.empty()) {
            pos.push_back(3); // keep every row non-empty
        }
        op.push_back(bs(pos));
    }
    Sc sc;
    sc.rebuild(op);
    BOOST_REQUIRE_EQUAL(sc.sealed_chunks(), 0U);
    // Nothing here ever seals, which is the state a small partition stays in for its whole life -- so
    // the tail must reach both of its containers on its own. Mode 0 is set every third row and takes a
    // bitmap; mode 2 has two postings in 300 rows and stays a delta stream.
    BOOST_REQUIRE((sc.container_tag(col_of(0), 0) == ChunkTag::Bitmap));
    BOOST_REQUIRE((sc.container_tag(col_of(2), 0) == ChunkTag::U8Delta));

    const size_t words = (kR + 63) / 64;
    const std::vector<size_t> cols{col_of(0), col_of(1), col_of(2)};

    std::vector<uint64_t> expected(words, 0);
    for (size_t r = 0; r < kR; ++r) {
        if (in_a[r] != (in_b[r] != in_sparse[r])) { // three-way XOR
            expected[r >> 6] |= uint64_t{1} << (r & 63U);
        }
    }

    std::vector<uint64_t> whole(words, 0xdeadbeefULL);
    combine_columns_block<N>(sc, cols, whole.data(), 0, words);
    BOOST_TEST(whole == expected, boost::test_tools::per_element());

    std::vector<uint64_t> pieced(words, 0xdeadbeefULL);
    for (size_t w = 0; w < words; ++w) {
        combine_columns_block<N>(sc, cols, pieced.data() + w, w, w + 1);
    }
    BOOST_TEST(pieced == expected, boost::test_tools::per_element());
}
