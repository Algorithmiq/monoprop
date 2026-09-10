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
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;
using namespace monoprop::detail;

BOOST_AUTO_TEST_CASE(operator_index_term_index_is_32_bit) {
    static_assert(sizeof(TermIndex) == 4, "TermIndex is fixed at 32 bits; the packed row stores assume it");
    BOOST_TEST(sizeof(TermIndex) == 4u);
}

namespace {
constexpr size_t N = 32;
using Store = OperatorIndex<N>;
using MSet = Monomial<N>;

// Owners hold the store by unique_ptr and share stable pointers into it, so it must stay
// non-copyable and non-movable; clone() is the only deep copy.
static_assert(!std::is_move_constructible_v<Store>, "OperatorIndex must remain non-movable");
static_assert(!std::is_copy_constructible_v<Store>, "OperatorIndex must remain non-copyable");

MSet bs(const VecZ &r) {
    return indices_to_bitset<N>(r);
}
} // namespace

BOOST_AUTO_TEST_CASE(rows_roundtrip_dense_popcount_positions) {
    Store s;
    s.push_back(bs({0, 3, 5}));
    s.push_back(bs({1, 2}));
    BOOST_TEST(s.size() == 2u);
    BOOST_TEST(s.popcount(0) == 3u);
    BOOST_TEST(s.popcount(1) == 2u);
    BOOST_TEST((s.row(0) == bs({0, 3, 5})));
    std::vector<size_t> pos;
    s.for_each_position(0, [&](size_t b) { pos.push_back(b); });
    BOOST_TEST(pos.size() == 3u);
    // for_each_position yields raw bit positions (ascending). indices_to_bitset<32>({0,3,5})
    // sets bits at 2*32-1-0=63, 2*32-1-3=60, 2*32-1-5=58, so find_first gives 58 first.
    BOOST_TEST(pos[0] == 58u);
    BOOST_TEST(pos[2] == 63u);
}

// The ceiling is declared, never materialised: 2^32 rows is hundreds of GiB, so every case here is
// arithmetic on the refusal path, which returns before a single row is allocated.
BOOST_AUTO_TEST_CASE(append_past_the_term_index_ceiling_is_refused_before_it_grows) {
    Store s;
    BOOST_CHECK_THROW(s.grow_rows_geometric(Store::kIndexCeiling + 1), TermIndexCeilingReached);
    BOOST_CHECK_EQUAL(s.size(), 0u);

    s.push_back(bs({0, 1}));
    // A store holding one term has room for kIndexCeiling - 1 more, so exactly kIndexCeiling is one too many.
    BOOST_CHECK_THROW(s.grow_rows_geometric(Store::kIndexCeiling), TermIndexCeilingReached);
    BOOST_CHECK_EQUAL(s.size(), 1u);
    BOOST_CHECK((s.row(0) == bs({0, 1})));

    // The count is checked as a subtraction, so a request that would wrap base + n is refused too.
    BOOST_CHECK_THROW(s.grow_rows_geometric(std::numeric_limits<size_t>::max()), TermIndexCeilingReached);
    BOOST_CHECK_EQUAL(s.size(), 1u);
}

// An empty append is legal at any size, including at the ceiling itself -- the last valid index is
// kIndexCeiling - 1, so a store of exactly kIndexCeiling terms is full, not over.
BOOST_AUTO_TEST_CASE(empty_append_never_throws_and_a_normal_append_still_grows) {
    Store s;
    BOOST_CHECK_NO_THROW(s.grow_rows_geometric(0));
    BOOST_CHECK_EQUAL(s.size(), 0u);
    s.push_back(bs({2}));
    BOOST_CHECK_NO_THROW(s.grow_rows_geometric(0));
    BOOST_CHECK_EQUAL(s.size(), 1u);
    const size_t base = s.grow_rows_geometric(2);
    BOOST_CHECK_EQUAL(base, 1u);
    BOOST_CHECK_EQUAL(s.size(), 3u);
}

BOOST_AUTO_TEST_CASE(index_emplace_then_find_roundtrip) {
    Store s;
    s.push_back(bs({0, 3, 5}));
    s.emplace(bs({0, 3, 5}), 0);
    s.push_back(bs({1, 2}));
    s.emplace(bs({1, 2}), 1);
    auto f = s.find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);
    BOOST_TEST(!s.find(bs({7, 9})).has_value());
}

BOOST_AUTO_TEST_CASE(width_is_a_construction_invariant) {
    Store s(4);                    // stride = 1 + 4, fixed at construction
    s.push_back(bs({0, 2, 4, 6})); // a 4-position row fits inline at width 4
    s.reserve(20);                 // capacity only -- width/stride are never touched by reserve
    BOOST_TEST(s.popcount(0) == 4u);
    BOOST_TEST((s.row(0) == bs({0, 2, 4, 6})));
}

BOOST_AUTO_TEST_CASE(overflow_is_lossless_above_width) {
    Store s(2); // width 2; a 3-position row must overflow
    s.push_back(bs({0, 1, 2}));
    BOOST_TEST(s.popcount(0) == 3u); // popcount recovered from the overflow map
    BOOST_TEST((s.row(0) == bs({0, 1, 2})));
}

BOOST_AUTO_TEST_CASE(index_survives_rehash_in_place) {
    Store a;
    // 64 distinct rows (positions i and (i+7)%62) force at least one rehash of the in-place index.
    for (int i = 0; i < 64; ++i) {
        a.push_back(bs({static_cast<size_t>(i % 62), static_cast<size_t>((i + 7) % 62)}));
        a.emplace(a.row(static_cast<size_t>(i)), static_cast<size_t>(i));
    }
    auto f = a.find(a.row(50));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 50u);
}

BOOST_AUTO_TEST_CASE(clone_is_deep_and_independent) {
    Store a(4); // non-default width must carry over
    a.push_back(bs({0, 3, 5}));
    a.emplace(bs({0, 3, 5}), 0);
    a.push_back(bs({1, 2}));
    a.emplace(bs({1, 2}), 1);

    auto b = a.clone();
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST((b->row(0) == bs({0, 3, 5})));
    auto f = b->find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);

    a.push_back(bs({6, 7}));
    a.emplace(bs({6, 7}), 2);
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST(!b->find(bs({6, 7})).has_value());

    // If the clone still referenced the source's rows, this find would read a->row(0) (now {8,9})
    // and fail.
    a.set(0, bs({8, 9}));
    auto g = b->find(bs({0, 3, 5}));
    BOOST_TEST(g.has_value());
    BOOST_TEST(*g == 0u);
}

BOOST_AUTO_TEST_CASE(clone_preserves_overflow_rows) {
    Store a(2); // width 2; a 3-position row overflows losslessly
    a.push_back(bs({0, 1, 2}));
    a.emplace(bs({0, 1, 2}), 0);

    auto b = a.clone();
    BOOST_TEST(b->popcount(0) == 3u);
    BOOST_TEST((b->row(0) == bs({0, 1, 2})));
    BOOST_TEST(*b->find(bs({0, 1, 2})) == 0u);
}

// find_batch (the group-prefetch pipelined lookup) must be semantically identical to n independent
// find() calls. The query mix below spans several G=16 groups plus a short tail and interleaves
// present and absent keys, so every branch but the h32-collision fallback runs; that one needs a
// real 32-bit hash collision, but the equivalence assertion pins it whichever path a key takes.
BOOST_AUTO_TEST_CASE(find_batch_matches_scalar_find) {
    Store s;
    constexpr size_t kRows = 200; // > 12 groups of G=16
    // (i/60, 4 + i%60) is a bijection for i < 240 over the disjoint ranges {0..3} and {4..63}.
    for (size_t i = 0; i < kRows; ++i) {
        const auto key = bs({i / 60, 4 + (i % 60)});
        s.push_back(key);
        s.emplace(key, i);
    }

    std::vector<MSet> queries;
    for (size_t i = 0; i < kRows; ++i) {
        queries.push_back(bs({i / 60, 4 + (i % 60)}));
        queries.push_back(bs({0, 1, 2 + (i % 20)}));
    }
    queries.push_back(bs({0, 1, 2})); // 401 total
    BOOST_TEST(queries.size() % 16u != 0u);

    std::vector<size_t> out(queries.size(), 424242);
    s.find_batch(queries.data(), queries.size(), out.data());

    bool all_match = true;
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto scalar = s.find(queries[i]);
        const size_t expected = scalar ? *scalar : Store::kNotFound;
        if (out[i] != expected) {
            all_match = false;
        }
    }
    BOOST_TEST(all_match);
    BOOST_TEST(out[0] == 0u);               // first present key -> row 0
    BOOST_TEST(out[1] == Store::kNotFound); // first absent key
}

// Pins find_batch's partition.count == 0 early-out.
BOOST_AUTO_TEST_CASE(find_batch_on_empty_store_is_all_missing) {
    Store s;
    const std::array<MSet, 3> keys{bs({0, 3}), bs({1, 2}), bs({4, 5, 6})};
    std::array<size_t, 3> out{0, 0, 0};
    s.find_batch(keys.data(), keys.size(), out.data());
    for (size_t i = 0; i < keys.size(); ++i) {
        BOOST_TEST(out[i] == Store::kNotFound);
    }
}

namespace {

// The smallest legal chunk: 64 rows, so a few hundred rows cross several boundaries. Production sizes
// its chunks from the store's height and tops out at Store::kMaxRowsPerChunk, which no unit test can
// afford to cross.
constexpr size_t kTinyChunkRows = 64;
constexpr size_t kBoundaryTestRows = 200;

// Distinct terms, with the four-position ones landing on and around the chunk boundaries so the spill
// path is exercised exactly where a row changes chunk.
auto boundary_term(size_t i) -> MSet {
    const size_t a = i % 61;
    const size_t b = (i * 7) % 59;
    const size_t c = (i * 13) % 53;
    VecZ pos{a};
    if (b != a) {
        pos.push_back(b);
    }
    if (c != a && c != b) {
        pos.push_back(c);
    }
    const size_t off = i % kTinyChunkRows;
    if (off == 0 || off == 1 || off == kTinyChunkRows - 1) {
        for (size_t extra = 17; extra < 2 * N; ++extra) {
            if (extra != a && extra != b && extra != c) {
                pos.push_back(extra);
                break;
            }
        }
    }
    std::sort(pos.begin(), pos.end());
    return bs(pos);
}

auto fill_boundary_store(Store &s) -> void {
    s.grow_rows_geometric(kBoundaryTestRows);
    for (size_t i = 0; i < kBoundaryTestRows; ++i) {
        s.set(i, boundary_term(i));
    }
}

} // namespace

// Every read path has to keep working when the row it wants is in a different chunk from the one before
// it -- and a row must never straddle, or the span row_positions() hands out would run off the end of a
// chunk into unrelated memory.
BOOST_AUTO_TEST_CASE(chunked_rows_read_back_across_chunk_boundaries) {
    Store s(3, kTinyChunkRows); // inline width 3, so the boundary rows spill
    fill_boundary_store(s);
    BOOST_REQUIRE_GT(kBoundaryTestRows, 3 * kTinyChunkRows); // really is multi-chunk
    BOOST_REQUIRE_GT(s.overflow_size(), 0U);                 // and really does spill

    for (size_t i = 0; i < kBoundaryTestRows; ++i) {
        BOOST_TEST_INFO("row " << i);
        const MSet want = boundary_term(i);
        BOOST_CHECK(s.row(i) == want);
        BOOST_CHECK_EQUAL(s.popcount(i), want.count());

        std::vector<size_t> seen;
        s.for_each_position(i, [&](size_t b) { seen.push_back(b); });
        std::vector<size_t> expected;
        for (size_t b = want.find_first(); b < want.size(); b = want.find_next(b)) {
            expected.push_back(b);
        }
        BOOST_CHECK_EQUAL_COLLECTIONS(seen.begin(), seen.end(), expected.begin(), expected.end());

        // An inline row's span is the row and nothing more, so it lies inside one chunk.
        const auto rp = s.row_positions(i);
        if (rp.inlined()) {
            BOOST_CHECK_EQUAL(rp.pos.size(), want.count());
            MSet rebuilt;
            for (const auto b : rp.pos) {
                rebuilt.set(b);
            }
            BOOST_CHECK(rebuilt == want);
        }
        else {
            BOOST_CHECK_GT(want.count(), 3U); // only a spilled row has no span
        }
    }
}

// The index has to keep finding rows that moved chunk, both by value and through the batch pipeline:
// a chunk lookup off by a row would confirm against a neighbour.
BOOST_AUTO_TEST_CASE(the_index_finds_rows_in_every_chunk) {
    Store s(3, kTinyChunkRows);
    fill_boundary_store(s);
    for (size_t i = 0; i < kBoundaryTestRows; ++i) {
        s.emplace(boundary_term(i), i);
    }

    std::vector<MSet> keys;
    std::vector<size_t> want;
    for (size_t i = 0; i < kBoundaryTestRows; ++i) {
        keys.push_back(boundary_term(i));
        want.push_back(i);
        keys.push_back(bs({0, 1, 2, 4, 6, 8})); // absent at inline width 3: a spilled query
        want.push_back(Store::kNotFound);
    }
    std::vector<size_t> out(keys.size(), 424242);
    s.find_batch(keys.data(), keys.size(), out.data());
    for (size_t q = 0; q < keys.size(); ++q) {
        BOOST_TEST_INFO("query " << q);
        const auto scalar = s.find(keys[q]);
        BOOST_CHECK_EQUAL(out[q], scalar ? *scalar : Store::kNotFound);
        if (want[q] != Store::kNotFound) {
            BOOST_CHECK_EQUAL(out[q], want[q]);
        }
        else {
            BOOST_CHECK_EQUAL(out[q], Store::kNotFound);
        }
    }
}

// The chunk length is a storage decision and nothing else: the same writes must produce the same store,
// row for row, at 64 rows per chunk and at the production ceiling of 2^18.
BOOST_AUTO_TEST_CASE(chunk_size_does_not_change_the_store) {
    Store tiny(3, kTinyChunkRows);
    Store production(3, Store::kMaxRowsPerChunk);
    fill_boundary_store(tiny);
    fill_boundary_store(production);

    BOOST_REQUIRE_EQUAL(tiny.size(), production.size());
    BOOST_CHECK_EQUAL(tiny.overflow_size(), production.overflow_size());
    for (size_t i = 0; i < tiny.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(tiny.row(i) == production.row(i));
        BOOST_CHECK_EQUAL(tiny.popcount(i), production.popcount(i));
        const auto rp = production.row_positions(i);
        const auto tp = tiny.row_positions(i);
        BOOST_CHECK_EQUAL(tp.inlined(), rp.inlined());
        if (rp.inlined()) {
            BOOST_CHECK_EQUAL_COLLECTIONS(tp.pos.begin(), tp.pos.end(), rp.pos.begin(), rp.pos.end());
        }
    }
}

// The chunk length a store of a given height asks for: a quarter of the height rounded down to a power
// of two, clamped to [2^12, 2^18].
BOOST_AUTO_TEST_CASE(chunk_rows_are_selected_from_the_row_count) {
    const std::vector<std::pair<size_t, size_t>> table = {
        {0, Store::kMinRowsPerChunk},
        {1, Store::kMinRowsPerChunk},
        {Store::kMinRowsPerChunk * 4 - 1, Store::kMinRowsPerChunk},
        {Store::kMinRowsPerChunk * 4, Store::kMinRowsPerChunk},
        {Store::kMinRowsPerChunk * 8, Store::kMinRowsPerChunk * 2},
        {1U << 20U, Store::kMaxRowsPerChunk},
        {9'259'094, Store::kMaxRowsPerChunk},
    };
    for (const auto &[rows, want] : table) {
        BOOST_TEST_INFO("rows " << rows);
        BOOST_CHECK_EQUAL(Store::chunk_rows_for_rows(rows), want);
    }
    // Every length is a power of two and a multiple of 64, so a 64-row window never straddles a chunk.
    for (size_t rows = 1; rows < (size_t{1} << 22U); rows *= 3) {
        const size_t c = Store::chunk_rows_for_rows(rows);
        BOOST_TEST_INFO("rows " << rows);
        BOOST_CHECK(std::has_single_bit(c));
        BOOST_CHECK_EQUAL(c % 64, 0U);
        // The bound the selector exists for: the tail is under a quarter of the store, or under one
        // minimum chunk while the store is smaller than four of those.
        BOOST_CHECK_LT(c, std::max(Store::kMinRowsPerChunk + 1, rows / 4 + 1));
    }
}

// The chunk length follows the store's height as it grows, because nobody can tell it that height up
// front: the propagator reserves the initial operator's size, a handful of terms even for a run ending
// in millions. Read through slack_bytes(), which is (capacity - size) * stride.
BOOST_AUTO_TEST_CASE(the_chunk_length_follows_the_row_count) {
    constexpr size_t kStrideBytes = 4 * sizeof(Store::PosT); // 1 popcount slot + 3 inline positions

    // A store nobody reserved holds nothing at all, and takes the floor on its first growth.
    Store s(3);
    BOOST_CHECK_EQUAL(s.memory_bytes(), 0U);
    BOOST_CHECK_EQUAL(s.slack_bytes(), 0U);
    s.grow_rows_geometric(100);
    BOOST_CHECK_EQUAL(s.slack_bytes(), (Store::kMinRowsPerChunk - 100) * kStrideBytes);

    // Rows written before a migration read back through it: the geometry moves, the rows do not.
    for (size_t i = 0; i < 100; ++i) {
        s.set(i, boundary_term(i));
    }
    s.grow_rows_geometric(8 * Store::kMinRowsPerChunk - 100);
    BOOST_CHECK_EQUAL(Store::chunk_rows_for_rows(s.size()), Store::kMinRowsPerChunk * 2);
    BOOST_CHECK_EQUAL(s.slack_bytes() % (Store::kMinRowsPerChunk * 2 * kStrideBytes), 0U);
    BOOST_CHECK_LE(s.slack_bytes(), Store::kMinRowsPerChunk * 2 * kStrideBytes);
    for (size_t i = 0; i < 100; ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(s.row(i) == boundary_term(i));
    }

    // A forced length never moves, whatever the store grows to.
    Store pinned(3, kTinyChunkRows);
    pinned.grow_rows_geometric(64 * kTinyChunkRows);
    BOOST_CHECK_EQUAL(pinned.slack_bytes(), 0U);

    // A clone of a store that never grew is empty too, and one of a grown store keeps its geometry.
    BOOST_CHECK_EQUAL(Store(3).clone()->memory_bytes(), 0U);
    BOOST_CHECK_EQUAL(s.clone()->slack_bytes(), s.slack_bytes());
    BOOST_CHECK_EQUAL(s.clone()->memory_bytes(), s.memory_bytes());
}

// Growth appends chunks instead of reallocating, so the store never holds more spare than one chunk's
// tail, and the pool's mapping bounds the chunks it has handed out.
BOOST_AUTO_TEST_CASE(chunked_growth_bounds_the_slack_by_one_chunk) {
    Store s(3, kTinyChunkRows);
    fill_boundary_store(s);
    const size_t stride_bytes = 4 * sizeof(Store::PosT);
    BOOST_CHECK_LT(s.slack_bytes(), kTinyChunkRows * stride_bytes);
    BOOST_CHECK_GE(s.pool_mapped_bytes(), s.memory_bytes() - s.overflow_size() * 64);
    BOOST_CHECK_LE(s.pool_free_chunk_bytes(), s.pool_mapped_bytes());

    // A row that lands exactly on a boundary leaves no slack at all.
    Store exact(3, kTinyChunkRows);
    exact.grow_rows_geometric(2 * kTinyChunkRows);
    BOOST_CHECK_EQUAL(exact.slack_bytes(), 0U);
    // And one row past it costs one chunk, not one reallocation of everything.
    exact.grow_rows_geometric(1);
    BOOST_CHECK_EQUAL(exact.slack_bytes(), (kTinyChunkRows - 1) * stride_bytes);

    // A clone takes its own chunks: writing through one must not be visible in the other.
    const auto copy = s.clone();
    BOOST_REQUIRE_EQUAL(copy->size(), s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(copy->row(i) == s.row(i));
    }
    s.set(65, bs({0, 2, 4})); // a row in the second chunk
    BOOST_CHECK(copy->row(65) == boundary_term(65));
}

namespace {

// A term of exactly `slots` ascending positions, distinct per (i, slots).
auto term_of_width(size_t i, size_t slots) -> MSet {
    VecZ pos;
    for (size_t j = 0; j < slots; ++j) {
        pos.push_back((i * 3 + j * 5) % (2 * N));
    }
    std::sort(pos.begin(), pos.end());
    pos.erase(std::unique(pos.begin(), pos.end()), pos.end());
    for (size_t extra = 0; pos.size() < slots && extra < 2 * N; ++extra) {
        if (std::find(pos.begin(), pos.end(), extra) == pos.end()) {
            pos.push_back(extra);
        }
    }
    std::sort(pos.begin(), pos.end());
    return bs(pos);
}

} // namespace

// A row wider than the inline slot but no wider than the structural bound goes to the fixed-stride wide
// tier, not the side-map: it reads back the same through every accessor, and the side-map stays empty.
BOOST_AUTO_TEST_CASE(wide_rows_go_to_the_second_tier_not_the_side_map) {
    constexpr size_t kInline = 6;
    constexpr size_t kBound = 10;
    Store s(kInline, kTinyChunkRows, kBound);
    BOOST_REQUIRE_EQUAL(s.inline_width(), kInline);
    BOOST_REQUIRE_EQUAL(s.wide_width(), kBound);

    // Widths on both sides of the inline slot, and one over the bound so the side-map is still used.
    const std::vector<size_t> widths = {1, kInline - 1, kInline, kInline + 1, kBound - 1, kBound, kBound + 1};
    std::vector<MSet> want;
    for (size_t i = 0; i < widths.size(); ++i) {
        want.push_back(term_of_width(i, widths[i]));
        s.push_back(want.back());
    }
    BOOST_CHECK_EQUAL(s.wide_size(), 3U);     // inline+1, bound-1, bound
    BOOST_CHECK_EQUAL(s.overflow_size(), 1U); // only the row over the bound
    BOOST_CHECK_EQUAL(s.restrides(), 0U);

    for (size_t i = 0; i < widths.size(); ++i) {
        BOOST_TEST_INFO("row " << i << " width " << widths[i]);
        BOOST_CHECK(s.row(i) == want[i]);
        BOOST_CHECK_EQUAL(s.popcount(i), widths[i]);
        // Only the row past the bound loses its position array; a wide row keeps one.
        const auto rp = s.row_positions(i);
        BOOST_CHECK_EQUAL(rp.inlined(), widths[i] <= kBound);
        if (rp.inlined()) {
            BOOST_CHECK_EQUAL(rp.pos.size(), widths[i]);
        }
        std::vector<size_t> seen;
        s.for_each_position(i, [&](size_t b) { seen.push_back(b); });
        BOOST_CHECK_EQUAL(seen.size(), widths[i]);
    }

    // set_positions() takes the same two-tier route as set(), and the index finds either tier.
    Store t(kInline, kTinyChunkRows, kBound);
    t.grow_rows_geometric(widths.size());
    for (size_t i = 0; i < widths.size(); ++i) {
        const auto rp = s.row_positions(i);
        if (rp.inlined()) {
            t.set_positions(i, rp.pos);
        }
        else {
            t.set(i, want[i]);
        }
        t.emplace(want[i], i);
    }
    BOOST_CHECK_EQUAL(t.wide_size(), s.wide_size());
    BOOST_CHECK_EQUAL(t.overflow_size(), s.overflow_size());
    for (size_t i = 0; i < widths.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(t.row(i) == want[i]);
        BOOST_CHECK(t.find(want[i]) == std::optional<size_t>{i});
    }
}

// The policy: a tier under the threshold is left alone; over it the store re-lays itself at the bound,
// once. A store with no tier can never trigger it.
BOOST_AUTO_TEST_CASE(a_store_restrides_once_when_the_wide_tier_grows_past_the_threshold) {
    constexpr size_t kInline = 6;
    constexpr size_t kBound = 10;
    constexpr size_t kRows = 400;

    // One row in fifty is wide: 2 %, under the 3 % threshold, so the guess stands.
    Store lean(kInline, kTinyChunkRows, kBound);
    for (size_t i = 0; i < kRows; ++i) {
        lean.push_back(term_of_width(i, i % 50 == 0 ? kBound : kInline));
    }
    BOOST_CHECK_EQUAL(lean.wide_size(), kRows / 50);
    BOOST_CHECK(!lean.should_restride());

    // One row in ten is wide: over the threshold, so the inline width was the wrong guess.
    Store fat(kInline, kTinyChunkRows, kBound);
    std::vector<MSet> want;
    for (size_t i = 0; i < kRows; ++i) {
        want.push_back(term_of_width(i, i % 10 == 0 ? kBound : kInline));
        fat.push_back(want.back());
        fat.emplace(want.back(), i);
    }
    BOOST_REQUIRE(fat.should_restride());
    // What the index answered before the layout moved. term_of_width repeats a term every 64 rows, so
    // these are not all distinct -- which is beside the point: the claim is that the index answers
    // exactly as it did, whatever it answered.
    std::vector<size_t> found_before;
    for (size_t i = 0; i < kRows; ++i) {
        found_before.push_back(fat.find(want[i]).value_or(Store::kNotFound));
    }

    fat.restride_to_bound();
    BOOST_CHECK_EQUAL(fat.restrides(), 1U);
    BOOST_CHECK_EQUAL(fat.inline_width(), kBound);
    BOOST_CHECK_EQUAL(fat.wide_size(), 0U);
    BOOST_CHECK(!fat.should_restride()); // the tier is gone, so it can never fire again
    BOOST_CHECK_EQUAL(fat.size(), kRows);

    // Index-preserving: every row survives at its own index, and the hash index -- which keys on the
    // row's value, not on its layout -- still resolves each term to the row it was inserted at. This is
    // what lets the inverted index and the graph's endpoints stand across a restride.
    for (size_t i = 0; i < kRows; ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(fat.row(i) == want[i]);
        BOOST_CHECK_EQUAL(fat.popcount(i), i % 10 == 0 ? kBound : kInline);
        BOOST_CHECK_EQUAL(fat.find(want[i]).value_or(Store::kNotFound), found_before[i]);
    }
    // A restride is idempotent, and a store built at its bound has no tier to begin with.
    fat.restride_to_bound();
    BOOST_CHECK_EQUAL(fat.restrides(), 1U);
    Store flat(kBound, kTinyChunkRows, kBound);
    flat.push_back(term_of_width(0, kBound));
    BOOST_CHECK_EQUAL(flat.wide_size(), 0U);
    BOOST_CHECK(!flat.should_restride());
}

// raise_bound follows a cutoff widened after construction: it drains the tier first, so the rows the
// old bound forbade are laid out rather than spilled into the side-map an entry at a time.
BOOST_AUTO_TEST_CASE(raising_the_bound_drains_the_tier_and_widens_it) {
    constexpr size_t kInline = 6;
    constexpr size_t kBound = 8;
    Store s(kInline, kTinyChunkRows, kBound);
    std::vector<MSet> want;
    for (size_t i = 0; i < 40; ++i) {
        want.push_back(term_of_width(i, i % 4 == 0 ? kBound : kInline));
        s.push_back(want.back());
    }
    BOOST_REQUIRE_EQUAL(s.wide_size(), 10U);

    s.raise_bound(12);
    BOOST_CHECK_EQUAL(s.wide_width(), 12U);
    BOOST_CHECK_EQUAL(s.inline_width(), kBound); // the drain re-laid the rows at the old bound
    BOOST_CHECK_EQUAL(s.wide_size(), 0U);
    for (size_t i = 0; i < want.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(s.row(i) == want[i]);
    }
    // A row the old bound would have spilled now takes the fresh tier.
    s.push_back(term_of_width(100, 11));
    BOOST_CHECK_EQUAL(s.wide_size(), 1U);
    BOOST_CHECK_EQUAL(s.overflow_size(), 0U);
    // Narrowing is a no-op: the rows are already laid out wider than that.
    s.raise_bound(2);
    BOOST_CHECK_EQUAL(s.wide_width(), 12U);
}

// A clone carries the tier, not just the narrow rows, and shares no storage with its source.
BOOST_AUTO_TEST_CASE(a_clone_carries_the_wide_tier) {
    constexpr size_t kInline = 6;
    constexpr size_t kBound = 10;
    Store s(kInline, kTinyChunkRows, kBound);
    for (size_t i = 0; i < 40; ++i) {
        s.push_back(term_of_width(i, i % 4 == 0 ? kBound : kInline));
    }
    BOOST_REQUIRE_EQUAL(s.wide_size(), 10U);

    const auto copy = s.clone();
    BOOST_CHECK_EQUAL(copy->wide_size(), s.wide_size());
    BOOST_CHECK_EQUAL(copy->inline_width(), s.inline_width());
    BOOST_CHECK_EQUAL(copy->wide_width(), s.wide_width());
    for (size_t i = 0; i < s.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(copy->row(i) == s.row(i));
    }
    // Restriding the copy must not disturb the original's tier.
    copy->restride_to_bound();
    BOOST_CHECK_EQUAL(copy->wide_size(), 0U);
    BOOST_CHECK_EQUAL(s.wide_size(), 10U);
    for (size_t i = 0; i < s.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(copy->row(i) == s.row(i));
    }
}
