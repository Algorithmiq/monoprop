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
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;
using namespace monoprop::detail;

BOOST_AUTO_TEST_CASE(operator_index_term_index_width_matches_build) {
#if defined(monoprop_WIDE_TERM_INDEX)
    static_assert(sizeof(TermIndex) == 8, "wide build must use 64-bit TermIndex");
    BOOST_TEST(sizeof(TermIndex) == 8u);
#else
    static_assert(sizeof(TermIndex) == 4, "default build must use 32-bit TermIndex");
    BOOST_TEST(sizeof(TermIndex) == 4u);
#endif
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

BOOST_AUTO_TEST_CASE(clone_is_deep_and_independent) {
    Store a(4); // non-default width must carry over
    a.push_back(bs({0, 3, 5}));
    a.push_back(bs({1, 2}));

    auto b = a.clone();
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST((b->row(0) == bs({0, 3, 5})));
    BOOST_TEST((b->row(1) == bs({1, 2})));

    a.push_back(bs({6, 7}));
    BOOST_TEST(b->size() == 2u);

    // If the clone still referenced the source's rows, row 0 would now read {8,9}.
    a.set(0, bs({8, 9}));
    BOOST_TEST((b->row(0) == bs({0, 3, 5})));
    BOOST_TEST((a.row(0) == bs({8, 9})));
}

BOOST_AUTO_TEST_CASE(clone_preserves_overflow_rows) {
    Store a(2); // width 2; a 3-position row overflows losslessly
    a.push_back(bs({0, 1, 2}));

    auto b = a.clone();
    BOOST_TEST(b->popcount(0) == 3u);
    BOOST_TEST((b->row(0) == bs({0, 1, 2})));
    BOOST_TEST(b->overflow_size() == 1u);
}

// for_each walks every row in index order: the Python-visible enumeration.
BOOST_AUTO_TEST_CASE(for_each_visits_rows_in_index_order) {
    Store s;
    for (size_t i = 0; i < 40; ++i) {
        s.push_back(bs({i % 62, (i + 7) % 62}));
    }
    size_t expect = 0;
    bool in_order = true;
    s.for_each([&](const MSet &mono, size_t i) {
        in_order = in_order && (i == expect) && (mono == s.row(i));
        ++expect;
    });
    BOOST_TEST(in_order);
    BOOST_TEST(expect == 40u);
}

// row_eq_positions is the confirm behind every fingerprint match: exact on inline rows (popcount first,
// then the positions) and on spilled rows through the dense compare.
BOOST_AUTO_TEST_CASE(row_eq_positions_confirms_exactly) {
    Store s(3);
    s.push_back(bs({0, 3, 5}));    // inline
    s.push_back(bs({0, 1, 2, 4})); // spilled at width 3
    const auto pos_of = [](const MSet &m) {
        std::vector<Store::PosT> pos;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            pos.push_back(static_cast<Store::PosT>(b));
        }
        return pos;
    };
    BOOST_TEST(s.row_eq_positions(0, pos_of(bs({0, 3, 5}))));
    BOOST_TEST(!s.row_eq_positions(0, pos_of(bs({0, 3}))));    // popcount differs
    BOOST_TEST(!s.row_eq_positions(0, pos_of(bs({0, 3, 6})))); // one position differs
    BOOST_TEST(s.row_eq_positions(1, pos_of(bs({0, 1, 2, 4}))));
    BOOST_TEST(!s.row_eq_positions(1, pos_of(bs({0, 1, 2, 5}))));
    BOOST_TEST(!s.row_eq_positions(1, pos_of(bs({0, 3, 5}))));
}

// grow_rows_geometric guards the TermIndex ceiling: it is the only door rows enter through.
BOOST_AUTO_TEST_CASE(grow_rows_geometric_returns_base_and_extends_size) {
    Store s;
    BOOST_TEST(s.grow_rows_geometric(3) == 0u);
    BOOST_TEST(s.size() == 3u);
    BOOST_TEST(s.grow_rows_geometric(2) == 3u);
    BOOST_TEST(s.size() == 5u);
    BOOST_TEST(s.grow_rows_geometric(0) == 5u);
}

// The per-row join key is what BucketJoin stages a gate's anticommuting rows from, so it must agree with
// the fingerprint of the row's own positions no matter which writer wrote the row -- dense set(), packed
// set_positions(), a row overwritten in place, and a spilled row.
BOOST_AUTO_TEST_CASE(operator_index_row_key_matches_the_fingerprint_of_the_row) {
    Store s(3); // inline width 3, so the four-position rows below spill
    const std::array<VecZ, 4> rows{VecZ{0, 3, 5}, VecZ{}, VecZ{1, 2, 4, 7}, VecZ{6}};
    s.grow_rows_geometric(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        s.set(i, bs(rows[i]));
    }
    const auto key_from_dense = [](const MSet &m) { return Store::join_tag(routing::linear_hash<2 * N>(m)); };
    const auto positions_of = [](const MSet &m) {
        std::vector<Store::PosT> pos;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            pos.push_back(static_cast<Store::PosT>(b));
        }
        return pos;
    };
    for (size_t i = 0; i < rows.size(); ++i) {
        BOOST_TEST(s.key(i) == key_from_dense(bs(rows[i])));
        const auto pos = positions_of(bs(rows[i]));
        BOOST_TEST(s.key(i) == Store::key_of_positions(pos.data(), pos.size()));
    }
    // set_positions writes the same key as set() for the same term, on both the inline and spill paths.
    Store t(3);
    t.grow_rows_geometric(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto pos = positions_of(bs(rows[i]));
        t.set_positions(i, std::span<const Store::PosT>(pos));
        BOOST_TEST(t.key(i) == s.key(i));
    }
    // Overwriting a row replaces its key; a clone carries the keys across.
    t.set(0, bs({2, 9}));
    BOOST_TEST(t.key(0) == key_from_dense(bs({2, 9})));
    const auto copy = t.clone();
    for (size_t i = 0; i < rows.size(); ++i) {
        BOOST_TEST(copy->key(i) == t.key(i));
    }
    // push_back goes through the same door, and the keys are priced on their own.
    t.push_back(bs({0, 1}));
    BOOST_TEST(t.key(t.size() - 1) == key_from_dense(bs({0, 1})));
    BOOST_TEST(t.row_keys_bytes() >= t.size() * sizeof(uint32_t));
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
    // Rows at and next to every chunk boundary get a fourth position, which spills at inline width 3.
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
        BOOST_CHECK_EQUAL(s.key(i), Store::join_tag(routing::linear_hash<2 * N>(want)));

        std::vector<size_t> seen;
        s.for_each_position(i, [&](size_t b) { seen.push_back(b); });
        std::vector<size_t> expected;
        for (size_t b = want.find_first(); b < want.size(); b = want.find_next(b)) {
            expected.push_back(b);
        }
        BOOST_CHECK_EQUAL_COLLECTIONS(seen.begin(), seen.end(), expected.begin(), expected.end());

        // An inline row's span is the row and nothing more: it must lie inside one chunk, which is what
        // "a row never straddles" buys the callers that hold these pointers.
        const auto rp = s.row_positions(i);
        if (rp.inlined()) {
            BOOST_CHECK_EQUAL(rp.pos.size(), want.count());
            BOOST_CHECK(s.row_eq_positions(i, rp.pos));
        }
        else {
            BOOST_CHECK_GT(want.count(), 3U); // only a spilled row has no span
        }
    }
}

// The confirm behind every fingerprint match, at the rows most likely to be resolved to the wrong chunk.
BOOST_AUTO_TEST_CASE(row_eq_positions_holds_at_chunk_boundaries) {
    Store s(3, kTinyChunkRows);
    fill_boundary_store(s);
    const std::vector<size_t> edges{0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 191, 192, 193, 199};
    for (const size_t i : edges) {
        BOOST_TEST_INFO("row " << i);
        std::vector<Store::PosT> pos;
        const MSet want = boundary_term(i);
        for (size_t b = want.find_first(); b < want.size(); b = want.find_next(b)) {
            pos.push_back(static_cast<Store::PosT>(b));
        }
        BOOST_CHECK(s.row_eq_positions(i, std::span<const Store::PosT>(pos)));
        // A neighbour's positions must not confirm, or the chunk arithmetic is off by a row.
        const MSet other = boundary_term((i + 1) % kBoundaryTestRows);
        if (other != want) {
            std::vector<Store::PosT> other_pos;
            for (size_t b = other.find_first(); b < other.size(); b = other.find_next(b)) {
                other_pos.push_back(static_cast<Store::PosT>(b));
            }
            BOOST_CHECK(!s.row_eq_positions(i, std::span<const Store::PosT>(other_pos)));
        }
    }
}

// The hoist the chunking exists to make cheap: one chunk lookup per 64-row window, used by
// BucketJoin::stage_rows and by the scan's orbital-gate loop. It has to agree with the per-row door.
BOOST_AUTO_TEST_CASE(row_block_agrees_with_the_per_row_accessors) {
    Store s(3, kTinyChunkRows);
    fill_boundary_store(s);
    for (size_t first = 0; first < kBoundaryTestRows; first += 64) {
        const auto block = s.row_block(first);
        for (size_t i = first; i < std::min(first + 64, kBoundaryTestRows); ++i) {
            BOOST_TEST_INFO("row " << i);
            BOOST_CHECK_EQUAL(Store::block_key(block, i), s.key(i));
            const Store::PosT *const row = Store::block_row(block, i);
            const auto rp = s.row_positions(i);
            if (rp.inlined()) {
                BOOST_CHECK_EQUAL(static_cast<size_t>(row[0]), rp.pos.size());
                BOOST_CHECK(row + 1 == rp.pos.data());
            }
            else {
                BOOST_CHECK_EQUAL(row[0], Store::kOverflowMarker);
            }
        }
    }
}

// The chunk length is a storage decision and nothing else: the same writes must produce the same store,
// row for row and key for key, at 64 rows per chunk and at the production ceiling of 2^18.
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
        BOOST_CHECK_EQUAL(tiny.key(i), production.key(i));
        BOOST_CHECK_EQUAL(tiny.popcount(i), production.popcount(i));
        const auto rp = production.row_positions(i);
        BOOST_CHECK_EQUAL(tiny.row_positions(i).inlined(), rp.inlined());
        if (rp.inlined()) {
            BOOST_CHECK(tiny.row_eq_positions(i, rp.pos));
        }
    }
    // And the chunk-by-chunk walk keeps index order across boundaries.
    std::vector<size_t> order_tiny;
    std::vector<size_t> order_prod;
    tiny.for_each([&](const MSet &, size_t i) { order_tiny.push_back(i); });
    production.for_each([&](const MSet &, size_t i) { order_prod.push_back(i); });
    BOOST_CHECK_EQUAL_COLLECTIONS(order_tiny.begin(), order_tiny.end(), order_prod.begin(), order_prod.end());
    BOOST_CHECK_EQUAL(order_tiny.size(), kBoundaryTestRows);
}

// The chunk length a store of a given height asks for. A quarter of the height rounded down to a power
// of two, clamped to [2^12, 2^18]: flat at the floor until the height clears four minimum chunks, then
// stepping by powers of two, then flat at the ceiling from 2^20 rows on.
BOOST_AUTO_TEST_CASE(chunk_rows_are_selected_from_the_row_count) {
    const std::vector<std::pair<size_t, size_t>> table = {
        {0, Store::kMinRowsPerChunk},
        {1, Store::kMinRowsPerChunk},
        {45'471, Store::kMinRowsPerChunk * 2}, // the 98 B/term case: 11 367 -> 8192
        {Store::kMinRowsPerChunk * 4 - 1, Store::kMinRowsPerChunk},
        {Store::kMinRowsPerChunk * 4, Store::kMinRowsPerChunk},
        {Store::kMinRowsPerChunk * 8, Store::kMinRowsPerChunk * 2},
        {1U << 20U, Store::kMaxRowsPerChunk},
        {9'259'094, Store::kMaxRowsPerChunk}, // the reproducer: already at the ceiling, unchanged
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
// in millions. Read through slack_bytes(), which is (capacity - size) * stride and so reports the chunk
// geometry exactly; at inline width 3 a row is 4 bytes.
BOOST_AUTO_TEST_CASE(the_chunk_length_follows_the_row_count) {
    constexpr size_t kStrideBytes = 4 * sizeof(Store::PosT);

    // A store nobody reserved holds nothing at all, and takes the floor on its first growth.
    Store s(3);
    BOOST_CHECK_EQUAL(s.memory_bytes(), 0U);
    BOOST_CHECK_EQUAL(s.slack_bytes(), 0U);
    BOOST_CHECK_EQUAL(s.row_keys_bytes(), 0U);
    s.grow_rows_geometric(100);
    BOOST_CHECK_EQUAL(s.slack_bytes(), (Store::kMinRowsPerChunk - 100) * kStrideBytes);

    // Rows written before a migration read back through it: the geometry moves, the rows do not.
    for (size_t i = 0; i < 100; ++i) {
        s.set(i, boundary_term(i));
    }
    // Past four floor chunks the length steps up, and the store re-lays itself rather than keeping a
    // length that no longer matches its height.
    s.grow_rows_geometric(8 * Store::kMinRowsPerChunk - 100);
    BOOST_CHECK_EQUAL(Store::chunk_rows_for_rows(s.size()), Store::kMinRowsPerChunk * 2);
    BOOST_CHECK_EQUAL(s.slack_bytes() % (Store::kMinRowsPerChunk * 2 * kStrideBytes), 0U);
    BOOST_CHECK_LE(s.slack_bytes(), Store::kMinRowsPerChunk * 2 * kStrideBytes);
    for (size_t i = 0; i < 100; ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_CHECK(s.row(i) == boundary_term(i));
        BOOST_CHECK_EQUAL(s.key(i), Store::key_of(boundary_term(i)));
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

// The point of the lever: growth appends chunks instead of reallocating, so the store never holds more
// spare than one chunk's tail -- where the 1.5x reserve held up to half the operator, and held the old
// buffer alongside the new one while it copied.
BOOST_AUTO_TEST_CASE(chunked_growth_bounds_the_slack_by_one_chunk) {
    Store s(3, kTinyChunkRows);
    fill_boundary_store(s);
    const size_t stride_bytes = 4 * sizeof(Store::PosT); // 1 popcount byte + 3 inline positions
    BOOST_CHECK_LT(s.slack_bytes(), kTinyChunkRows * stride_bytes);

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
        BOOST_CHECK_EQUAL(copy->key(i), s.key(i));
    }
    s.set(65, bs({0, 2, 4})); // a row in the second chunk
    BOOST_CHECK(copy->row(65) == boundary_term(65));
}
