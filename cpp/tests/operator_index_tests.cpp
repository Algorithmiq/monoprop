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
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
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

// i -> a distinct 3-position monomial, for i < 8000. This is just i written in base 20 with each
// digit placed in its own disjoint range, so injectivity is by construction and every row fits
// inside the default inline width of 11.
MSet tri(size_t i) {
    return bs({i % 20, 20 + ((i / 20) % 20), 40 + ((i / 400) % 20)});
}

// Same construction with a fourth digit, for i < 32000 -- enough to walk past several row-store chunks.
// Monomial<32> has 64 positions, so the last digit gets the four that are left.
MSet quad(size_t i) {
    return bs({i % 20, 20 + ((i / 20) % 20), 40 + ((i / 400) % 20), 60 + ((i / 8000) % 4)});
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

// Every rehash re-derives each live entry's hash from its row, so rows that live in the lossless
// overflow side-map -- not in the packed inline area -- must survive a rehash too. Width 2 with
// 3-position rows sends every single row through overflow, which the other cases never do at scale.
BOOST_AUTO_TEST_CASE(rehash_preserves_overflow_rows) {
    Store s(2); // width 2; every 3-position row overflows
    constexpr size_t kN = 600;
    for (size_t i = 0; i < kN; ++i) {
        const auto key = tri(i);
        s.push_back(key);
        s.emplace(key, i);
    }
    BOOST_TEST(s.index_entry_count() == kN);
    bool all_found = true;
    for (size_t i = 0; i < kN; ++i) {
        const auto f = s.find(tri(i));
        if (!f.has_value() || *f != i) {
            all_found = false;
        }
    }
    BOOST_TEST(all_found);
    BOOST_TEST(s.popcount(kN - 1) == 3u);
}

// Growing the store through many rehashes must not strand a key. This is the regression guard for
// anything that touches the probe loop or the table's sizing rule: `find` and `find_batch` agree with
// each other and with insertion at every table size the growth path visits, and the load-factor
// ceiling holds throughout. It is deliberately larger than the other cases here (~8 doublings) --
// a probe or wraparound bug typically strands keys near the top of the table, silently, rather than
// failing on a small example. At this size it also drives ~20 fingerprint collisions through
// find_batch's exact-find fallback, a path a handful of keys would essentially never reach.
BOOST_AUTO_TEST_CASE(probing_is_correct_across_rehashes) {
    Store s;
    constexpr size_t kN = 4000;
    for (size_t i = 0; i < kN; ++i) {
        const auto key = tri(i);
        s.push_back(key);
        s.emplace(key, i);
    }
    BOOST_TEST(s.index_entry_count() == kN);

    // The 0.7 load ceiling is what keeps the group-prefetch probe short; pin it rather than trusting
    // the comment on Table.
    BOOST_TEST(s.index_slot_count() * 7u >= kN * 10u);

    bool all_found = true;
    for (size_t i = 0; i < kN; ++i) {
        const auto f = s.find(tri(i));
        if (!f.has_value() || *f != i) {
            all_found = false;
        }
    }
    BOOST_TEST(all_found);

    // An absent key must terminate on an empty slot rather than wrap forever.
    BOOST_TEST(!s.find(bs({61, 62, 63})).has_value());

    // find_batch walks the same probe loop through its own prefetch pipeline; it must agree.
    std::vector<MSet> q;
    for (size_t i = 0; i < 64; ++i) {
        q.push_back(tri(i * 61)); // strided, so the queries are spread over the whole table
    }
    q.push_back(bs({61, 62, 63}));
    std::vector<size_t> out(q.size(), 0);
    s.find_batch(q.data(), q.size(), out.data());
    bool batch_agrees = true;
    for (size_t i = 0; i < q.size(); ++i) {
        const auto scalar = s.find(q[i]);
        if (out[i] != (scalar ? *scalar : Store::kNotFound)) {
            batch_agrees = false;
        }
    }
    BOOST_TEST(batch_agrees);
    BOOST_TEST(out.back() == Store::kNotFound);
}

// The dedup table is the largest structure in the operator after the rows, and its size is entirely
// slots x slot width. The width is 1 fingerprint byte + one TermIndex, interleaved -- widening it back
// to a hash-sized slot would cost ~3 B/term across the whole operator without failing any other test
// here, so pin the per-slot cost directly.
BOOST_AUTO_TEST_CASE(index_slot_costs_one_byte_plus_a_term_index) {
    Store s;
    constexpr size_t kN = 4000;
    for (size_t i = 0; i < kN; ++i) {
        const auto key = tri(i);
        s.push_back(key);
        s.emplace(key, i);
    }
    // Subtract the fixed object so what is left is the slot array alone.
    const size_t table_bytes = s.index_estimated_memory_bytes() - sizeof(Store);
    BOOST_TEST(table_bytes == s.index_slot_count() * (1u + sizeof(TermIndex)));
}

// The rehash rebuilds the table by walking term indices in order rather than old slots in order, via a
// live-index bitmap. That is only equivalent if the bitmap reproduces the live set EXACTLY: a row that
// is indexed but not marked vanishes from the table, and a row marked but never indexed invents an
// entry for a key nobody inserted. Neither shows up as a crash -- only as a silently wrong find() --
// so pin it on a store where the indexed rows are a strict, scattered subset of the rows present.
BOOST_AUTO_TEST_CASE(rehash_rebuilds_exactly_the_indexed_subset) {
    Store s;
    constexpr size_t kN = 3000;
    // Every row is pushed, but only every third is indexed, so index order and slot order differ and
    // the bitmap has to carry the gaps. ~9 doublings over this range.
    for (size_t i = 0; i < kN; ++i) {
        const auto key = tri(i);
        s.push_back(key);
        if (i % 3 == 0) {
            s.emplace(key, i);
        }
    }
    BOOST_TEST(s.size() == kN);
    BOOST_TEST(s.index_entry_count() == (kN + 2) / 3);

    bool exact = true;
    for (size_t i = 0; i < kN; ++i) {
        const auto f = s.find(tri(i));
        const bool want = (i % 3 == 0);
        if (f.has_value() != want || (want && *f != i)) {
            exact = false;
        }
    }
    BOOST_TEST(exact);
}

// The row store's dead capacity is bounded by one chunk no matter how large the operator gets. This is
// what replaced shrink_rows_to_fit: a gated shrink measured `propagate` 1.030x slower on Hubbard
// (6/6, p=0.031) because geometric growth plus shrink-to-fit churns when both run repeatedly, so the
// overshoot is now bounded at the source instead of handed back afterwards.
BOOST_AUTO_TEST_CASE(row_slack_never_exceeds_one_chunk) {
    constexpr size_t kRowBytes = (1 + Store::kDefaultInlinePositions) * sizeof(Store::PosT);
    constexpr size_t kChunkBytes = Store::kChunkRows * kRowBytes;
    // quad() is injective below 32000; that covers three chunks at every swept kChunkRows up to 8192.
    const size_t kStop = std::min<size_t>((3 * Store::kChunkRows) + 7, 32000);
    Store s;
    // push_back writes the row only; the hash index is a separate emplace, exactly as the propagator's
    // setup loop does it. Without the emplace, find() below would dereference an empty optional.
    s.push_back(bs({0, 1}));
    s.emplace(bs({0, 1}), 0);
    s.push_back(bs({2, 3}));
    s.emplace(bs({2, 3}), 1);
    BOOST_TEST(s.slack_bytes() < kChunkBytes);

    // Cross several chunk boundaries: the bound holds at every size, and no row moves while it happens.
    bool bounded = true;
    for (size_t i = 2; i < kStop; ++i) {
        s.push_back(quad(i));
        s.emplace(quad(i), i);
        if (s.slack_bytes() >= kChunkBytes) {
            bounded = false;
        }
    }
    BOOST_TEST(bounded);

    // Rows written before the growth are still readable and still findable.
    BOOST_TEST(s.size() == kStop);
    BOOST_TEST(s.popcount(0) == 2u);
    BOOST_TEST(s.row(0) == bs({0, 1}));
    BOOST_TEST(s.row(1) == bs({2, 3}));
    BOOST_TEST(*s.find(bs({2, 3})) == 1u);
    BOOST_TEST(*s.find(quad(Store::kChunkRows + 1)) == Store::kChunkRows + 1);
    BOOST_TEST(s.row(Store::kChunkRows) == quad(Store::kChunkRows));
}

// An explicit reserve rounds up to whole chunks and nothing beyond, so a caller that knows its final
// size pays no overshoot at all.
BOOST_AUTO_TEST_CASE(reserve_allocates_whole_chunks_and_no_more) {
    Store s;
    s.reserve(Store::kChunkRows + 1);
    BOOST_TEST(s.size() == 0u);
    const size_t rows_held = s.slack_bytes() / ((1 + Store::kDefaultInlinePositions) * sizeof(Store::PosT));
    BOOST_TEST(rows_held == 2 * Store::kChunkRows);
}
