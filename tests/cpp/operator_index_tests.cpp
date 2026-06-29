#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;
using namespace monoprop::detail;

// Preserved from the former index_map_tests.cpp.
BOOST_AUTO_TEST_CASE(operator_index_term_index_width_matches_build) {
#if defined(MONOPROP_WIDE_TERM_INDEX)
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
using MSet = MajoranaSet<N>;

// The store caches a fixed back-pointer to itself in RowEq, so it must never move. Owners that
// relocate a store hold it by unique_ptr. Lock this design invariant at compile time.
static_assert(!std::is_move_constructible_v<Store>, "OperatorIndex must remain non-movable");
static_assert(!std::is_copy_constructible_v<Store>, "OperatorIndex must remain non-copyable");

MSet bs(const VecZ &r) { return indices_to_bitset<N>(r); }
} // namespace

BOOST_AUTO_TEST_CASE(rows_roundtrip_dense_popcount_positions) {
    Store s;
    s.push_back(bs({0, 3, 5}));
    s.push_back(bs({1, 2}));
    BOOST_TEST(s.size() == 2u);
    BOOST_TEST(s.popcount(0) == 3u);
    BOOST_TEST(s.popcount(1) == 2u);
    BOOST_TEST((s.dense(0) == bs({0, 3, 5})));
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
    BOOST_TEST(s.index_size() == 2u);
    auto f = s.find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);
    BOOST_TEST(!s.find(bs({7, 9})).has_value());
}

BOOST_AUTO_TEST_CASE(width_is_a_construction_invariant) {
    Store s(4);                  // stride = 1 + 4, fixed at construction
    BOOST_TEST(s.inline_width() == 4u);
    s.push_back(bs({0, 2, 4, 6}));
    s.reserve(20);               // capacity only -- width is never touched by reserve
    BOOST_TEST(s.inline_width() == 4u);
}

BOOST_AUTO_TEST_CASE(overflow_is_lossless_above_width) {
    Store s(2);                  // width 2; a 3-position row must overflow
    s.push_back(bs({0, 1, 2}));
    BOOST_TEST(s.overflow_count() == 1u);
    BOOST_TEST(s.popcount(0) == 3u);
    BOOST_TEST((s.dense(0) == bs({0, 1, 2})));
}

// The store is now intentionally non-movable (RowEq holds a fixed back-pointer into the rows,
// so owners hold it by unique_ptr). The former `index_survives_store_move` case exercised a
// move that no longer exists by design; index integrity in its final, stable location is
// covered by the find/emplace round-trip below.
BOOST_AUTO_TEST_CASE(index_survives_rehash_in_place) {
    Store a;
    // Insert 64 distinct rows (varying both positions) to force >=1 rehash in the flat_set.
    // Using i and i+7 (mod 62) as positions; since 64 > 31, we vary the second axis too so
    // all 64 monomials are distinct.
    for (int i = 0; i < 64; ++i) {
        a.push_back(bs({static_cast<size_t>(i % 62), static_cast<size_t>((i + 7) % 62)}));
        a.emplace(a.dense(static_cast<size_t>(i)), static_cast<size_t>(i));
    }
    auto f = a.find(a.dense(50));             // RowEq confirms against a's rows after rehash
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 50u);
}

// clone() is the only deep-copy entry point: the store stays non-copyable/non-movable (the
// static_asserts above), so clone() must hand back a fresh heap store whose index back-pointer
// (RowEq::store) has been repaired to point at the CLONE, not the source.
BOOST_AUTO_TEST_CASE(clone_is_deep_with_local_backpointer) {
    Store a(4);                       // non-default width must carry over
    a.push_back(bs({0, 3, 5}));
    a.emplace(bs({0, 3, 5}), 0);
    a.push_back(bs({1, 2}));
    a.emplace(bs({1, 2}), 1);

    auto b = a.clone();               // std::unique_ptr<Store>
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST(b->index_size() == 2u);
    BOOST_TEST(b->inline_width() == 4u);
    BOOST_TEST((b->dense(0) == bs({0, 3, 5})));
    auto f = b->find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);

    // Deep independence: growing the source must not touch the clone.
    a.push_back(bs({6, 7}));
    a.emplace(bs({6, 7}), 2);
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST(!b->find(bs({6, 7})).has_value());

    // Back-pointer locality: corrupting the SOURCE's row 0 must not perturb the clone's find,
    // which has to confirm against the CLONE's own rows. If clone kept RowEq.store == &a, this
    // find would read a->dense(0) (now {8,9}) and fail.
    a.set(0, bs({8, 9}));
    auto g = b->find(bs({0, 3, 5}));
    BOOST_TEST(g.has_value());
    BOOST_TEST(*g == 0u);
}

BOOST_AUTO_TEST_CASE(clone_preserves_overflow_rows) {
    Store a(2);                       // width 2; a 3-position row overflows losslessly
    a.push_back(bs({0, 1, 2}));
    a.emplace(bs({0, 1, 2}), 0);

    auto b = a.clone();
    BOOST_TEST(b->overflow_count() == 1u);
    BOOST_TEST(b->popcount(0) == 3u);
    BOOST_TEST((b->dense(0) == bs({0, 1, 2})));
    BOOST_TEST(*b->find(bs({0, 1, 2})) == 0u);
}
