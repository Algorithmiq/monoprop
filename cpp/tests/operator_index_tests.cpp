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
using Store = OperatorIndex;
constexpr size_t kBits = 2 * N; // the store is runtime-width now
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
    Store s(kBits);
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
    Store s(kBits);
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
    Store s(kBits, 4);             // stride = 1 + 4, fixed at construction
    s.push_back(bs({0, 2, 4, 6})); // a 4-position row fits inline at width 4
    s.reserve(20);                 // capacity only -- width/stride are never touched by reserve
    BOOST_TEST(s.popcount(0) == 4u);
    BOOST_TEST((s.row(0) == bs({0, 2, 4, 6})));
}

BOOST_AUTO_TEST_CASE(overflow_is_lossless_above_width) {
    Store s(kBits, 2); // width 2; a 3-position row must overflow
    s.push_back(bs({0, 1, 2}));
    BOOST_TEST(s.popcount(0) == 3u); // popcount recovered from the overflow map
    BOOST_TEST((s.row(0) == bs({0, 1, 2})));
}

BOOST_AUTO_TEST_CASE(index_survives_rehash_in_place) {
    Store a(kBits);
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
    Store a(kBits, 4); // non-default width must carry over
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
    Store a(kBits, 2); // width 2; a 3-position row overflows losslessly
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
    Store s(kBits);
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
    Store s(kBits);
    const std::array<MSet, 3> keys{bs({0, 3}), bs({1, 2}), bs({4, 5, 6})};
    std::array<size_t, 3> out{0, 0, 0};
    s.find_batch(keys.data(), keys.size(), out.data());
    for (size_t i = 0; i < keys.size(); ++i) {
        BOOST_TEST(out[i] == Store::kNotFound);
    }
}
