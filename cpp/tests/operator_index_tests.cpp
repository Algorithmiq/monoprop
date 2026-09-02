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
#include "monoprop/core/Monomial.h"
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
