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

// The dense-vector and packed OperatorIndex backends must agree through every RowAccess.h accessor.

#include <boost/test/unit_test.hpp>

#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/RowAccess.h"

using namespace monoprop;

namespace {

template <size_t N>
auto positions_of(const auto &backend, size_t i) -> std::vector<size_t> {
    std::vector<size_t> out;
    for_each_row_position<N>(backend, i, [&](size_t b) { out.push_back(b); });
    return out;
}

template <size_t N>
auto check_backends_agree(const std::vector<std::vector<size_t>> &raw_rows) -> void {
    std::vector<Monomial<N>> dense;
    detail::OperatorIndex<N> packed;
    for (const auto &bits : raw_rows) {
        Monomial<N> m;
        for (size_t b : bits) {
            m.set(b);
        }
        dense.push_back(m);
        packed.push_back(m);
    }

    BOOST_REQUIRE(packed.size() == dense.size());
    for (size_t i = 0; i < dense.size(); ++i) {
        BOOST_TEST((materialize_row<N>(dense, i) == materialize_row<N>(packed, i)));
        BOOST_TEST(row_popcount<N>(dense, i) == row_popcount<N>(packed, i));
        BOOST_TEST(row_popcount<N>(dense, i) == materialize_row<N>(dense, i).count());
        BOOST_TEST(positions_of<N>(dense, i) == positions_of<N>(packed, i));
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(row_accessor_backends_agree_single_word) {
    check_backends_agree<32>({{0, 3, 5}, {1, 2}, {}, {63}, {0, 1, 2, 3, 62, 63}});
}

BOOST_AUTO_TEST_CASE(row_accessor_backends_agree_multi_word) {
    check_backends_agree<96>({{0, 64, 191}, {5, 63, 64, 65}, {}, {128, 190}});
}

BOOST_AUTO_TEST_CASE(row_accessor_assign_row_overwrites) {
    constexpr size_t N = 32;
    std::vector<Monomial<N>> dense;
    detail::OperatorIndex<N> packed;
    Monomial<N> original;
    original.set(1);
    original.set(2);
    dense.push_back(original);
    packed.push_back(original);

    Monomial<N> replacement;
    replacement.set(10);
    replacement.set(20);
    replacement.set(30);
    assign_row<N>(dense, 0, replacement);
    assign_row<N>(packed, 0, replacement);

    BOOST_TEST((materialize_row<N>(dense, 0) == replacement));
    BOOST_TEST((materialize_row<N>(packed, 0) == replacement));
    BOOST_TEST(row_popcount<N>(packed, 0) == 3U);
    BOOST_TEST(positions_of<N>(dense, 0) == positions_of<N>(packed, 0));
}
