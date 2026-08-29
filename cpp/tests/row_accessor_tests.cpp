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

// The dense-vector, packed OperatorIndex and sparse SparseRowStore backends must agree through every
// RowAccess.h accessor.

#include <boost/test/unit_test.hpp>

#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/RowAccess.h"
#include "monoprop/detail/operator/SparseRowStore.h"

using namespace monoprop;

namespace {

template <size_t N>
auto positions_of(const auto &backend, size_t i) -> std::vector<size_t> {
    std::vector<size_t> out;
    for_each_row_position<N>(backend, i, [&](size_t b) { out.push_back(b); });
    return out;
}

// `slots` is the sparse backend's per-row mode capacity: pass one below a row's occupied-mode count to
// drive that row down the overflow path, which must stay invisible through the accessors.
template <size_t N>
auto check_backends_agree(const std::vector<std::vector<size_t>> &raw_rows, size_t slots = 8) -> void {
    std::vector<Monomial<N>> dense;
    detail::OperatorIndex<N> packed;
    detail::SparseRowStore<N> sparse(slots);
    for (const auto &bits : raw_rows) {
        Monomial<N> m;
        for (size_t b : bits) {
            m.set(b);
        }
        dense.push_back(m);
        packed.push_back(m);
        sparse.push_back(m);
    }

    BOOST_REQUIRE(packed.size() == dense.size());
    BOOST_REQUIRE(sparse.size() == dense.size());
    for (size_t i = 0; i < dense.size(); ++i) {
        BOOST_TEST((materialize_row<N>(dense, i) == materialize_row<N>(packed, i)));
        BOOST_TEST((materialize_row<N>(dense, i) == materialize_row<N>(sparse, i)));
        BOOST_TEST(row_popcount<N>(dense, i) == row_popcount<N>(packed, i));
        BOOST_TEST(row_popcount<N>(dense, i) == row_popcount<N>(sparse, i));
        BOOST_TEST(row_popcount<N>(dense, i) == materialize_row<N>(dense, i).count());
        BOOST_TEST(positions_of<N>(dense, i) == positions_of<N>(packed, i));
        BOOST_TEST(positions_of<N>(dense, i) == positions_of<N>(sparse, i));
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(row_accessor_backends_agree_single_word) {
    check_backends_agree<32>({{0, 3, 5}, {1, 2}, {}, {63}, {0, 1, 2, 3, 62, 63}});
}

BOOST_AUTO_TEST_CASE(row_accessor_backends_agree_multi_word) {
    check_backends_agree<96>({{0, 64, 191}, {5, 63, 64, 65}, {}, {128, 190}});
}

// Four occupied modes against a two-slot capacity: the first two rows spill, the empty row and the
// one-mode row do not, so the same store serves both kinds.
BOOST_AUTO_TEST_CASE(row_accessor_backends_agree_sparse_overflow) {
    check_backends_agree<32>({{0, 3, 5, 8, 20, 21}, {1, 2, 40, 41, 62, 63}, {}, {10, 11}}, 2);
}

BOOST_AUTO_TEST_CASE(row_accessor_assign_row_overwrites) {
    constexpr size_t N = 32;
    std::vector<Monomial<N>> dense;
    detail::OperatorIndex<N> packed;
    // Two slots, and the original occupies three modes: the row starts spilled and the overwrite must
    // pull it back inline rather than leaving the stale side-map entry to shadow it.
    detail::SparseRowStore<N> sparse(2);
    Monomial<N> original;
    original.set(1);
    original.set(2);
    original.set(40);
    original.set(41);
    original.set(60);
    dense.push_back(original);
    packed.push_back(original);
    sparse.push_back(original);
    BOOST_TEST(sparse.spilled(0));

    // Three set bits over two modes, so it fits the sparse store's two slots.
    Monomial<N> replacement;
    replacement.set(10);
    replacement.set(11);
    replacement.set(30);
    assign_row<N>(dense, 0, replacement);
    assign_row<N>(packed, 0, replacement);
    assign_row<N>(sparse, 0, replacement);

    BOOST_TEST((materialize_row<N>(dense, 0) == replacement));
    BOOST_TEST((materialize_row<N>(packed, 0) == replacement));
    BOOST_TEST((materialize_row<N>(sparse, 0) == replacement));
    BOOST_TEST(row_popcount<N>(packed, 0) == 3U);
    BOOST_TEST(row_popcount<N>(sparse, 0) == 3U);
    BOOST_TEST(!sparse.spilled(0));
    BOOST_TEST(positions_of<N>(dense, 0) == positions_of<N>(packed, 0));
    BOOST_TEST(positions_of<N>(dense, 0) == positions_of<N>(sparse, 0));
}
