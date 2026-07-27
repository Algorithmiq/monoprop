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
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/operator/InvertedIndex.h"

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
} // namespace

// row_parity_words() packs popcount(|M|)&1 over the current rows; the oracle is each row's popcount.
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
    // The lone sparse hit is recorded losslessly.
    BOOST_TEST(sc.sparse_column_rows(col_of(1)).size() == 1u);
    BOOST_TEST(sc.sparse_column_rows(col_of(1))[0] == 0u);
}

// rebuild fills columns in row order, so sparse row-lists come out ASCENDING — the invariant
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
        if (!std::is_sorted(rows.begin(), rows.end())) {
            all_sorted = false;
        }
    }
    BOOST_TEST(saw_nonempty_sparse); // the fill actually populated sparse columns
    BOOST_TEST(all_sorted);
}
