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

// Internals of the even-parity scan inverted index: the tiered column store (sparse row-lists vs dense
// bit-vectors, promoted at the 1/kPromoteDensityInv density crossover), the lazily-built per-row
// parity(|M|) bitmap, and the row-block-parallel fill. The inverted index reads its rows through the
// backend-agnostic for_each_row_position accessor, which is defined for a plain
// std::vector<Monomial<N>> — so these tests build one directly, no operator store required.

using namespace monoprop;
using namespace monoprop::detail;

namespace {
constexpr size_t N = 32; // 2N = 64 majorana columns
using Sc = InvertedIndex<N>;
using MSet = Monomial<N>;
MSet bs(const VecZ &r) {
    return indices_to_bitset<N>(r);
}
// indices_to_bitset maps mode index m to bit position 2N-1-m, and the inverted index indexes its columns by
// raw bit position — so mode m populates column col_of(m).
constexpr size_t col_of(size_t mode) {
    return 2 * N - 1 - mode;
}
} // namespace

// row_parity_words() builds the packed popcount(|M|)&1 bitmap over the current rows. Verify each
// bit against the known parity of the row it was built from.
BOOST_AUTO_TEST_CASE(inverted_index_row_parity_matches_popcount) {
    const std::vector<MSet> op{
        bs({0, 1}),       // |M| = 2 -> even
        bs({0, 1, 2}),    // |M| = 3 -> odd
        bs({5}),          // |M| = 1 -> odd
        bs({4, 5, 6, 7}), // |M| = 4 -> even
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

// A column crosses to DENSE when set_rows.size() * kPromoteDensityInv >= row_count (density >= 1/64);
// below that it stays a sparse row-list. rebuild decides tiers from the final per-column counts.
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
    BOOST_TEST(sc.sparse_column_count(col_of(1)) == 1u);
    BOOST_TEST(sc.column_rows(col_of(1)) == std::vector<TermIndex>{0u}, boost::test_tools::per_element());
}

// rebuild fills columns in row order, so sparse row-lists come out ASCENDING — the invariant the fold
// recompute relies on (combine_columns_block lower_bounds them). Build a many-mode operator kept below
// the promote threshold so columns stay sparse, then assert every sparse list is sorted.
BOOST_AUTO_TEST_CASE(inverted_index_fill_yields_ascending_sparse_rows) {
    constexpr size_t M = 64; // 2M = 128 columns
    using ScW = InvertedIndex<M>;
    constexpr size_t kR = 16'385; // large operator, many sparse columns
    std::vector<Monomial<M>> op;
    op.reserve(kR);
    for (size_t i = 0; i < kR; ++i) {
        // one mode per row spread over all 128 columns: each column set ~128 times, 128*64 < 16385,
        // so every column stays sparse.
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
        if (sc.sparse_column_count(c) != 0) {
            saw_nonempty_sparse = true;
        }
        // Walk the tier through its public iteration entry point: strictly ascending is the contract
        // combine_columns_block's range query depends on, so assert it on what the fold actually sees.
        TermIndex prev = 0;
        bool first = true;
        sc.for_each_sparse_row(c, [&](TermIndex r) {
            if (!first && r <= prev) {
                all_sorted = false;
            }
            prev = r;
            first = false;
        });
    }
    BOOST_TEST(saw_nonempty_sparse); // the fill actually populated sparse columns
    BOOST_TEST(all_sorted);          // and they are ascending, at any thread count
}
