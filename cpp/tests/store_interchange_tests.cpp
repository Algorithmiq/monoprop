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

// Which of the operator store's consumers actually care which store they are given. The answer decides
// how much the backend swap has to touch, so it is asserted rather than reasoned about:
//
//   InvertedIndex  -- no: it reads rows only through the RowAccess.h accessors, so both stores build the
//                     same columns, the same parity words and the same tiering.
//   MonomialMap    -- no: keyed by a dense monomial, which both backends accept as a find() key, so
//                     init_op_map keeps its key type.
//   for_each       -- no: SparseRowStore offers OperatorIndex's fn(monomial, row_index) signature.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <map>
#include <random>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/InvertedIndex.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

#include "RandomMonomial.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

// Every observable of a built index, so a difference cannot hide in a field the test forgot.
template <size_t NumModes>
auto columns_agree(const InvertedIndex<NumModes> &a, const InvertedIndex<NumModes> &b) -> bool {
    if (a.rows() != b.rows() || a.words() != b.words()) {
        return false;
    }
    for (size_t c = 0; c < InvertedIndex<NumModes>::kNumColumns; ++c) {
        if (a.column_is_dense(c) != b.column_is_dense(c)) {
            return false;
        }
        if (a.column_is_dense(c)) {
            for (size_t w = 0; w < a.words(); ++w) {
                if (a.dense_column_data(c)[w] != b.dense_column_data(c)[w]) {
                    return false;
                }
            }
        }
        else if (a.sparse_column_rows(c) != b.sparse_column_rows(c)) {
            return false;
        }
    }
    for (size_t w = 0; w < a.words(); ++w) {
        if (a.row_parity_words()[w] != b.row_parity_words()[w]) {
            return false;
        }
    }
    return a.tier_memory_bytes() == b.tier_memory_bytes();
}

template <size_t NumModes>
auto check_inverted_index_is_store_agnostic(std::mt19937_64 &rng) -> void {
    OperatorIndex<NumModes> packed;
    // Capacity 4 on purpose, so a good share of rows spill and the index reads them through the
    // accessors' overflow path rather than off a codes word.
    SparseRowStore<NumModes> sparse(4);
    size_t spilled = 0;
    for (size_t t = 0; t < 400; ++t) {
        const auto mono = test_utils::random_monomial<NumModes>(rng, 8);
        packed.push_back(mono);
        sparse.push_back(mono);
        spilled += sparse.spilled(sparse.size() - 1) ? 1 : 0;
    }
    BOOST_TEST(spilled > 0U);

    InvertedIndex<NumModes> from_packed;
    InvertedIndex<NumModes> from_sparse;
    from_packed.rebuild(packed);
    from_sparse.rebuild(sparse);
    BOOST_TEST(columns_agree<NumModes>(from_packed, from_sparse));

    // append_rows is the incremental path evolution actually takes, so it needs its own comparison --
    // and a like-for-like one. An appended index and a rebuilt index legitimately differ in tiering:
    // rebuild() counts every column's postings up front and pre-promotes, while append_rows can only
    // promote as rows arrive. That is InvertedIndex's own behaviour and says nothing about the store,
    // so what is compared here is append-vs-append.
    InvertedIndex<NumModes> appended_sparse;
    InvertedIndex<NumModes> appended_packed;
    appended_sparse.rebuild(sparse);
    appended_packed.rebuild(packed);
    for (size_t t = 0; t < 50; ++t) {
        const auto mono = test_utils::random_monomial<NumModes>(rng, 8);
        const size_t base = sparse.size();
        packed.push_back(mono);
        sparse.push_back(mono);
        appended_sparse.append_rows(sparse, base, 1);
        appended_packed.append_rows(packed, base, 1);
    }
    BOOST_TEST(columns_agree<NumModes>(appended_sparse, appended_packed));
}

} // namespace

BOOST_AUTO_TEST_CASE(store_interchange_inverted_index_is_store_agnostic) {
    std::mt19937_64 rng(20260812U);
    check_inverted_index_is_store_agnostic<32>(rng);
    check_inverted_index_is_store_agnostic<64>(rng);
}

// MonomialMap is keyed by a dense monomial, and both backends accept one as a find() key, so the map
// needs no change at all when the store is swapped -- which is what is asserted here.
BOOST_AUTO_TEST_CASE(store_interchange_monomial_map_keys_still_resolve) {
    std::mt19937_64 rng(1234U);
    constexpr size_t kNumModes = 32;
    OperatorIndex<kNumModes> packed;
    SparseRowStore<kNumModes> sparse(4);

    MonomialMap<kNumModes> pending;
    std::vector<Monomial<kNumModes>> stored;
    for (size_t t = 0; t < 200; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 8);
        // One row per distinct monomial: emplace is insert-or-no-op, so a duplicate would resolve to the
        // first row holding it and the coefficient below would name the wrong index.
        if (sparse.find(mono).has_value()) {
            continue;
        }
        const size_t i = sparse.size();
        packed.push_back(mono);
        sparse.push_back(mono);
        packed.emplace(mono, i);
        sparse.emplace(mono, i);
        stored.push_back(mono);
        pending[mono] = static_cast<double>(i);
    }
    // Terms the store does not hold, which MPOperator::get_operator must leave pending.
    size_t absent = 0;
    for (size_t t = 0; t < 200; ++t) {
        Monomial<kNumModes> mono;
        for (size_t k = 0; k < 1 + (rng() % 5); ++k) {
            mono.set(rng() % (2 * kNumModes));
        }
        // emplace, not [], and count only what it inserted: the same absent monomial can be drawn twice,
        // and assigning would leave the map smaller than the count.
        if (!sparse.find(mono).has_value() && pending.emplace(mono, -1.0).second) {
            ++absent;
        }
    }
    BOOST_TEST(absent > 0U);

    // get_operator's loop, run against both stores: a key present in the store drains to its row, one
    // absent stays. Both stores must agree on which is which, and on the row.
    size_t drained = 0;
    size_t retained = 0;
    for (const auto &[mono, coeff] : pending) {
        const auto in_sparse = sparse.find(mono);
        const auto in_packed = packed.find(mono);
        BOOST_TEST(in_sparse.has_value() == in_packed.has_value());
        if (in_sparse) {
            BOOST_TEST(*in_sparse == *in_packed);
            BOOST_TEST(coeff == static_cast<double>(*in_sparse));
            ++drained;
        }
        else {
            ++retained;
        }
    }
    BOOST_TEST(drained == stored.size());
    BOOST_TEST(retained == absent);
}

// evolved_operator_terms iterates the index, so both stores must offer the same signature and -- since
// the shared RowHashTable fixes slot order for a given insertion sequence -- visit the same row indices
// in the same order. The monomials come out equal even though the two stores hash differently, because
// that order is the *table's*, and both tables saw the same sequence of (index, hash) pairs from their
// own hash.
BOOST_AUTO_TEST_CASE(store_interchange_for_each_visits_every_row) {
    std::mt19937_64 rng(4321U);
    constexpr size_t kNumModes = 32;
    OperatorIndex<kNumModes> packed;
    SparseRowStore<kNumModes> sparse(4);
    std::vector<Monomial<kNumModes>> stored;
    for (size_t t = 0; t < 200; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 8);
        if (sparse.find(mono).has_value()) {
            continue; // one row per distinct monomial, so an index maps to one term
        }
        const size_t i = sparse.size();
        packed.push_back(mono);
        sparse.push_back(mono);
        packed.emplace(mono, i);
        sparse.emplace(mono, i);
        stored.push_back(mono);
    }

    std::map<size_t, Monomial<kNumModes>> from_packed;
    std::map<size_t, Monomial<kNumModes>> from_sparse;
    packed.for_each([&](const auto &mono, size_t idx) { from_packed.emplace(idx, mono); });
    sparse.for_each([&](const auto &mono, size_t idx) { from_sparse.emplace(idx, mono); });

    BOOST_REQUIRE(from_packed.size() == stored.size());
    BOOST_REQUIRE(from_sparse.size() == stored.size());
    for (size_t i = 0; i < stored.size(); ++i) {
        BOOST_TEST((from_packed.at(i) == stored[i]));
        BOOST_TEST((from_sparse.at(i) == stored[i]));
    }
}
