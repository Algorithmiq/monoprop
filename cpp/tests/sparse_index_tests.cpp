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

// SparseRowStore's keyless index must answer lookups exactly as OperatorIndex's does -- same hits, same
// misses, same insert-or-no-op -- because a propagator swaps one for the other. The hash *value* differs
// by design (that is the documented re-baseline); nothing else may.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/SparseRowStore.h"

#include "RandomMonomial.h"

using namespace monoprop;
using namespace monoprop::detail;

namespace {

// Distinct rows only: bulk_insert and the emplace-is-idempotent check both require it, and a duplicate
// would make "found at index i" ambiguous.
template <size_t NumModes>
struct RowSet {
    std::vector<Monomial<NumModes>> rows;
    std::set<std::vector<size_t>> seen;

    auto add(const Monomial<NumModes> &mono) -> bool {
        if (!seen.insert(positions(mono)).second) {
            return false;
        }
        rows.push_back(mono);
        return true;
    }
    [[nodiscard]] auto contains(const Monomial<NumModes> &mono) const -> bool {
        return seen.contains(positions(mono));
    }

    static auto positions(const Monomial<NumModes> &mono) -> std::vector<size_t> {
        std::vector<size_t> out;
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            out.push_back(b);
        }
        return out;
    }
};

// A spilled row has no codes word, so it can only be hashed by walking the dense monomial. That walk and
// the row walk are two producers of one hash, and if they ever disagreed a spilled row would become
// unfindable -- silently, and only for the small share of rows that spill.
template <size_t NumModes>
auto check_hash_agrees(std::mt19937_64 &rng) -> size_t {
    using Store = SparseRowStore<NumModes>;
    Store store(Store::kMaxSlots);
    std::vector<Monomial<NumModes>> rows;
    for (size_t t = 0; t < 200; ++t) {
        rows.push_back(test_utils::random_monomial<NumModes>(rng, Store::kMaxSlots));
        store.push_back(rows.back());
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        BOOST_REQUIRE(!store.spilled(i));
        BOOST_TEST(sparse_row_hash(store.view(i)) == sparse_row_hash(rows[i]));
    }
    return rows.size();
}

// The lookup contract, against OperatorIndex as the oracle.
template <size_t NumModes>
auto check_lookups_match_packed(std::mt19937_64 &rng, size_t slots, size_t &spilled_rows, size_t &misses) -> void {
    constexpr size_t kNumBits = 2 * NumModes;
    SparseRowStore<NumModes> sparse(slots);
    OperatorIndex<NumModes> packed;
    RowSet<NumModes> set;
    for (size_t t = 0; t < 300; ++t) {
        const auto mono = test_utils::random_monomial<NumModes>(rng, 12);
        if (!set.add(mono)) {
            continue;
        }
        const size_t i = sparse.size();
        sparse.push_back(mono);
        packed.push_back(mono);
        sparse.emplace(mono, i);
        packed.emplace(mono, i);
        spilled_rows += sparse.spilled(i) ? 1 : 0;
    }
    BOOST_REQUIRE(sparse.size() == set.rows.size());

    for (size_t i = 0; i < set.rows.size(); ++i) {
        const auto by_mono = sparse.find(set.rows[i]);
        BOOST_REQUIRE(by_mono.has_value());
        BOOST_TEST(*by_mono == i);
        BOOST_TEST(*by_mono == packed.find(set.rows[i]).value());
        if (!sparse.spilled(i)) {
            const auto by_row = sparse.find(sparse.view(i));
            BOOST_REQUIRE(by_row.has_value());
            BOOST_TEST(*by_row == i);
        }
    }

    // Absent keys must miss, not land on a hash neighbour.
    for (size_t t = 0; t < 300; ++t) {
        Monomial<NumModes> mono;
        for (size_t k = 0; k < 1 + (rng() % 6); ++k) {
            mono.set(rng() % kNumBits);
        }
        if (set.contains(mono)) {
            continue;
        }
        BOOST_TEST(!sparse.find(mono).has_value());
        BOOST_TEST(!packed.find(mono).has_value());
        ++misses;
    }

    // emplace is insert-or-no-op, so replaying every key must add no slot.
    const size_t indexed = sparse.indexed_count();
    for (size_t i = 0; i < set.rows.size(); ++i) {
        sparse.emplace(set.rows[i], i);
    }
    BOOST_TEST(sparse.indexed_count() == indexed);

    const auto copy = sparse.clone();
    BOOST_TEST(copy->indexed_count() == indexed);
    for (size_t i = 0; i < set.rows.size(); ++i) {
        const auto found = copy->find(set.rows[i]);
        BOOST_REQUIRE(found.has_value());
        BOOST_TEST(*found == i);
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_index_hash_agrees_between_row_and_monomial) {
    std::mt19937_64 rng(20260812U);
    size_t compared = 0;
    compared += check_hash_agrees<32>(rng);
    compared += check_hash_agrees<64>(rng);
    compared += check_hash_agrees<300>(rng);
    BOOST_TEST(compared == 600U);
}

// The row capacity is a tuning parameter. If it leaked into the hash, changing it would move probe order
// and MPI owner routing, so two stores tuned differently must agree on every row they can both hold.
BOOST_AUTO_TEST_CASE(sparse_index_hash_is_independent_of_row_capacity) {
    std::mt19937_64 rng(4321U);
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> narrow(8);
    SparseRowStore<kNumModes> wide(SparseRowStore<kNumModes>::kMaxSlots);
    size_t compared = 0;
    for (size_t t = 0; t < 300; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 8);
        narrow.push_back(mono);
        wide.push_back(mono);
        const size_t i = narrow.size() - 1;
        if (narrow.spilled(i)) {
            continue;
        }
        BOOST_TEST(sparse_row_hash(narrow.view(i)) == sparse_row_hash(wide.view(i)));
        ++compared;
    }
    BOOST_TEST(compared > 200U);
}

// Three widths and three row capacities, so that spilled and inline rows both occur.
BOOST_AUTO_TEST_CASE(sparse_index_lookups_match_the_packed_backend) {
    std::mt19937_64 rng(99U);
    size_t spilled_rows = 0;
    size_t misses = 0;
    for (const size_t slots : {4U, 8U, 32U}) {
        check_lookups_match_packed<32>(rng, slots, spilled_rows, misses);
        check_lookups_match_packed<64>(rng, slots, spilled_rows, misses);
        check_lookups_match_packed<300>(rng, slots, spilled_rows, misses);
    }
    // Both row kinds have to have occurred, or the spill path above was never taken.
    BOOST_TEST(spilled_rows > 0U);
    BOOST_TEST(misses > 100U);
}

// find_batch is a pipelined re-implementation of find, so it needs its own equality check -- against
// find, for both key forms, over a query list mixing hits and misses.
BOOST_AUTO_TEST_CASE(sparse_index_find_batch_matches_find) {
    std::mt19937_64 rng(555U);
    constexpr size_t kNumModes = 64;
    constexpr size_t kNumBits = 2 * kNumModes;
    SparseRowStore<kNumModes> store(8);
    RowSet<kNumModes> set;
    for (size_t t = 0; t < 400; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 10);
        if (!set.add(mono)) {
            continue;
        }
        const size_t i = store.size();
        store.push_back(mono);
        store.emplace(mono, i);
    }

    // More than one group of 16, and interleaved absentees so a group holds both.
    std::vector<Monomial<kNumModes>> queries;
    for (size_t i = 0; i < set.rows.size(); ++i) {
        queries.push_back(set.rows[i]);
        if (i % 3 == 0) {
            Monomial<kNumModes> absent;
            absent.set(rng() % kNumBits);
            if (!set.contains(absent)) {
                queries.push_back(absent);
            }
        }
    }
    BOOST_REQUIRE(queries.size() > 16U);

    std::vector<size_t> batched(queries.size());
    store.find_batch(queries.data(), queries.size(), batched.data());
    for (size_t j = 0; j < queries.size(); ++j) {
        const auto scalar = store.find(queries[j]);
        BOOST_TEST(batched[j] == (scalar ? *scalar : SparseRowStore<kNumModes>::kNotFound));
    }

    // The row-key form, which is what the scan will hand it.
    std::vector<SparseRow> row_queries;
    std::vector<size_t> expected;
    for (size_t i = 0; i < store.size(); ++i) {
        if (!store.spilled(i)) {
            row_queries.push_back(store.view(i));
            expected.push_back(i);
        }
    }
    BOOST_REQUIRE(row_queries.size() > 16U);
    std::vector<size_t> row_batched(row_queries.size());
    store.find_batch(row_queries.data(), row_queries.size(), row_batched.data());
    BOOST_TEST(row_batched == expected, boost::test_tools::per_element());
}

// bulk_insert skips the duplicate probe, so it is only correct on provably distinct keys; what it must
// still produce is an index that finds every one of them.
BOOST_AUTO_TEST_CASE(sparse_index_bulk_insert_indexes_every_row) {
    std::mt19937_64 rng(777U);
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> store(8);
    RowSet<kNumModes> set;
    for (size_t t = 0; t < 200; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 6);
        if (set.add(mono)) {
            store.push_back(mono);
        }
    }
    store.bulk_insert(store.size(), 0, [&](size_t k) { return set.rows[k]; });
    BOOST_TEST(store.indexed_count() == store.size());
    for (size_t i = 0; i < set.rows.size(); ++i) {
        const auto found = store.find(set.rows[i]);
        BOOST_REQUIRE(found.has_value());
        BOOST_TEST(*found == i);
    }
}

// The sparse hash is a different function from the dense one, which is exactly why the store swap needs
// a re-baseline. Pinned so that "the results moved" is never a surprise, and so a future change that
// accidentally reunified them would be noticed.
BOOST_AUTO_TEST_CASE(sparse_index_hash_differs_from_the_dense_hash) {
    std::mt19937_64 rng(8888U);
    constexpr size_t kNumModes = 32;
    size_t differing = 0;
    size_t total = 0;
    for (size_t t = 0; t < 200; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 8);
        if (mono.count() == 0) {
            continue;
        }
        differing += sparse_row_hash(mono) != monomial_hash<kNumModes>(mono) ? 1 : 0;
        ++total;
    }
    BOOST_REQUIRE(total > 100U);
    BOOST_TEST(differing == total);
}

// A pre-filter that collided often would still be correct but would degrade every probe into a lane
// compare. Over distinct rows the 32-bit folds should be near-injective.
BOOST_AUTO_TEST_CASE(sparse_index_hash_folds_are_near_injective) {
    std::mt19937_64 rng(31337U);
    constexpr size_t kNumModes = 64;
    SparseRowStore<kNumModes> store(SparseRowStore<kNumModes>::kMaxSlots);
    RowSet<kNumModes> set;
    for (size_t t = 0; t < 4000; ++t) {
        const auto mono = test_utils::random_monomial<kNumModes>(rng, 10);
        if (set.add(mono)) {
            store.push_back(mono);
        }
    }
    std::set<uint32_t> folds;
    for (const auto &mono : set.rows) {
        const size_t full = sparse_row_hash(mono);
        folds.insert(static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32)));
    }
    BOOST_REQUIRE(set.rows.size() > 3000U);
    // Birthday-bound expectation for n draws from 2^32 is ~n^2/2^33 collisions, i.e. under 1 for n=4000;
    // allowing 4 keeps this from being flaky while still failing on a hash that structurally collides.
    BOOST_TEST(set.rows.size() - folds.size() <= 4U);
}
