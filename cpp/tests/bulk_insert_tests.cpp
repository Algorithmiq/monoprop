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

// bulk_insert issues a group of 16 slot addresses before walking any of them, the way find_batch
// already does on the probe side. It is a pure hint: it must leave the table in EXACTLY the state an
// unpipelined loop leaves it in.
//
// "Exactly" is stronger than "every key is findable", and the extra strength is load-bearing.
// OperatorIndex::for_each walks table_.slots in slot order, and it is the operator's public term
// ENUMERATOR -- evolved_operator_terms goes through it -- so slot order is Python-visible. A change
// that reshuffled the table while keeping every key findable would still change the order in which the
// Python API returns terms.
//
// WHAT THIS COMPARES AGAINST. This was a two-arm differential while the unpipelined loop was still
// selectable at runtime. That loop measured slower on every instrument (incoming_s 0.9008x/0.9106x,
// insert_s 0.8786x/0.8763x, 8/8 paired reps in two cells at p=0.0078) and is gone. What replaced it as
// the oracle is a naive insert written HERE -- one key at a time, through the public single-key insert
// path, in the same order -- so the grouped loop is still checked against something that is not
// itself. A round-trip on the grouped loop alone would only ask whether it agrees with itself.
//
// The reference deliberately does not reuse bulk_insert or any of its helpers. If it did, a bug in a
// shared helper would be invisible to the comparison, which is the failure mode that made two earlier
// oracles vacuous during this work.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;

namespace {

constexpr size_t kN = 250;
using Index = detail::OperatorIndex<kN>;

auto draw_distinct(std::mt19937_64 &rng, size_t n) -> std::vector<Monomial<kN>> {
    std::vector<Monomial<kN>> out;
    std::set<std::vector<uint64_t>> seen;
    std::uniform_int_distribution<size_t> bit(0, Monomial<kN>::size() - 1);
    std::uniform_int_distribution<size_t> pop(0, 12);
    while (out.size() < n) {
        Monomial<kN> m;
        const size_t k = pop(rng);
        for (size_t placed = 0; placed < k;) {
            const size_t b = bit(rng);
            if (!m.test(b)) {
                m.set(b);
                ++placed;
            }
        }
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<kN>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        if (seen.insert(key).second) {
            out.push_back(m);
        }
    }
    return out;
}

// Fill an index with `terms` in one bulk_insert. Deliberately NOT reserved: the interesting case for a
// prefetch is the one where rehash_if_needed fires mid-group and the addresses already issued point
// into a table that has just been freed. That must be harmless -- a prefetch is a hint, and the insert
// recomputes its slot from the current mask -- and the only way to know is to run it, so the growth
// path is what these cases exercise.
auto build(const std::vector<Monomial<kN>> &terms) -> std::unique_ptr<Index> {
    auto idx = std::make_unique<Index>();
    const size_t base = idx->grow_rows_geometric(terms.size());
    for (size_t k = 0; k < terms.size(); ++k) {
        idx->set(base + k, terms[k]);
    }
    idx->bulk_insert(terms.size(), base, [&](size_t k) -> const Monomial<kN> & { return terms[k]; });
    return idx;
}

// The oracle: the same rows in the same order, but as N separate one-key calls instead of one N-key
// call. Every group is then trivially of size one, so none of the grouped loop's boundary arithmetic
// is exercised -- which is exactly what makes it a check on that arithmetic.
//
// Be precise about what this does and does not isolate. It shares insert_slot_ with the code under
// test, so it cannot catch a wrong slot policy; both would be wrong together. It does not share the
// GROUPING, and the grouping is the thing that changed: the group width, the partial final group, and
// the stale prefetch addresses left behind when rehash_if_needed fires mid-group.
auto build_reference(const std::vector<Monomial<kN>> &terms) -> std::unique_ptr<Index> {
    auto idx = std::make_unique<Index>();
    const size_t base = idx->grow_rows_geometric(terms.size());
    for (size_t k = 0; k < terms.size(); ++k) {
        idx->set(base + k, terms[k]);
        idx->bulk_insert(1, base + k, [&](size_t) -> const Monomial<kN> & { return terms[k]; });
    }
    return idx;
}

} // namespace

BOOST_AUTO_TEST_CASE(bulk_insert_finds_every_key) {
    std::mt19937_64 rng(20260814);
    const auto terms = draw_distinct(rng, 4000);
    const auto idx = build(terms);
    BOOST_REQUIRE_EQUAL(idx->size(), terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        const auto found = idx->find(terms[i]);
        BOOST_REQUIRE(found.has_value());
        BOOST_TEST(*found == i);
    }
}

BOOST_AUTO_TEST_CASE(bulk_insert_batch_find_agrees_with_one_key_at_a_time) {
    std::mt19937_64 rng(20260815);
    const auto terms = draw_distinct(rng, 4000);
    const auto absent = draw_distinct(rng, 500);
    const auto grouped = build(terms);
    const auto one_by_one = build_reference(terms);

    std::vector<size_t> a(terms.size(), 0);
    std::vector<size_t> b(terms.size(), 0);
    one_by_one->find_batch(terms.data(), terms.size(), a.data());
    grouped->find_batch(terms.data(), terms.size(), b.data());
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(a[i] == b[i]);
    }

    // A miss must stay a miss both ways: a table that answered everything would satisfy the loop above
    // and prove nothing about it.
    std::vector<size_t> ma(absent.size(), 0);
    std::vector<size_t> mb(absent.size(), 0);
    one_by_one->find_batch(absent.data(), absent.size(), ma.data());
    grouped->find_batch(absent.data(), absent.size(), mb.data());
    size_t genuinely_absent = 0;
    for (size_t i = 0; i < absent.size(); ++i) {
        BOOST_TEST(ma[i] == mb[i]);
        if (!one_by_one->find(absent[i]).has_value()) {
            ++genuinely_absent;
        }
    }
    BOOST_TEST(genuinely_absent > 0U);
}

BOOST_AUTO_TEST_CASE(bulk_insert_preserves_the_enumeration_order) {
    // for_each walks slots in slot order and is what evolved_operator_terms enumerates through, so this
    // sequence is Python-visible. Identical, not merely equal as a set.
    std::mt19937_64 rng(20260816);
    const auto terms = draw_distinct(rng, 4000);
    const auto grouped = build(terms);
    const auto one_by_one = build_reference(terms);

    std::vector<size_t> order_ref;
    std::vector<size_t> order_grouped;
    one_by_one->for_each([&](const Monomial<kN> &, size_t i) { order_ref.push_back(i); });
    grouped->for_each([&](const Monomial<kN> &, size_t i) { order_grouped.push_back(i); });

    BOOST_REQUIRE_EQUAL(order_grouped.size(), order_ref.size());
    BOOST_REQUIRE_EQUAL(order_ref.size(), terms.size());
    // Guard against the whole comparison passing because both happened to come out in index order,
    // which would make it blind to a reshuffle that preserved that.
    bool is_sorted_by_index = true;
    for (size_t i = 1; i < order_ref.size(); ++i) {
        if (order_ref[i] < order_ref[i - 1]) {
            is_sorted_by_index = false;
            break;
        }
    }
    BOOST_TEST(!is_sorted_by_index);
    for (size_t i = 0; i < order_ref.size(); ++i) {
        BOOST_TEST(order_grouped[i] == order_ref[i]);
    }
}

BOOST_AUTO_TEST_CASE(bulk_insert_handles_a_partial_final_group) {
    // The group width is 16; n % 16 != 0 must not read or insert past the end. Sizes chosen to land on
    // every residue class around the boundary, including fewer elements than one whole group.
    std::mt19937_64 rng(20260817);
    for (const size_t n : {size_t{1}, size_t{15}, size_t{16}, size_t{17}, size_t{31}, size_t{33}}) {
        const auto terms = draw_distinct(rng, n);
        const auto grouped = build(terms);
        const auto one_by_one = build_reference(terms);
        BOOST_REQUIRE_EQUAL(grouped->size(), n);
        for (size_t i = 0; i < n; ++i) {
            const auto fa = one_by_one->find(terms[i]);
            const auto fb = grouped->find(terms[i]);
            BOOST_REQUIRE(fa.has_value());
            BOOST_REQUIRE(fb.has_value());
            BOOST_TEST(*fb == *fa);
        }
    }
}

BOOST_AUTO_TEST_CASE(bulk_insert_of_nothing_is_a_no_op) {
    std::mt19937_64 rng(20260818);
    const auto terms = draw_distinct(rng, 100);
    auto idx = build(terms);
    const size_t before = idx->size();
    const auto key_at = [&](size_t k) -> const Monomial<kN> & { return terms[k]; };
    idx->bulk_insert(0, 0, key_at);
    BOOST_TEST(idx->size() == before);
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_REQUIRE(idx->find(terms[i]).has_value());
    }
}
