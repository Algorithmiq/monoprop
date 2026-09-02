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

// The per-gate partner table (AntiTable.h): the algebraic fact it rests on, the probe against a
// TermLookup oracle over rows that include spilled and fully paired ones, and the leader-pass marks.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/AntiTable.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/TermLookup.h"

using namespace monoprop;

namespace {

template <size_t NumModes>
auto random_monomial(std::mt19937_64 &rng, size_t k) -> Monomial<NumModes> {
    Monomial<NumModes> m;
    std::uniform_int_distribution<size_t> bit(0, Monomial<NumModes>::size() - 1);
    k = std::min(k, Monomial<NumModes>::size());
    while (m.count() < k) {
        m.set(bit(rng));
    }
    return m;
}

template <size_t NumModes>
auto random_paired_monomial(std::mt19937_64 &rng, size_t d) -> Monomial<NumModes> {
    Monomial<NumModes> m;
    std::uniform_int_distribution<size_t> mode(0, NumModes - 1);
    while (m.count() < 2 * d) {
        const size_t mo = mode(rng);
        m.set(2 * mo);
        m.set((2 * mo) + 1);
    }
    return m;
}

template <size_t NumModes>
auto draw_distinct(std::mt19937_64 &rng, size_t n) -> std::vector<Monomial<NumModes>> {
    std::vector<Monomial<NumModes>> out;
    std::set<std::vector<uint64_t>> seen;
    const std::vector<size_t> pops = {1, 2, 3, 4, 5, 6, 8, 12, 14};
    std::uniform_int_distribution<size_t> pick(0, pops.size() - 1);
    while (out.size() < n) {
        const size_t k = pops[pick(rng)];
        const auto m =
            ((rng() & 3U) == 0U) ? random_paired_monomial<NumModes>(rng, k / 2 + 1) : random_monomial<NumModes>(rng, k);
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        if (seen.insert(key).second) {
            out.push_back(m);
        }
    }
    return out;
}

template <size_t NumModes>
auto positions_of(const Monomial<NumModes> &m) -> std::vector<typename detail::OperatorIndex<NumModes>::PosT> {
    std::vector<typename detail::OperatorIndex<NumModes>::PosT> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<typename detail::OperatorIndex<NumModes>::PosT>(b));
    }
    return pos;
}

// Majorana anticommutation: |M||G| - |M ∩ G| odd.
template <size_t NumModes>
auto majorana_anticommutes(const Monomial<NumModes> &m, const Monomial<NumModes> &g) -> bool {
    return ((m.count() * g.count()) - m.count_and(g)) % 2 == 1;
}

} // namespace

// The lemma the table rests on, checked as pure algebra in both bases: if M anticommutes with G then so
// does M ^ G. Odd-length Majorana generators are the case that needs |G|^2 ≡ |G|.
BOOST_AUTO_TEST_CASE(anti_table_partner_of_an_anticommuting_term_anticommutes) {
    constexpr size_t kN = 40;
    std::mt19937_64 rng(20260901);
    size_t majorana_pairs = 0;
    size_t pauli_pairs = 0;
    for (size_t trial = 0; trial < 4000; ++trial) {
        const auto m = random_monomial<kN>(rng, 1 + (rng() % 9));
        const auto g = random_monomial<kN>(rng, 1 + (rng() % 6));
        if (majorana_anticommutes<kN>(m, g)) {
            BOOST_REQUIRE(majorana_anticommutes<kN>(m ^ g, g));
            ++majorana_pairs;
        }
        if (pauli_anticommutes<kN>(m, g)) {
            BOOST_REQUIRE(pauli_anticommutes<kN>(m ^ g, g));
            ++pauli_pairs;
        }
    }
    BOOST_TEST(majorana_pairs > 500U);
    BOOST_TEST(pauli_pairs > 500U);
}

// The probe against the oracle: every tabled row is found from its positions and nothing else is,
// across inline, spilled (width 6) and fully paired rows.
BOOST_AUTO_TEST_CASE(anti_table_probe_matches_the_lookup_oracle) {
    constexpr size_t kN = 96;
    std::mt19937_64 rng(20260902);
    const auto terms = draw_distinct<kN>(rng, 500);
    detail::OperatorIndex<kN> store(6);
    store.grow_rows_geometric(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        store.set(i, terms[i]);
    }
    BOOST_REQUIRE(store.overflow_size() > 0U);
    const auto lookup = detail::build_term_lookup<kN>(store, 0, store.size());
    const uint64_t *labels = routing::linear_basis<2 * kN>().data();

    // Only every third row is "anticommuting": the others must miss even though they are tracked.
    detail::AntiTable<kN> table;
    std::vector<size_t> tabled;
    for (size_t i = 0; i < terms.size(); i += 3) {
        tabled.push_back(i);
    }
    table.begin(tabled.size());
    for (const size_t i : tabled) {
        const auto pos = positions_of<kN>(terms[i]);
        const uint64_t fp = store.row_positions(i).inlined()
                                ? routing::fingerprint_positions(labels, pos.data(), pos.size())
                                : routing::linear_hash<2 * kN>(terms[i]);
        BOOST_REQUIRE_EQUAL(fp, routing::linear_hash<2 * kN>(terms[i]));
        const auto ord = table.add(static_cast<TermIndex>(i), fp);
        BOOST_REQUIRE_EQUAL(table.row_of(ord), i);
    }
    BOOST_REQUIRE_EQUAL(table.size(), tabled.size());

    for (size_t i = 0; i < terms.size(); ++i) {
        const auto pos = positions_of<kN>(terms[i]);
        const uint64_t fp = routing::fingerprint_positions(labels, pos.data(), pos.size());
        const auto ord = table.probe(store, fp, pos);
        const bool expect_hit = (i % 3) == 0;
        BOOST_REQUIRE_EQUAL(ord != detail::AntiTable<kN>::kNone, expect_hit);
        if (expect_hit) {
            BOOST_REQUIRE_EQUAL(table.row_of(ord), i);
            BOOST_REQUIRE_EQUAL(table.ordinal_of_row(i), ord);
            BOOST_REQUIRE_EQUAL(lookup.at(terms[i]), i);
        }
        else {
            BOOST_REQUIRE_EQUAL(table.ordinal_of_row(i), detail::AntiTable<kN>::kNone);
        }
    }
    for (const auto &m : draw_distinct<kN>(rng, 100)) {
        if (lookup.find(m) != lookup.end()) {
            continue;
        }
        const auto pos = positions_of<kN>(m);
        BOOST_REQUIRE(table.probe(store, routing::fingerprint_positions(labels, pos.data(), pos.size()), pos)
                      == detail::AntiTable<kN>::kNone);
    }
}

// An equal fingerprint is a prefilter, never a match: a row whose fingerprint collides with the query
// but whose positions differ must miss. Forced here by handing the probe a wrong fingerprint on purpose.
BOOST_AUTO_TEST_CASE(anti_table_confirms_positions_after_a_fingerprint_match) {
    constexpr size_t kN = 16;
    detail::OperatorIndex<kN> store(4);
    const auto a = indices_to_bitset<kN>({0, 3});
    const auto b = indices_to_bitset<kN>({1, 5, 7});
    store.grow_rows_geometric(2);
    store.set(0, a);
    store.set(1, b);
    const uint64_t *labels = routing::linear_basis<2 * kN>().data();
    const auto pa = positions_of<kN>(a);
    const auto pb = positions_of<kN>(b);
    const uint64_t fa = routing::fingerprint_positions(labels, pa.data(), pa.size());

    detail::AntiTable<kN> table;
    table.begin(2);
    table.add(0, fa);
    table.add(1, fa); // a colliding fingerprint on a different row
    BOOST_TEST(table.probe(store, fa, pa) == 0U);
    BOOST_TEST(table.probe(store, fa, pb) == 1U); // found through the confirm
    BOOST_TEST(table.probe(store, fa, positions_of<kN>(indices_to_bitset<kN>({0, 4}))) == detail::AntiTable<kN>::kNone);
}

// Marks are per ordinal, cleared by begin(), and reachable by row.
BOOST_AUTO_TEST_CASE(anti_table_marks_are_per_ordinal_and_cleared_by_begin) {
    detail::AntiTable<8> table;
    table.begin(3);
    table.add(2, 0x11);
    table.add(5, 0x22);
    table.add(9, 0x33);
    BOOST_TEST(!table.is_marked(1));
    table.mark(1);
    BOOST_TEST(table.is_marked(1));
    BOOST_TEST(table.is_marked_row(5));
    BOOST_TEST(!table.is_marked_row(2));
    BOOST_TEST(!table.is_marked_row(7)); // not in the table at all
    BOOST_TEST(table.ordinal_of_row(9) == 2U);
    table.begin(1);
    table.add(5, 0x22);
    BOOST_TEST(!table.is_marked(0));
    BOOST_TEST(table.size() == 1U);

    // Retained capacity is bounded: a large gate followed by a tiny one releases the large table.
    table.begin(100000);
    const size_t big = table.memory_bytes();
    table.begin(4);
    BOOST_TEST(table.memory_bytes() < big / 4);
}
