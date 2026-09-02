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

// The per-gate bucketed join (BucketJoin.h): the algebraic fact it rests on, hit/miss against an
// unordered_map oracle over rows that include spilled and fully paired ones, the position confirm behind
// an engineered key collision, and the per-row marks a gate leaves behind (GateScratch.h).

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/BucketJoin.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/OperatorIndex.h"

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
        const auto m = ((rng() & 3U) == 0U) ? random_paired_monomial<NumModes>(rng, (k / 2) + 1)
                                            : random_monomial<NumModes>(rng, k);
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
    using PosT = typename detail::OperatorIndex<NumModes>::PosT;
    std::vector<PosT> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<PosT>(b));
    }
    return pos;
}

// Majorana anticommutation: |M||G| - |M ∩ G| odd.
template <size_t NumModes>
auto majorana_anticommutes(const Monomial<NumModes> &m, const Monomial<NumModes> &g) -> bool {
    return ((m.count() * g.count()) - m.count_and(g)) % 2 == 1;
}

// A hash over the word representation, so the oracle is an ordinary map and shares nothing with the
// routing fingerprint the join keys on.
template <size_t NumModes>
struct MonoHash {
    auto operator()(const Monomial<NumModes> &m) const -> size_t {
        size_t h = 0xcbf2'9ce4'8422'2325ULL;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            h = (h ^ m.word(w)) * 0x0000'0100'0000'01B3ULL;
        }
        return h;
    }
};

template <size_t NumModes>
auto fp_of(const Monomial<NumModes> &m) -> uint64_t {
    return routing::linear_hash<2 * NumModes>(m);
}

// One gate: `rows` staged as the anticommuting side, `queries` asked for in that order. Returns the
// join, so the caller can read hit(q).
template <size_t NumModes>
auto run_join(const detail::OperatorIndex<NumModes> &store,
              std::span<const size_t> rows,
              const std::vector<std::vector<typename detail::OperatorIndex<NumModes>::PosT>> &queries,
              const std::vector<uint64_t> &query_fp) -> detail::BucketJoin<NumModes> {
    using PosT = typename detail::OperatorIndex<NumModes>::PosT;
    detail::BucketJoin<NumModes> join;
    join.begin_rows(rows.size());
    const uint64_t *labels = routing::linear_basis<2 * NumModes>().data();
    for (const size_t row : rows) {
        const auto src = store.row_positions(row);
        const uint64_t fp = src.inlined() ? routing::fingerprint_positions(labels, src.pos.data(), src.pos.size())
                                          : routing::linear_hash<2 * NumModes>(store.row(row));
        join.add_row(fp, row);
    }
    join.begin_queries(queries.size());
    for (size_t q = 0; q < queries.size(); ++q) {
        join.add_query(q, query_fp[q]);
    }
    join.run(store, [&](size_t q) { return std::span<const PosT>(queries[q]); });
    return join;
}

} // namespace

// The lemma the join rests on, checked as pure algebra in both bases: if M anticommutes with G then so
// does M ^ G, so a tracked partner is inside the gate's own anticommuting set and nothing outside it has
// to be searched. Odd-length Majorana generators are the case that needs |G|^2 ≡ |G|.
BOOST_AUTO_TEST_CASE(bucket_join_partner_of_an_anticommuting_term_anticommutes) {
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

namespace {

// Every third row is "anticommuting": the others are tracked but not staged, so a query naming one of
// them must miss. The asked-for keys are a shuffle of the tracked terms and of genuinely absent ones, so
// all three arms of the oracle are exercised at every size. They must be pairwise distinct -- ⊕G is
// injective over distinct sources, so no gate ever asks for one monomial twice, and the join asserts it.
template <size_t NumModes>
auto check_against_oracle(size_t n_terms, size_t n_queries, uint64_t seed) -> void {
    using PosT = typename detail::OperatorIndex<NumModes>::PosT;
    std::mt19937_64 rng(seed);
    const auto terms = draw_distinct<NumModes>(rng, std::max<size_t>(n_terms, 1));
    detail::OperatorIndex<NumModes> store(6); // narrow inline width, so wide terms spill
    store.grow_rows_geometric(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        store.set(i, terms[i]);
    }
    if (terms.size() > 20) {
        BOOST_REQUIRE(store.overflow_size() > 0U);
    }

    std::vector<size_t> staged;
    std::unordered_map<Monomial<NumModes>, size_t, MonoHash<NumModes>> oracle;
    std::set<std::vector<uint64_t>> tracked_keys;
    for (size_t i = 0; i < terms.size(); ++i) {
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(terms[i].word(w));
        }
        tracked_keys.insert(key);
        if (i % 3 == 0) {
            staged.push_back(i);
            oracle.emplace(terms[i], i);
        }
    }

    std::vector<Monomial<NumModes>> pool = terms;
    for (const auto &m : draw_distinct<NumModes>(rng, (n_queries / 2) + 8)) {
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        if (tracked_keys.count(key) == 0) {
            pool.push_back(m); // genuinely absent from the store
        }
    }
    std::ranges::shuffle(pool, rng);
    BOOST_REQUIRE(pool.size() >= n_queries);
    const std::vector<Monomial<NumModes>> asked(pool.begin(), pool.begin() + static_cast<ptrdiff_t>(n_queries));

    std::vector<std::vector<PosT>> queries;
    std::vector<uint64_t> query_fp;
    for (const auto &m : asked) {
        queries.push_back(positions_of<NumModes>(m));
        query_fp.push_back(fp_of<NumModes>(m));
    }

    const auto join = run_join<NumModes>(store, staged, queries, query_fp);
    size_t hits = 0;
    for (size_t q = 0; q < asked.size(); ++q) {
        const auto it = oracle.find(asked[q]);
        const size_t got = join.hit(q);
        if (it == oracle.end()) {
            BOOST_REQUIRE_MESSAGE(got == detail::BucketJoin<NumModes>::kMissing,
                                  "query " << q << " matched a row it must not");
        }
        else {
            BOOST_REQUIRE_EQUAL(got, it->second);
            BOOST_REQUIRE((store.row(got) == asked[q]));
            ++hits;
        }
    }
    if (n_queries > 8) {
        BOOST_TEST(hits > 0U); // a vacuous pass would prove nothing about the confirm
    }
}

} // namespace

// The oracle check across the sizes that select every bucket count the formula can produce, from the
// 16-bucket floor to a run whose rows and queries both spill L2.
BOOST_AUTO_TEST_CASE(bucket_join_hits_and_misses_match_a_map_oracle) {
    check_against_oracle<96>(/*n_terms=*/1, /*n_queries=*/0, 20260902);
    check_against_oracle<96>(/*n_terms=*/4, /*n_queries=*/1, 20260903);
    check_against_oracle<96>(/*n_terms=*/22, /*n_queries=*/7, 20260904);
    check_against_oracle<96>(/*n_terms=*/3000, /*n_queries=*/1000, 20260905);
    check_against_oracle<40>(/*n_terms=*/150000, /*n_queries=*/200000, 20260906);
}

// Both position widths, since the store's PosT narrows below 129 modes and the confirm compares raw
// stored positions.
BOOST_AUTO_TEST_CASE(bucket_join_matches_at_both_position_widths) {
    static_assert(sizeof(detail::OperatorIndex<32>::PosT) == 1, "narrow store");
    static_assert(sizeof(detail::OperatorIndex<250>::PosT) == 2, "wide store");
    check_against_oracle<32>(/*n_terms=*/900, /*n_queries=*/700, 20260907);
    check_against_oracle<250>(/*n_terms=*/900, /*n_queries=*/700, 20260908);
}

// An equal key is a prefilter, never a match: the fingerprint maps 2*NumModes bits onto 64, so the join
// confirms every candidate against the query's positions. Forced here by staging two rows under ONE
// fingerprint, which puts them in the same bucket with the same 32-bit tag.
BOOST_AUTO_TEST_CASE(bucket_join_confirms_positions_after_a_key_collision) {
    constexpr size_t kN = 16;
    using PosT = detail::OperatorIndex<kN>::PosT;
    detail::OperatorIndex<kN> store(4);
    const auto a = indices_to_bitset<kN>({0, 3});
    const auto b = indices_to_bitset<kN>({1, 5, 7});
    const auto c = indices_to_bitset<kN>({0, 4}); // neither row, same forced key
    store.grow_rows_geometric(2);
    store.set(0, a);
    store.set(1, b);
    const uint64_t collide = fp_of<kN>(a);

    detail::BucketJoin<kN> join;
    join.begin_rows(2);
    join.add_row(collide, 0);
    join.add_row(collide, 1); // a colliding key on a different row
    const std::vector<std::vector<PosT>> queries = {positions_of<kN>(a), positions_of<kN>(b), positions_of<kN>(c)};
    join.begin_queries(queries.size());
    for (size_t q = 0; q < queries.size(); ++q) {
        join.add_query(q, collide);
    }
    join.run(store, [&](size_t q) { return std::span<const PosT>(queries[q]); });
    BOOST_TEST(join.hit(0) == 0U);
    BOOST_TEST(join.hit(1) == 1U); // found through the confirm, past the first equal-key entry
    BOOST_TEST(join.hit(2) == detail::BucketJoin<kN>::kMissing);
}

// A query whose key belongs to no staged row must miss even when its key lands in an occupied bucket:
// the join reports kMissing rather than the bucket's nearest entry.
BOOST_AUTO_TEST_CASE(bucket_join_reports_missing_for_an_unstaged_key) {
    constexpr size_t kN = 16;
    using PosT = detail::OperatorIndex<kN>::PosT;
    detail::OperatorIndex<kN> store(4);
    const auto a = indices_to_bitset<kN>({2, 6});
    const auto absent = indices_to_bitset<kN>({3, 9});
    store.grow_rows_geometric(1);
    store.set(0, a);

    detail::BucketJoin<kN> join;
    join.begin_rows(1);
    join.add_row(fp_of<kN>(a), 0);
    const std::vector<std::vector<PosT>> queries = {positions_of<kN>(absent)};
    join.begin_queries(1);
    join.add_query(0, fp_of<kN>(a)); // the staged row's own key, but the positions are another term's
    join.run(store, [&](size_t q) { return std::span<const PosT>(queries[q]); });
    BOOST_TEST(join.hit(0) == detail::BucketJoin<kN>::kMissing);
}

// A gate reuses the join and the marks, so both must come back clean: the second gate's queries see only
// the second gate's rows, and no mark survives into it. The two gates share rows on purpose, since a
// stale `received` bit on a shared row is exactly what would rotate a pair twice.
BOOST_AUTO_TEST_CASE(bucket_join_and_marks_are_clean_between_gates) {
    constexpr size_t kN = 32;
    using PosT = detail::OperatorIndex<kN>::PosT;
    std::mt19937_64 rng(20260909);
    const auto terms = draw_distinct<kN>(rng, 200);
    detail::OperatorIndex<kN> store(6);
    store.grow_rows_geometric(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        store.set(i, terms[i]);
    }

    detail::BucketJoin<kN> join;
    detail::RowMarks marks;
    // Gate 1: rows 0..63 (word 0), gate 2: rows 32..95 (words 0 and 1) -- 32 rows in common.
    const std::vector<detail::EvenParityNzWord> nz1 = {
        {.base = 0, .overlap = ~uint64_t{0}, .foll = 0x5555'5555'5555'5555ULL}};
    const std::vector<detail::EvenParityNzWord> nz2 = {{.base = 0, .overlap = 0xFFFF'FFFF'0000'0000ULL, .foll = 0},
                                                       {.base = 64, .overlap = 0x0000'0000'FFFF'FFFFULL, .foll = 0}};

    // What the scan does per gate: clear the marks over this gate's words, stage every anticommuting row
    // for the join, and record each row's pivot bit.
    auto stage = [&](const std::vector<detail::EvenParityNzWord> &nz) {
        marks.begin(terms.size(), nz);
        join.begin_rows(64);
        for (const auto &w : nz) {
            for (uint64_t m = w.overlap; m != 0U; m &= m - 1) {
                const size_t tz = static_cast<size_t>(std::countr_zero(m));
                const size_t row = w.base + tz;
                join.add_row(fp_of<kN>(terms[row]), row);
                if (((w.foll >> tz) & 1U) != 0U) {
                    marks.set_foll(row);
                }
            }
        }
    };
    auto ask = [&](const std::vector<size_t> &rows) {
        std::vector<std::vector<PosT>> queries;
        join.begin_queries(rows.size());
        for (size_t q = 0; q < rows.size(); ++q) {
            queries.push_back(positions_of<kN>(terms[rows[q]]));
            join.add_query(q, fp_of<kN>(terms[rows[q]]));
        }
        join.run(store, [&](size_t q) { return std::span<const PosT>(queries[q]); });
    };

    stage(nz1);
    for (size_t row = 0; row < 64; ++row) {
        BOOST_REQUIRE(!marks.rot(row));
        BOOST_REQUIRE(!marks.received(row));
        BOOST_REQUIRE_EQUAL(marks.foll(row), (row % 2) == 0);
        marks.set_rot(row);
        marks.set_received(row);
        marks.set_partner_rot(row);
    }
    ask({0, 5, 63});
    BOOST_TEST(join.hit(0) == 0U);
    BOOST_TEST(join.hit(1) == 5U);
    BOOST_TEST(join.hit(2) == 63U);

    stage(nz2);
    // Rows 32..95 are this gate's; every mark the first gate set on the shared rows is gone, and the
    // foll bits are this gate's (none), not the first gate's alternating pattern.
    for (size_t row = 32; row < 96; ++row) {
        BOOST_REQUIRE(!marks.rot(row));
        BOOST_REQUIRE(!marks.received(row));
        BOOST_REQUIRE(!marks.partner_rot(row));
        BOOST_REQUIRE(!marks.foll(row));
    }
    // A row of the first gate only (row 5) is no longer staged, so its own key must miss now.
    ask({5, 40, 95});
    BOOST_TEST(join.hit(0) == detail::BucketJoin<kN>::kMissing);
    BOOST_TEST(join.hit(1) == 40U);
    BOOST_TEST(join.hit(2) == 95U);
}

// Retained capacity is bounded: a large gate followed by a tiny one releases the large buffers, the same
// rule the per-gate scratch has always had.
BOOST_AUTO_TEST_CASE(bucket_join_releases_capacity_after_a_large_gate) {
    constexpr size_t kN = 16;
    using PosT = detail::OperatorIndex<kN>::PosT;
    detail::OperatorIndex<kN> store(4);
    store.grow_rows_geometric(1);
    store.set(0, indices_to_bitset<kN>({1, 2}));

    detail::BucketJoin<kN> join;
    join.begin_rows(200000);
    join.begin_queries(200000);
    for (size_t i = 0; i < 200000; ++i) {
        join.add_row(routing::mix64(i), 0);
        join.add_query(i, routing::mix64(i));
    }
    join.run(store, [](size_t /*q*/) { return std::span<const PosT>(); });
    const size_t big = join.memory_bytes();
    // clear_rows() is the reset the gates that stage nothing take, so it must keep every buffer: those
    // gates outnumber the rest, and releasing on each of them would refault the row side every time.
    join.clear_rows();
    BOOST_TEST(join.rows() == 0U);
    BOOST_TEST(join.memory_bytes() == big);
    join.begin_rows(4);
    join.begin_queries(4);
    BOOST_TEST(join.memory_bytes() < big / 4);
}
