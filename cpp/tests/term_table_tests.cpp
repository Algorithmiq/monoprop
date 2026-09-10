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

// The persistent join key -> row table: every stored row is found under its own key and positions,
// absent keys miss, the batched probe agrees with a transient map over the same rows -- answer for
// answer and in the miss order the resolve assigns row indices in -- growth keeps the load bound and
// every earlier row, a 32-bit key collision is settled by the row confirm, and the staleness guard
// catches a growth that bypassed reindex_after_growth.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/RowKey.h"
#include "monoprop/detail/operator/TermTable.h"

using namespace monoprop;

namespace {

constexpr size_t kN = 32; // 64 positions: more than the 32 key bits, so a key collision is constructible
using Store = detail::OperatorIndex<kN>;
using PosT = Store::PosT;
using Op = detail::MPOperator<kN>;

auto key_of(const Monomial<kN> &m) -> uint32_t {
    return detail::key_of<2 * kN>(m);
}

auto positions_of(const Monomial<kN> &m) -> std::vector<PosT> {
    std::vector<PosT> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<PosT>(b));
    }
    return pos;
}

auto random_monomial(std::mt19937_64 &rng, size_t k) -> Monomial<kN> {
    Monomial<kN> m;
    std::uniform_int_distribution<size_t> bit(0, Monomial<kN>::size() - 1);
    while (m.count() < k) {
        m.set(bit(rng));
    }
    return m;
}

auto words_of(const Monomial<kN> &m) -> std::vector<uint64_t> {
    std::vector<uint64_t> key;
    for (size_t w = 0; w < Monomial<kN>::num_words(); ++w) {
        key.push_back(m.word(w));
    }
    return key;
}

// `n` pairwise distinct monomials, none of them in `avoid`, with popcounts 1..12 (some spill the
// narrow stores below and land in the wide tier or the side-map).
auto draw_distinct(std::mt19937_64 &rng, size_t n, const std::set<std::vector<uint64_t>> &avoid = {})
    -> std::vector<Monomial<kN>> {
    std::vector<Monomial<kN>> out;
    std::set<std::vector<uint64_t>> seen = avoid;
    std::uniform_int_distribution<size_t> pop(1, 12);
    while (out.size() < n) {
        const auto m = random_monomial(rng, pop(rng));
        if (seen.insert(words_of(m)).second) {
            out.push_back(m);
        }
    }
    return out;
}

// Appends `terms` as rows [base, base + n) through the engine's own growth door, so the table follows.
auto append_terms(Op &op, const std::vector<Monomial<kN>> &terms) -> size_t {
    return detail::insert_absent_terms<kN>(op, terms.size(), [&](size_t k, size_t base) {
        const auto pos = positions_of(terms[k]);
        op.store->set_positions(base + k, std::span<const PosT>(pos));
    });
}

auto find_term(const Op &op, const Monomial<kN> &m) -> size_t {
    const auto pos = positions_of(m);
    return op.term_table().find(*op.store, key_of(m), std::span<const PosT>(pos));
}

// out[q] = the row of query q, or TermTable::kNotFound, through the batched pipeline. `on_hit` fires once
// per confirmed row, which is what the gate join marks its rows with, so it is counted here too: the
// hit count the probe returns must equal the calls it made.
auto probe_batch(const Op &op, const std::vector<Monomial<kN>> &asked) -> std::vector<size_t> {
    std::vector<uint32_t> keys;
    std::vector<std::vector<PosT>> pos;
    for (const auto &m : asked) {
        keys.push_back(key_of(m));
        pos.push_back(positions_of(m));
    }
    constexpr TermIndex kMissing = std::numeric_limits<TermIndex>::max();
    std::vector<TermIndex> rows(asked.size(), 0);
    size_t on_hit_calls = 0;
    const size_t hits = op.term_table().find_batch(
        *op.store,
        asked.size(),
        [&keys](size_t q) { return keys[q]; },
        [&pos](size_t q) { return std::span<const PosT>(pos[q]); },
        [&on_hit_calls](size_t /*q*/, size_t /*row*/) { ++on_hit_calls; },
        std::span<TermIndex>(rows),
        kMissing);
    BOOST_TEST(on_hit_calls == hits);
    std::vector<size_t> out;
    out.reserve(asked.size());
    size_t found = 0;
    for (const TermIndex row : rows) {
        out.push_back(row == kMissing ? detail::TermTable::kNotFound : static_cast<size_t>(row));
        found += static_cast<size_t>(row != kMissing);
    }
    BOOST_TEST(found == hits);
    return out;
}

// The oracle the engine's own lookup replaces: a dense map over the same rows, in the same order.
auto oracle_of(const Op &op) -> std::unordered_map<Monomial<kN>, size_t, MonomialHash<kN>, MonomialEqual<kN>> {
    std::unordered_map<Monomial<kN>, size_t, MonomialHash<kN>, MonomialEqual<kN>> map;
    for (size_t i = 0; i < op.store->size(); ++i) {
        map.emplace(op.store->row(i), i);
    }
    return map;
}

// A non-empty set of positions whose join key is 0: Gaussian elimination over the labels' top 32 bits.
// 64 vectors in a 32-dimensional space are dependent, so this always finds one within the first 33.
auto zero_key_positions() -> Monomial<kN> {
    const auto &labels = routing::linear_basis<2 * kN>();
    std::array<uint32_t, 32> basis{}; // basis[b] has its highest set bit at b
    std::array<uint64_t, 32> combo{}; // the positions XORed into basis[b]
    for (size_t p = 0; p < 2 * kN; ++p) {
        auto v = static_cast<uint32_t>(labels[p] >> 32U);
        uint64_t c = uint64_t{1} << p;
        while (v != 0) {
            const auto b = static_cast<size_t>(31 - std::countl_zero(v));
            if (basis[b] == 0) {
                basis[b] = v;
                combo[b] = c;
                break;
            }
            v ^= basis[b];
            c ^= combo[b];
        }
        if (v == 0) {
            Monomial<kN> d;
            for (size_t q = 0; q < 2 * kN; ++q) {
                if (((c >> q) & 1U) != 0) {
                    d.set(q);
                }
            }
            return d;
        }
    }
    BOOST_FAIL("64 labels in a 32-bit space must be dependent");
    return {};
}

} // namespace

BOOST_AUTO_TEST_CASE(term_table_finds_every_row_under_its_own_key_and_misses_absent_ones) {
    std::mt19937_64 rng(11);
    const auto terms = draw_distinct(rng, 3000);
    Op op;
    op.store = std::make_unique<Store>(/*inline_width=*/3);
    append_terms(op, terms);
    const detail::TermTable &t = op.term_table();
    BOOST_TEST(t.rows() == terms.size());
    BOOST_TEST(std::has_single_bit(t.slots()));
    BOOST_TEST(t.rows() * 10 <= t.slots() * 7); // the load bound
    BOOST_TEST(t.memory_bytes() >= t.slots() * sizeof(uint32_t));

    size_t spilled = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(find_term(op, terms[i]) == i);
        if (!op.store->row_positions(i).inlined()) {
            ++spilled;
        }
    }
    BOOST_TEST(spilled > 0); // the side-map / wide-tier rows are reached through the same confirm

    std::set<std::vector<uint64_t>> seeded;
    for (const auto &m : terms) {
        seeded.insert(words_of(m));
    }
    for (const auto &m : draw_distinct(rng, 200, seeded)) {
        BOOST_TEST(find_term(op, m) == detail::TermTable::kNotFound);
    }
}

// The two properties the resolve depends on, against a dense map over the same rows: the batched probe
// answers exactly what a by-value lookup answers, and the misses -- taken in query-stream order -- get
// the indices base+j that the insert then writes rows at.
BOOST_AUTO_TEST_CASE(term_table_batch_probe_agrees_with_a_dense_map_answer_and_miss_order) {
    std::mt19937_64 rng(12);
    const auto terms = draw_distinct(rng, 2500);
    Op op;
    append_terms(op, terms);
    std::set<std::vector<uint64_t>> seeded;
    for (const auto &m : terms) {
        seeded.insert(words_of(m));
    }
    const auto absent = draw_distinct(rng, 500, seeded);

    // Present and absent keys interleaved, in a shuffled order, so every pipeline group mixes both.
    std::vector<Monomial<kN>> asked = terms;
    asked.insert(asked.end(), absent.begin(), absent.end());
    std::shuffle(asked.begin(), asked.end(), rng);

    const auto out = probe_batch(op, asked);
    const auto oracle = oracle_of(op);
    // Every query's resolved index, hit or miss: a miss takes the next free row, base + j, in
    // query-stream order -- the assignment the insert then writes rows at.
    const size_t base = op.store->size();
    std::vector<size_t> table_idx;
    std::vector<size_t> oracle_idx;
    size_t table_next = base;
    size_t oracle_next = base;
    size_t hits = 0;
    for (size_t q = 0; q < asked.size(); ++q) {
        BOOST_TEST_INFO("query " << q);
        const auto it = oracle.find(asked[q]);
        table_idx.push_back(out[q] == detail::TermTable::kNotFound ? table_next++ : out[q]);
        oracle_idx.push_back(it == oracle.end() ? oracle_next++ : it->second);
        if (it != oracle.end()) {
            BOOST_TEST(find_term(op, asked[q]) == it->second); // and the one-at-a-time find agrees
            ++hits;
        }
    }
    BOOST_TEST(hits == terms.size());
    BOOST_TEST(table_next == base + absent.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(table_idx.begin(), table_idx.end(), oracle_idx.begin(), oracle_idx.end());
}

// The empty-table probe: no slots exist yet, so every query misses without a load (the early return).
BOOST_AUTO_TEST_CASE(term_table_probe_of_an_empty_table_is_all_missing) {
    std::mt19937_64 rng(17);
    Op op;
    const auto asked = draw_distinct(rng, 40);
    BOOST_TEST(op.term_table().rows() == 0U);
    for (const size_t row : probe_batch(op, asked)) {
        BOOST_TEST(row == detail::TermTable::kNotFound);
    }
    BOOST_TEST(find_term(op, asked[0]) == detail::TermTable::kNotFound);
}

BOOST_AUTO_TEST_CASE(term_table_growth_keeps_the_load_bound_and_every_earlier_row) {
    std::mt19937_64 rng(13);
    const auto terms = draw_distinct(rng, 20000);
    Op op;
    // Materialise the table on an empty store: the first append must size it from nothing.
    BOOST_TEST(op.term_table().rows() == 0U);

    std::uniform_int_distribution<size_t> batch(1, 1500);
    size_t inserted = 0;
    size_t growths = 0;
    size_t last_slots = op.term_table().slots();
    while (inserted < terms.size()) {
        const size_t n = std::min(batch(rng), terms.size() - inserted);
        const std::vector<Monomial<kN>> chunk(terms.begin() + static_cast<std::ptrdiff_t>(inserted),
                                              terms.begin() + static_cast<std::ptrdiff_t>(inserted + n));
        const size_t base = append_terms(op, chunk);
        BOOST_TEST(base == inserted);
        inserted += n;
        const detail::TermTable &t = op.term_table();
        BOOST_TEST(t.rows() == inserted);
        BOOST_TEST(t.rows() * 10 <= t.slots() * 7);
        if (t.slots() != last_slots) {
            BOOST_TEST(t.slots() > last_slots);
            ++growths;
            last_slots = t.slots();
        }
        // A sample of everything inserted so far, the oldest rows included.
        for (size_t k = 0; k < 64; ++k) {
            const size_t i = (k * 7919U) % inserted;
            BOOST_TEST(find_term(op, terms[i]) == i);
        }
    }
    BOOST_TEST(growths >= 5U); // 16 slots up to 20000 rows is at least ten doublings
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(find_term(op, terms[i]) == i);
    }
}

// The table holds no key of its own: it folds a row's key off the row when it indexes it
// (OperatorIndex::key_of_row), so a row must be findable under the key of its own positions whichever
// tier holds it -- inline, the wide tier, or the side-map, where there is no position array at all and
// the fold falls back to the dense form. A rebuild (forced through the staleness guard) has to reach
// the same verdict as the appends that indexed the rows one batch at a time.
BOOST_AUTO_TEST_CASE(term_table_folds_a_rows_key_off_every_storage_tier) {
    std::mt19937_64 rng(29);
    // Inline width 4 (kMinInlineForWideTier, the narrowest that still leaves room for a tier slot) and
    // a structural bound of 5: popcounts 1..12 put rows in all three tiers.
    Op op;
    op.store = std::make_unique<Store>(/*inline_width=*/4, /*forced_rows_per_chunk=*/0, /*wide_width=*/5);
    const auto terms = draw_distinct(rng, 4000);
    append_terms(op, terms);
    BOOST_REQUIRE_GT(op.store->wide_size(), 0U);
    BOOST_REQUIRE_GT(op.store->overflow_size(), 0U);

    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        const auto pos = positions_of(terms[i]);
        BOOST_TEST(op.store->key_of_row(i) == key_of(terms[i]));
        BOOST_TEST(op.store->key_of_row(i) == detail::key_of_positions<2 * kN>(pos.data(), pos.size()));
        BOOST_TEST(find_term(op, terms[i]) == i);
    }

    // Same table, built in one pass from the store instead of appended batch by batch.
    const size_t slots_appended = op.term_table().slots();
    Op rebuilt;
    rebuilt.store = std::make_unique<Store>(/*inline_width=*/4, /*forced_rows_per_chunk=*/0, /*wide_width=*/5);
    append_terms(rebuilt, terms);
    (void)rebuilt.term_table();
    rebuilt.term_table_.reset(); // force the rebuild arm
    BOOST_TEST(rebuilt.term_table().slots() == slots_appended);
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST_INFO("row " << i);
        BOOST_TEST(find_term(rebuilt, terms[i]) == i);
    }
}

// Two distinct terms with the same 32-bit key share a home slot AND a prefilter (both are functions of
// the key), so the second is only ever reached by refuting the first against the row: the collision arm.
// A collision therefore costs a compare and can never answer with the wrong row.
BOOST_AUTO_TEST_CASE(term_table_settles_a_key_collision_by_the_row_confirm) {
    const Monomial<kN> d = zero_key_positions();
    BOOST_REQUIRE(d.any());
    BOOST_REQUIRE_EQUAL(key_of(d), 0U);

    std::mt19937_64 rng(14);
    const auto filler = draw_distinct(rng, 700);
    std::vector<Monomial<kN>> terms;
    // Three mutually colliding terms spread through the store: s, s ^ d and a third with the same key
    // (t ^ d for another t), interleaved with filler so the chain has other occupants too.
    Monomial<kN> s = random_monomial(rng, 5);
    while ((s ^ d).count() == 0 || s == (s ^ d)) {
        s = random_monomial(rng, 5);
    }
    const Monomial<kN> s2 = s ^ d;
    BOOST_REQUIRE_EQUAL(key_of(s), key_of(s2));
    BOOST_REQUIRE(s != s2);
    Monomial<kN> t = random_monomial(rng, 4);
    while (key_of(t) == key_of(s) || t == s || t == s2) {
        t = random_monomial(rng, 4);
    }
    const Monomial<kN> t2 = t ^ d;
    std::set<std::vector<uint64_t>> used;
    for (const auto &m : {s, s2, t, t2}) {
        used.insert(words_of(m));
    }
    size_t f = 0;
    auto take_filler = [&](size_t n) {
        for (size_t k = 0; k < n && f < filler.size(); ++k, ++f) {
            if (used.insert(words_of(filler[f])).second) {
                terms.push_back(filler[f]);
            }
        }
    };
    take_filler(100);
    terms.push_back(s);
    take_filler(200);
    terms.push_back(s2);
    take_filler(150);
    terms.push_back(t);
    take_filler(50);
    terms.push_back(t2);
    take_filler(200);

    Op op;
    append_terms(op, terms);
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(find_term(op, terms[i]) == i);
    }
    // And through the batched probe, colliding queries adjacent in one pipeline group.
    const std::vector<Monomial<kN>> asked = {s, s2, t2, t, s ^ t /* absent unless drawn */};
    const auto out = probe_batch(op, asked);
    const auto oracle = oracle_of(op);
    for (size_t q = 0; q < asked.size(); ++q) {
        const auto it = oracle.find(asked[q]);
        BOOST_TEST(out[q] == (it == oracle.end() ? detail::TermTable::kNotFound : it->second));
        if (out[q] != detail::TermTable::kNotFound) {
            BOOST_CHECK(op.store->row(out[q]) == asked[q]);
        }
    }
}

BOOST_AUTO_TEST_CASE(term_table_staleness_guard_catches_growth_that_bypassed_reindex) {
    std::mt19937_64 rng(15);
    const auto terms = draw_distinct(rng, 300);
    Op op;
    append_terms(op, std::vector<Monomial<kN>>(terms.begin(), terms.begin() + 200));
    BOOST_TEST(op.term_table().rows() == 200U);
    // append_term grows the store without telling the indices (the setup path).
    for (size_t i = 200; i < terms.size(); ++i) {
        op.append_term(terms[i]);
    }
    BOOST_TEST(op.term_table().rows() == 300U);
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST(find_term(op, terms[i]) == i);
    }
    // A copy carries its own table and follows its own growth.
    Op copy(op);
    BOOST_TEST(copy.term_table().rows() == 300U);
    std::set<std::vector<uint64_t>> seeded;
    for (const auto &m : terms) {
        seeded.insert(words_of(m));
    }
    const auto more = draw_distinct(rng, 50, seeded);
    append_terms(copy, more);
    BOOST_TEST(copy.term_table().rows() == 350U);
    BOOST_TEST(op.term_table().rows() == 300U);
    BOOST_TEST(find_term(copy, more[0]) == 300U);
    BOOST_TEST(find_term(op, more[0]) == detail::TermTable::kNotFound);
}
