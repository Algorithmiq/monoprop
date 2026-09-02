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

// The position-form resolve path, differentially against the queries the caller built, the dense
// Monomial-keyed insert path and a transient TermLookup oracle, none of which is the code under test.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <span>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/AntiTable.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/TermLookup.h"

using namespace monoprop;

namespace {

template <size_t NumModes>
auto random_monomial(std::mt19937_64 &rng, size_t k) -> Monomial<NumModes> {
    Monomial<NumModes> m;
    std::uniform_int_distribution<size_t> bit(0, Monomial<NumModes>::size() - 1);
    size_t placed = 0;
    while (placed < k) {
        const size_t b = bit(rng);
        if (!m.test(b)) {
            m.set(b);
            ++placed;
        }
    }
    return m;
}

// Fully paired terms are the only source of wide records in production, so drawn here explicitly.
template <size_t NumModes>
auto random_paired_monomial(std::mt19937_64 &rng, size_t d) -> Monomial<NumModes> {
    Monomial<NumModes> m;
    std::uniform_int_distribution<size_t> mode(0, NumModes - 1);
    size_t placed = 0;
    while (placed < d) {
        const size_t mo = mode(rng);
        if (!m.test(2 * mo)) {
            m.set(2 * mo);
            m.set((2 * mo) + 1);
            ++placed;
        }
    }
    return m;
}

// 0 and 1 for the degenerate records, up to 20 for multi-word ones, 14 for the overflow spill.
const std::vector<size_t> kPopcounts = {0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 14, 20};

template <size_t NumModes>
auto make_op(const std::vector<Monomial<NumModes>> &terms) -> detail::MPOperator<NumModes> {
    detail::MPOperator<NumModes> op;
    op.basis = Basis::Majorana;
    if (terms.empty()) {
        return op;
    }
    detail::insert_absent_terms<NumModes>(op, terms.size(), [&](size_t k, size_t base) {
        assign_row<NumModes>(*op.store, base + k, terms[k]);
    });
    return op;
}

// The partner table a gate that finds EVERY row anticommuting would build: the receiver-side oracle the
// probe runs against. Spilled rows take the dense fingerprint, exactly as the scan does.
template <size_t NumModes>
auto table_over_all_rows(const detail::MPOperator<NumModes> &op) -> detail::AntiTable<NumModes> {
    detail::AntiTable<NumModes> table;
    const size_t n = op.store->size();
    table.begin(n);
    const uint64_t *labels = routing::linear_basis<2 * NumModes>().data();
    for (size_t i = 0; i < n; ++i) {
        const auto src = op.store->row_positions(i);
        const uint64_t fp = src.inlined() ? routing::fingerprint_positions(labels, src.pos.data(), src.pos.size())
                                          : routing::linear_hash<2 * NumModes>(op.store->row(i));
        table.add(static_cast<TermIndex>(i), fp);
    }
    return table;
}

template <size_t NumModes>
auto draw_distinct(std::mt19937_64 &rng, size_t n) -> std::vector<Monomial<NumModes>> {
    std::vector<Monomial<NumModes>> out;
    std::set<std::vector<uint64_t>> seen;
    std::uniform_int_distribution<size_t> pick(0, kPopcounts.size() - 1);
    while (out.size() < n) {
        const size_t k = kPopcounts[pick(rng)];
        const auto m =
            ((rng() & 1U) != 0U) ? random_paired_monomial<NumModes>(rng, k / 2) : random_monomial<NumModes>(rng, k);
        std::vector<uint64_t> key;
        key.reserve(Monomial<NumModes>::num_words());
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        if (seen.insert(key).second) {
            out.push_back(m);
        }
    }
    return out;
}

// Extracts an ascending position vector from a Monomial: the wire record is built from positions, and
// production never encodes straight from a bitset.
template <size_t NumModes>
auto positions_of(const Monomial<NumModes> &m) -> std::vector<uint16_t> {
    std::vector<uint16_t> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<uint16_t>(b));
    }
    return pos;
}

template <size_t NumModes>
auto serialize(const std::vector<std::vector<Monomial<NumModes>>> &queries, bool fused, mpi::SlotWindow window)
    -> mpi::WindowVec<VecZ> {
    mpi::WindowVec<VecZ> incoming(window);
    for (size_t s = 0; s < queries.size(); ++s) {
        VecZ &buf = incoming[mpi::WindowIndex{s}];
        for (size_t q = 0; q < queries[s].size(); ++q) {
            const int phase = ((q % 2) == 0) ? 1 : -1;
            const auto pos = positions_of<NumModes>(queries[s][q]);
            detail::QueryWire<NumModes>::push(buf, pos, phase);
            if (fused) {
                detail::QueryWire<NumModes>::push_value(buf, 0.5 + static_cast<double>(q));
            }
        }
    }
    return incoming;
}

template <size_t NumModes>
auto check_probe_matches_the_queries(std::mt19937_64 &rng,
                                     size_t n_seed,
                                     size_t n_query,
                                     size_t rank_count,
                                     bool fused,
                                     size_t window_base = 0) -> void {
    // A non-zero base is the case a re-basing bug survives: sender_slot must still name the flat slot.
    const mpi::SlotWindow window{.base = window_base, .count = rank_count};
    const auto seed_terms = draw_distinct<NumModes>(rng, n_seed);
    const auto fresh_terms = draw_distinct<NumModes>(rng, n_query);

    // Hits matter even though the production hit rate is ~0: only they exercise the confirm.
    std::vector<std::vector<Monomial<NumModes>>> queries(rank_count);
    std::set<std::vector<uint64_t>> queried;
    size_t hits_planned = 0;
    size_t misses_planned = 0;
    for (size_t i = 0; i < n_query; ++i) {
        const bool want_hit = (i % 3) == 0 && !seed_terms.empty();
        const auto m = want_hit ? seed_terms[i % seed_terms.size()] : fresh_terms[i];
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        // A repeat would violate bulk_insert's precondition; the engine gets distinctness from ^G.
        if (!queried.insert(key).second) {
            continue;
        }
        (want_hit ? hits_planned : misses_planned) += 1;
        queries[i % rank_count].push_back(m);
    }
    BOOST_REQUIRE(hits_planned > 0);
    BOOST_REQUIRE(misses_planned > 0);

    std::vector<Monomial<NumModes>> expect_mono;
    std::vector<int> expect_phase;
    std::vector<size_t> expect_sender;
    for (size_t s = 0; s < rank_count; ++s) {
        for (size_t q = 0; q < queries[s].size(); ++q) {
            expect_mono.push_back(queries[s][q]);
            expect_phase.push_back(((q % 2) == 0) ? 1 : -1);
            expect_sender.push_back(s);
        }
    }

    const auto incoming = serialize<NumModes>(queries, fused, window);
    const detail::QueryForm form = fused ? detail::QueryForm::Fused : detail::QueryForm::Plain;

    auto op = make_op<NumModes>(seed_terms);
    const auto table = table_over_all_rows<NumModes>(op);
    const auto pr = detail::probe_incoming_queries<NumModes>(incoming, op, form, table);
    BOOST_REQUIRE_EQUAL(pr.window.base, window.base);
    BOOST_REQUIRE_EQUAL(pr.window.count, window.count);

    BOOST_REQUIRE_EQUAL(pr.nq_total, expect_mono.size());
    BOOST_REQUIRE(pr.nq_total > 0);
    BOOST_REQUIRE_EQUAL(pr.pos_off.size(), pr.nq_total);

    std::set<std::vector<uint64_t>> seeded;
    for (const auto &m : seed_terms) {
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        seeded.insert(key);
    }

    size_t hits_seen = 0;
    size_t wide_seen = 0;
    std::vector<Monomial<NumModes>> expected_misses;
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const Monomial<NumModes> &want = expect_mono[g];
        BOOST_TEST((pr.mono_at(g) == want));
        BOOST_TEST(pr.k_of[g] == want.count());
        BOOST_TEST(pr.phase_of[g] == expect_phase[g]);
        BOOST_TEST(pr.sender_index(g).value == expect_sender[g]);
        BOOST_TEST(pr.sender_slot(g) == window.base + expect_sender[g]);
        BOOST_TEST(pr.is_paired_at(g) == monoprop::is_paired<NumModes>(want));

        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(want.word(w));
        }
        const bool want_hit = seeded.count(key) != 0;
        BOOST_TEST((pr.idx_of[g] < pr.base) == want_hit);
        BOOST_TEST((pr.ord_of[g] != detail::AntiTable<NumModes>::kNone) == want_hit);
        if (want_hit) {
            BOOST_TEST(table.row_of(pr.ord_of[g]) == pr.idx_of[g]);
            ++hits_seen;
        }
        else {
            expected_misses.push_back(want);
        }
        VecZ scratch;
        const auto want_pos = positions_of<NumModes>(want);
        if (detail::QueryWire<NumModes>::push(scratch, want_pos, expect_phase[g]) > 1U) {
            ++wide_seen;
        }
    }
    // Vacuous-pass guards: no hit means the confirm never ran, no wide record means the cursor didn't.
    BOOST_TEST(hits_seen > 0);
    BOOST_TEST(wide_seen > 0);

    BOOST_REQUIRE_EQUAL(pr.miss_g.size(), expected_misses.size());
    for (size_t j = 0; j < pr.miss_g.size(); ++j) {
        BOOST_TEST((expect_mono[pr.miss_g[j]] == expected_misses[j]));
        BOOST_TEST(pr.idx_of[pr.miss_g[j]] == pr.base + j);
    }

    detail::insert_incoming_misses<NumModes>(op, pr);

    // The second implementation: the dense Monomial-keyed path, sharing no code with set_positions.
    auto ref = make_op<NumModes>(seed_terms);
    detail::insert_absent_terms<NumModes>(ref, expected_misses.size(), [&](size_t j, size_t base) {
        assign_row<NumModes>(*ref.store, base + j, expected_misses[j]);
    });

    BOOST_REQUIRE_EQUAL(op.store->size(), ref.store->size());
    BOOST_TEST(op.store->size() > pr.base);
    size_t overflow_seen = 0;
    for (size_t i = 0; i < ref.store->size(); ++i) {
        BOOST_TEST((op.store->row(i) == ref.store->row(i)));
        BOOST_TEST(op.store->popcount(i) == ref.store->popcount(i));
        if (!ref.store->row_positions(i).inlined()) {
            ++overflow_seen;
        }
    }
    BOOST_TEST(overflow_seen > 0);

    // Every inserted row is findable again by value, and distinct: the next gate's table is built from
    // exactly these rows, so a duplicate or an unreadable row would surface there.
    const auto lookup = detail::build_term_lookup<NumModes>(*op.store, 0, op.store->size());
    BOOST_REQUIRE_EQUAL(lookup.size(), op.store->size());
    for (size_t i = 0; i < ref.store->size(); ++i) {
        const auto it = lookup.find(ref.store->row(i));
        BOOST_REQUIRE(it != lookup.end());
        BOOST_TEST(it->second == i);
    }
}

} // namespace

/* ── The check, across both position widths and both buffer layouts ── */

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_narrow_positions) {
    std::mt19937_64 rng(20260814);
    static_assert(sizeof(detail::OperatorIndex<32>::PosT) == 1, "this case exists to cover the narrowing decode");
    check_probe_matches_the_queries<32>(rng, /*n_seed=*/40, /*n_query=*/90, /*rank_count=*/3, /*fused=*/false);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_wide_positions) {
    std::mt19937_64 rng(20260815);
    static_assert(sizeof(detail::OperatorIndex<250>::PosT) == 2, "this case exists to cover the wide store");
    check_probe_matches_the_queries<250>(rng,
                                         /*n_seed=*/60,
                                         /*n_query=*/140,
                                         /*rank_count=*/4,
                                         /*fused=*/false,
                                         /*window_base=*/16);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_fused_layout) {
    std::mt19937_64 rng(20260816);
    check_probe_matches_the_queries<250>(rng,
                                         /*n_seed=*/50,
                                         /*n_query=*/120,
                                         /*rank_count=*/2,
                                         /*fused=*/true,
                                         /*window_base=*/6);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_single_sender) {
    std::mt19937_64 rng(20260817);
    check_probe_matches_the_queries<32>(rng, /*n_seed=*/25, /*n_query=*/60, /*rank_count=*/1, /*fused=*/true);
}

/* ── The pieces, pinned individually ──────────────────────────────────────── */

BOOST_AUTO_TEST_CASE(sparse_resolve_set_positions_matches_set) {
    constexpr size_t kN = 250;
    constexpr size_t kInlineWidth = 11;
    std::mt19937_64 rng(20260818);
    const auto terms = draw_distinct<kN>(rng, 200);

    detail::OperatorIndex<kN> from_mono(kInlineWidth);
    detail::OperatorIndex<kN> from_pos(kInlineWidth);
    from_mono.grow_rows_geometric(terms.size());
    from_pos.grow_rows_geometric(terms.size());

    size_t spilled = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        from_mono.set(i, terms[i]);
        std::vector<detail::OperatorIndex<kN>::PosT> pos;
        for (size_t b = terms[i].find_first(); b < terms[i].size(); b = terms[i].find_next(b)) {
            pos.push_back(static_cast<detail::OperatorIndex<kN>::PosT>(b));
        }
        from_pos.set_positions(i, pos);
        if (pos.size() > kInlineWidth) {
            ++spilled;
        }
    }
    BOOST_TEST(spilled > 0);
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_TEST((from_pos.row(i) == from_mono.row(i)));
        BOOST_TEST((from_pos.row(i) == terms[i]));
        BOOST_TEST(from_pos.popcount(i) == from_mono.popcount(i));
        BOOST_TEST(from_pos.row_positions(i).inlined() == from_mono.row_positions(i).inlined());
    }
    BOOST_TEST(from_pos.overflow_size() == from_mono.overflow_size());
}

BOOST_AUTO_TEST_CASE(sparse_resolve_table_probe_matches_the_lookup_oracle) {
    // The partner table against a TermLookup over the same rows: every seeded term hits its own row from
    // its positions, and a genuinely absent term misses. Spilled (wide) rows are in the draw.
    constexpr size_t kN = 250;
    std::mt19937_64 rng(20260819);
    const auto terms = draw_distinct<kN>(rng, 300);
    auto op = make_op<kN>(terms);
    const auto table = table_over_all_rows<kN>(op);
    const auto lookup = detail::build_term_lookup<kN>(*op.store, 0, op.store->size());
    const uint64_t *labels = routing::linear_basis<2 * kN>().data();

    size_t spilled = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        std::vector<detail::OperatorIndex<kN>::PosT> pos;
        for (size_t b = terms[i].find_first(); b < terms[i].size(); b = terms[i].find_next(b)) {
            pos.push_back(static_cast<detail::OperatorIndex<kN>::PosT>(b));
        }
        const uint64_t fp = routing::fingerprint_positions(labels, pos.data(), pos.size());
        BOOST_REQUIRE_EQUAL(fp, routing::linear_hash<2 * kN>(terms[i]));
        const auto ord = table.probe(*op.store, fp, pos);
        BOOST_REQUIRE(ord != detail::AntiTable<kN>::kNone);
        BOOST_TEST(table.row_of(ord) == i);
        BOOST_TEST(lookup.at(terms[i]) == i);
        if (!op.store->row_positions(i).inlined()) {
            ++spilled;
        }
    }
    BOOST_TEST(spilled > 0);

    const auto absent = draw_distinct<kN>(rng, 50);
    size_t genuinely_absent = 0;
    for (const auto &m : absent) {
        // draw_distinct may re-draw a seeded term; only genuinely absent ones are evidence.
        if (lookup.find(m) != lookup.end()) {
            continue;
        }
        std::vector<detail::OperatorIndex<kN>::PosT> pos;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            pos.push_back(static_cast<detail::OperatorIndex<kN>::PosT>(b));
        }
        const uint64_t fp = routing::fingerprint_positions(labels, pos.data(), pos.size());
        BOOST_TEST(table.probe(*op.store, fp, pos) == detail::AntiTable<kN>::kNone);
        ++genuinely_absent;
    }
    BOOST_TEST(genuinely_absent > 0);
}
