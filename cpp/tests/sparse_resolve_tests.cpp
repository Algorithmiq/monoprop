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

// The position-form resolve path, differentially against the queries the caller built and the dense
// Monomial-keyed insert path, neither of which is the code under test.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/QueryCodec.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"

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

// Fully paired terms are the only source of wide records: 94 in 20.9M in production, so drawn here.
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
    detail::insert_absent_terms<NumModes>(
        op,
        terms.size(),
        [&](size_t k) -> const Monomial<NumModes> & { return terms[k]; },
        [&](size_t k, size_t base) { assign_row<NumModes>(*op.store, base + k, terms[k]); });
    return op;
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

template <size_t NumModes>
auto serialize(const std::vector<std::vector<Monomial<NumModes>>> &queries, bool fused) -> std::vector<VecZ> {
    std::vector<VecZ> incoming(queries.size());
    for (size_t s = 0; s < queries.size(); ++s) {
        for (size_t q = 0; q < queries[s].size(); ++q) {
            const int phase = ((q % 2) == 0) ? 1 : -1;
            detail::QueryCodec<NumModes>::push(incoming[s], queries[s][q], phase);
            if (fused) {
                detail::QueryCodec<NumModes>::push_value(incoming[s], 0.5 + static_cast<double>(q));
            }
        }
    }
    return incoming;
}

template <size_t NumModes>
auto check_probe_matches_the_queries(std::mt19937_64 &rng, size_t n_seed, size_t n_query, size_t rank_count, bool fused)
    -> void {
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

    const auto incoming = serialize<NumModes>(queries, fused);
    const detail::QueryLayout layout{fused};

    auto op = make_op<NumModes>(seed_terms);
    const auto pr = detail::probe_incoming_queries<NumModes>(incoming, op, rank_count, layout);

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
        BOOST_TEST(pr.sender_of[g] == expect_sender[g]);
        BOOST_TEST(pr.is_paired_at(g) == monoprop::is_paired<NumModes>(want));

        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(want.word(w));
        }
        const bool want_hit = seeded.count(key) != 0;
        BOOST_TEST((pr.idx_of[g] < pr.base) == want_hit);
        if (want_hit) {
            ++hits_seen;
        }
        else {
            expected_misses.push_back(want);
        }
        VecZ scratch;
        if (detail::QueryCodec<NumModes>::push(scratch, want, expect_phase[g]) > 1U) {
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
    detail::insert_absent_terms<NumModes>(
        ref,
        expected_misses.size(),
        [&](size_t j) -> const Monomial<NumModes> & { return expected_misses[j]; },
        [&](size_t j, size_t base) { assign_row<NumModes>(*ref.store, base + j, expected_misses[j]); });

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

    // The index, not just the rows: a wrong hash leaves the row correct and unfindable.
    for (size_t i = 0; i < ref.store->size(); ++i) {
        const auto key = ref.store->row(i);
        const auto in_op = op.store->find(key);
        const auto in_ref = ref.store->find(key);
        BOOST_REQUIRE(in_ref.has_value());
        BOOST_REQUIRE(in_op.has_value());
        BOOST_TEST(*in_op == *in_ref);
        BOOST_TEST(*in_ref == i);
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
    check_probe_matches_the_queries<250>(rng, /*n_seed=*/60, /*n_query=*/140, /*rank_count=*/4, /*fused=*/false);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_fused_layout) {
    std::mt19937_64 rng(20260816);
    check_probe_matches_the_queries<250>(rng, /*n_seed=*/50, /*n_query=*/120, /*rank_count=*/2, /*fused=*/true);
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
        from_pos.set_positions(i, pos.data(), pos.size());
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

BOOST_AUTO_TEST_CASE(sparse_resolve_finds_dense_inserted_keys) {
    // The hash identity, isolated: fold_hash_positions differing from fold_hash misses, and legally.
    constexpr size_t kN = 250;
    std::mt19937_64 rng(20260819);
    const auto terms = draw_distinct<kN>(rng, 300);
    auto op = make_op<kN>(terms);

    std::vector<detail::OperatorIndex<kN>::PosT> flat;
    std::vector<size_t> off;
    std::vector<uint32_t> kk;
    for (const auto &m : terms) {
        off.push_back(flat.size());
        size_t k = 0;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            flat.push_back(static_cast<detail::OperatorIndex<kN>::PosT>(b));
            ++k;
        }
        kk.push_back(static_cast<uint32_t>(k));
    }
    std::vector<size_t> out(terms.size(), 0);
    std::vector<uint32_t> hashes(terms.size(), 0);
    op.store->find_batch_positions(flat.data(), off.data(), kk.data(), terms.size(), out.data(), hashes.data());

    std::vector<size_t> out_dense(terms.size(), 0);
    op.store->find_batch(terms.data(), terms.size(), out_dense.data());
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_REQUIRE(out[i] != detail::OperatorIndex<kN>::kNotFound);
        BOOST_TEST(out[i] == i);
        BOOST_TEST(out[i] == out_dense[i]);
        BOOST_TEST(hashes[i] == detail::OperatorIndex<kN>::fold_hash_positions(flat.data() + off[i], kk[i]));
    }

    const auto absent = draw_distinct<kN>(rng, 50);
    std::vector<detail::OperatorIndex<kN>::PosT> aflat;
    std::vector<size_t> aoff;
    std::vector<uint32_t> akk;
    for (const auto &m : absent) {
        aoff.push_back(aflat.size());
        size_t k = 0;
        for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
            aflat.push_back(static_cast<detail::OperatorIndex<kN>::PosT>(b));
            ++k;
        }
        akk.push_back(static_cast<uint32_t>(k));
    }
    std::vector<size_t> aout(absent.size(), 0);
    op.store->find_batch_positions(aflat.data(), aoff.data(), akk.data(), absent.size(), aout.data(), nullptr);
    size_t genuinely_absent = 0;
    for (size_t i = 0; i < absent.size(); ++i) {
        // draw_distinct may re-draw a seeded term; only genuinely absent ones are evidence.
        if (!op.store->find(absent[i]).has_value()) {
            BOOST_TEST(aout[i] == detail::OperatorIndex<kN>::kNotFound);
            ++genuinely_absent;
        }
    }
    BOOST_TEST(genuinely_absent > 0);
}
