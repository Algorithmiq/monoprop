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

// The resolve path keeps an incoming query in the position form it arrived in instead of inflating it
// into a dense Monomial, hashing that, and then unpacking it again inside OperatorIndex::set. Nothing
// about the RESULT may change: same probe answers, same miss order, same rows, same index.
//
// WHAT THIS COMPARES AGAINST, NOW THAT THERE IS ONE ARM. This was a two-arm differential while the
// dense-inflate path still existed. It is not written as a round-trip in its place, because a
// round-trip against the same codec only asks whether the decoder agrees with its own encoder and is
// blind to the two of them being wrong together. It compares against two things neither of which is
// the code under test:
//
//   * THE QUERIES THE TEST SENT. The caller builds the monomials, so the expected key, popcount,
//     phase and hit/miss classification of every query g are known independently of anything the probe
//     does. That is the oracle for Phase 1-2.
//   * THE DENSE INSERT PATH. insert_absent_terms + assign_row still exist and still key by Monomial,
//     so the same misses can be inserted a second, entirely different way. Requiring the two operators
//     to agree row for row AND index answer for index answer is the oracle for Phase 4. This is the
//     part that would otherwise have been lost, and it is why that path is worth keeping reachable.
//
// The three ways this change can be silently wrong, and where each is covered:
//
//   1. THE HASH DISAGREES. fold_hash_positions must equal fold_hash of the monomial those positions
//      describe. A mismatch does not throw: the probe simply misses a term that is present, the miss is
//      inserted as a second row for one monomial, and every later gate rotates both copies. Covered by
//      seeding the operator through the DENSE path and requiring the probe to find every seeded key
//      (sparse_resolve_finds_dense_inserted_keys), and again end to end below.
//   2. THE ROW IS WRITTEN DIFFERENTLY. set_positions must leave the row set() would have left,
//      including the overflow spill, where there is no position array and a bitset has to be rebuilt.
//   3. CONTINUATION RECORDS. k is not bounded by the cutoff -- a fully paired term is kept
//      unconditionally -- so a query can span more than one record. At the benchmark configuration this
//      fires 94 times in 20.9M and a fuzz run there would prove nothing, so every case below forces the
//      case with terms at k = 6, 7, 8, 14 and 20: the inline boundary from both sides, one continuation,
//      and two.
//
// Widths cover both position types: NumModes = 32 gives 2*NumModes = 64 and OperatorIndex::PosT =
// uint8_t, which is NARROWER than the wire's uint16_t, so the decode narrows; NumModes = 250 gives
// PosT = uint16_t, eight words, and a bit count that is not a multiple of 64.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/CompactQuery.h"
#include "monoprop/detail/evolution/layer_build/QueryCodec.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;

namespace {

// Distinct random bit sets of a given popcount. Arbitrary rather than "well-formed": every property
// under test is a pure function of the bit set, and arbitrary sets reach parts of the position range a
// production monomial never would.
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

// A fully paired monomial of k = 2*d positions. These are the terms the cutoff keeps unconditionally,
// so they are the ones that can exceed the record's six inline positions -- i.e. the only production
// source of continuation records.
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

// The popcounts every case draws from: 0 and 1 for the degenerate records, 6 and 7 for the inline
// boundary from both sides, 8 and 14 for one continuation, 20 for two, and 14 again for the overflow
// spill at the default inline width of 11.
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

// Draw `n` pairwise-distinct monomials with popcounts spanning kPopcounts.
template <size_t NumModes>
auto draw_distinct(std::mt19937_64 &rng, size_t n) -> std::vector<Monomial<NumModes>> {
    std::vector<Monomial<NumModes>> out;
    std::set<std::vector<uint64_t>> seen;
    std::uniform_int_distribution<size_t> pick(0, kPopcounts.size() - 1);
    while (out.size() < n) {
        const size_t k = kPopcounts[pick(rng)];
        // Half the draws are fully paired, which is what makes wide terms (and so continuations) common
        // here instead of one-in-a-million as they are at the benchmark configuration.
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

// Serialize `queries[s]` into one compact buffer per sender, optionally fused (a value word after each
// query, which is what a ContractSink resolver receives).
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

// The core check: probe the wire buffers, verify every observable against the queries the caller built
// (not against a second run of the same code), then insert the misses BOTH ways -- through the resolve
// path under test and through the dense Monomial-keyed insert_absent_terms -- and require the two
// operators to be indistinguishable.
template <size_t NumModes>
auto check_probe_matches_the_queries(std::mt19937_64 &rng, size_t n_seed, size_t n_query, size_t rank_count, bool fused)
    -> void {
    const auto seed_terms = draw_distinct<NumModes>(rng, n_seed);
    const auto fresh_terms = draw_distinct<NumModes>(rng, n_query);

    // Queries: a mix of terms already in the operator (hits) and terms that are not (misses), spread
    // over the senders. Hits matter even though the production hit rate is ~0: the confirm and the
    // hit/miss bookkeeping are only exercised by them.
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
        // A repeated query would be a duplicate insert, which violates bulk_insert's precondition. The
        // engine guarantees distinctness (queries are source^G over distinct sources, ^G injective);
        // this file has to arrange it by hand.
        if (!queried.insert(key).second) {
            continue;
        }
        (want_hit ? hits_planned : misses_planned) += 1;
        queries[i % rank_count].push_back(m);
    }
    BOOST_REQUIRE(hits_planned > 0);
    BOOST_REQUIRE(misses_planned > 0);

    // g runs in (sender, record) order, so this is the expected key and phase of query g, built from
    // what the caller serialized rather than from anything the decode produced.
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

    // The seeded set, so a query's hit/miss status is known without asking the probe.
    std::set<std::vector<uint64_t>> seeded;
    for (const auto &m : seed_terms) {
        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(m.word(w));
        }
        seeded.insert(key);
    }

    size_t hits_seen = 0;
    size_t continued_seen = 0;
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
        // A hit resolves below base (it was already there); a miss is assigned an index at or above it.
        BOOST_TEST((pr.idx_of[g] < pr.base) == want_hit);
        if (want_hit) {
            ++hits_seen;
        }
        else {
            expected_misses.push_back(want);
        }
        if (pr.k_of[g] > detail::CompactQuery<NumModes>::kInlinePositions) {
            ++continued_seen;
        }
    }
    // Both of these guard against a vacuous pass: no hit means the confirm never ran, and no
    // continuation means the multi-record path never ran.
    BOOST_TEST(hits_seen > 0);
    BOOST_TEST(continued_seen > 0);

    // Miss order is (sender, record) order, which is the order expected_misses was built in.
    BOOST_REQUIRE_EQUAL(pr.miss_g.size(), expected_misses.size());
    for (size_t j = 0; j < pr.miss_g.size(); ++j) {
        BOOST_TEST((expect_mono[pr.miss_g[j]] == expected_misses[j]));
        BOOST_TEST(pr.idx_of[pr.miss_g[j]] == pr.base + j);
    }

    detail::insert_incoming_misses<NumModes>(op, pr);

    // The second implementation: the same seeds and the same misses, in the same order, inserted
    // through the dense Monomial-keyed path instead. Nothing it does shares code with set_positions or
    // with the hash the probe folded from the wire.
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
    // The spill path writes the overflow map from a rebuilt bitset rather than from the position array,
    // and is the one place set_positions cannot be a copy. kPopcounts reaches past the default inline
    // width of 11 so that it runs.
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
    // 2*NumModes = 64 -> OperatorIndex::PosT is uint8_t, narrower than the wire's uint16_t.
    std::mt19937_64 rng(20260814);
    static_assert(sizeof(detail::OperatorIndex<32>::PosT) == 1, "this case exists to cover the narrowing decode");
    check_probe_matches_the_queries<32>(rng, /*n_seed=*/40, /*n_query=*/90, /*rank_count=*/3, /*fused=*/false);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_wide_positions) {
    // 2*NumModes = 500: eight words, not a multiple of 64, PosT = uint16_t.
    std::mt19937_64 rng(20260815);
    static_assert(sizeof(detail::OperatorIndex<250>::PosT) == 2, "this case exists to cover the wide store");
    check_probe_matches_the_queries<250>(rng, /*n_seed=*/60, /*n_query=*/140, /*rank_count=*/4, /*fused=*/false);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_fused_layout) {
    // What a ContractSink resolver receives: one value word after each query's records. The walk has to
    // step over it, and a query ordinal names nothing.
    std::mt19937_64 rng(20260816);
    check_probe_matches_the_queries<250>(rng, /*n_seed=*/50, /*n_query=*/120, /*rank_count=*/2, /*fused=*/true);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_single_sender) {
    std::mt19937_64 rng(20260817);
    check_probe_matches_the_queries<32>(rng, /*n_seed=*/25, /*n_query=*/60, /*rank_count=*/1, /*fused=*/true);
}

/* ── The pieces, pinned individually ──────────────────────────────────────── */

BOOST_AUTO_TEST_CASE(sparse_resolve_set_positions_matches_set) {
    // set_positions must leave exactly the row set() leaves, at every width including the spill.
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
    // The hash identity, isolated: seed the index entirely through the DENSE path, then look every key
    // up through the POSITION path. If fold_hash_positions were not exactly fold_hash, these lookups
    // would miss -- silently, since a miss is a legal answer.
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
        // The hash the probe hands the insert must be the hash of that key, not of its neighbour.
        BOOST_TEST(hashes[i] == detail::OperatorIndex<kN>::fold_hash_positions(flat.data() + off[i], kk[i]));
    }

    // Absent keys must come back absent through the position path too: a find that always hits would
    // pass every assertion above.
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
        // draw_distinct may re-draw a seeded term; only the ones that really are absent are evidence.
        if (!op.store->find(absent[i]).has_value()) {
            BOOST_TEST(aout[i] == detail::OperatorIndex<kN>::kNotFound);
            ++genuinely_absent;
        }
    }
    BOOST_TEST(genuinely_absent > 0);
}
