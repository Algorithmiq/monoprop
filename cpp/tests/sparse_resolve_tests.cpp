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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "KernelTestSupport.h"

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/mpi/Comm.h"
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
    // A non-zero base is the case a re-basing bug survives: the sender must still re-base to its flat slot.
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
    const auto pr = detail::probe_incoming_queries<NumModes>(incoming, op, form);
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
        BOOST_TEST(pr.window.slot(pr.sender_index(g)) == window.base + expect_sender[g]);
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
    op.store->find_batch_positions(flat, off, kk, out, hashes);

    std::vector<size_t> out_dense(terms.size(), 0);
    op.store->find_batch(terms.data(), terms.size(), out_dense.data());
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_REQUIRE(out[i] != detail::OperatorIndex<kN>::kNotFound);
        BOOST_TEST(out[i] == i);
        BOOST_TEST(out[i] == out_dense[i]);
        BOOST_TEST(hashes[i]
                   == detail::OperatorIndex<kN>::fold_hash_positions(
                       std::span<const detail::OperatorIndex<kN>::PosT>(flat).subspan(off[i], kk[i])));
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
    op.store->find_batch_positions(aflat, aoff, akk, aout);
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

/* ── Task 6: checked, threaded incoming resolution and bounded self probing ─────────────────────── */

namespace {

using detail::KernelRange;
using kernel_test::AccumulatingObserver;
using kernel_test::PhaseLog;

template <size_t N>
auto key_of(const Monomial<N> &m) -> std::vector<uint64_t> {
    std::vector<uint64_t> key;
    for (size_t w = 0; w < Monomial<N>::num_words(); ++w) {
        key.push_back(m.word(w));
    }
    return key;
}

template <size_t N>
auto rows_of(const detail::MPOperator<N> &op) -> std::vector<Monomial<N>> {
    std::vector<Monomial<N>> rows;
    for (size_t i = 0; i < op.store->size(); ++i) {
        rows.push_back(op.store->row(i));
    }
    return rows;
}

// Every array exactly, including its length: the probe allocates each once, at the query count.
template <size_t N>
auto same_probe(const detail::IncomingProbe<N> &a, const detail::IncomingProbe<N> &b) -> bool {
    const auto eq = [](const auto &x, const auto &y) { return std::ranges::equal(x, y); };
    return a.window == b.window && a.base == b.base && a.nq_total == b.nq_total && eq(a.goff, b.goff)
           && eq(a.sender_wi, b.sender_wi) && eq(a.phase_of, b.phase_of) && eq(a.off_of, b.off_of)
           && eq(a.idx_of, b.idx_of) && eq(a.miss_g, b.miss_g) && eq(a.pos_flat, b.pos_flat) && eq(a.pos_off, b.pos_off)
           && eq(a.k_of, b.k_of) && eq(a.hash_of, b.hash_of);
}

template <size_t N>
auto exact_sizes(const detail::IncomingProbe<N> &p, size_t positions) -> bool {
    const size_t n = p.nq_total;
    return p.goff.size() == p.window.count + 1 && p.sender_wi.size() == n && p.phase_of.size() == n
           && p.off_of.size() == n && p.idx_of.size() == n && p.pos_off.size() == n && p.k_of.size() == n
           && p.hash_of.size() == n && p.pos_flat.size() == positions && p.miss_g.capacity() == p.miss_g.size();
}

// The rows a column holds, whichever representation it is in.
template <size_t N>
auto column_rows(const detail::InvertedIndex<N> &index, size_t c) -> std::vector<size_t> {
    std::vector<size_t> rows;
    if (index.column_is_dense(c)) {
        const uint64_t *w = index.dense_column_data(c);
        for (size_t r = 0; r < index.rows(); ++r) {
            if (((w[r >> 6] >> (r & 63U)) & 1U) != 0U) {
                rows.push_back(r);
            }
        }
    }
    else {
        for (const auto r : index.sparse_column_rows(c)) {
            rows.push_back(r);
        }
    }
    return rows;
}

// An index kept up to date by reindex_after_growth against one built from scratch over the same rows.
template <size_t N>
auto same_inverted_index(const detail::MPOperator<N> &grown, const detail::MPOperator<N> &fresh) -> bool {
    const auto &a = grown.inverted_index();
    const auto &b = fresh.inverted_index();
    if (a.rows() != b.rows()) {
        return false;
    }
    for (size_t c = 0; c < 2 * N; ++c) {
        if (column_rows<N>(a, c) != column_rows<N>(b, c)) {
            return false;
        }
    }
    return true;
}

// Distinct keys for one resolution: `hits` drawn from the seeded store, the rest fresh, in shuffled order.
template <size_t N>
struct QueryPlan {
    std::vector<Monomial<N>> seeds;
    std::vector<std::vector<Monomial<N>>> per_sender;
    size_t hits = 0;
    size_t misses = 0;
};

enum class HitMix { none, all, mixed };

template <size_t N>
auto plan_queries(uint64_t seed, const std::vector<size_t> &per_sender, HitMix mix, size_t n_seed = 300)
    -> QueryPlan<N> {
    std::mt19937_64 rng(seed);
    QueryPlan<N> plan;
    size_t total = 0;
    for (const auto n : per_sender) {
        total += n;
    }
    // Enough stored terms for every planned hit: all of them, or one query in three.
    plan.seeds = draw_distinct<N>(rng, std::max(n_seed, mix == HitMix::all ? total : (total / 3) + 1));
    std::set<std::vector<uint64_t>> used;
    for (const auto &m : plan.seeds) {
        used.insert(key_of<N>(m));
    }
    size_t next_seed = 0;
    size_t i = 0;
    plan.per_sender.resize(per_sender.size());
    for (size_t s = 0; s < per_sender.size(); ++s) {
        for (size_t q = 0; q < per_sender[s]; ++q, ++i) {
            const bool hit = mix == HitMix::all || (mix == HitMix::mixed && i % 3 == 0);
            if (hit) {
                BOOST_REQUIRE(next_seed < plan.seeds.size());
                plan.per_sender[s].push_back(plan.seeds[next_seed++]);
                ++plan.hits;
                continue;
            }
            for (;;) {
                const auto m = draw_distinct<N>(rng, 1)[0];
                if (used.insert(key_of<N>(m)).second) {
                    plan.per_sender[s].push_back(m);
                    ++plan.misses;
                    break;
                }
            }
        }
    }
    return plan;
}

// Serial reference (budget 1) against budgets 2, 3, 4 and 8 on identical fresh stores: the whole probe, the
// published rows in row-ID order, every row findable at its ID, and -- with a prebuilt inverted index -- the
// index maintained by reindex_after_growth equal to one built from scratch.
template <size_t N>
auto check_incoming_budgets(const char *label,
                            const QueryPlan<N> &plan,
                            mpi::SlotWindow window,
                            bool fused,
                            bool prebuilt_index) -> void {
    BOOST_TEST_CONTEXT(label << " N=" << N << " base=" << window.base << " fused=" << fused
                             << " prebuilt=" << prebuilt_index) {
        const auto incoming = serialize<N>(plan.per_sender, fused, window);
        const auto form = fused ? detail::QueryForm::Fused : detail::QueryForm::Plain;
        std::vector<Monomial<N>> expect;
        std::vector<size_t> expect_sender;
        std::vector<int> expect_phase;
        std::vector<double> expect_value;
        for (size_t s = 0; s < plan.per_sender.size(); ++s) {
            for (size_t q = 0; q < plan.per_sender[s].size(); ++q) {
                expect.push_back(plan.per_sender[s][q]);
                expect_sender.push_back(s);
                expect_phase.push_back(((q % 2) == 0) ? 1 : -1);
                expect_value.push_back(0.5 + static_cast<double>(q));
            }
        }
        size_t positions = 0;
        for (const auto &m : expect) {
            positions += m.count();
        }

        auto serial_op = make_op<N>(plan.seeds);
        if (prebuilt_index) {
            (void)serial_op.inverted_index();
        }
        const auto serial = detail::probe_incoming_queries<N>(incoming, serial_op, form, {.threads = 1});
        BOOST_TEST(exact_sizes<N>(serial, positions));
        BOOST_REQUIRE_EQUAL(serial.nq_total, expect.size());
        BOOST_TEST(serial.base == plan.seeds.size());
        size_t hits = 0;
        std::vector<size_t> expect_misses;
        for (size_t g = 0; g < serial.nq_total; ++g) {
            BOOST_TEST((serial.mono_at(g) == expect[g]));
            BOOST_TEST(serial.sender_index(g).value == expect_sender[g]);
            BOOST_TEST(serial.window.slot(serial.sender_index(g)) == window.base + expect_sender[g]);
            BOOST_TEST(serial.phase_of[g] == expect_phase[g]);
            BOOST_TEST(serial.hash_of[g] == detail::OperatorIndex<N>::fold_hash_positions(serial.positions_at(g)));
            if (fused) {
                const double v = detail::QueryWire<N>::value_at(incoming[serial.sender_index(g)], serial.off_of[g]);
                BOOST_TEST(std::bit_cast<uint64_t>(v) == std::bit_cast<uint64_t>(expect_value[g]));
            }
            const bool hit = serial.idx_of[g] < serial.base;
            if (hit) {
                ++hits;
                BOOST_TEST((serial_op.store->row(serial.idx_of[g]) == expect[g]));
            }
            else {
                BOOST_TEST(serial.idx_of[g] == serial.base + expect_misses.size());
                expect_misses.push_back(g);
            }
        }
        BOOST_TEST(hits == plan.hits);
        BOOST_TEST(std::ranges::equal(serial.miss_g, expect_misses));
        detail::insert_incoming_misses<N>(serial_op, serial, {.threads = 1});
        const auto serial_rows = rows_of<N>(serial_op);
        BOOST_TEST(serial_rows.size() == plan.seeds.size() + plan.misses);
        for (size_t j = 0; j < expect_misses.size(); ++j) {
            BOOST_TEST((serial_rows[serial.base + j] == expect[expect_misses[j]]));
        }
        for (size_t i = 0; i < serial_rows.size(); ++i) {
            const auto found = serial_op.store->find(serial_rows[i]);
            BOOST_TEST((found.has_value() && *found == i));
        }
        if (prebuilt_index) {
            BOOST_TEST(same_inverted_index<N>(serial_op, make_op<N>(serial_rows)));
        }

        for (const int threads : {2, 3, 4, 8}) {
            auto op = make_op<N>(plan.seeds);
            if (prebuilt_index) {
                (void)op.inverted_index();
            }
            const auto pr = detail::probe_incoming_queries<N>(incoming, op, form, {.threads = threads});
            BOOST_TEST(same_probe<N>(serial, pr), "budget " << threads);
            detail::insert_incoming_misses<N>(op, pr, {.threads = threads});
            BOOST_TEST((rows_of<N>(op) == serial_rows), "budget " << threads);
            if (prebuilt_index) {
                BOOST_TEST(same_inverted_index<N>(op, make_op<N>(serial_rows)));
            }
        }
    }
}

} // namespace

// The plan's fixture: a window at base 4, senders 4 and 6 (5 is empty), predicted and published IDs. Repeated
// at base zero. This preserves behavior that existed before Task 6 and may pass before it.
BOOST_AUTO_TEST_CASE(openmp_incoming_miss_order) {
    using namespace monoprop;
    const auto a = indices_to_bitset<8>({0, 1});
    const auto b = indices_to_bitset<8>({2, 3});
    const auto c = indices_to_bitset<8>({4, 5});
    for (const size_t window_base : {size_t{4}, size_t{0}}) {
        for (const auto threads : {1, 3}) {
            auto op = make_op<8>(std::vector<Monomial<8>>{a});
            const mpi::SlotWindow window{.base = window_base, .count = 3};
            mpi::WindowVec<VecZ> incoming(window);
            detail::QueryWire<8>::push(incoming.at_slot(window_base), positions_of<8>(b), +1);
            detail::QueryWire<8>::push(incoming.at_slot(window_base + 2), positions_of<8>(a), -1);
            detail::QueryWire<8>::push(incoming.at_slot(window_base + 2), positions_of<8>(c), +1);
            const auto pr =
                detail::probe_incoming_queries<8>(incoming, op, detail::QueryForm::Plain, {.threads = threads});
            BOOST_TEST(pr.window.slot(pr.sender_index(0)) == window_base);
            BOOST_TEST(pr.window.slot(pr.sender_index(1)) == window_base + 2);
            const std::vector<size_t> offsets{0, 1, 1, 3}, ids{1, 0, 2}, misses{0, 2};
            BOOST_TEST(pr.base == 1U);
            BOOST_CHECK_EQUAL_COLLECTIONS(pr.goff.begin(), pr.goff.end(), offsets.begin(), offsets.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(pr.idx_of.begin(), pr.idx_of.end(), ids.begin(), ids.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(pr.miss_g.begin(), pr.miss_g.end(), misses.begin(), misses.end());
            detail::insert_incoming_misses(op, pr);
            BOOST_TEST(op.size() == 3U);
            BOOST_REQUIRE(op.store->find(a).has_value());
            BOOST_REQUIRE(op.store->find(b).has_value());
            BOOST_REQUIRE(op.store->find(c).has_value());
            BOOST_TEST(op.store->find(a).value() == 0U);
            BOOST_TEST(op.store->find(b).value() == 1U);
            BOOST_TEST(op.store->find(c).value() == 2U);
        }
    }
}

// miss_g holds query ordinals, which are not TermIndex values and may exceed their range.
BOOST_AUTO_TEST_CASE(openmp_incoming_miss_ordinals_are_size_t) {
    static_assert(std::is_same_v<decltype(detail::IncomingProbe<8>::miss_g), std::vector<size_t>>);
    BOOST_TEST(true);
}

// Serial against threaded over the input shapes the resolver meets: empty, all-hit, all-miss and mixed, block
// boundaries 255/256/257 and larger partial blocks, empty intervening senders, window bases zero and non-zero,
// narrow and wide positions with inline and spilled rows and multiword/escaped records, Plain and Fused codecs,
// and a prebuilt inverted index maintained across the publication.
BOOST_AUTO_TEST_CASE(openmp_incoming_probe_threaded_matches_serial) {
    const mpi::SlotWindow base0{.base = 0, .count = 1};
    check_incoming_budgets<32>("empty", plan_queries<32>(1, {0}, HitMix::none), base0, false, false);
    check_incoming_budgets<32>("empty senders",
                               plan_queries<32>(2, {0, 0, 0}, HitMix::none),
                               {.base = 3, .count = 3},
                               true,
                               false);
    check_incoming_budgets<32>("255 all-hit", plan_queries<32>(3, {255}, HitMix::all), base0, false, true);
    check_incoming_budgets<32>("256 all-miss", plan_queries<32>(4, {256}, HitMix::none), base0, true, true);
    check_incoming_budgets<32>("257 mixed", plan_queries<32>(5, {257}, HitMix::mixed), base0, false, false);
    check_incoming_budgets<250>("empty intervening senders",
                                plan_queries<250>(6, {0, 300, 0, 413}, HitMix::mixed),
                                {.base = 5, .count = 4},
                                false,
                                true);
    check_incoming_budgets<250>("fused partial blocks",
                                plan_queries<250>(7, {1, 600, 130}, HitMix::mixed),
                                {.base = 2, .count = 3},
                                true,
                                false);
    check_incoming_budgets<128>("narrow store, wide wire",
                                plan_queries<128>(8, {513, 0}, HitMix::mixed),
                                {.base = 0, .count = 2},
                                true,
                                true);
    // Escaped counts: fully paired 20-mode terms carry k >= 31 on the wire.
    {
        auto plan = plan_queries<1024>(9, {40, 40}, HitMix::mixed, 60);
        std::mt19937_64 rng(10);
        for (auto &sender : plan.per_sender) {
            for (size_t q = 1; q < sender.size(); q += 2) {
                auto &m = sender[q];
                if (m.count() < 31U) {
                    auto wide = random_paired_monomial<1024>(rng, 16 + (rng() % 20));
                    if (std::ranges::none_of(plan.seeds, [&](const auto &s) { return s == wide; })) {
                        m = wide;
                    }
                }
            }
        }
        // Rebuild the hit/miss counts from the store: some hits were replaced by fresh escaped terms.
        auto op = make_op<1024>(plan.seeds);
        plan.hits = 0;
        plan.misses = 0;
        size_t escaped = 0;
        for (const auto &sender : plan.per_sender) {
            for (const auto &m : sender) {
                (op.store->find(m).has_value() ? plan.hits : plan.misses) += 1;
                escaped += static_cast<size_t>(m.count() >= detail::QueryWire<1024>::kKEscape);
            }
        }
        BOOST_TEST(escaped > 0U);
        check_incoming_budgets<1024>("escaped counts", plan, {.base = 1, .count = 2}, true, false);
    }
}

// Invalid headers and boundaries are rejected on the caller before any decode worker starts: nothing is
// decoded, probed or published, and the store is untouched.
BOOST_AUTO_TEST_CASE(openmp_incoming_probe_rejects_malformed_streams_before_decode) {
    using QW = detail::QueryWire<128>;
    std::mt19937_64 rng(11);
    const auto seeds = draw_distinct<128>(rng, 20);
    // Three one-word records: read as Fused, the second becomes the first's value word and the third lacks one.
    VecZ good;
    for (const auto &pos : {std::vector<uint16_t>{1, 2}, std::vector<uint16_t>{3, 9}, std::vector<uint16_t>{0, 5}}) {
        BOOST_REQUIRE(QW::push(good, pos, 1) == 1U);
    }
    VecZ multiword;
    std::vector<uint16_t> spread;
    for (uint16_t p = 0; p < 250; p += 7) {
        spread.push_back(p);
    }
    const size_t words = QW::push(multiword, spread, -1);
    BOOST_REQUIRE(words >= 2U);
    VecZ truncated = multiword;
    truncated.pop_back();
    VecZ trailing = good;
    trailing.push_back(multiword[0]);
    VecZ fused_missing_value = good; // Fused form: every record needs a value word, and none has one
    VecZ bad_phase{size_t{3}};
    struct Bad {
        const char *label;
        VecZ stream;
        detail::QueryForm form;
    };
    const std::vector<Bad> cases{{"noncanonical escape 0x8037e", VecZ{size_t{0x8037e}}, detail::QueryForm::Plain},
                                 {"truncated multiword record", truncated, detail::QueryForm::Plain},
                                 {"trailing partial record", trailing, detail::QueryForm::Plain},
                                 {"fused stream without value words", fused_missing_value, detail::QueryForm::Fused},
                                 {"phase field 3", bad_phase, detail::QueryForm::Plain}};
    for (const auto &bad : cases) {
        for (const int threads : {1, 3}) {
            BOOST_TEST_CONTEXT(bad.label << " budget " << threads) {
                auto op = make_op<128>(seeds);
                const mpi::SlotWindow window{.base = 3, .count = 3};
                mpi::WindowVec<VecZ> incoming(window);
                if (bad.form == detail::QueryForm::Plain) {
                    incoming.at_slot(3) = good; // a valid sender first, an empty one, then the malformed one
                }
                incoming.at_slot(5) = bad.stream;
                PhaseLog log;
                BOOST_CHECK_THROW((void)detail::probe_incoming_queries<128>(incoming,
                                                                            op,
                                                                            bad.form,
                                                                            {.threads = threads},
                                                                            AccumulatingObserver{&log}),
                                  detail::MalformedQueryStream);
                BOOST_TEST(log.calls.empty()); // no decode, no probe
                BOOST_TEST(op.size() == seeds.size());
            }
        }
    }
}

// Valid headers with invalid decoded positions fail inside a decode worker; the failure joins and is rethrown
// before the probe starts or anything is published.
BOOST_AUTO_TEST_CASE(openmp_incoming_decode_rejects_invalid_positions_before_probing) {
    using QW = detail::QueryWire<250>;
    const auto plan = plan_queries<250>(12, {400, 300}, HitMix::mixed);
    auto incoming = serialize<250>(plan.per_sender, false, {.base = 0, .count = 2});
    QW::push(incoming[mpi::WindowIndex{1}], std::vector<uint16_t>{7, 505}, 1); // query 700, decode block 2
    for (const int threads : {1, 3}) {
        BOOST_TEST_CONTEXT("budget " << threads) {
            auto op = make_op<250>(plan.seeds);
            PhaseLog log;
            BOOST_CHECK_THROW((void)detail::probe_incoming_queries<250>(incoming,
                                                                        op,
                                                                        detail::QueryForm::Plain,
                                                                        {.threads = threads},
                                                                        AccumulatingObserver{&log}),
                              detail::MalformedQueryStream);
            BOOST_TEST(log.of(KernelRange::decode).size() == 1U);
            BOOST_TEST(log.of(KernelRange::decode)[0]->worker.size() == 3U);
            BOOST_TEST(log.of(KernelRange::incoming_probe).empty());
            BOOST_TEST(op.size() == plan.seeds.size());
        }
    }
}

// A decode worker's own failure (not a malformed input) is joined and rethrown before probing, too.
BOOST_AUTO_TEST_CASE(openmp_incoming_decode_worker_failure_joins_before_probing) {
    const auto plan = plan_queries<250>(13, {600, 424}, HitMix::mixed);
    const auto incoming = serialize<250>(plan.per_sender, true, {.base = 1, .count = 2});
    auto op = make_op<250>(plan.seeds);
    PhaseLog log;
    BOOST_CHECK_THROW((void)detail::probe_incoming_queries<250>(incoming,
                                                                op,
                                                                detail::QueryForm::Fused,
                                                                {.threads = 4},
                                                                AccumulatingObserver{&log, KernelRange::decode, 0, 3}),
                      std::runtime_error);
    BOOST_TEST(log.of(KernelRange::decode).size() == 1U);
    BOOST_TEST(log.of(KernelRange::decode)[0]->visits.at(3) == 1);
    BOOST_TEST(log.of(KernelRange::incoming_probe).empty());
    BOOST_TEST(op.size() == plan.seeds.size());
}

// Decode and probe each run in their own region, one logical range per 256 queries.
BOOST_AUTO_TEST_CASE(openmp_incoming_decode_and_probe_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    const auto plan = plan_queries<250>(14, {500, 0, 524}, HitMix::mixed);
    const auto incoming = serialize<250>(plan.per_sender, false, {.base = 0, .count = 3});
    auto op = make_op<250>(plan.seeds);
    PhaseLog log;
    const auto pr = detail::probe_incoming_queries<250>(incoming,
                                                        op,
                                                        detail::QueryForm::Plain,
                                                        {.threads = 4},
                                                        AccumulatingObserver{&log});
    BOOST_TEST(pr.nq_total == 1024U);
    BOOST_REQUIRE(log.of(KernelRange::decode).size() == 1U);
    BOOST_REQUIRE(log.of(KernelRange::incoming_probe).size() == 1U);
    kernel_test::check_participation(kernel_test::slots_of(*log.of(KernelRange::decode)[0]), 4, "decode");
    kernel_test::check_participation(kernel_test::slots_of(*log.of(KernelRange::incoming_probe)[0]), 4, "probe");
    // Budget 1 and a single block stay on the caller.
    for (const auto &[threads, senders] :
         std::vector<std::pair<int, std::vector<size_t>>>{{1, {500, 0, 524}}, {4, {200, 0, 56}}}) {
        const auto small = plan_queries<250>(15, senders, HitMix::mixed);
        auto op_small = make_op<250>(small.seeds);
        PhaseLog serial_log;
        (void)detail::probe_incoming_queries<250>(serialize<250>(small.per_sender, false, {.base = 0, .count = 3}),
                                                  op_small,
                                                  detail::QueryForm::Plain,
                                                  {.threads = threads},
                                                  AccumulatingObserver{&serial_log});
        const size_t blocks = detail::logical_ranges(senders[0] + senders[2], detail::kProbeBlockQueries);
        kernel_test::check_serial(kernel_test::slots_of(*serial_log.of(KernelRange::decode)[0]), blocks, "decode");
        kernel_test::check_serial(kernel_test::slots_of(*serial_log.of(KernelRange::incoming_probe)[0]),
                                  blocks,
                                  "probe");
    }
}

// probe_frozen_positions: 256-query blocks over absolute offsets into one shared buffer give exactly the single
// find_batch_positions call, hashes included, at every budget and with more workers than blocks.
BOOST_AUTO_TEST_CASE(openmp_probe_frozen_positions_matches_one_batch) {
    const auto run = []<size_t N>(std::integral_constant<size_t, N>, uint64_t seed) {
        using PosT = typename detail::OperatorIndex<N>::PosT;
        std::mt19937_64 rng(seed);
        const auto seeds = draw_distinct<N>(rng, 400);
        const auto op = make_op<N>(seeds);
        const auto fresh = draw_distinct<N>(rng, 700);
        for (const size_t n : {size_t{0}, size_t{1}, size_t{255}, size_t{256}, size_t{257}, size_t{1000}}) {
            std::vector<PosT> flat(5, PosT{1}); // unreferenced prefix
            std::vector<size_t> off(n);
            std::vector<uint32_t> k_of(n);
            // Queries laid out in reverse, so offsets are absolute and not monotonic in the query ordinal.
            for (size_t q = n; q-- > 0;) {
                const auto &m = (q % 2 == 0) ? seeds[q % seeds.size()] : fresh[q % fresh.size()];
                off[q] = flat.size();
                for (const auto p : positions_of<N>(m)) {
                    flat.push_back(static_cast<PosT>(p));
                }
                k_of[q] = static_cast<uint32_t>(m.count());
            }
            std::vector<size_t> want(n);
            std::vector<uint32_t> want_hash(n);
            op.store->find_batch_positions(flat, off, k_of, want, want_hash);
            for (const int threads : {1, 2, 3, 4, 8}) {
                std::vector<size_t> out(n, 17);
                std::vector<uint32_t> hash(n, 17);
                detail::probe_frozen_positions<N>(*op.store, flat, off, k_of, out, hash, {.threads = threads});
                BOOST_TEST(out == want, "N=" << N << " n=" << n << " budget " << threads);
                BOOST_TEST(hash == want_hash, "N=" << N << " n=" << n << " budget " << threads);
                std::vector<size_t> no_hash(n, 17);
                detail::probe_frozen_positions<N>(*op.store, flat, off, k_of, no_hash, {}, {.threads = threads});
                BOOST_TEST(no_hash == want);
            }
        }
    };
    run(std::integral_constant<size_t, 32>{}, 16);
    run(std::integral_constant<size_t, 250>{}, 17); // wide positions, spilled rows
}

// The helper's contracts are checked on the caller, before any block runs.
BOOST_AUTO_TEST_CASE(openmp_probe_frozen_positions_rejects_bad_spans) {
    using PosT = detail::OperatorIndex<32>::PosT;
    const auto op = make_op<32>(std::vector<Monomial<32>>{indices_to_bitset<32>({1, 2})});
    const std::vector<PosT> flat{1, 2, 3, 4};
    std::vector<size_t> out(2);
    std::vector<uint32_t> hash(2);
    const auto expect_throw = [&](std::vector<size_t> off, std::vector<uint32_t> k, size_t n_out, size_t n_hash) {
        std::vector<size_t> o(n_out);
        std::vector<uint32_t> h(n_hash);
        PhaseLog log;
        BOOST_CHECK_THROW(detail::probe_frozen_positions<32>(*op.store,
                                                             flat,
                                                             off,
                                                             k,
                                                             o,
                                                             h,
                                                             {.threads = 2},
                                                             AccumulatingObserver{&log},
                                                             KernelRange::incoming_probe),
                          std::invalid_argument);
        BOOST_TEST(log.calls.empty());
    };
    expect_throw({0, 2}, {2}, 2, 2);                                     // counts shorter than offsets
    expect_throw({0, 2}, {2, 2}, 1, 2);                                  // output shorter
    expect_throw({0, 2}, {2, 2}, 2, 1);                                  // hash output neither empty nor full
    expect_throw({0, 3}, {2, 2}, 2, 2);                                  // span past the buffer
    expect_throw({0, std::numeric_limits<size_t>::max()}, {2, 1}, 2, 2); // offset + count wraps
}

BOOST_AUTO_TEST_CASE(openmp_probe_frozen_positions_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    using PosT = detail::OperatorIndex<32>::PosT;
    std::mt19937_64 rng(18);
    const auto seeds = draw_distinct<32>(rng, 300);
    const auto op = make_op<32>(seeds);
    std::vector<PosT> flat;
    std::vector<size_t> off;
    std::vector<uint32_t> k_of;
    for (size_t q = 0; q < 1024; ++q) {
        off.push_back(flat.size());
        for (const auto p : positions_of<32>(seeds[q % seeds.size()])) {
            flat.push_back(static_cast<PosT>(p));
        }
        k_of.push_back(static_cast<uint32_t>(seeds[q % seeds.size()].count()));
    }
    std::vector<size_t> out(off.size());
    std::vector<uint32_t> hash(off.size());
    PhaseLog log;
    detail::probe_frozen_positions<32>(*op.store,
                                       flat,
                                       off,
                                       k_of,
                                       out,
                                       hash,
                                       {.threads = 4},
                                       AccumulatingObserver{&log},
                                       KernelRange::self_probe);
    BOOST_REQUIRE(log.of(KernelRange::self_probe).size() == 1U);
    kernel_test::check_participation(kernel_test::slots_of(log.calls[0]), 4, "frozen probe");
}

/* ── Phase-3 scatter: capability-gated parallel on_resolved ─────────────────────────────────────── */

namespace {

// A generic sink with no parallel capability: on_resolved appends to a shared vector, so a scatter that ignored
// the missing capability would race (TSan) and scramble the order.
template <size_t N>
struct AppendingSink {
    using Response = TermIndex;
    static auto init_response() -> Response { return std::numeric_limits<TermIndex>::max(); }
    [[nodiscard]] auto incoming_form() const -> detail::QueryForm { return detail::QueryForm::Plain; }
    std::vector<size_t> order;
    auto prepare(const detail::IncomingProbe<N> &,
                 detail::MPOperator<N> &,
                 const mpi::WindowVec<std::vector<Response>> &) -> void {}
    auto on_resolved(size_t g,
                     mpi::WindowIndex,
                     size_t,
                     size_t ip,
                     const detail::IncomingProbe<N> &,
                     detail::QueryWire<N>::WireView) -> Response {
        order.push_back(g);
        return static_cast<TermIndex>(ip);
    }
};

template <size_t N>
struct ExplicitlySerialSink : AppendingSink<N> {
    static constexpr bool parallel_resolve = false;
};

struct ScatterRun {
    std::vector<std::vector<double>> responses; // bit patterns compared through bit_cast
    std::vector<std::vector<uint64_t>> sink_state;
    std::vector<uint8_t> marks;
    std::vector<std::vector<uint64_t>> rows;
    PhaseLog log;
};

template <size_t N, typename Sink, typename Capture>
auto run_scatter(const QueryPlan<N> &plan,
                 mpi::SlotWindow window,
                 bool leader,
                 int threads,
                 Sink &sink,
                 detail::MPOperator<N> &op,
                 Capture &&capture) -> ScatterRun {
    ScatterRun run;
    const auto incoming = serialize<N>(plan.per_sender, sink.incoming_form() == detail::QueryForm::Fused, window);
    detail::MatchedEpochSet matched;
    const size_t combined = op.size();
    matched.begin_gate(combined);
    const auto responses = detail::resolve_incoming<N>(incoming,
                                                       op,
                                                       leader,
                                                       matched,
                                                       combined,
                                                       sink,
                                                       {.threads = threads},
                                                       AccumulatingObserver{&run.log});
    for (const auto wi : window.indices()) {
        std::vector<double> r;
        for (const auto v : responses[wi]) {
            r.push_back(static_cast<double>(v));
        }
        run.responses.push_back(r);
    }
    for (size_t i = 0; i < combined; ++i) {
        run.marks.push_back(static_cast<uint8_t>(matched.is_marked(i)));
    }
    for (const auto &m : rows_of<N>(op)) {
        run.rows.push_back(key_of<N>(m));
    }
    run.sink_state = capture(sink);
    return run;
}

auto same_scatter(const ScatterRun &a, const ScatterRun &b) -> bool {
    if (a.responses.size() != b.responses.size()) {
        return false;
    }
    for (size_t s = 0; s < a.responses.size(); ++s) {
        if (a.responses[s].size() != b.responses[s].size()
            || !std::ranges::equal(a.responses[s], b.responses[s], [](double x, double y) {
                   return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y);
               })) {
            return false;
        }
    }
    return a.sink_state == b.sink_state && a.marks == b.marks && a.rows == b.rows;
}

template <size_t N>
auto graph_state(const detail::GraphSink<N> &sink) -> std::vector<std::vector<uint64_t>> {
    std::vector<std::vector<uint64_t>> out;
    for (const auto &a : sink.acc) {
        std::vector<uint64_t> v;
        for (const auto &e : a.in_entries) {
            v.push_back(e.idx);
            v.push_back(static_cast<uint64_t>(static_cast<int64_t>(e.phase)));
        }
        out.push_back(v);
    }
    return out;
}

inline auto contract_state(const detail::FusedContract &fc) -> std::vector<std::vector<uint64_t>> {
    std::vector<uint64_t> v;
    for (const auto &h : fc.cross_half) {
        v.push_back(h.local_idx);
        v.push_back(std::bit_cast<uint64_t>(h.v_partner));
        v.push_back(static_cast<uint64_t>(static_cast<int64_t>(h.phase_signed)));
        v.push_back(static_cast<uint64_t>(h.is_insert));
    }
    return {v};
}

} // namespace

// GraphSink and the Heisenberg ContractSink certify parallel on_resolved; their scatter runs on workers and
// matches the serial scatter exactly: responses, sink records, leader marks and the published rows.
BOOST_AUTO_TEST_CASE(openmp_resolve_scatter_threaded_matches_serial) {
    constexpr size_t N = 250;
    const mpi::SlotWindow window{.base = 2, .count = 3};
    const auto plan = plan_queries<N>(19, {1300, 0, 1250}, HitMix::mixed);
    VecD coeffs;
    for (size_t i = 0; i < plan.seeds.size(); ++i) {
        coeffs.push_back(0.125 + (0.001 * static_cast<double>(i)));
    }
    for (const bool leader : {true, false}) {
        std::vector<ScatterRun> graph;
        std::vector<ScatterRun> contract;
        std::vector<ScatterRun> contract_swept;
        for (const int threads : {1, 2, 3, 4}) {
            {
                auto op = make_op<N>(plan.seeds);
                detail::GraphSink<N> sink(window.stop(), 0);
                graph.push_back(run_scatter<N>(plan, window, leader, threads, sink, op, [](const auto &s) {
                    return graph_state<N>(s);
                }));
            }
            for (const bool swept : {false, true}) {
                auto op = make_op<N>(plan.seeds);
                detail::FusedContract fc;
                detail::ContractSink<N> sink{.R = window.stop(),
                                             .my_rank = 0,
                                             .fc = fc,
                                             .op_coeffs = coeffs,
                                             .fused_scale = swept,
                                             .inv_cos = swept ? 1.25 : 1.0,
                                             .schrodinger = false,
                                             .basis = Basis::Majorana};
                (swept ? contract_swept : contract)
                    .push_back(run_scatter<N>(plan, window, leader, threads, sink, op, [&](const auto &) {
                        return contract_state(fc);
                    }));
            }
        }
        for (size_t t = 1; t < graph.size(); ++t) {
            BOOST_TEST(same_scatter(graph[0], graph[t]), "graph leader=" << leader << " budget " << t + 1);
            BOOST_TEST(same_scatter(contract[0], contract[t]), "contract leader=" << leader << " budget " << t + 1);
            BOOST_TEST(same_scatter(contract_swept[0], contract_swept[t]), "swept leader=" << leader);
        }
        BOOST_TEST(std::ranges::count(graph[0].marks, 1) == (leader ? static_cast<std::ptrdiff_t>(plan.hits) : 0));
    }
}

BOOST_AUTO_TEST_CASE(openmp_resolve_scatter_workers_participate,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    constexpr size_t N = 250;
    const mpi::SlotWindow window{.base = 0, .count = 2};
    const auto plan = plan_queries<N>(20, {2048, 2048}, HitMix::mixed, 1400);
    VecD coeffs(plan.seeds.size(), 0.5);
    {
        auto op = make_op<N>(plan.seeds);
        detail::GraphSink<N> sink(2, 0);
        const auto run =
            run_scatter<N>(plan, window, true, 4, sink, op, [](const auto &s) { return graph_state<N>(s); });
        BOOST_REQUIRE(run.log.of(KernelRange::scatter).size() == 1U);
        kernel_test::check_participation(kernel_test::slots_of(*run.log.of(KernelRange::scatter)[0]), 4, "graph");
    }
    {
        auto op = make_op<N>(plan.seeds);
        detail::FusedContract fc;
        detail::ContractSink<N> sink{.R = 2,
                                     .my_rank = 0,
                                     .fc = fc,
                                     .op_coeffs = coeffs,
                                     .fused_scale = false,
                                     .inv_cos = 1.0,
                                     .schrodinger = false,
                                     .basis = Basis::Majorana};
        const auto run =
            run_scatter<N>(plan, window, true, 4, sink, op, [&](const auto &) { return contract_state(fc); });
        kernel_test::check_participation(kernel_test::slots_of(*run.log.of(KernelRange::scatter)[0]), 4, "contract");
    }
}

// Sinks that do not certify parallel on_resolved keep the serial scatter at any budget: a generic sink without
// the member, one declaring it false, and the Schrodinger ContractSink (state scoring stays serial).
BOOST_AUTO_TEST_CASE(openmp_resolve_scatter_stays_serial_without_the_capability) {
    constexpr size_t N = 128;
    const mpi::SlotWindow window{.base = 1, .count = 2};
    const auto plan = plan_queries<N>(21, {1500, 1100}, HitMix::mixed);
    const size_t blocks = detail::logical_ranges(2600, detail::kResolveScatterBlock);
    const auto order_state = [](const auto &s) {
        return std::vector<std::vector<uint64_t>>{std::vector<uint64_t>(s.order.begin(), s.order.end())};
    };
    for (const int threads : {1, 4}) {
        auto op = make_op<N>(plan.seeds);
        AppendingSink<N> generic;
        const auto a = run_scatter<N>(plan, window, true, threads, generic, op, order_state);
        kernel_test::check_serial(kernel_test::slots_of(*a.log.of(KernelRange::scatter)[0]), blocks, "no member");
        std::vector<uint64_t> in_order(2600);
        std::iota(in_order.begin(), in_order.end(), 0);
        BOOST_TEST(a.sink_state[0] == in_order);

        auto op2 = make_op<N>(plan.seeds);
        ExplicitlySerialSink<N> declared;
        const auto b = run_scatter<N>(plan, window, true, threads, declared, op2, order_state);
        kernel_test::check_serial(kernel_test::slots_of(*b.log.of(KernelRange::scatter)[0]), blocks, "false");
        BOOST_TEST(b.sink_state[0] == in_order);
    }
    // Schrodinger: fresh cross-rank misses are state-scored from their positions (paired-only mono_at).
    const auto plan_s = plan_queries<N>(22, {1400, 1200}, HitMix::mixed);
    VecD coeffs(plan_s.seeds.size(), 0.25);
    std::vector<ScatterRun> runs;
    for (const int threads : {1, 4}) {
        auto op = make_op<N>(plan_s.seeds);
        op.initial_state = {0, 3, 5, 9};
        detail::FusedContract fc;
        detail::ContractSink<N> sink{.R = 3,
                                     .my_rank = 0,
                                     .fc = fc,
                                     .op_coeffs = coeffs,
                                     .fused_scale = false,
                                     .inv_cos = 1.0,
                                     .schrodinger = true,
                                     .basis = Basis::Majorana};
        runs.push_back(
            run_scatter<N>(plan_s, window, true, threads, sink, op, [&](const auto &) { return contract_state(fc); }));
        kernel_test::check_serial(kernel_test::slots_of(*runs.back().log.of(KernelRange::scatter)[0]),
                                  blocks,
                                  "schrodinger");
    }
    BOOST_TEST(same_scatter(runs[0], runs[1]));
}

// The parallel scatter's matched marks are disjoint uint16 writes only because distinct queries resolve to
// distinct rows: queries are source XOR G over distinct sources. Pinned on generator-derived queries, with a
// duplicated query as the negative control the debug assertion would catch.
BOOST_AUTO_TEST_CASE(openmp_resolve_leader_marks_are_unique) {
    constexpr size_t N = 32;
    std::mt19937_64 rng(23);
    const auto store_terms = draw_distinct<N>(rng, 500);
    const auto gen = indices_to_bitset<N>({3, 17});
    // Sources whose partner is stored (hits) and others (misses): partners are pairwise distinct.
    std::vector<std::vector<Monomial<N>>> per_sender(2);
    std::set<std::vector<uint64_t>> seen;
    for (size_t i = 0; i < store_terms.size(); ++i) {
        const auto query = (i % 4 == 0) ? store_terms[i] : (store_terms[i] ^ gen ^ indices_to_bitset<N>({i % 64}));
        if (seen.insert(key_of<N>(query)).second) {
            per_sender[i % 2].push_back(query);
        }
    }
    auto op = make_op<N>(store_terms);
    const auto incoming = serialize<N>(per_sender, false, {.base = 0, .count = 2});
    const auto pr = detail::probe_incoming_queries<N>(incoming, op, detail::QueryForm::Plain, {.threads = 3});
    BOOST_TEST(detail::leader_marks_unique(pr, op.size()));
    auto dup = per_sender;
    dup[1].push_back(dup[0][0]);
    const auto pr_dup = detail::probe_incoming_queries<N>(serialize<N>(dup, false, {.base = 0, .count = 2}),
                                                          op,
                                                          detail::QueryForm::Plain,
                                                          {.threads = 3});
    BOOST_TEST(!detail::leader_marks_unique(pr_dup, op.size()));
}

/* ── Self resolution in bounded windows, and the publication order across passes ────────────────── */

namespace {

template <size_t N, typename Sink>
using ObservedEngine = detail::LayerBuildEngine<N, Sink, AccumulatingObserver>;

template <size_t N, typename Eng>
auto stage_self(Eng &eng, const std::vector<Monomial<N>> &keys, const std::vector<size_t> &srcs) -> void {
    using PosT = typename Eng::RowPosT;
    for (size_t q = 0; q < keys.size(); ++q) {
        std::vector<PosT> pos;
        for (const auto p : positions_of<N>(keys[q])) {
            pos.push_back(static_cast<PosT>(p));
        }
        eng.self_stage_.push(pos, (q % 3 == 0) ? -1 : 1);
    }
    eng.src_idx_r.at_slot(eng.my_rank) = srcs;
}

struct SelfRun {
    std::vector<std::vector<uint64_t>> state; // sink entries, deferred misses, marks, rows
    PhaseLog log;
};

// Leader then follower self passes over stages spanning several 4096-query windows, then finish(): the sink's
// hits in order, every deferred miss's positions, hash, source, phase and v_src, the leader marks, and the rows
// published after both passes.
template <size_t N, bool Contract>
auto run_self(const std::vector<Monomial<N>> &seeds,
              const std::vector<Monomial<N>> &leader_keys,
              const std::vector<size_t> &leader_srcs,
              const std::vector<Monomial<N>> &follower_keys,
              const std::vector<size_t> &follower_srcs,
              int threads) -> SelfRun {
    SelfRun run;
    auto op = make_op<N>(seeds);
    detail::MatchedEpochSet matched;
    const size_t combined = op.size();
    detail::FusedContract fc;
    VecD coeffs;
    for (size_t i = 0; i < seeds.size(); ++i) {
        coeffs.push_back(1.0 / static_cast<double>(i + 3));
    }
    const mpi::SlotWindow window{.base = 0, .count = 1};
    auto body = [&](auto &eng) {
        const auto values = [](const std::vector<size_t> &srcs) {
            std::vector<double> v;
            for (const auto s : srcs) {
                v.push_back(0.5 + static_cast<double>(s));
            }
            return v;
        };
        stage_self<N>(eng, leader_keys, leader_srcs);
        if constexpr (Contract) {
            eng.src_val_r.reset(window);
            eng.src_val_r.at_slot(0) = values(leader_srcs);
        }
        eng.resolve_self_queries(true);
        stage_self<N>(eng, follower_keys, follower_srcs);
        if constexpr (Contract) {
            eng.src_val_r.at_slot(0) = values(follower_srcs);
        }
        eng.resolve_self_queries(false);
        std::vector<uint64_t> deferred;
        for (const auto &m : eng.deferred_self_misses) {
            deferred.insert(deferred.end(),
                            {m.k,
                             m.hash,
                             m.src,
                             static_cast<uint64_t>(static_cast<int64_t>(m.phase)),
                             std::bit_cast<uint64_t>(m.v_src)});
            for (size_t j = 0; j < m.k; ++j) {
                deferred.push_back(eng.deferred_pos_flat_[m.pos_at + j]);
            }
        }
        run.state.push_back(deferred);
        std::vector<uint64_t> marks;
        for (size_t i = 0; i < combined; ++i) {
            marks.push_back(static_cast<uint64_t>(matched.is_marked(i)));
        }
        run.state.push_back(marks);
        (void)eng.finish(CosMask{}, nullptr);
    };
    if constexpr (Contract) {
        ObservedEngine<N, detail::ContractSink<N>> eng(op,
                                                       1,
                                                       0,
                                                       matched,
                                                       combined,
                                                       detail::ContractSink<N>{.R = 1,
                                                                               .my_rank = 0,
                                                                               .fc = fc,
                                                                               .op_coeffs = coeffs,
                                                                               .fused_scale = false,
                                                                               .inv_cos = 1.0,
                                                                               .schrodinger = false,
                                                                               .basis = Basis::Majorana},
                                                       window,
                                                       {.threads = threads},
                                                       AccumulatingObserver{&run.log});
        body(eng);
        std::vector<uint64_t> recs;
        for (const auto *list : {&fc.hits, &fc.inserts}) {
            for (const auto &r : *list) {
                recs.insert(recs.end(),
                            {r.src,
                             r.tgt,
                             std::bit_cast<uint64_t>(r.v_src),
                             std::bit_cast<uint64_t>(r.v_tgt),
                             static_cast<uint64_t>(static_cast<int64_t>(r.phase))});
            }
        }
        run.state.push_back(recs);
    }
    else {
        ObservedEngine<N, detail::GraphSink<N>> eng(op,
                                                    1,
                                                    0,
                                                    matched,
                                                    combined,
                                                    detail::GraphSink<N>(1, 0),
                                                    window,
                                                    {.threads = threads},
                                                    AccumulatingObserver{&run.log});
        body(eng);
        std::vector<uint64_t> entries;
        for (const auto *list : {&eng.sink.acc[0].in_entries, &eng.sink.acc[0].out_entries}) {
            for (const auto &e : *list) {
                entries.push_back(e.idx);
                entries.push_back(static_cast<uint64_t>(static_cast<int64_t>(e.phase)));
            }
        }
        run.state.push_back(entries);
    }
    const auto rows = rows_of<N>(op);
    size_t findable = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto found = op.store->find(rows[i]);
        findable += static_cast<size_t>(found.has_value() && *found == i);
        run.state.push_back(key_of<N>(rows[i]));
    }
    // A deferred miss published with a wrong cached hash is a correct row that no lookup can find.
    BOOST_TEST(findable == rows.size());
    return run;
}

} // namespace

// Stages spanning two full 4096-query windows plus a short final one; followers whose source a leader matched
// are filtered before windowing. Serial and threaded self resolution agree exactly, and each window probes in
// at most 16 blocks of 256 queries.
BOOST_AUTO_TEST_CASE(openmp_self_windows_match_serial) {
    constexpr size_t N = 128;
    std::mt19937_64 rng(24);
    const auto seeds = draw_distinct<N>(rng, 6000);
    std::set<std::vector<uint64_t>> used;
    for (const auto &m : seeds) {
        used.insert(key_of<N>(m));
    }
    const auto fresh = [&] {
        for (;;) {
            const auto m = draw_distinct<N>(rng, 1)[0];
            if (used.insert(key_of<N>(m)).second) {
                return m;
            }
        }
    };
    const size_t n_leader = (2 * detail::kSelfProbeWindow) + 37;
    std::vector<Monomial<N>> leader_keys;
    std::vector<size_t> leader_srcs;
    for (size_t q = 0; q < n_leader; ++q) {
        leader_keys.push_back(q % 2 == 0 ? seeds[q / 2] : fresh());
        leader_srcs.push_back(q % seeds.size());
    }
    // Followers: every fifth source is a row a leader hit (seeds[q/2] for even q), so it is filtered.
    std::vector<Monomial<N>> follower_keys;
    std::vector<size_t> follower_srcs;
    size_t filtered = 0;
    for (size_t q = 0; q < 5000; ++q) {
        const bool matched_source = q % 5 == 0;
        follower_keys.push_back(q % 3 == 0 ? seeds[4200 + (q % 1700)] : fresh());
        follower_srcs.push_back(matched_source ? (q % 4000) : 5000 + (q % 900));
        filtered += static_cast<size_t>(matched_source);
    }
    // Duplicate hit keys across the two passes would be two rotations with one target; keep them distinct.
    {
        std::set<std::vector<uint64_t>> keys;
        for (const auto &m : leader_keys) {
            BOOST_REQUIRE(keys.insert(key_of<N>(m)).second);
        }
        for (const auto &m : follower_keys) {
            BOOST_REQUIRE(keys.insert(key_of<N>(m)).second);
        }
    }
    const size_t kept_followers = follower_keys.size() - filtered;
    for (const bool contract : {false, true}) {
        const auto run_at = [&](int threads) {
            return contract
                       ? run_self<N, true>(seeds, leader_keys, leader_srcs, follower_keys, follower_srcs, threads)
                       : run_self<N, false>(seeds, leader_keys, leader_srcs, follower_keys, follower_srcs, threads);
        };
        const auto serial = run_at(1);
        for (const int threads : {2, 3, 4}) {
            const auto threaded = run_at(threads);
            BOOST_TEST((serial.state == threaded.state), "contract=" << contract << " budget " << threads);
            const auto calls = threaded.log.of(KernelRange::self_probe);
            const size_t leader_windows = detail::logical_ranges(n_leader, detail::kSelfProbeWindow);
            const size_t follower_windows = detail::logical_ranges(kept_followers, detail::kSelfProbeWindow);
            BOOST_REQUIRE(calls.size() == leader_windows + follower_windows);
            BOOST_TEST(calls[0]->worker.size() == 16U);
            BOOST_TEST(calls[leader_windows - 1]->worker.size() == 1U); // the 37-query tail
            BOOST_TEST(calls[leader_windows]->worker.size() == 16U);
            BOOST_TEST(
                calls.back()->worker.size()
                == detail::logical_ranges(kept_followers % detail::kSelfProbeWindow, detail::kProbeBlockQueries));
            for (const auto *c : calls) {
                BOOST_TEST(c->worker.size() <= detail::kSelfProbeWindow / detail::kProbeBlockQueries);
            }
        }
        if (!!kernel_test::runtime_offers_two_workers(0)) {
            const auto threaded = run_at(4);
            const auto calls = threaded.log.of(KernelRange::self_probe);
            kernel_test::check_participation(kernel_test::slots_of(*calls[0]), 16, "self window");
            kernel_test::check_serial(kernel_test::slots_of(*calls[2]), 1, "self tail");
        }
    }
}

// Mixed self and remote hits and misses across the leader and follower passes, as run_exchange orders them:
// remote leader misses publish in the leader round, remote follower misses in the follower round, and the
// deferred self misses (leader, then follower) only after both passes. Identical at every budget.
BOOST_AUTO_TEST_CASE(openmp_mixed_self_and_remote_publication_order) {
    constexpr size_t N = 250;
    std::mt19937_64 rng(25);
    const auto seeds = draw_distinct<N>(rng, 3000);
    std::set<std::vector<uint64_t>> used;
    for (const auto &m : seeds) {
        used.insert(key_of<N>(m));
    }
    size_t next_seed = 0;
    const auto pick = [&](bool hit) {
        if (hit) {
            return seeds[next_seed++];
        }
        for (;;) {
            const auto m = draw_distinct<N>(rng, 1)[0];
            if (used.insert(key_of<N>(m)).second) {
                return m;
            }
        }
    };
    const auto keys = [&](size_t n) {
        std::vector<Monomial<N>> out;
        for (size_t q = 0; q < n; ++q) {
            out.push_back(pick(q % 3 == 0));
        }
        return out;
    };
    const auto self_leader = keys(900);
    const auto remote_leader = keys(700);
    const auto self_follower = keys(600);
    const auto remote_follower = keys(650);
    std::vector<size_t> self_leader_srcs(self_leader.size());
    std::iota(self_leader_srcs.begin(), self_leader_srcs.end(), 0);
    // Follower sources 0..299 are targets a leader hit? Leaders hit seeds[0..], i.e. rows 0,1,2,..; filter a few.
    std::vector<size_t> self_follower_srcs;
    for (size_t q = 0; q < self_follower.size(); ++q) {
        self_follower_srcs.push_back(q % 7 == 0 ? q / 7 : 2800 + (q % 190));
    }
    const mpi::SlotWindow window{.base = 0, .count = 2};
    const auto remote = [&](const std::vector<Monomial<N>> &k) {
        return serialize<N>(std::vector<std::vector<Monomial<N>>>{{}, k}, false, window);
    };
    std::vector<std::vector<std::vector<uint64_t>>> results;
    for (const int threads : {1, 2, 3, 4}) {
        auto op = make_op<N>(seeds);
        detail::MatchedEpochSet matched;
        const size_t combined = op.size();
        detail::LayerBuildEngine<N, detail::GraphSink<N>>
            eng(op, 2, 0, matched, combined, detail::GraphSink<N>(2, 0), window, {.threads = threads});
        stage_self<N>(eng, self_leader, self_leader_srcs);
        eng.resolve_self_queries(true);
        const auto resp_leader = detail::resolve_incoming<N>(remote(remote_leader),
                                                             op,
                                                             true,
                                                             matched,
                                                             combined,
                                                             eng.sink,
                                                             {.threads = threads});
        const size_t after_leader = op.size();
        stage_self<N>(eng, self_follower, self_follower_srcs);
        eng.resolve_self_queries(false);
        const auto resp_follower = detail::resolve_incoming<N>(remote(remote_follower),
                                                               op,
                                                               false,
                                                               matched,
                                                               combined,
                                                               eng.sink,
                                                               {.threads = threads});
        const size_t after_follower = op.size();
        (void)eng.finish(CosMask{}, nullptr);

        // Expected physical order: remote leader misses, remote follower misses, self leader, self follower.
        std::vector<Monomial<N>> expect(seeds);
        const auto misses = [&](const std::vector<Monomial<N>> &k, const std::vector<size_t> *srcs) {
            for (size_t q = 0; q < k.size(); ++q) {
                if (srcs != nullptr && matched.is_marked((*srcs)[q])) {
                    continue;
                }
                if (std::ranges::find(seeds, k[q]) == seeds.end()) {
                    expect.push_back(k[q]);
                }
            }
        };
        misses(remote_leader, nullptr);
        BOOST_TEST(after_leader == expect.size());
        misses(remote_follower, nullptr);
        BOOST_TEST(after_follower == expect.size());
        misses(self_leader, nullptr);
        misses(self_follower, &self_follower_srcs);
        BOOST_TEST((rows_of<N>(op) == expect), "budget " << threads);
        std::vector<std::vector<uint64_t>> r;
        r.emplace_back(resp_leader[mpi::WindowIndex{1}].begin(), resp_leader[mpi::WindowIndex{1}].end());
        r.emplace_back(resp_follower[mpi::WindowIndex{1}].begin(), resp_follower[mpi::WindowIndex{1}].end());
        for (const auto &m : rows_of<N>(op)) {
            r.push_back(key_of<N>(m));
        }
        for (const auto &slot : graph_state<N>(eng.sink)) {
            r.push_back(slot);
        }
        results.push_back(r);
    }
    for (size_t t = 1; t < results.size(); ++t) {
        BOOST_TEST((results[0] == results[t]), "budget " << t + 1);
    }
}
