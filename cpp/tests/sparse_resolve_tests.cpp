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

// The receiver side of the one-round exchange: the incoming records decoded (IncomingRecords), matched
// against the operator's term table (TableJoin), joined under the receiver rule (join_incoming) and the
// misses inserted (MissStage / insert_misses) -- differentially against the records the caller built, the
// dense Monomial-keyed insert path and the table's by-value find, none of which is the code under test.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <span>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/TableJoin.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/RowKey.h"
#include "monoprop/detail/operator/TermTable.h"

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

// The per-gate state a gate that finds EVERY row anticommuting would leave behind: marks cleared over
// all of them. This is the receiver-side oracle the decode and join run against; the rows themselves are
// reached through the operator's term table, spilled ones included, exactly as the engine reaches them.
template <size_t NumModes>
struct AllRowsGate {
    detail::TableJoin<NumModes> join;
    detail::RowMarks marks;
    std::vector<detail::EvenParityNzWord> nz;

    explicit AllRowsGate(const detail::MPOperator<NumModes> &op) {
        const size_t n = op.store->size();
        for (size_t base = 0; base < n; base += 64) {
            const size_t k = std::min<size_t>(64, n - base);
            nz.push_back(detail::EvenParityNzWord{.base = base,
                                                  .overlap = (k == 64) ? ~uint64_t{0} : ((uint64_t{1} << k) - 1U),
                                                  .foll = 0});
        }
        marks.begin(n, nz);
    }

    // Probes a decoded batch's keys in record order against the operator's term table.
    template <typename Records>
    auto match(const detail::MPOperator<NumModes> &op, const Records &pr) -> void {
        join.begin_queries(pr.nq_total);
        join.run(
            op.term_table(),
            *op.store,
            [&](size_t q) { return pr.tag_of[q]; },
            [&](size_t q) { return pr.positions_at(q); },
            [](size_t, size_t) {});
    }
};

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

// Record q of sender s carries phase (+1, -1 alternating), rot = (q % 3 != 2) and value 0.5 + q.
auto record_phase(size_t q) -> int {
    return ((q % 2) == 0) ? 1 : -1;
}
auto record_rot(size_t q) -> bool {
    return (q % 3) != 2;
}
auto record_value(size_t q) -> double {
    return 0.5 + static_cast<double>(q);
}

template <size_t NumModes>
auto serialize(const std::vector<std::vector<Monomial<NumModes>>> &queries, bool fused) -> std::vector<VecZ> {
    std::vector<VecZ> incoming(queries.size());
    for (size_t s = 0; s < queries.size(); ++s) {
        VecZ &buf = incoming[s];
        for (size_t q = 0; q < queries[s].size(); ++q) {
            const auto pos = positions_of<NumModes>(queries[s][q]);
            detail::QueryWire<NumModes>::push(buf, pos, record_phase(q), record_rot(q));
            if (fused) {
                detail::QueryWire<NumModes>::push_value(buf, record_value(q));
            }
        }
    }
    return incoming;
}

// Records every sink call with its slot, so the join order and the slot attribution are both pinned.
struct RecordingSink {
    // No value column: the decode's own value checks above cover it, and join_incoming reads val_of only
    // when a sink asks for it -- which the Plain half of this case has none of.
    static constexpr bool wants_values = false;
    static constexpr bool wants_responses = false;

    struct Rec {
        char kind; // 'h' hit, 'm' mint
        size_t slot;
        size_t idx;
        double v;
        int phase;
        bool foll;
    };
    std::vector<Rec> recs;
    auto hit(size_t slot, size_t row, double v, int phase, bool foll, bool /*own_rot*/) -> void {
        recs.push_back({'h', slot, row, v, phase, foll});
    }
    auto mint(size_t slot, size_t idx, double v, int phase) -> void {
        recs.push_back({'m', slot, idx, v, phase, false});
    }
    auto out_pair(size_t, size_t, int) -> void { BOOST_FAIL("the join never reports out-side entries"); }
    auto out_unanswered(size_t, size_t, double, int) -> void { BOOST_FAIL("the join never reports out-side entries"); }
};

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
        // A repeat would violate the mint's distinctness precondition; the engine gets it from ^G.
        if (!queried.insert(key).second) {
            continue;
        }
        (want_hit ? hits_planned : misses_planned) += 1;
        queries[i % rank_count].push_back(m);
    }
    BOOST_REQUIRE(hits_planned > 0);
    BOOST_REQUIRE(misses_planned > 0);

    // The flat record order the receiver must reproduce: senders ascending, each in stream order.
    std::vector<Monomial<NumModes>> expect_mono;
    std::vector<int> expect_phase;
    std::vector<bool> expect_rot;
    std::vector<double> expect_value;
    std::vector<size_t> expect_sender;
    for (size_t s = 0; s < rank_count; ++s) {
        for (size_t q = 0; q < queries[s].size(); ++q) {
            expect_mono.push_back(queries[s][q]);
            expect_phase.push_back(record_phase(q));
            expect_rot.push_back(record_rot(q));
            expect_value.push_back(fused ? record_value(q) : 0.0);
            expect_sender.push_back(s);
        }
    }

    const auto incoming = serialize<NumModes>(queries, fused);
    const detail::QueryForm form = fused ? detail::QueryForm::Fused : detail::QueryForm::Plain;

    auto op = make_op<NumModes>(seed_terms);
    AllRowsGate<NumModes> gate(op);
    std::vector<std::span<const size_t>> views;
    detail::IncomingRecords<NumModes> pr;
    detail::decode_incoming_records<NumModes>(detail::slot_streams(incoming, views), form, pr);
    gate.match(op, pr);
    BOOST_REQUIRE_EQUAL(pr.nq_total, expect_mono.size());
    BOOST_REQUIRE(pr.nq_total > 0);
    BOOST_REQUIRE_EQUAL(pr.goff.size(), rank_count + 1);
    BOOST_REQUIRE_EQUAL(pr.goff.back(), pr.nq_total);
    BOOST_REQUIRE_EQUAL(pr.pos_off.size(), pr.nq_total);
    BOOST_REQUIRE_EQUAL(pr.val_of.empty(), !fused);

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
    size_t dropped_seen = 0;
    std::vector<Monomial<NumModes>> expected_mints; // rot=1 misses, in flat record order
    std::vector<size_t> expected_mint_slot;
    std::vector<size_t> expected_hit_row;
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const Monomial<NumModes> &want = expect_mono[g];
        const auto pos = pr.positions_at(g);
        Monomial<NumModes> got;
        for (const auto p : pos) {
            got.set(static_cast<size_t>(p));
        }
        BOOST_TEST((got == want));
        BOOST_TEST(pr.k_of[g] == want.count());
        BOOST_TEST(pr.phase_of[g] == expect_phase[g]);
        BOOST_TEST((pr.rot_of[g] != 0) == expect_rot[g]);
        BOOST_TEST(pr.value_at(g) == expect_value[g]);
        const size_t s = expect_sender[g];
        BOOST_TEST(pr.goff[s] <= g);
        BOOST_TEST(g < pr.goff[s + 1]);

        std::vector<uint64_t> key;
        for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
            key.push_back(want.word(w));
        }
        const bool want_hit = seeded.count(key) != 0;
        const size_t row = gate.join.hit(g);
        BOOST_TEST((row != detail::TableJoin<NumModes>::kMissing) == want_hit);
        if (want_hit) {
            BOOST_TEST((op.store->row(row) == want));
            expected_hit_row.push_back(row);
            ++hits_seen;
        }
        else if (expect_rot[g]) {
            expected_mints.push_back(want);
            expected_mint_slot.push_back(s);
        }
        else {
            ++dropped_seen;
        }
        VecZ scratch;
        const auto want_pos = positions_of<NumModes>(want);
        if (detail::QueryWire<NumModes>::push(scratch, want_pos, expect_phase[g]) > 1U) {
            ++wide_seen;
        }
    }
    // Vacuous-pass guards: no hit means the confirm never ran, no wide record means the cursor didn't,
    // no dropped record means the rot=0 miss arm never ran.
    BOOST_TEST(hits_seen > 0);
    BOOST_TEST(wide_seen > 0);
    BOOST_TEST(dropped_seen > 0);
    BOOST_REQUIRE(!expected_mints.empty());

    // The join: every hit applies (no row has its rot bit set, so only rot=1 records rotate a hit),
    // every rot=1 miss mints at base + j in flat record order, every rot=0 miss is dropped.
    const size_t base = op.store->size();
    detail::MissStage<NumModes> misses;
    RecordingSink sink;
    std::vector<VecZ> responses; // wants_responses is false, so the join never touches it
    detail::join_incoming<NumModes>(pr, gate.join, /*q_base=*/0, gate.marks, base, misses, sink, responses);
    BOOST_REQUIRE_EQUAL(misses.size(), expected_mints.size());
    size_t mints_seen = 0;
    size_t hit_calls = 0;
    for (const auto &r : sink.recs) {
        if (r.kind == 'm') {
            BOOST_REQUIRE(mints_seen < expected_mints.size());
            BOOST_TEST(r.idx == base + mints_seen);
            BOOST_TEST(r.slot == expected_mint_slot[mints_seen]);
            Monomial<NumModes> minted;
            for (const auto p : misses.positions_at(mints_seen)) {
                minted.set(static_cast<size_t>(p));
            }
            BOOST_TEST((minted == expected_mints[mints_seen]));
            ++mints_seen;
        }
        else {
            ++hit_calls;
            BOOST_TEST(r.idx < base);
            BOOST_TEST((std::find(expected_hit_row.begin(), expected_hit_row.end(), r.idx) != expected_hit_row.end()));
            BOOST_TEST(r.slot < rank_count);
        }
    }
    BOOST_TEST(mints_seen == expected_mints.size());
    // Only rot=1 records rotate an unmarked hit, so the hit calls are the rot=1 hits.
    size_t rot_hits = 0;
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t row = gate.join.hit(g);
        if (row != detail::TableJoin<NumModes>::kMissing) {
            BOOST_TEST(gate.marks.received(row));
            if (expect_rot[g]) {
                ++rot_hits;
            }
        }
    }
    BOOST_TEST(hit_calls == rot_hits);
    BOOST_TEST(rot_hits > 0);

    detail::insert_misses<NumModes>(op, misses, base);

    // The second implementation: the dense Monomial-keyed path, sharing no code with set_positions.
    auto ref = make_op<NumModes>(seed_terms);
    detail::insert_absent_terms<NumModes>(ref, expected_mints.size(), [&](size_t j, size_t b) {
        assign_row<NumModes>(*ref.store, b + j, expected_mints[j]);
    });

    BOOST_REQUIRE_EQUAL(op.store->size(), ref.store->size());
    BOOST_TEST(op.store->size() > base);
    size_t overflow_seen = 0;
    for (size_t i = 0; i < ref.store->size(); ++i) {
        BOOST_TEST((op.store->row(i) == ref.store->row(i)));
        BOOST_TEST(op.store->popcount(i) == ref.store->popcount(i));
        if (!ref.store->row_positions(i).inlined()) {
            ++overflow_seen;
        }
    }
    BOOST_TEST(overflow_seen > 0);

    // The term table, not just the rows: an unindexed row is correct and unfindable, and the next gate
    // would mint a duplicate of it.
    for (size_t i = 0; i < ref.store->size(); ++i) {
        const auto want = ref.store->row(i);
        BOOST_TEST(op.term_table().find(*op.store, want) == i);
        BOOST_TEST(ref.term_table().find(*ref.store, want) == i);
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
                                         /*fused=*/false);
}

BOOST_AUTO_TEST_CASE(sparse_resolve_probe_matches_fused_layout) {
    std::mt19937_64 rng(20260816);
    check_probe_matches_the_queries<250>(rng,
                                         /*n_seed=*/50,
                                         /*n_query=*/120,
                                         /*rank_count=*/2,
                                         /*fused=*/true);
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

BOOST_AUTO_TEST_CASE(sparse_resolve_join_matches_the_by_value_oracle) {
    // The gate join against the term table's own by-value find over the same rows: every seeded term,
    // asked for as a query, matches its own row, and a genuinely absent term matches nothing. Spilled
    // (wide) rows are in the draw, which is what pins the fingerprint of a row with no position array.
    constexpr size_t kN = 250;
    using PosT = detail::OperatorIndex<kN>::PosT;
    std::mt19937_64 rng(20260819);
    const auto terms = draw_distinct<kN>(rng, 300);
    auto op = make_op<kN>(terms);
    const uint64_t *labels = routing::linear_basis<2 * kN>().data();

    const auto absent = draw_distinct<kN>(rng, 50);
    std::vector<Monomial<kN>> asked = terms;
    size_t genuinely_absent = 0;
    for (const auto &m : absent) {
        // draw_distinct may re-draw a seeded term; only genuinely absent ones are evidence.
        if (op.term_table().find(*op.store, m) == detail::TermTable::kNotFound) {
            asked.push_back(m);
            ++genuinely_absent;
        }
    }
    BOOST_TEST(genuinely_absent > 0);

    std::vector<std::vector<PosT>> query_pos;
    query_pos.reserve(asked.size());
    for (const auto &m : asked) {
        query_pos.push_back(positions_of<kN>(m));
    }
    AllRowsGate<kN> gate(op);
    std::vector<uint32_t> keys;
    for (size_t q = 0; q < asked.size(); ++q) {
        const uint64_t fp = routing::fingerprint_positions(labels, query_pos[q].data(), query_pos[q].size());
        BOOST_REQUIRE_EQUAL(fp, routing::linear_hash<2 * kN>(asked[q]));
        keys.push_back(detail::join_tag(fp));
    }
    gate.join.begin_queries(asked.size());
    gate.join.run(
        op.term_table(),
        *op.store,
        [&](size_t q) { return keys[q]; },
        [&](size_t q) { return std::span<const PosT>(query_pos[q]); },
        [](size_t, size_t) {});

    size_t spilled = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        BOOST_REQUIRE(gate.join.hit(i) != detail::TableJoin<kN>::kMissing);
        BOOST_TEST(gate.join.hit(i) == i);
        BOOST_TEST(op.term_table().find(*op.store, terms[i]) == i);
        if (!op.store->row_positions(i).inlined()) {
            ++spilled;
        }
    }
    BOOST_TEST(spilled > 0);
    for (size_t q = terms.size(); q < asked.size(); ++q) {
        BOOST_TEST(gate.join.hit(q) == detail::TableJoin<kN>::kMissing);
    }
}
