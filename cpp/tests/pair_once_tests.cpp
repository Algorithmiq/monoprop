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

// Pair-once (Resolve.h join_self): the value path settles a mutual pair from the leader's record alone
// when that record is resolved first, and skips the follower's record at the probe. This runs the real
// engine with the real ContractSink over a hand-built gate that has every case at once -- twenty
// leader-first mutual pairs (far enough apart that the follower's probe is skipped by the pipeline, not
// only by the resolve), one follower-first mutual pair, a rotating leader onto a silent follower, a
// rotating follower onto a silent leader, and two absent partners -- and compares the halves, the mints
// and the marks bit for bit against the same gate resolved with two records per pair. The exact number
// of probes the skip saves is deliberately not asserted: it is a property of the transport, and the
// pair exchange changes which records reach the resolve first.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/GateSinks.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/evolution/layer_build/TableJoin.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowKey.h"

using namespace monoprop;

namespace {

constexpr size_t kN = 64;
using Op = detail::MPOperator<kN>;
using Store = detail::OperatorIndex<kN>;
using PosT = Store::PosT;

auto key_of(const Monomial<kN> &m) -> uint32_t {
    return detail::key_of<2 * kN>(m);
}

auto mono(std::initializer_list<size_t> bits) -> Monomial<kN> {
    Monomial<kN> m;
    for (const size_t b : bits) {
        m.set(b);
    }
    return m;
}

auto positions_of(const Monomial<kN> &m) -> std::vector<PosT> {
    std::vector<PosT> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<PosT>(b));
    }
    return pos;
}

// Appends `terms` as rows [base, base + n) through the engine's own growth door, so the table follows.
auto append_terms(Op &op, const std::vector<Monomial<kN>> &terms) -> size_t {
    return detail::insert_absent_terms<kN>(op, terms.size(), [&](size_t k, size_t base) {
        const auto pos = positions_of(terms[k]);
        op.store->set_positions(base + k, std::span<const PosT>(pos));
    });
}

auto draw_distinct(std::mt19937_64 &rng, size_t n) -> std::vector<Monomial<kN>> {
    std::vector<Monomial<kN>> out;
    std::set<std::vector<uint64_t>> seen;
    std::uniform_int_distribution<size_t> bit(0, Monomial<kN>::size() - 1);
    std::uniform_int_distribution<size_t> pop(1, 8);
    while (out.size() < n) {
        Monomial<kN> m;
        const size_t k = pop(rng);
        while (m.count() < k) {
            m.set(bit(rng));
        }
        std::vector<uint64_t> words;
        for (size_t w = 0; w < Monomial<kN>::num_words(); ++w) {
            words.push_back(m.word(w));
        }
        if (seen.insert(words).second) {
            out.push_back(m);
        }
    }
    return out;
}

// The gate. Pivot bit 0: a follower has it set, a leader has not. Rows and their records:
//   [0, 20)    leaders, rotating, partner row 20 + i        leader-first mutual pairs
//   [20, 40)   followers, rotating, partner row i - 20
//   40 / 41    follower C / leader D, both rotating         follower-first mutual pair
//   42 / 43    leader E rotating / follower F silent        answered on the self slot
//   44 / 45    follower G rotating / leader H silent        answered on the self slot
//   46         leader I, rotating, partner X absent         mint at base + 0
//   47         follower J, rotating, partner Y absent       mint at base + 1
struct Gate {
    static constexpr size_t kPairs = 20;
    static constexpr size_t kRows = 48;
    static constexpr size_t kTag = 100; // a position every row carries, so no row is a single position

    std::vector<Monomial<kN>> terms;
    std::vector<Monomial<kN>> record_key; // the partner each rotating row names; empty for a silent row
    std::vector<bool> foll;
    std::vector<bool> rot;
    std::vector<int> phase;
    std::vector<double> v;
    Op op;
    detail::GateScratch<kN> scratch;
    VecD coeffs;
    detail::FusedContract fc;
    double cos_build = std::cos(2.0 * 0.37);
    double inv_cos = 1.0 / cos_build;

    Gate() : terms(kRows), record_key(kRows), foll(kRows), rot(kRows), phase(kRows), v(kRows), coeffs(kRows) {
        auto leader = [](size_t k) { return mono({k, kTag}); };
        auto follower = [](size_t k) { return mono({0, k, kTag}); };
        auto set = [&](size_t row, Monomial<kN> m, bool is_foll, bool rotates, Monomial<kN> key, int ph) {
            terms[row] = m;
            foll[row] = is_foll;
            rot[row] = rotates;
            record_key[row] = key;
            phase[row] = ph;
        };
        for (size_t i = 0; i < kPairs; ++i) {
            set(i, leader(i + 1), false, true, follower(i + 1), +1);
            set(kPairs + i, follower(i + 1), true, true, leader(i + 1), -1);
        }
        set(40, follower(41), true, true, leader(41), -1);
        set(41, leader(41), false, true, follower(41), +1);
        set(42, leader(43), false, true, follower(43), +1);
        set(43, follower(43), true, false, {}, 0);
        set(44, follower(45), true, true, leader(45), -1);
        set(45, leader(45), false, false, {}, 0);
        set(46, leader(47), false, true, follower(47), +1); // {0, 47, kTag} is no row
        set(47, follower(48), true, true, leader(48), -1);  // {48, kTag} is no row
        for (size_t i = 0; i < kRows; ++i) {
            // Pre-cos values no rounding identity holds for by accident: irrational-looking, mixed signs.
            v[i] = (i % 2 == 0 ? 1.0 : -1.0) / (static_cast<double>(i) + 3.0);
            if (!rot[i]) {
                v[i] *= 1e-9; // silent: below any threshold, still swept
            }
            // The fused cos sweep: every anticommuting slot holds fl(v * cos) when the join runs.
            coeffs[i] = v[i] * cos_build;
        }
        detail::insert_absent_terms<kN>(op, kRows, [&](size_t k, size_t base) {
            const auto pos = positions_of(terms[k]);
            op.store->set_positions(base + k, std::span<const PosT>(pos));
        });
        op.op_coeffs = coeffs;
        uint64_t foll_bits = 0;
        for (size_t i = 0; i < kRows; ++i) {
            foll_bits |= static_cast<uint64_t>(foll[i]) << i;
        }
        scratch.nz = {detail::EvenParityNzWord{.base = 0, .overlap = (uint64_t{1} << kRows) - 1U, .foll = foll_bits}};
        scratch.marks.begin(kRows, scratch.nz);
        for (size_t i = 0; i < kRows; ++i) {
            if (foll[i]) {
                scratch.marks.set_foll(i);
            }
            if (rot[i]) {
                scratch.marks.set_rot(i);
            }
        }
    }

    // The value-path scan's product: one rot=1 record per rotating row, in ascending row order.
    auto scan() -> detail::FusedScanResult<kN> {
        detail::FusedScanResult<kN> res;
        res.window = mpi::SlotWindow{.base = 0, .count = 1};
        res.queries.reset(res.window);
        res.sent.reset(res.window);
        res.self.keeps_values = true;
        for (size_t i = 0; i < kRows; ++i) {
            if (!rot[i]) {
                continue;
            }
            const auto pos = positions_of(record_key[i]);
            res.self.push(pos, phase[i], key_of(record_key[i]), /*rot=*/true, v[i]);
            res.sent.at_slot(0).push_back(
                detail::SentRecord{.row = static_cast<TermIndex>(i), .phase = static_cast<int8_t>(phase[i])});
        }
        return res;
    }

    auto sink() -> detail::ContractSink<kN> {
        return detail::ContractSink<kN>{.fc = fc,
                                        .fused_scale = true,
                                        .op_coeffs = op.op_coeffs,
                                        .cos_build = cos_build,
                                        .inv_cos = inv_cos,
                                        .absences_carry_c0 = false};
    }
};

// ContractSink with pair-once declared off: the two-record resolution, the oracle.
struct TwoRecordSink : detail::ContractSink<kN> {
    static constexpr bool pairs_once = false;
};

static_assert(detail::sink_pairs_once<detail::ContractSink<kN>>());
static_assert(!detail::sink_pairs_once<TwoRecordSink>());
static_assert(!detail::sink_pairs_once<detail::GraphSink<kN>>());

// One half as bits, so a value that differs by one ULP is a difference.
using HalfBits = std::tuple<TermIndex, int, bool, uint64_t>;

struct Outcome {
    std::vector<HalfBits> halves;                    // sorted by slot; every slot appears at most once
    std::vector<std::tuple<bool, bool, bool>> marks; // received, partner_rot, answered per row
    std::vector<Monomial<kN>> rows;                  // the store after the mints
    size_t skipped = 0;
};

template <typename Sink>
auto run(Gate &g, Sink sink) -> Outcome {
    detail::LayerBuildEngine<kN, Sink> eng(g.op,
                                           mpi::Comm{},
                                           /*R_=*/1,
                                           /*my_rank_=*/0,
                                           g.scratch,
                                           /*combined_size_=*/Gate::kRows,
                                           std::move(sink));
    eng.exchange_and_join(g.scan());
    Outcome out;
    for (const auto &h : g.fc.halves) {
        out.halves.emplace_back(h.local_idx,
                                static_cast<int>(h.phase_signed),
                                h.is_insert,
                                std::bit_cast<uint64_t>(h.v_partner));
    }
    std::sort(out.halves.begin(), out.halves.end());
    for (size_t i = 0; i < Gate::kRows; ++i) {
        out.marks.emplace_back(g.scratch.marks.received(i),
                               g.scratch.marks.partner_rot(i),
                               g.scratch.marks.answered(i));
    }
    for (size_t i = 0; i < g.op.store->size(); ++i) {
        out.rows.push_back(g.op.store->row(i));
    }
    out.skipped = g.scratch.join.skipped_count();
    return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(pair_once_settles_leader_first_pairs_from_one_record_and_is_bit_identical) {
    Gate once;
    Gate twice;
    const Outcome a = run(once, once.sink());
    const Outcome b = run(twice, TwoRecordSink{twice.sink()});

    // Some leader-first follower was skipped, no more than one per pair, and nothing at all without
    // the rule. How many is the transport's business, so it is not pinned here.
    BOOST_TEST(a.skipped > 0U);
    BOOST_TEST(a.skipped <= Gate::kPairs);
    BOOST_TEST(b.skipped == 0U);

    // Every rotating pair endpoint gets exactly one half, plus the two mints: 46 pre-gate slots + 2.
    BOOST_REQUIRE_EQUAL(a.halves.size(), Gate::kRows - 2 + 2);
    BOOST_REQUIRE_EQUAL(b.halves.size(), a.halves.size());
    for (size_t k = 1; k < a.halves.size(); ++k) {
        BOOST_TEST(std::get<0>(a.halves[k]) != std::get<0>(a.halves[k - 1])); // one half per slot
    }
    BOOST_TEST(a.halves == b.halves);
    BOOST_TEST(a.marks == b.marks);
    BOOST_TEST(a.rows == b.rows);

    // The identity the saving rests on, pinned on one pair: the leader's half carries the follower's
    // value recovered from its swept slot, the follower's half the leader's exact pre-cos value.
    auto half_of = [&](const Outcome &o, size_t slot) {
        const auto it =
            std::find_if(o.halves.begin(), o.halves.end(), [&](const HalfBits &h) { return std::get<0>(h) == slot; });
        BOOST_REQUIRE(it != o.halves.end());
        return *it;
    };
    const HalfBits leader = half_of(a, 0);
    const HalfBits follower = half_of(a, Gate::kPairs);
    BOOST_TEST(std::get<3>(leader) == std::bit_cast<uint64_t>((once.v[Gate::kPairs] * once.cos_build) * once.inv_cos));
    BOOST_TEST(std::get<1>(leader) == -once.phase[0]);
    BOOST_TEST(std::get<3>(follower) == std::bit_cast<uint64_t>(once.v[0]));
    BOOST_TEST(std::get<1>(follower) == once.phase[0]);
    BOOST_TEST(!std::get<2>(leader));
    BOOST_TEST(!std::get<2>(follower));

    // The mints, in record order: X from I's record, then Y from J's, and the table followed the store.
    BOOST_REQUIRE_EQUAL(a.rows.size(), Gate::kRows + 2);
    BOOST_TEST((a.rows[Gate::kRows] == mono({0, 47, Gate::kTag})));
    BOOST_TEST((a.rows[Gate::kRows + 1] == mono({48, Gate::kTag})));
    BOOST_TEST(once.op.term_table().rows() == Gate::kRows + 2);
    const auto x = positions_of(a.rows[Gate::kRows]);
    BOOST_TEST(once.op.term_table().find(*once.op.store, key_of(a.rows[Gate::kRows]), std::span<const PosT>(x))
               == Gate::kRows);

    // Marks after the gate: both endpoints of every tracked pair received with the partner's rot; the
    // silent partners answered their leaders; the absent partners left their senders unanswered.
    const auto &m = once.scratch.marks;
    for (size_t i = 0; i < 42; ++i) {
        BOOST_TEST(m.received(i));
        BOOST_TEST(m.partner_rot(i));
        BOOST_TEST(!m.answered(i));
    }
    BOOST_TEST((m.received(43) && m.partner_rot(43) && m.answered(42) && !m.received(42)));
    BOOST_TEST((m.received(45) && m.partner_rot(45) && m.answered(44) && !m.received(44)));
    BOOST_TEST((!m.received(46) && !m.answered(46)));
    BOOST_TEST((!m.received(47) && !m.answered(47)));
}

// The follower-first pair alone, in both resolutions: the leader's record takes the ordinary arm because
// the follower's already set received on it, and nothing is pushed twice.
BOOST_AUTO_TEST_CASE(pair_once_follower_first_pair_takes_the_two_record_path) {
    Gate g;
    const Outcome a = run(g, g.sink());
    size_t on_c = 0;
    size_t on_d = 0;
    for (const auto &h : a.halves) {
        on_c += static_cast<size_t>(std::get<0>(h) == 40);
        on_d += static_cast<size_t>(std::get<0>(h) == 41);
    }
    BOOST_TEST(on_c == 1U);
    BOOST_TEST(on_d == 1U);
    BOOST_TEST((g.scratch.marks.received(40) && g.scratch.marks.received(41)));
}

// The probe's skip bookkeeping, on the join alone: a declined query is recorded as skipped, counted as a
// hit (it can never mint) and never confirmed, and the predicate is consulted once per query in order.
BOOST_AUTO_TEST_CASE(table_join_records_skipped_queries_as_settled) {
    std::mt19937_64 rng(16);
    const auto terms = draw_distinct(rng, 100);
    Op op;
    append_terms(op, terms);
    std::vector<uint32_t> keys;
    std::vector<std::vector<PosT>> pos;
    for (const auto &m : terms) {
        keys.push_back(key_of(m));
        pos.push_back(positions_of(m));
    }
    detail::TableJoin<kN> join;
    join.begin_queries(terms.size());
    std::vector<size_t> asked_skip;
    std::vector<size_t> confirmed;
    join.run(
        op.term_table(),
        *op.store,
        [&](size_t q) { return keys[q]; },
        [&](size_t q) { return std::span<const PosT>(pos[q]); },
        [&](size_t q) {
            asked_skip.push_back(q);
            return q % 3 == 0;
        },
        [&](size_t q, size_t) { confirmed.push_back(q); });
    BOOST_TEST(std::is_sorted(asked_skip.begin(), asked_skip.end()));
    BOOST_TEST(asked_skip.size() == terms.size());
    size_t skipped = 0;
    for (size_t q = 0; q < terms.size(); ++q) {
        if (q % 3 == 0) {
            BOOST_TEST(join.skipped(q));
            BOOST_TEST((std::find(confirmed.begin(), confirmed.end(), q) == confirmed.end()));
            ++skipped;
        }
        else {
            BOOST_TEST(!join.skipped(q));
            BOOST_TEST(join.hit(q) == q);
        }
    }
    BOOST_TEST(join.skipped_count() == skipped);
    BOOST_TEST(join.hits() == terms.size()); // confirmed + skipped
    BOOST_TEST(join.queries() - join.hits() == 0U);
}

// The same bookkeeping with nothing to probe: an empty table confirms no query, and a declined one is
// still recorded as skipped rather than as a miss.
BOOST_AUTO_TEST_CASE(table_join_skips_on_an_empty_table_without_probing) {
    Op op;
    const auto asked = mono({1, 3, 5});
    const auto pos = positions_of(asked);
    detail::TableJoin<kN> join;
    join.begin_queries(2);
    join.run(
        op.term_table(),
        *op.store,
        [&](size_t) { return key_of(asked); },
        [&](size_t) { return std::span<const PosT>(pos); },
        [](size_t q) { return q == 0; },
        [](size_t, size_t) { BOOST_FAIL("an empty table cannot confirm a row"); });
    BOOST_TEST(join.skipped(0));
    BOOST_TEST(!join.skipped(1));
    BOOST_TEST(join.hit(1) == detail::TableJoin<kN>::kMissing);
    BOOST_TEST(join.skipped_count() == 1U);
    BOOST_TEST(join.hits() == 1U);
}
