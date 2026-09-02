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

// The one-round gate exchange (Engine.h): the join and absence rules driven directly on a hand-built scan
// result, the graph sink's endpoint layout, the phase antisymmetry the protocol's exactness rests on, and
// the receiver rule end to end through propagate() on two-term operators. Plus the CutoffContext predicates.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/BucketJoin.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

using namespace monoprop;
using monoprop::detail::CutoffContext;

namespace {

// Records every sink surface in call order.
struct RecordingSink {
    static constexpr bool wants_values = true;
    [[nodiscard]] auto incoming_form() const -> detail::QueryForm { return detail::QueryForm::Fused; }

    struct Rec {
        std::string kind;
        size_t slot;
        size_t idx;
        double v;
        int phase;
        bool foll;
    };
    std::vector<Rec> recs;

    auto hit(size_t slot, size_t row, double v, int phase, bool foll) -> void {
        recs.push_back({"hit", slot, row, v, phase, foll});
    }
    auto mint(size_t slot, size_t idx, double v, int phase) -> void {
        recs.push_back({"mint", slot, idx, v, phase, false});
    }
    auto out_pair(size_t slot, size_t row, int phase) -> void {
        recs.push_back({"out_pair", slot, row, 0.0, phase, false});
    }
    auto out_unanswered(size_t slot, size_t row, double c0, int phase) -> void {
        recs.push_back({"out_unanswered", slot, row, c0, phase, false});
    }
};

constexpr size_t kN = 8;
using Op = detail::MPOperator<kN>;
using RowPosT = detail::OperatorIndex<kN>::PosT;

auto indexed_op(const std::vector<Monomial<kN>> &terms) -> Op {
    Op op;
    detail::insert_absent_terms<kN>(op, terms.size(), [&](size_t k, size_t base) {
        assign_row<kN>(*op.store, base + k, terms[k]);
    });
    return op;
}

auto positions_of(const Monomial<kN> &m) -> std::vector<RowPosT> {
    std::vector<RowPosT> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<RowPosT>(b));
    }
    return pos;
}

auto fp_of(const Monomial<kN> &m) -> uint64_t {
    const auto pos = positions_of(m);
    return routing::fingerprint_positions(routing::linear_basis<2 * kN>().data(), pos.data(), pos.size());
}

// The scenario every engine case below runs: six tracked anticommuting rows t0..t5, each sending one
// self-addressed record in stream order. Odd rows carry the pivot (foll).
//
//   src  key         rot  φ    v      expectation at the join
//   t0   t1 (hit)    1    +1   0.5    hit: rotates by the record's rot
//   t1   X  (absent) 1    -1   0.25   mint at base+0
//   t2   t3 (hit)    0    +1   0.75   hit: rotates by t3's own rot
//   t3   t2 (hit)    1    -1   1.5    hit on a leader row (foll clear)
//   t4   t5 (hit)    0    +1   2.0    hit: rotates by t5's own rot
//   t5   Y  (absent) 0    +1   3.0    dropped: a silent record that missed mints nothing
//
// Own rot bits: t0, t1, t3, t5. Nobody sends key t0 or t4, so those are the unanswered ones.
struct Scenario {
    std::vector<Monomial<kN>> terms;
    Monomial<kN> absent_x = indices_to_bitset<kN>({2, 3});
    Monomial<kN> absent_y = indices_to_bitset<kN>({4, 5});
    Op op;
    detail::GateScratch<kN> scratch;
    // The scan's product: rows 0..5 are the gate's anticommuting set, all in word 0.
    std::vector<detail::EvenParityNzWord> nz{{.base = 0, .overlap = 0b11'1111, .foll = 0b10'1010}};

    Scenario() {
        for (size_t i = 0; i < 6; ++i) {
            terms.push_back(indices_to_bitset<kN>({i, i + 8}));
        }
        op = indexed_op(terms);
        scratch.marks.begin(terms.size(), nz);
        scratch.join.begin_rows(terms.size());
        for (size_t i = 0; i < terms.size(); ++i) {
            scratch.join.add_row(fp_of(terms[i]), i);
            if (i % 2 == 1) {
                scratch.marks.set_foll(i);
            }
        }
        for (const size_t row : {0U, 1U, 3U, 5U}) {
            scratch.marks.set_rot(row);
        }
    }

    auto scan(bool with_c0) -> detail::FusedScanResult<kN> {
        detail::FusedScanResult<kN> res;
        res.window = mpi::SlotWindow{.base = 0, .count = 1};
        res.queries.reset(res.window);
        res.sent.reset(res.window);
        if (with_c0) {
            res.sent_c0.reset(res.window);
            res.sent_c0.at_slot(0) = {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        auto push = [&](const Monomial<kN> &key, int phase, bool rot, double v, size_t row) {
            res.self.push(positions_of(key), phase, fp_of(key), rot, v);
            res.sent.at_slot(0).push_back(
                detail::SentRecord{.row = static_cast<TermIndex>(row), .phase = static_cast<int8_t>(phase)});
        };
        push(terms[1], 1, true, 0.5, 0);
        push(absent_x, -1, true, 0.25, 1);
        push(terms[3], 1, false, 0.75, 2);
        push(terms[2], -1, true, 1.5, 3);
        push(terms[5], 1, false, 2.0, 4);
        push(absent_y, 1, false, 3.0, 5);
        return res;
    }
};

} // namespace

// The receiver rule on the self slot: hits rotate iff rot_rec ∨ rot_own, a rot=1 miss mints at base+j, a
// rot=0 miss is dropped, and the absence pass reports the unanswered E-record and the answered leader.
BOOST_AUTO_TEST_CASE(one_round_join_applies_the_receiver_rule) {
    Scenario sc;
    detail::LayerBuildEngine<kN, RecordingSink> eng(sc.op,
                                                    mpi::Comm{},
                                                    /*R_=*/1,
                                                    /*my_rank_=*/0,
                                                    sc.scratch,
                                                    /*combined_size_=*/6,
                                                    RecordingSink{});
    eng.exchange_and_join(sc.scan(/*with_c0=*/true));

    const auto &r = eng.sink.recs;
    BOOST_REQUIRE_EQUAL(r.size(), 7U);
    // Join, in stream order.
    BOOST_TEST(r[0].kind == "hit");
    BOOST_TEST(r[0].idx == 1U);
    BOOST_TEST(r[0].v == 0.5);
    BOOST_TEST(r[0].phase == 1);
    BOOST_TEST(r[0].foll);
    BOOST_TEST(r[1].kind == "mint");
    BOOST_TEST(r[1].idx == 6U); // base + 0
    BOOST_TEST(r[1].v == 0.25);
    BOOST_TEST(r[1].phase == -1);
    BOOST_TEST(r[2].kind == "hit"); // rot=0 record, but t3's own rot is set
    BOOST_TEST(r[2].idx == 3U);
    BOOST_TEST(r[2].v == 0.75);
    BOOST_TEST(r[2].foll);
    BOOST_TEST(r[3].kind == "hit");
    BOOST_TEST(r[3].idx == 2U);
    BOOST_TEST(r[3].phase == -1);
    BOOST_TEST(!r[3].foll);
    BOOST_TEST(r[4].kind == "hit"); // rot=0 record, t5's own rot is set
    BOOST_TEST(r[4].idx == 5U);
    BOOST_TEST(r[4].v == 2.0);
    // Absence pass, ascending ordinal: t0 unanswered (nobody sent key t0) with its c0; t2 an answered
    // leader whose partner's record carried rot. t4 sent rot=0 and received nothing: neither list. The
    // followers t1/t3/t5 never appear on the out side.
    BOOST_TEST(r[5].kind == "out_unanswered");
    BOOST_TEST(r[5].idx == 0U);
    BOOST_TEST(r[5].v == -1.0);
    BOOST_TEST(r[5].phase == 1);
    BOOST_TEST(r[6].kind == "out_pair");
    BOOST_TEST(r[6].idx == 2U);
    BOOST_TEST(r[6].phase == 1);

    // The mint landed as a row, and only it.
    BOOST_REQUIRE_EQUAL(sc.op.store->size(), 7U);
    BOOST_TEST((sc.op.store->row(6) == sc.absent_x));

    const auto &t = sc.scratch.marks;
    BOOST_TEST(!t.received(0));
    BOOST_TEST(t.received(1));
    BOOST_TEST(t.received(2));
    BOOST_TEST(t.received(3));
    BOOST_TEST(!t.received(4));
    BOOST_TEST(t.received(5));
    BOOST_TEST(t.partner_rot(1)); // t0's record
    BOOST_TEST(t.partner_rot(2)); // t3's record
    BOOST_TEST(!t.partner_rot(3));
    BOOST_TEST(!t.partner_rot(5));
}

// The absence pass alone, so both out lists are pinned with their order and payload: t0 is the unanswered
// E-record (with the sender-side c0), t2 the answered leader (rotating through its partner's rot).
BOOST_AUTO_TEST_CASE(one_round_absence_pass_reports_unanswered_and_answered_leaders) {
    Scenario sc;
    RecordingSink sink;
    auto scan = sc.scan(/*with_c0=*/true);
    detail::MissStage<kN> misses;
    sc.scratch.join.begin_queries(scan.self.size());
    for (size_t q = 0; q < scan.self.size(); ++q) {
        sc.scratch.join.add_query(q, scan.self.fp_of[q]);
    }
    sc.scratch.join.run(*sc.op.store, [&](size_t q) { return scan.self.positions_at(q); });
    detail::join_self<kN>(scan.self,
                          sc.scratch.join,
                          /*q_base=*/0,
                          sc.scratch.marks,
                          /*my_rank=*/0,
                          /*base=*/6,
                          misses,
                          sink);
    sink.recs.clear();
    detail::absence_pass<kN>(sc.scratch.marks, scan.sent, scan.sent_c0, sink);
    BOOST_REQUIRE_EQUAL(sink.recs.size(), 2U);
    BOOST_TEST(sink.recs[0].kind == "out_unanswered");
    BOOST_TEST(sink.recs[0].idx == 0U);
    BOOST_TEST(sink.recs[0].v == -1.0);
    BOOST_TEST(sink.recs[0].phase == 1);
    BOOST_TEST(sink.recs[1].kind == "out_pair");
    BOOST_TEST(sink.recs[1].idx == 2U);
    BOOST_TEST(sink.recs[1].phase == 1);
    BOOST_TEST(misses.size() == 1U);
}

// The fused sink turns the same joins into half-rotations: +φ_rec·v_rec on hits and mints (mints flagged as
// inserts), −φ_own·c0 on the unanswered E-record, nothing for the answered leader.
BOOST_AUTO_TEST_CASE(one_round_contract_sink_records_one_half_per_touched_slot) {
    Scenario sc;
    detail::FusedContract fc;
    detail::LayerBuildEngine<kN, detail::ContractSink<kN>>
        eng(sc.op, mpi::Comm{}, 1, 0, sc.scratch, 6, detail::ContractSink<kN>{.fc = fc, .fused_scale = true});
    eng.exchange_and_join(sc.scan(/*with_c0=*/true));
    BOOST_REQUIRE_EQUAL(fc.halves.size(), 6U);
    const auto expect = [&](size_t k, size_t idx, double v, int phase, bool insert) {
        BOOST_TEST_CONTEXT("half " << k) {
            BOOST_TEST(fc.halves[k].local_idx == idx);
            BOOST_TEST(fc.halves[k].v_partner == v);
            BOOST_TEST(fc.halves[k].phase_signed == phase);
            BOOST_TEST(fc.halves[k].is_insert == insert);
        }
    };
    expect(0, 1, 0.5, 1, false);
    expect(1, 6, 0.25, -1, true);
    expect(2, 3, 0.75, 1, false);
    expect(3, 2, 1.5, -1, false);
    expect(4, 5, 2.0, 1, false);
    expect(5, 0, -1.0, -1, false); // the absence: −φ_t0 · c0
    // Every slot is touched at most once, which is what lets the apply run in any order.
    std::vector<size_t> touched;
    for (const auto &h : fc.halves) {
        touched.push_back(h.local_idx);
    }
    std::ranges::sort(touched);
    BOOST_TEST((std::ranges::adjacent_find(touched) == touched.end()));
}

// The graph sink's layout: in = [in_pairs (hits on followers), in_mints], out = [out_pairs (answered
// leaders), out_unanswered], and finalize's sin_send = [in…, out…] / sin_recv = [(out, −φ)…, (in, +φ)…].
BOOST_AUTO_TEST_CASE(one_round_graph_sink_lays_out_in_and_out_blocks) {
    Scenario sc;
    detail::LayerBuildEngine<kN, detail::GraphSink<kN>>
        eng(sc.op, mpi::Comm{}, 1, 0, sc.scratch, 6, detail::GraphSink<kN>{1, 0});
    eng.exchange_and_join(sc.scan(/*with_c0=*/false));
    const auto &a = eng.sink.acc[0];
    const auto entries = [](const std::vector<detail::PhasedEntry> &v) {
        std::vector<std::pair<size_t, int>> out;
        for (const auto &e : v) {
            out.emplace_back(e.idx, e.phase);
        }
        return out;
    };
    using P = std::vector<std::pair<size_t, int>>;
    BOOST_TEST((entries(a.in_pairs) == P{{1, 1}, {3, 1}, {5, 1}}));
    BOOST_TEST((entries(a.in_mints) == P{{6, -1}}));
    BOOST_TEST((entries(a.out_pairs) == P{{2, 1}}));
    BOOST_TEST((entries(a.out_unanswered) == P{{0, 1}}));

    CosMask cos;
    const auto core = eng.finish(CosMask{}, &cos);
    BOOST_REQUIRE(core != nullptr);
    const auto slot = detail::cross_rank_slot(core->cross_rank, 0);
    BOOST_REQUIRE_EQUAL(slot.sin_send_count, 6U);
    BOOST_TEST(slot.in_count == 4U);
    const std::vector<size_t> b = {1, 3, 5, 6, 2, 0};
    for (size_t k = 0; k < 6; ++k) {
        BOOST_TEST(detail::slot_sin_send_index(slot, k) == b[k]);
    }
    const std::vector<std::pair<size_t, int>> d = {{2, -1}, {0, -1}, {1, 1}, {3, 1}, {5, 1}, {6, -1}};
    for (size_t k = 0; k < 6; ++k) {
        BOOST_TEST_CONTEXT("sin_recv " << k) {
            BOOST_TEST(detail::slot_sin_recv_index(slot, k) == d[k].first);
            BOOST_TEST(detail::slot_sin_recv_phase(slot, k) == d[k].second);
        }
    }
    // The mint is a rotation endpoint born after the scan, so the cos mask covers it.
    BOOST_TEST(cos.total_count == 1U);
}

namespace {

template <size_t N>
auto majorana_anticommutes(const Monomial<N> &m, const Monomial<N> &g) -> bool {
    return ((m.count() * g.count()) - m.count_and(g)) % 2 == 1;
}

template <size_t N, Algebra A>
auto emit_phase_of(const Monomial<N> &m, const Monomial<N> &g) -> int {
    const auto ctx = A::make_gen_context(g);
    const Monomial<N> partner = m ^ g;
    return A::emit_phase(A::rotation_sign(ctx, m, partner), m.count(), g.count(), m.count_and(g));
}

template <size_t N>
auto random_monomial(std::mt19937_64 &rng, size_t k) -> Monomial<N> {
    Monomial<N> m;
    std::uniform_int_distribution<size_t> bit(0, Monomial<N>::size() - 1);
    while (m.count() < k) {
        m.set(bit(rng));
    }
    return m;
}

} // namespace

// The identity the one-round protocol's exactness rests on: the two endpoints of a pair emit opposite
// phases, φ(M⊕G, G) = −φ(M, G), in both algebras. Each side then applies the OTHER side's record with its
// phase as sent, and gets exactly today's two adds.
BOOST_AUTO_TEST_CASE(emit_phase_antisymmetry) {
    constexpr size_t N = 24;
    std::mt19937_64 rng(20260902);
    size_t majorana = 0;
    size_t pauli = 0;
    for (size_t trial = 0; trial < 6000; ++trial) {
        const auto m = random_monomial<N>(rng, 1 + (rng() % 8));
        const auto g = random_monomial<N>(rng, 1 + (rng() % 6));
        if (majorana_anticommutes<N>(m, g)) {
            const int fwd = emit_phase_of<N, MajoranaAlgebra<N>>(m, g);
            const int back = emit_phase_of<N, MajoranaAlgebra<N>>(m ^ g, g);
            BOOST_REQUIRE_EQUAL(back, -fwd);
            ++majorana;
        }
        if (pauli_anticommutes<N>(m, g)) {
            const int fwd = emit_phase_of<N, PauliAlgebra<N>>(m, g);
            const int back = emit_phase_of<N, PauliAlgebra<N>>(m ^ g, g);
            BOOST_REQUIRE_EQUAL(back, -fwd);
            ++pauli;
        }
    }
    BOOST_TEST(majorana > 1000U);
    BOOST_TEST(pauli > 1000U);
}

/* ── The receiver rule end to end, on two-term Majorana operators ─────────────────────────────────── */

namespace {

constexpr size_t kM = 4;

// A Majorana term with the given ENCODED (real) coefficient: the dict holds coeff·i^C(k,2).
auto term(OperatorDict &dict, const VecZ &idx, double encoded) -> void {
    dict[idx] = hermitian_coefficient<kM>(indices_to_bitset<kM>(idx)) * encoded;
}

struct Knobs {
    std::optional<double> lower_atol = std::nullopt;
    std::optional<double> upper_atol = std::nullopt;
    unsigned int cutoff = 2 * kM;
    std::optional<unsigned int> schrodinger = std::nullopt;
    VecZ initial_state = {};
};

auto make(const OperatorDict &dict, const Knobs &k) -> MonomialPropagator<kM> {
    return MonomialPropagator<kM>(dict,
                                  k.cutoff,
                                  k.initial_state,
                                  k.schrodinger,
                                  MPI_COMM_SELF,
                                  k.lower_atol,
                                  k.upper_atol,
                                  CutoffType::Length,
                                  std::nullopt);
}

// The evolved coefficient of `idx` (nullopt when the term is not tracked), from the live picture vector.
auto coeff_of(MonomialPropagator<kM> &sim, const VecZ &idx) -> std::optional<double> {
    const auto want = indices_to_bitset<kM>(idx);
    const VecD &c = sim.mp_op().get_operator();
    std::optional<double> out;
    sim.indexing().for_each([&](const Monomial<kM> &mono, size_t i) {
        if (mono == want && i < c.size()) {
            out = c[i];
        }
    });
    return out;
}

// One gate G at angle θ over `dict`, with the knobs; returns the propagator for inspection.
auto run_gate(const OperatorDict &dict,
              const VecZ &gate,
              double theta,
              const Knobs &k,
              std::optional<size_t> rot_k = {}) -> MonomialPropagator<kM> {
    auto sim = make(dict, k);
    sim.propagate({gate}, VecZ{0}, VecD{1.0}, VecD{theta}, rot_k);
    return sim;
}

const VecZ kG = {0, 1};  // |G| = 2
const VecZ kNu = {0, 2}; // anticommutes with G (2·2 − 1 odd); partner μ = {1, 2}
const VecZ kMu = {1, 2};
constexpr double kTheta = 0.37;

} // namespace

// Both endpoints below lower_atol: the pair does not rotate, so each coefficient equals its solo
// evolution (cos-scaled only; a solo term's absent partner contributes nothing).
BOOST_AUTO_TEST_CASE(receiver_rule_both_below_atol_do_not_rotate) {
    const Knobs k{.lower_atol = 1e-6};
    OperatorDict both;
    term(both, kNu, 1e-12);
    term(both, kMu, -3e-12);
    OperatorDict solo_nu;
    term(solo_nu, kNu, 1e-12);
    OperatorDict solo_mu;
    term(solo_mu, kMu, -3e-12);
    auto pair = run_gate(both, kG, kTheta, k);
    auto nu = run_gate(solo_nu, kG, kTheta, k);
    auto mu = run_gate(solo_mu, kG, kTheta, k);
    BOOST_TEST(pair.size() == 2U);
    BOOST_TEST(*coeff_of(pair, kNu) == *coeff_of(nu, kNu));
    BOOST_TEST(*coeff_of(pair, kMu) == *coeff_of(mu, kMu));
    // And the solo runs minted nothing: below the threshold a term does not emit.
    BOOST_TEST(nu.size() == 1U);
    BOOST_TEST(mu.size() == 1U);
}

// One endpoint below lower_atol: the other's emission rotates the pair, and BOTH adds happen with the
// same values as with the threshold off -- the below-threshold side's silent record is what makes that
// bit-identical.
BOOST_AUTO_TEST_CASE(receiver_rule_one_below_atol_still_rotates_both) {
    OperatorDict both;
    term(both, kNu, 1.0);
    term(both, kMu, 1e-12);
    auto gated = run_gate(both, kG, kTheta, Knobs{.lower_atol = 1e-6});
    auto exact = run_gate(both, kG, kTheta, Knobs{});
    BOOST_TEST(gated.size() == 2U);
    BOOST_TEST(*coeff_of(gated, kNu) == *coeff_of(exact, kNu));
    BOOST_TEST(*coeff_of(gated, kMu) == *coeff_of(exact, kMu));
    // The rotation really happened: the small side moved by sin·|c_ν| ≫ 1e-12.
    BOOST_TEST(std::abs(*coeff_of(gated, kMu)) > 0.1);
}

// A source over the rotation length cap does not emit and is not cos-scaled, but its in-cap partner's
// record still rotates it: c'_μ = c_μ + sin·φ·c_ν against c'_μ = cos·c_μ + sin·φ·c_ν uncapped, while the
// in-cap side is bit-identical either way.
BOOST_AUTO_TEST_CASE(receiver_rule_capped_source_is_rotated_by_its_partner) {
    const VecZ g = {0, 1, 2};     // odd |G|: anticommutes with disjoint odd-weight terms
    const VecZ nu = {3};          // pop 1: in cap
    const VecZ mu = {0, 1, 2, 3}; // pop 4: over cap 3
    OperatorDict both;
    term(both, nu, 0.8);
    term(both, mu, -0.6);
    auto capped = run_gate(both, g, kTheta, Knobs{}, /*rot_k=*/3);
    auto uncapped = run_gate(both, g, kTheta, Knobs{});
    // The same two operations on ν either way (cos·c_ν, then + sin·φ_μ·c_μ); the two apply paths (fused cos
    // sweep vs the two-pass cos mask) are separate loops the compiler may contract differently, so this is
    // a 1-ULP check rather than a bit-identity one.
    BOOST_TEST(std::abs(*coeff_of(capped, nu) - *coeff_of(uncapped, nu))
               <= std::numeric_limits<double>::epsilon() * std::abs(*coeff_of(uncapped, nu)));
    const double cos2 = std::cos(2 * kTheta);
    // Both runs added the same sine term to μ; only the cos scale differs.
    BOOST_TEST(*coeff_of(capped, mu) - (-0.6) == *coeff_of(uncapped, mu) - (cos2 * -0.6),
               boost::test_tools::tolerance(1e-15));
    BOOST_TEST(std::abs(*coeff_of(capped, mu) - (-0.6)) > 0.1); // it did rotate
}

// upper_atol rescues a partner the structural cutoff rejects: the partner is minted and the source's
// value is what the wide-cutoff run gives.
BOOST_AUTO_TEST_CASE(receiver_rule_rescued_partner_is_minted) {
    const VecZ g = {0, 1, 2};
    const VecZ nu = {4};          // pop 1 passes cutoff 2
    const VecZ mu = {0, 1, 2, 4}; // pop 4 fails cutoff 2 and is not paired
    OperatorDict one;
    term(one, nu, 0.9);
    auto rescued = run_gate(one, g, kTheta, Knobs{.upper_atol = 0.0, .cutoff = 2});
    auto dropped = run_gate(one, g, kTheta, Knobs{.cutoff = 2});
    auto wide = run_gate(one, g, kTheta, Knobs{});
    BOOST_TEST(rescued.size() == 2U);
    BOOST_TEST(dropped.size() == 1U);
    BOOST_TEST(*coeff_of(rescued, mu) == *coeff_of(wide, mu));
    BOOST_TEST(*coeff_of(rescued, nu) == *coeff_of(wide, nu));
    BOOST_TEST(*coeff_of(dropped, nu) == std::cos(2 * kTheta) * 0.9);
}

// An initial term over the structural cutoff (the initial operator is never filtered) whose partner passes:
// only the partner emits, and the pair must still rotate on both sides. This is what the per-call
// over_cutoff_possible flag exists for -- without it the over-cutoff side would never send, and its
// partner would see it as absent.
BOOST_AUTO_TEST_CASE(receiver_rule_initial_over_cutoff_term_rotates_with_its_partner) {
    const VecZ g = {0, 1, 2};
    const VecZ nu = {4};          // pop 1: passes cutoff 1
    const VecZ mu = {0, 1, 2, 4}; // pop 4: tracked (initial) but over cutoff 1
    OperatorDict both;
    term(both, nu, 0.7);
    term(both, mu, 0.4);
    auto tight = run_gate(both, g, kTheta, Knobs{.cutoff = 1});
    auto wide = run_gate(both, g, kTheta, Knobs{});
    BOOST_TEST(tight.size() == 2U);
    BOOST_TEST(*coeff_of(tight, nu) == *coeff_of(wide, nu));
    BOOST_TEST(*coeff_of(tight, mu) == *coeff_of(wide, mu));
    BOOST_TEST(std::abs(*coeff_of(tight, mu) - std::cos(2 * kTheta) * 0.4) > 0.1); // rotated, not just scaled

    // With cutoff 0 neither side passes and nothing is rescued: the pair only cos-scales.
    auto zero = run_gate(both, g, kTheta, Knobs{.cutoff = 0});
    BOOST_TEST(*coeff_of(zero, nu) == std::cos(2 * kTheta) * 0.7);
    BOOST_TEST(*coeff_of(zero, mu) == std::cos(2 * kTheta) * 0.4);
}

// Schrödinger: the sender computes c0(μ), the coefficient an absent paired partner would be minted with,
// off the partner's positions -- the state score for a fully paired μ, 0 otherwise -- and the scan carries
// it parallel to the sent ordinals so the absence pass can supply the source's half without the partner.
BOOST_AUTO_TEST_CASE(scan_c0_is_the_state_score_of_a_paired_absent_partner) {
    using A = MajoranaAlgebra<kM>;
    // G = {1,3,4,5}: ν1={1,2} → μ1={2,3,4,5} paired, ν2={3} → μ2={1,4,5} unpaired, ν3={0,3} → μ3={0,1,4,5}
    // paired. Mode 1 occupied ⇒ mask bit 2 ⇒ c0(μ1) = (−1)^(1+2) = −1, c0(μ3) = (−1)^(0+2) = +1.
    const auto gen = indices_to_bitset<kM>({1, 3, 4, 5});
    const std::vector<Monomial<kM>> rows = {indices_to_bitset<kM>({1, 2}),
                                            indices_to_bitset<kM>({3}),
                                            indices_to_bitset<kM>({0, 3})};
    detail::MPOperator<kM> op;
    detail::insert_absent_terms<kM>(op, rows.size(), [&](size_t k, size_t base) {
        assign_row<kM>(*op.store, base + k, rows[k]);
    });
    const VecD coeffs(rows.size(), 1.0);
    const CutoffFn<kM> fn = detail::LengthCutoff<kM>{2 * kM, kM};
    const detail::CutoffEvaluator<kM> eval(fn);
    const auto cut = detail::build_majorana_evolution_cutoff_state(std::nullopt, std::cref(coeffs), std::nullopt, 0.3);
    const auto router = routing::Router::splitmix(1);
    const mpi::SlotWindow window{.base = 0, .count = 1};
    const auto mask = initial_state_mask<kM>(VecZ{1});
    detail::GateScratch<kM> scratch;
    const auto res = detail::fused_find_and_collect<kM, A>(op,
                                                           gen,
                                                           eval,
                                                           cut,
                                                           coeffs,
                                                           std::nullopt,
                                                           /*over_cutoff_possible=*/false,
                                                           window,
                                                           /*my_rank=*/0,
                                                           router,
                                                           scratch,
                                                           /*capture_values=*/true,
                                                           nullptr,
                                                           1.0,
                                                           &mask);
    BOOST_REQUIRE_EQUAL(scratch.join.rows(), 3U);
    BOOST_REQUIRE_EQUAL(res.self.size(), 3U); // R = 1: every record is self-addressed and staged
    const auto &sent = res.sent.at_slot(0);
    const auto &c0 = res.sent_c0.at_slot(0);
    BOOST_REQUIRE_EQUAL(sent.size(), 3U);
    BOOST_REQUIRE_EQUAL(c0.size(), 3U);
    for (size_t j = 0; j < 3; ++j) {
        BOOST_TEST(sent[j].row == static_cast<TermIndex>(j));
    }
    BOOST_TEST(c0[0] == -1.0);
    BOOST_TEST(c0[1] == 0.0);
    BOOST_TEST(c0[2] == 1.0);
    // The rot bit rides with the record: all three sources are above threshold with a structurally
    // admitted partner, so all three rotate.
    for (size_t q = 0; q < 3; ++q) {
        BOOST_TEST(res.self.rot_of[q] == 1);
        BOOST_TEST(res.self.val_of[q] == 1.0);
        BOOST_TEST(scratch.marks.rot(q));
    }
    // Without a state mask (Heisenberg) no c0 is carried at all.
    detail::GateScratch<kM> scratch_h;
    const auto heis = detail::fused_find_and_collect<kM, A>(op,
                                                            gen,
                                                            eval,
                                                            cut,
                                                            coeffs,
                                                            std::nullopt,
                                                            false,
                                                            window,
                                                            0,
                                                            router,
                                                            scratch_h,
                                                            true,
                                                            nullptr,
                                                            1.0,
                                                            nullptr);
    BOOST_TEST(heis.sent_c0.size() == 0U);
    BOOST_TEST(heis.sent.at_slot(0).size() == 3U);
}

// The position-form state score agrees with the dense one in both algebras.
BOOST_AUTO_TEST_CASE(state_phase_positions_matches_state_phase) {
    constexpr size_t N = 24;
    std::mt19937_64 rng(20260903);
    for (size_t trial = 0; trial < 400; ++trial) {
        VecZ occupied;
        for (size_t m = 0; m < N; ++m) {
            if ((rng() & 1U) != 0U) {
                occupied.push_back(m);
            }
        }
        const auto mask = initial_state_mask<N>(occupied);
        Monomial<N> paired;
        for (size_t m = 0; m < N; ++m) {
            if ((rng() & 1U) != 0U) {
                paired.set(2 * m);
                paired.set((2 * m) + 1);
            }
        }
        std::vector<uint16_t> pos;
        for (size_t b = paired.find_first(); b < paired.size(); b = paired.find_next(b)) {
            pos.push_back(static_cast<uint16_t>(b));
        }
        BOOST_TEST(MajoranaAlgebra<N>::state_phase_positions(pos.data(), pos.size(), mask)
                   == MajoranaAlgebra<N>::state_phase(paired, mask));
        const auto z_string = random_monomial<N>(rng, 1 + (rng() % 10));
        std::vector<uint16_t> zpos;
        for (size_t b = z_string.find_first(); b < z_string.size(); b = z_string.find_next(b)) {
            zpos.push_back(static_cast<uint16_t>(b));
        }
        BOOST_TEST(PauliAlgebra<N>::state_phase_positions(zpos.data(), zpos.size(), mask)
                   == PauliAlgebra<N>::state_phase(z_string, mask));
    }
}

BOOST_AUTO_TEST_CASE(cutoff_context_abs_coeff_for) {
    const VecD coeffs{-3.0, 2.0, 0.0};

    CutoffContext off; // use_coeff_checks defaults false
    BOOST_TEST(off.abs_coeff_for(0, coeffs) == 0.0);

    CutoffContext on;
    on.use_coeff_checks = true;
    BOOST_TEST(on.abs_coeff_for(0, coeffs) == 3.0);
    BOOST_TEST(on.abs_coeff_for(1, coeffs) == 2.0);
    BOOST_TEST(on.abs_coeff_for(3, coeffs) == 0.0); // out of range -> 0
}

// is_above_upper is the rescue predicate: enabled AND |sin|·|coeff| >= upper_atol (inclusive).
BOOST_AUTO_TEST_CASE(cutoff_context_is_above_upper) {
    CutoffContext ctx;
    ctx.abs_sin_val = 0.5;

    ctx.check_upper_atol = false;
    BOOST_TEST(!ctx.is_above_upper(100.0));

    ctx.check_upper_atol = true;
    ctx.upper_atol_value = 1.0;
    BOOST_TEST(ctx.is_above_upper(2.0)); // 0.5*2.0 == 1.0 -> boundary inclusive
    BOOST_TEST(ctx.is_above_upper(4.0));
    BOOST_TEST(!ctx.is_above_upper(1.0));
}

// is_below_sin is the lower-atol drop predicate: enabled AND |sin|·|coeff| <= atol (inclusive).
BOOST_AUTO_TEST_CASE(cutoff_context_is_below_sin) {
    CutoffContext ctx;
    ctx.abs_sin_val = 2.0;

    ctx.check_atol = false;
    BOOST_TEST(!ctx.is_below_sin(0.0));

    ctx.check_atol = true;
    ctx.atol_value = 1.0;
    BOOST_TEST(ctx.is_below_sin(0.5)); // 2.0*0.5 == 1.0 -> boundary inclusive
    BOOST_TEST(ctx.is_below_sin(0.1));
    BOOST_TEST(!ctx.is_below_sin(1.0));
}
