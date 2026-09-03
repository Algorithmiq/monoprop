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

// Graph mode's positional pairing under the one-round exchange, over an in-process ShmComm world: for
// every ordered pair of slots (p, q) the out-part p recorded for q has the length of the in-part q
// recorded for p (that is the whole contract replay's index-free sin exchange rests on), and replaying the
// graph reproduces the fused propagate on the same operator within a few ULP, term for term.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "monoprop/Evolution.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/graph/MPGraphLayers.h"
#include "monoprop/detail/graph_encoding/MPGraphEncodingStorage.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/ShmComm.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/operator/RowAccess.h"

#include "ThreadHarness.h"

using namespace monoprop;

namespace {

constexpr size_t kN = 8;

struct Problem {
    std::vector<Monomial<kN>> terms;
    std::vector<double> coeffs; // parallel to terms
    std::vector<Monomial<kN>> gates;
    std::vector<double> angles;
};

auto random_monomial(std::mt19937_64 &rng, size_t k) -> Monomial<kN> {
    Monomial<kN> m;
    std::uniform_int_distribution<size_t> bit(0, (2 * kN) - 1);
    while (m.count() < k) {
        m.set(bit(rng));
    }
    return m;
}

// Weights 1..6 keep every initial term inside the structural cutoff below (so over_cutoff_possible is
// false, as the propagator would compute it); coefficients span six decades so lower_atol splits the
// pairs into rotating, silent and mixed ones.
auto make_problem(uint64_t seed, size_t n_terms, size_t n_gates) -> Problem {
    std::mt19937_64 rng(seed);
    Problem p;
    std::uniform_real_distribution<double> mag(-6.0, 0.0);
    std::uniform_real_distribution<double> ang(0.1, 1.2);
    while (p.terms.size() < n_terms) {
        const auto m = random_monomial(rng, 1 + (rng() % 6));
        bool dup = false;
        for (const auto &t : p.terms) {
            dup |= (t == m);
        }
        if (!dup) {
            p.terms.push_back(m);
            p.coeffs.push_back(((rng() & 1U) != 0U ? 1.0 : -1.0) * std::pow(10.0, mag(rng)));
        }
    }
    for (size_t g = 0; g < n_gates; ++g) {
        p.gates.push_back(random_monomial(rng, 1 + (rng() % 4)));
        p.angles.push_back(ang(rng));
    }
    return p;
}

// ULP distance of two doubles with the same sign; a sign change or a NaN is "infinite".
auto ulp_distance(double a, double b) -> uint64_t {
    if (a == b) {
        return 0;
    }
    if (std::isnan(a) || std::isnan(b) || std::signbit(a) != std::signbit(b)) {
        return UINT64_MAX;
    }
    const auto ia = std::bit_cast<int64_t>(a);
    const auto ib = std::bit_cast<int64_t>(b);
    return static_cast<uint64_t>(ia > ib ? ia - ib : ib - ia);
}

struct SlotResult {
    std::vector<std::shared_ptr<LayerCore>> cores; // one per gate
    std::vector<Monomial<kN>> rows_fused;
    std::vector<Monomial<kN>> rows_graph;
    VecD coeffs_fused;
    VecD coeffs_graph;
    // Per gate, before the arms are re-synchronised (so each gate is compared on identical inputs): the
    // worst ULP distance between the two arms' coefficients, and the worst deviation in units of the
    // one-product rounding bound (see run_slot). `compared` counts rows.
    std::vector<uint64_t> worst_ulp;
    std::vector<double> worst_bound_ratio;
    size_t compared = 0;
    bool rows_agree = true;
};

auto rows_of(const detail::MPOperator<kN> &op) -> std::vector<Monomial<kN>> {
    std::vector<Monomial<kN>> out;
    for (size_t i = 0; i < op.store->size(); ++i) {
        out.push_back(op.store->row(i));
    }
    return out;
}

// One slot's run of both arms over the same gates. Each arm owns its operator: the two must grow the
// same rows in the same order, which the comparison in the caller checks.
auto run_slot(mpi::Comm comm, size_t my_rank, const Problem &p, std::optional<double> lower_atol) -> SlotResult {
    const auto router = router_for<kN>(comm);
    std::vector<Monomial<kN>> owned;
    VecD owned_c;
    for (size_t i = 0; i < p.terms.size(); ++i) {
        if (find_rank<kN>(p.terms[i], router) == my_rank) {
            owned.push_back(p.terms[i]);
            owned_c.push_back(p.coeffs[i]);
        }
    }
    auto seed = [&](detail::MPOperator<kN> &op) {
        op.basis = Basis::Majorana;
        detail::insert_absent_terms<kN>(op, owned.size(), [&](size_t k, size_t base) {
            assign_row<kN>(*op.store, base + k, owned[k]);
        });
    };
    detail::MPOperator<kN> op_f;
    detail::MPOperator<kN> op_g;
    seed(op_f);
    seed(op_g);
    const CutoffFn<kN> cutoff_fn = detail::LengthCutoff<kN>{6, kN};

    SlotResult res;
    res.coeffs_fused = owned_c;
    res.coeffs_graph = owned_c;
    detail::GateScratch<kN> scratch_f;
    detail::GateScratch<kN> scratch_g;
    for (size_t g = 0; g < p.gates.size(); ++g) {
        const double theta = p.angles[g];
        const VecD pre = res.coeffs_graph; // both arms start the gate from these values
        // Fused arm, as evolve_mode_contract_immediately_ drives it.
        {
            CosMask cos;
            detail::FusedContract fc;
            bool fused_scale = false;
            detail::build_layer<kN>(op_f,
                                    p.gates[g],
                                    cutoff_fn,
                                    lower_atol,
                                    detail::CoeffSpan(res.coeffs_fused),
                                    std::nullopt,
                                    theta,
                                    std::nullopt,
                                    /*over_cutoff_possible=*/false,
                                    scratch_f,
                                    comm,
                                    &cos,
                                    &fc,
                                    /*schrodinger=*/false,
                                    detail::MutCoeffSpan(res.coeffs_fused),
                                    &fused_scale);
            res.coeffs_fused.resize(op_f.store->size(), 0.0);
            detail::apply_fused_contract(fc, detail::MutCoeffSpan(res.coeffs_fused), cos, theta, fused_scale);
        }
        // Graph arm, as evolve_mode_graph_with_coeffs_ drives it: build, extend, replay.
        {
            auto cos = std::make_shared<CosMask>();
            auto core = detail::build_layer<kN>(op_g,
                                                p.gates[g],
                                                cutoff_fn,
                                                lower_atol,
                                                detail::CoeffSpan(res.coeffs_graph),
                                                std::nullopt,
                                                theta,
                                                std::nullopt,
                                                /*over_cutoff_possible=*/false,
                                                scratch_g,
                                                comm,
                                                cos.get());
            if (core == nullptr) {
                throw std::runtime_error("graph build returned no layer");
            }
            res.coeffs_graph.resize(op_g.store->size(), 0.0);
            Layer layer(core);
            const detail::LayerCosScale cos_scale = [cos](size_t, double *c, double v) {
                detail::scale_cos_mask(c, *cos, v);
            };
            evolve_step(res.coeffs_graph, layer, theta, comm, cos_scale);
            res.cores.push_back(std::move(core));
        }
        // Compare this gate on its own, then hand the graph arm's values to the fused arm: the two arms
        // round differently (the fused mint folds cos and sin into one expression, the replay scales then
        // adds), and letting a 1-ULP difference feed 13 more gates of cancellation measures accumulation,
        // not equivalence.
        if (op_f.store->size() != op_g.store->size()) {
            res.rows_agree = false;
            throw std::runtime_error("the two arms grew different row counts");
        }
        for (size_t i = 0; i < op_f.store->size(); ++i) {
            if (!(op_f.store->row(i) == op_g.store->row(i))) {
                res.rows_agree = false;
                throw std::runtime_error("the two arms grew different rows");
            }
        }
        // Both arms compute cos·c then c += sin·φ·v with the same operands; the compiler is free to
        // contract the product into the add in one loop and not the other, so the two results may differ
        // by one rounding of the product p = sin·φ·v, i.e. by eps·|p| ≤ eps·(|c_before| + |c_after|).
        // Where c_before and p cancel that is many ULP of the result, so the ULP distance alone is not the
        // criterion: a row passes if it is within 4 ULP or within 4 such roundings.
        uint64_t worst = 0;
        double worst_ratio = 0.0;
        for (size_t i = 0; i < res.coeffs_fused.size(); ++i) {
            const double a = res.coeffs_fused[i];
            const double b = res.coeffs_graph[i];
            const double before = i < pre.size() ? pre[i] : 0.0;
            const double bound =
                std::numeric_limits<double>::epsilon() * (std::abs(before) + std::max(std::abs(a), std::abs(b)));
            worst = std::max(worst, ulp_distance(a, b));
            worst_ratio = std::max(worst_ratio, bound > 0.0 ? std::abs(a - b) / bound : (a == b ? 0.0 : 1e300));
            ++res.compared;
        }
        res.worst_ulp.push_back(worst);
        res.worst_bound_ratio.push_back(worst_ratio);
        res.coeffs_fused = res.coeffs_graph;
    }
    res.rows_fused = rows_of(op_f);
    res.rows_graph = rows_of(op_g);
    return res;
}

auto check_world(size_t S, uint64_t seed, std::optional<double> lower_atol) -> void {
    const Problem p = make_problem(seed, /*n_terms=*/400, /*n_gates=*/14);
    mpi::ShmComm sh(static_cast<int>(S));
    std::vector<SlotResult> results(S);
    const auto errs = test_utils::run_comm_threads(sh, static_cast<int>(S), [&](mpi::ShmComm &world, int r) {
        results[static_cast<size_t>(r)] =
            run_slot(mpi::Comm::make_shm(&world, r), static_cast<size_t>(r), p, lower_atol);
    });
    for (const auto &e : errs) {
        if (e) {
            std::rethrow_exception(e);
        }
    }

    // (1) The pairing contract: my out-part for q is exactly as long as q's in-part for me, every gate.
    size_t cross_pairs = 0;
    size_t self_pairs = 0;
    for (size_t g = 0; g < p.gates.size(); ++g) {
        for (size_t a = 0; a < S; ++a) {
            for (size_t b = 0; b < S; ++b) {
                const auto ab = detail::cross_rank_slot(results[a].cores[g]->cross_rank, b);
                const auto ba = detail::cross_rank_slot(results[b].cores[g]->cross_rank, a);
                BOOST_REQUIRE_GE(ab.sin_send_count, ab.in_count);
                const size_t out_ab = ab.sin_send_count - ab.in_count;
                BOOST_TEST_CONTEXT("gate " << g << " slots " << a << "->" << b) {
                    BOOST_TEST(out_ab == ba.in_count);
                }
                (a == b ? self_pairs : cross_pairs) += ba.in_count;
            }
        }
    }
    BOOST_TEST(cross_pairs > 0);
    BOOST_TEST(self_pairs > 0);

    // (2) Both arms grew the same rows in the same order, and per gate the replay reproduces the fused
    // values to within a few ULP on every row.
    uint64_t worst = 0;
    double worst_ratio = 0.0;
    size_t compared = 0;
    for (size_t r = 0; r < S; ++r) {
        const auto &res = results[r];
        BOOST_REQUIRE(res.rows_agree);
        BOOST_REQUIRE_EQUAL(res.rows_fused.size(), res.rows_graph.size());
        BOOST_REQUIRE_EQUAL(res.worst_ulp.size(), p.gates.size());
        for (size_t g = 0; g < p.gates.size(); ++g) {
            BOOST_TEST_CONTEXT("slot " << r << " gate " << g << " worst_ulp " << res.worst_ulp[g]
                                       << " worst_bound_ratio " << res.worst_bound_ratio[g]) {
                BOOST_TEST((res.worst_ulp[g] <= 4U || res.worst_bound_ratio[g] <= 4.0));
            }
            worst = std::max(worst, res.worst_ulp[g]);
            worst_ratio = std::max(worst_ratio, res.worst_bound_ratio[g]);
        }
        compared += res.compared;
    }
    BOOST_TEST_MESSAGE("S=" << S << " atol=" << lower_atol.value_or(0.0) << " compared=" << compared
                            << " worst_ulp=" << worst << " worst_bound_ratio=" << worst_ratio
                            << " cross_pairs=" << cross_pairs << " self_pairs=" << self_pairs);
    BOOST_TEST(compared > p.terms.size());
}

} // namespace

BOOST_AUTO_TEST_CASE(graph_pair_order_two_slots_exact) {
    check_world(2, 0x0A11CE, std::nullopt);
}

BOOST_AUTO_TEST_CASE(graph_pair_order_two_slots_with_lower_atol) {
    check_world(2, 0x0B0B, 1e-4);
}

BOOST_AUTO_TEST_CASE(graph_pair_order_four_slots_exact) {
    check_world(4, 0xC0FFEE, std::nullopt);
}

BOOST_AUTO_TEST_CASE(graph_pair_order_four_slots_with_lower_atol) {
    check_world(4, 0xD00D, 1e-3);
}
