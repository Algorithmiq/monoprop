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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;

namespace {

constexpr size_t kModes = 2;

auto make_sim() -> MonomialPropagator<kModes> {
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    VecZ initial_state{0, 1};
    return MonomialPropagator<kModes>(ham,
                                      2 * kModes,
                                      initial_state,
                                      std::nullopt,
                                      MPI_COMM_SELF,
                                      std::nullopt,
                                      std::nullopt,
                                      CutoffType::Length,
                                      std::nullopt);
}

} // namespace

BOOST_AUTO_TEST_CASE(n_gates_defaults_to_one_per_generator) {
    auto sim = make_sim();
    const std::vector<VecZ> monos{{0}, {1}, {2}};
    sim.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0});
    BOOST_TEST(sim.graph_layers() == 3u);
    BOOST_TEST(sim.n_gates() == sim.graph_layers());
}

BOOST_AUTO_TEST_CASE(gate_indices_group_layers) {
    auto sim = make_sim();
    const std::vector<VecZ> monos{{0}, {1}, {2}};
    // Two monomials belong to gate 0, one to gate 1.
    sim.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0}, VecZ{0, 0, 1});
    BOOST_TEST(sim.graph_layers() == 3u);
    BOOST_TEST(sim.n_gates() == 2u);
}

BOOST_AUTO_TEST_CASE(set_parameter_mapping_per_gate_ties_layers) {
    auto sim = make_sim();
    const std::vector<VecZ> monos{{0}, {1}, {2}};
    sim.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0}, VecZ{0, 0, 1});

    // Length n_gates (2) selects the per-gate branch; both gates tied to one angle.
    sim.set_parameter_mapping(VecZ{0, 0});
    auto per_layer = sim.parameter_mapping();
    BOOST_TEST(per_layer.size() == 3u);
    BOOST_TEST(std::ranges::all_of(per_layer, [](size_t p) { return p == 0; }));

    // Distinct angles: gate 0's two layers share one param, gate 1's layer gets the other -- counted
    // rather than compared positionally, so the check holds regardless of layer ordering.
    sim.set_parameter_mapping(VecZ{0, 1});
    per_layer = sim.parameter_mapping();
    const auto zeros = std::ranges::count(per_layer, 0u);
    const auto ones = std::ranges::count(per_layer, 1u);
    BOOST_TEST(zeros == 2);
    BOOST_TEST(ones == 1);
}

BOOST_AUTO_TEST_CASE(set_parameter_mapping_per_layer_still_works) {
    auto sim = make_sim();
    const std::vector<VecZ> monos{{0}, {1}, {2}};
    sim.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0}, VecZ{0, 0, 1});

    // Length graph_layers() -> per-layer branch.
    sim.set_parameter_mapping(VecZ{0, 0, 0});
    BOOST_TEST(sim.parameter_mapping().size() == 3u);
    BOOST_TEST(std::ranges::all_of(sim.parameter_mapping(), [](size_t p) { return p == 0; }));
}

// relabel copies each LayerCore, whose lazily-built derivative exchange layout is eval-time cache, not
// data: a mapping set after a gradient must behave exactly like one set before it.
BOOST_AUTO_TEST_CASE(set_parameter_mapping_after_gradient_matches_before) {
    const std::vector<VecZ> monos{{0}, {1}, {2}};
    const VecD params{0.3, 0.4};

    auto before = make_sim();
    before.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0}, VecZ{0, 0, 1});
    before.set_parameter_mapping(VecZ{0, 1});
    const auto [value_before, grad_before] = before.expectation_value_and_gradient(params);

    auto after = make_sim();
    after.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0}, VecZ{0, 0, 1});
    // Materialize the derivative layout first, then relabel: the copied cores must not inherit it.
    static_cast<void>(after.expectation_value_and_gradient(VecD{0.1, 0.2, 0.5}));
    after.set_parameter_mapping(VecZ{0, 1});
    const auto [value_after, grad_after] = after.expectation_value_and_gradient(params);

    BOOST_TEST(value_after == value_before, boost::test_tools::tolerance(1e-12));
    BOOST_REQUIRE_EQUAL(grad_after.size(), grad_before.size());
    for (size_t i = 0; i < grad_before.size(); ++i) {
        BOOST_TEST(grad_after[i] == grad_before[i], boost::test_tools::tolerance(1e-12));
    }
}

BOOST_AUTO_TEST_CASE(set_parameter_mapping_rejects_bad_length) {
    auto sim = make_sim();
    const std::vector<VecZ> monos{{0}, {1}, {2}};
    sim.build_graph(monos, VecZ{0, 1, 2}, VecD{1.0, 1.0, 1.0}, VecZ{0, 0, 1});
    // Length 4 matches neither graph_layers (3) nor n_gates (2).
    BOOST_CHECK_THROW(sim.set_parameter_mapping(VecZ{0, 1, 2, 3}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(n_gates_accumulates_across_builds) {
    auto sim = make_sim();
    sim.build_graph(std::vector<VecZ>{{0}}, VecZ{0}, VecD{1.0});
    BOOST_TEST(sim.n_gates() == 1u);
    // Second call's local gate indices (iota) are offset by the existing gate count.
    sim.build_graph(std::vector<VecZ>{{1}}, VecZ{0}, VecD{1.0});
    BOOST_TEST(sim.n_gates() == 2u);
}

BOOST_AUTO_TEST_CASE(build_graph_rejects_malformed_gate_indices) {
    auto sim = make_sim();
    const std::vector<VecZ> monos{{0}, {1}};
    // A jump from 0 to 2 is not a contiguous run.
    BOOST_CHECK_THROW(sim.build_graph(monos, VecZ{0, 1}, VecD{1.0, 1.0}, VecZ{0, 2}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(coeff_informed_build_graph_rejects_too_few_parameters) {
    auto sim = make_sim();
    sim.build_graph(std::vector<VecZ>{{0}, {1}}, VecZ{0, 1}, VecD{1.0, 1.0});

    // A coefficient-informed second build must supply enough parameters to replay that graph
    // (>= 2). Its own mapping references only parameter 0, so a length-1 vector passes the
    // per-mapping length check but is too short to contract the existing graph -- the guard
    // rejects it up front rather than seeding from a silently truncated prefix.
    BOOST_CHECK_THROW(
        sim.build_graph(std::vector<VecZ>{{2}}, VecZ{0}, VecD{1.0}, std::nullopt, std::optional<VecD>{VecD{0.5}}),
        std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(contract_partially_replays_existing_graph_and_supports_inplace) {
    auto sim = make_sim();
    sim.build_graph(std::vector<VecZ>{{0}, {1}}, VecZ{0, 1}, VecD{1.0, 1.0});
    BOOST_TEST(sim.graph_layers() == 2u);

    // Coefficient-informed extend on a non-empty graph: build_graph internally calls
    // contract_partially(existing_params, /*inplace=*/false) to reseed atol truncation. The new layer's
    // own parameter index (2) must be >= the existing graph's max index (1), so `parameters` here can
    // simultaneously satisfy this call's own length check (against its local mapping, {2}) and be long
    // enough to replay the existing graph (which needs 2 values).
    sim.build_graph(std::vector<VecZ>{{0}},
                    VecZ{2},
                    VecD{1.0},
                    std::nullopt,
                    std::optional<VecD>{VecD{0.5, 0.25, 0.1}});
    BOOST_TEST(sim.graph_layers() == 3u);

    // inplace=false: returns coefficients, leaves the graph intact.
    const auto peeked = sim.contract_partially(VecD{0.5, 0.25, 0.1}, false);
    BOOST_TEST(!peeked.empty());
    BOOST_TEST(sim.graph_layers() == 3u);

    // inplace=true: consumes the (entire) graph into the operator.
    const auto consumed = sim.contract_partially(VecD{0.5, 0.25, 0.1}, true);
    BOOST_TEST(consumed.size() == peeked.size());
    BOOST_TEST(sim.graph_layers() == 0u);
}

namespace {

constexpr size_t kBlockModes = 3;

// Every unpaired observable term is within cutoff 2, the regime where blockwise truncation is the
// literal Eq. 4 rule.
auto make_block_sim(unsigned int cutoff,
                    size_t partitions = 1,
                    std::optional<unsigned int> schrodinger_cutoff = std::nullopt,
                    std::optional<size_t> max_sins = std::nullopt) -> MonomialPropagator<kBlockModes> {
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 0.7};
    ham[VecZ{2, 3}] = std::complex<double>{0.0, -0.4};
    ham[VecZ{0, 2}] = std::complex<double>{0.0, 0.3};
    ham[VecZ{1, 4}] = std::complex<double>{0.0, 0.5};
    return MonomialPropagator<kBlockModes>(ham,
                                           cutoff,
                                           VecZ{0},
                                           schrodinger_cutoff,
                                           MPI_COMM_SELF,
                                           std::nullopt,
                                           std::nullopt,
                                           CutoffType::Length,
                                           std::nullopt,
                                           kBlockModes,
                                           Basis::Majorana,
                                           partitions,
                                           max_sins);
}

// Two-term "excitation-like" gates and one four-index gate, each gate its own parameter.
const std::vector<VecZ> kBlockMonos{{0, 3}, {1, 2}, {2, 5}, {3, 4}, {0, 1, 2, 5}, {1, 4}, {0, 5}};
const VecZ kBlockMapping{0, 0, 1, 1, 2, 3, 3};
const VecD kBlockCoeffs{0.9, -0.9, 0.6, -0.6, 0.4, 1.1, -1.1};
const VecD kBlockParams{0.37, 1.21, -0.83, 2.4};
const VecZ kGateBlocks{0, 0, 1, 1, 2, 3, 3};

// Heisenberg, and Schrödinger over the full initial paired basis.
const std::array<std::optional<unsigned int>, 2> kPictures{std::nullopt, 2 * kBlockModes};

} // namespace

BOOST_AUTO_TEST_CASE(build_graph_rejects_malformed_block_ids) {
    auto sim = make_block_sim(2);
    BOOST_CHECK_THROW(
        sim.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs, std::nullopt, std::nullopt, std::nullopt, VecZ{0, 2}),
        std::runtime_error);
    BOOST_CHECK_THROW(sim.build_graph(kBlockMonos,
                                      kBlockMapping,
                                      kBlockCoeffs,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      VecZ{0, 0, 1, 1, 2, 3, 2}),
                      std::runtime_error);
    BOOST_TEST(sim.graph_layers() == 0u);
}

// Invariant A: a block per monomial is standard truncation, bit for bit, with nothing retired.
BOOST_AUTO_TEST_CASE(one_block_per_monomial_is_standard_truncation) {
    for (const auto schrodinger_cutoff : kPictures) {
        BOOST_TEST_CONTEXT("schrodinger_cutoff " << schrodinger_cutoff.value_or(0)) {
            auto standard = make_block_sim(2, 1, schrodinger_cutoff);
            standard.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs);
            auto per_layer = make_block_sim(2, 1, schrodinger_cutoff);
            VecZ ids(kBlockMonos.size());
            std::iota(ids.begin(), ids.end(), size_t{0});
            per_layer
                .build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs, std::nullopt, std::nullopt, std::nullopt, ids);
            BOOST_TEST(per_layer.size() == standard.size());
            BOOST_TEST(per_layer.operator_memory_usage().retired_rows == 0u);
            BOOST_TEST(per_layer.expectation_value(kBlockParams) == standard.expectation_value(kBlockParams));
        }
    }
}

// Invariant B: one block over the whole circuit cuts only at the end, where it drops only unpaired
// terms over the cutoff. Neither picture reads those here -- the reference state scores paired terms
// only, and every unpaired observable term is within the cutoff -- so it is exact (the untruncated
// Heisenberg cutoff 2 * modes).
BOOST_AUTO_TEST_CASE(one_block_over_everything_is_exact) {
    auto exact = make_block_sim(2 * kBlockModes);
    exact.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs);
    for (const auto schrodinger_cutoff : kPictures) {
        BOOST_TEST_CONTEXT("schrodinger_cutoff " << schrodinger_cutoff.value_or(0)) {
            auto blocked = make_block_sim(2, 1, schrodinger_cutoff);
            blocked.build_graph(kBlockMonos,
                                kBlockMapping,
                                kBlockCoeffs,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                VecZ(kBlockMonos.size(), 0));
            BOOST_TEST(blocked.operator_memory_usage().retired_rows > 0u);
            BOOST_TEST(blocked.expectation_value(kBlockParams) == exact.expectation_value(kBlockParams),
                       boost::test_tools::tolerance(1e-12));
        }
    }
}

// Retirement is partition-local; the partition count must not change the result.
BOOST_AUTO_TEST_CASE(blockwise_truncation_is_partition_count_invariant) {
    const auto run = [](size_t partitions, bool seeded, std::optional<unsigned int> schrodinger_cutoff) {
        auto sim = make_block_sim(2, partitions, schrodinger_cutoff);
        sim.build_graph(kBlockMonos,
                        kBlockMapping,
                        kBlockCoeffs,
                        std::nullopt,
                        seeded ? std::optional<VecD>{kBlockParams} : std::nullopt,
                        std::nullopt,
                        kGateBlocks);
        return sim.expectation_value_and_gradient(kBlockParams);
    };
    for (const auto schrodinger_cutoff : kPictures) {
        for (const bool seeded : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger_cutoff " << schrodinger_cutoff.value_or(0) << ", seeded " << seeded) {
                const auto [e1, g1] = run(1, seeded, schrodinger_cutoff);
                const auto [e3, g3] = run(3, seeded, schrodinger_cutoff);
                BOOST_TEST(e1 == e3, boost::test_tools::tolerance(1e-12));
                BOOST_TEST_REQUIRE(g1.size() == g3.size());
                for (size_t i = 0; i < g1.size(); ++i) {
                    BOOST_TEST(g1[i] == g3[i], boost::test_tools::tolerance(1e-12));
                }
            }
        }
    }
}

// Max-sins truncation. At the largest limit no partner is ever declined, so the graph and results are
// those of a propagator without one; the counts alone take memory.
BOOST_AUTO_TEST_CASE(max_sins_at_the_limit_matches_off) {
    for (const auto schrodinger_cutoff : kPictures) {
        BOOST_TEST_CONTEXT("schrodinger_cutoff " << schrodinger_cutoff.value_or(0)) {
            auto off = make_block_sim(2, 1, schrodinger_cutoff);
            off.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs);
            auto on = make_block_sim(2, 1, schrodinger_cutoff);
            on.update_max_sins(MonomialPropagator<kBlockModes>::kMaxSinsLimit);
            on.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs);
            BOOST_TEST(on.size() == off.size());
            BOOST_TEST(on.expectation_value(kBlockParams) == off.expectation_value(kBlockParams));
            BOOST_TEST(off.operator_memory_usage().sin_count_bytes == 0u);
            BOOST_TEST(on.operator_memory_usage().sin_count_bytes >= on.size());
        }
    }
}

// max_sins = 0 never creates a row in Heisenberg: only rotations between initial rows survive.
BOOST_AUTO_TEST_CASE(max_sins_zero_creates_no_row) {
    auto sim = make_block_sim(2);
    sim.update_max_sins(0);
    const size_t initial = sim.size();
    sim.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs);
    BOOST_TEST(sim.size() == initial);
    BOOST_TEST(sim.graph_layers() == kBlockMonos.size());
}

// Uncounted sines (count_sins = false) never decline a partner, even at max_sins = 0.
BOOST_AUTO_TEST_CASE(max_sins_uncounted_call_matches_off) {
    auto off = make_block_sim(2);
    off.build_graph(kBlockMonos, kBlockMapping, kBlockCoeffs);
    auto uncounted = make_block_sim(2);
    uncounted.update_max_sins(0);
    uncounted.build_graph(kBlockMonos,
                          kBlockMapping,
                          kBlockCoeffs,
                          std::nullopt,
                          std::nullopt,
                          std::nullopt,
                          std::nullopt,
                          /*count_sins=*/false);
    BOOST_TEST(uncounted.size() == off.size());
    BOOST_TEST(uncounted.expectation_value(kBlockParams) == off.expectation_value(kBlockParams));
}

BOOST_AUTO_TEST_CASE(max_sins_rejects_out_of_range_and_propagate) {
    constexpr size_t kTooMany = MonomialPropagator<kBlockModes>::kMaxSinsLimit + 1;
    BOOST_CHECK_THROW(make_block_sim(2, 1, std::nullopt, kTooMany), PropagatorConfigError);
    auto sim = make_block_sim(2);
    BOOST_CHECK_THROW(sim.update_max_sins(kTooMany), PropagatorConfigError);
    BOOST_TEST(!sim.max_sins().has_value());
    sim.update_max_sins(1);
    BOOST_CHECK_THROW(sim.propagate(kBlockMonos, kBlockMapping, kBlockCoeffs, kBlockParams), PropagatorConfigError);
    sim.update_max_sins(std::nullopt);
    BOOST_TEST(sim.operator_memory_usage().sin_count_bytes == 0u);
}

// Counts are partition-local and cross partitions in the query records; the partition count must not
// change the result. The constructor argument reaches every partition.
BOOST_AUTO_TEST_CASE(max_sins_is_partition_count_invariant) {
    const auto run = [](size_t partitions, std::optional<unsigned int> schrodinger_cutoff, bool blocks) {
        auto sim = make_block_sim(2, partitions, schrodinger_cutoff, /*max_sins=*/1);
        BOOST_TEST(sim.max_sins().value_or(0) == 1u);
        sim.build_graph(kBlockMonos,
                        kBlockMapping,
                        kBlockCoeffs,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        blocks ? std::optional<VecZ>{kGateBlocks} : std::nullopt);
        const auto [e, g] = sim.expectation_value_and_gradient(kBlockParams);
        return std::tuple{sim.size(), e, g};
    };
    for (const auto schrodinger_cutoff : kPictures) {
        for (const bool blocks : {false, true}) {
            BOOST_TEST_CONTEXT("schrodinger_cutoff " << schrodinger_cutoff.value_or(0) << ", blocks " << blocks) {
                const auto [n1, e1, g1] = run(1, schrodinger_cutoff, blocks);
                const auto [n3, e3, g3] = run(3, schrodinger_cutoff, blocks);
                BOOST_TEST(n1 == n3);
                BOOST_TEST(e1 == e3, boost::test_tools::tolerance(1e-12));
                BOOST_TEST_REQUIRE(g1.size() == g3.size());
                for (size_t i = 0; i < g1.size(); ++i) {
                    BOOST_TEST(g1[i] == g3[i], boost::test_tools::tolerance(1e-12));
                }
            }
        }
    }
}
