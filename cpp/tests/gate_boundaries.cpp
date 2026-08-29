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
#include <complex>
#include <optional>
#include <stdexcept>

#include "TestPropagator.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;

namespace {

constexpr size_t kModes = 2;

auto make_sim() -> MonomialPropagator {
    OperatorDict ham;
    ham[VecZ{0, 1}] = std::complex<double>{0.0, 1.0};
    VecZ initial_state{0, 1};
    return test_utils::make_propagator(kModes,
                                       ham,
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
