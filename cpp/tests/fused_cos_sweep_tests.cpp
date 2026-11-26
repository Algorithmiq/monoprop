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

#include <cstddef>
#include <optional>

#include "TestUtilities.h"

// The fused cos sweep's one deliberate FP deviation from the two-pass path (≤1 ulp per hit endpoint,
// from resolve's stored·(1/cos) recovery), against the build_graph()+replay evaluation and the frozen exact energy as
// oracles, at the launch's T (cpp/tests/CMakeLists.txt reruns these cases at T = 4).

namespace {

using namespace test_utils;
using namespace monoprop;

// The two paths accumulate FP differently even before the sweep, so demand 1e-12: tight enough that a
// wrong cos factor on any endpoint (relative error O(1)) fails loudly, loose enough for reordering.
constexpr double agree_atol = 1e-12;
constexpr double exact_atol = 1e-9;

// In-place propagation against graph replay; `cap` is only_rotate_len_k for both.
void check_agreement(const CaseData &data,
                     const SimulatorConfig &cfg,
                     const char *label,
                     std::optional<size_t> cap = std::nullopt) {
    constexpr size_t n_modes = ExampleDataFix::n_modes;
    auto inplace_sim = build_simulator<n_modes>(data, cfg);
    inplace_sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, cap);
    const double inplace = inplace_sim.expectation_value_functional(std::nullopt)(VecD{});
    auto graph_sim = build_simulator<n_modes>(data, cfg);
    graph_sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs, std::nullopt, std::nullopt, cap);
    const double graph = graph_sim.expectation_value_functional(std::nullopt)(data.parameters);
    BOOST_TEST_CONTEXT(label << " inplace=" << inplace << " graph=" << graph) {
        BOOST_CHECK_SMALL(inplace - graph, agree_atol);
        BOOST_CHECK_SMALL(inplace - data.actual_expval, exact_atol);
    }
}

} // namespace

BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_heisenberg, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{}, "heisenberg");
}

// lower_atol active: the sin gate reads the pre-cos value, so the in-place store that follows must
// not change which terms are emitted.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_heisenberg_atol, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.atol = 1e-10}, "heisenberg atol=1e-10");
}

// Schrödinger: fresh inserts are born after the sweep with a nonzero coeff, so the apply's insert arm
// must fold the gate's cos into those slots itself.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_schrodinger, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes}, "schrodinger");
}

BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_schrodinger_atol, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes, .atol = 1e-10}, "schrodinger atol=1e-10");
}

// The two-pass apply (a length cap at the full width truncates nothing but disables the fused sweep) against
// graph replay, with the same tolerances: the insert snapshots must be gathered before the cosine phase.
BOOST_FIXTURE_TEST_CASE(two_pass_apply_matches_graph_replay, ExampleDataFix) {
    constexpr size_t cap = 2 * n_modes;
    check_agreement(data, SimulatorConfig{}, "two-pass heisenberg", cap);
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes}, "two-pass schrodinger", cap);
}
