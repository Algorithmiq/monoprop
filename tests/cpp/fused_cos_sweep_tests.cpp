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

#include <optional>

#include "TestUtilities.h"

// Fused cos sweep (ContractImmediately k==0): the scan multiplies every anticommuting coefficient by
// cos(2θ) in place during its own pass, and resolve recovers a hit partner's pre-cos value as
// stored·(1/cos) — see fused_find_and_collect / LayerBuildEngine. That recovery is the ONE deliberate
// FP deviation from the two-pass path (≤1 ulp per hit endpoint's sine term, physics-identical: same
// rotation set, same atol gating on the pre-cos load). These tests bound the accumulated drift by
// requiring the in-place propagate() (fused sweep) and the build_graph()+replay evaluation (untouched
// two-pass machinery) to agree far tighter than the physics tolerance, in both pictures and both with
// and without the lower_atol gate that steers the scan's emission.

namespace {

using namespace test_utils;
using namespace monoprop;

// propagate() and build_graph()+replay accumulate FP differently even before the sweep (parallel
// reduction order, replay recompute), so demand agreement to 1e-12 — tight enough that a wrong cos
// factor on any endpoint (relative error O(1)) fails loudly, loose enough for benign reordering.
constexpr double kAgreeAtol = 1e-12;
constexpr double kExactAtol = 1e-9;

template <size_t NumModes>
auto inplace_energy(const CaseData &data, const SimulatorConfig &cfg) -> double {
    auto sim = build_simulator<NumModes>(data, cfg);
    sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
    auto fn = sim.expectation_value_functional(std::nullopt);
    return fn(VecD{});
}

template <size_t NumModes>
auto graph_energy(const CaseData &data, const SimulatorConfig &cfg) -> double {
    auto sim = build_simulator<NumModes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    auto fn = sim.expectation_value_functional(std::nullopt);
    return fn(data.parameters);
}

void check_agreement(const CaseData &data, const SimulatorConfig &cfg, const char *label) {
    const double inplace = inplace_energy<ExampleDataFix::n_modes>(data, cfg);
    const double graph = graph_energy<ExampleDataFix::n_modes>(data, cfg);
    BOOST_TEST_CONTEXT(label << " inplace=" << inplace << " graph=" << graph) {
        BOOST_CHECK_SMALL(inplace - graph, kAgreeAtol);
        BOOST_CHECK_SMALL(inplace - data.actual_expval, kExactAtol);
    }
}

} // namespace

BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_heisenberg, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{}, "heisenberg");
}

// lower_atol active: the sin gate reads the PRE-cos value the sweep loads — emission (and therefore
// the rotation set) must be unchanged by the in-place store that follows it.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_heisenberg_atol, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.atol = 1e-10}, "heisenberg atol=1e-10");
}

// Schrödinger picture: fresh inserts carry a nonzero state-scored value born AFTER the sweep — the
// apply's in-place insert arm (c = cos·c + sin term) must fold the gate's cos into those slots.
BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_schrodinger, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes}, "schrodinger");
}

BOOST_FIXTURE_TEST_CASE(fused_sweep_matches_graph_replay_schrodinger_atol, ExampleDataFix) {
    check_agreement(data, SimulatorConfig{.schrodinger_cutoff = 2 * n_modes, .atol = 1e-10}, "schrodinger atol=1e-10");
}
