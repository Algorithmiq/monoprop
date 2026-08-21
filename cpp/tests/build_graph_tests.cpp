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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "TestUtilities.h"

using namespace test_utils;

TEST_CASE_METHOD(ExampleDataFix, "build_graph_cases") {
    const auto index = GENERATE(0U, 1U);
    const auto pare = ds_pare_values[index];
    const auto sch_enabled = ds_schrodinger_enabled[index];
    CAPTURE(pare, sch_enabled);
    const auto schrodinger_cutoff = make_schrodinger_cutoff(sch_enabled, cutoff);
    SimulatorConfig cfg{
        .schrodinger_cutoff = schrodinger_cutoff ? std::optional<unsigned int>(*schrodinger_cutoff) : std::nullopt,
        .cutoff_type = cutoff_type,
        .basis_change = basis_change,
    };
    test_evolve_build_graph<n_modes>(data, cfg, pare, data.actual_expval);
}

TEST_CASE_METHOD(ExampleDataFix, "build_graph_with_coeffs_cases") {
    const auto index = GENERATE(0U, 1U);
    const auto pare = ds_pare_values[index];
    const auto sch_enabled = ds_schrodinger_enabled[index];
    CAPTURE(pare, sch_enabled);
    const auto schrodinger_cutoff = make_schrodinger_cutoff(sch_enabled, cutoff);
    SimulatorConfig cfg{
        .schrodinger_cutoff = schrodinger_cutoff ? std::optional<unsigned int>(*schrodinger_cutoff) : std::nullopt,
        .cutoff_type = cutoff_type,
        .basis_change = basis_change,
    };
    test_evolve_build_graph_with_coeffs<n_modes>(data, cfg, pare, data.actual_expval);
}

// Schrodinger-only by construction; the reason is on test_evolve_build_graph_with_coeffs_extend.
TEST_CASE_METHOD(ExampleDataFix, "build_graph_with_coeffs_extend_cases") {
    const auto pare = GENERATE(false, true);
    CAPTURE(pare);
    const auto schrodinger_cutoff = make_schrodinger_cutoff(/*enabled=*/true, cutoff);
    SimulatorConfig cfg{
        .schrodinger_cutoff = std::optional<unsigned int>(*schrodinger_cutoff),
        .cutoff_type = cutoff_type,
        .basis_change = basis_change,
    };
    test_evolve_build_graph_with_coeffs_extend<n_modes>(data, cfg, pare, data.actual_expval);
}

// graph_size().first counts cos-scaled non-endpoints, recomputed from the operator's inverted index.
TEST_CASE("graph_size_reports_real_cosine_only_count") {
    constexpr size_t N = 8;
    const auto data = test_utils::load_case_data<N>("random_exact.msgpack");

    const auto sized = [&](unsigned int cutoff) {
        auto sim = MonomialPropagator<N>(data.hamiltonian, cutoff, data.initial_state, std::nullopt, MPI_COMM_SELF);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        return sim.graph_size();
    };

    // Truncating cutoff: some cos-scaled terms lose their sine partner, so cosine-only is positive.
    const auto truncated = sized(8);
    CHECK((truncated.first) > (0U));
    CHECK((truncated.second) > (0U));

    // Exact cutoff: every cos index is also a rotation endpoint, so cosine-only is genuinely zero.
    const auto exact = sized(2 * N);
    CHECK((exact.first) == (0U));
    CHECK((exact.second) > (truncated.second));
}
