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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "TestUtilities.h"

using namespace test_utils;

TEST_CASE("multi_rank_pare_expval_is_finite") {
    const int world_size = monoprop::mpi::size(MPI_COMM_WORLD);
    if (world_size < 2) {
        INFO("Skipping multi_rank_pare_expval_is_finite: MPI world size=" << world_size);
        return;
    }

    constexpr size_t NumModes = 8;
    constexpr double kExpvalAtol = 1e-9;

    const auto data = load_case_data<NumModes>("random_exact.msgpack");

    SimulatorConfig cfg{.comm = MPI_COMM_WORLD};

    auto baseline_sim = build_simulator<NumModes>(data, cfg);
    const double baseline_expval = evaluate_expval(baseline_sim, data, false);
    CHECK_THAT(std::abs(baseline_expval - data.actual_expval), Catch::Matchers::WithinAbs(0.0, kExpvalAtol));

    auto pared_sim = build_simulator<NumModes>(data, cfg);
    const double pared_expval = evaluate_expval(pared_sim, data, true);

    {
        INFO("world_size=" << world_size);
        INFO("pare=true produced non-finite expectation value across MPI ranks");
        CHECK(std::isfinite(pared_expval));
        CHECK_THAT(std::abs(pared_expval - data.actual_expval), Catch::Matchers::WithinAbs(0.0, kExpvalAtol));
    }
}
