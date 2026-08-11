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

#include <cmath>

#include "TestUtilities.h"

using namespace test_utils;

BOOST_AUTO_TEST_CASE(multi_rank_pare_expval_is_finite) {
    const int world_size = monoprop::mpi::size(MPI_COMM_WORLD);
    if (world_size < 2) {
        BOOST_TEST_MESSAGE("Skipping multi_rank_pare_expval_is_finite: MPI world size=" << world_size);
        return;
    }

    constexpr size_t NumModes = 8;
    constexpr double kExpvalAtol = 1e-9;

    const auto data = load_case_data<NumModes>("random_exact.msgpack");

    SimulatorConfig cfg{.comm = MPI_COMM_WORLD};

    auto baseline_sim = build_simulator<NumModes>(data, cfg);
    const double baseline_expval = evaluate_expval(baseline_sim, data, false);
    BOOST_CHECK_SMALL(std::abs(baseline_expval - data.actual_expval), kExpvalAtol);

    auto pared_sim = build_simulator<NumModes>(data, cfg);
    const double pared_expval = evaluate_expval(pared_sim, data, true);

    BOOST_TEST_CONTEXT("world_size=" << world_size) {
        BOOST_CHECK_MESSAGE(std::isfinite(pared_expval),
                            "pare=true produced non-finite expectation value across MPI ranks");
        BOOST_CHECK_SMALL(std::abs(pared_expval - data.actual_expval), kExpvalAtol);
    }
}
