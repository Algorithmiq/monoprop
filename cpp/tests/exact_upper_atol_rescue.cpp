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
#include <optional>
#include <vector>

#include "TestUtilities.h"

// upper_atol rescue: a structural cutoff of 0 rejects every non-identity partner, and upper_atol = 0
// rescues all of them (|sin|·|coeff| >= 0 always), so the evolution is exact. Oracle: data.actual_expval.
// The rescue reads the partner's coefficient, so only the coefficient-carrying in-place build path can
// take it — the graph-only build has no coefficients and is not exact at cutoff 0.

namespace {

using namespace test_utils;
using namespace monoprop;

constexpr double kEnergyAtol = 1e-9;

enum class CommMode { Self, World };

// build_simulator cannot express this: it hardcodes cutoff = 2*NumModes.
template <size_t NumModes>
auto build_zero_cutoff_full_rescue(const CaseData& data, MPI_Comm comm) -> MonomialPropagator<NumModes> {
    return MonomialPropagator<NumModes>(data.hamiltonian,
                                        /*cutoff=*/0U,
                                        data.initial_state,
                                        /*schrodinger_cutoff=*/std::nullopt,
                                        comm,
                                        /*atol=*/std::nullopt,
                                        /*upper_atol=*/std::optional<double>{0.0},
                                        CutoffType::Length,
                                        /*basis_change=*/std::nullopt);
}

template <size_t NumModes>
auto evaluate_zero_cutoff_full_rescue_energy(MonomialPropagator<NumModes>& simulator, const CaseData& data) -> double {
    simulator.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
    auto energy_fn = simulator.expectation_value_functional(std::nullopt);
    return energy_fn(VecD{});
}

} // namespace

// One test per (fixture, comm) so a failure pinpoints the configuration.
#define STRINGIFY_IMPL(Token) #Token
#define STRINGIFY(Token)      STRINGIFY_IMPL(Token)
#define MAKE_ZERO_CUTOFF_RESCUE_TEST(NAME, FixtureType, CommToken)                                            \
    TEST_CASE_METHOD(FixtureType, STRINGIFY(NAME) "_" STRINGIFY(CommToken)) {                                 \
        MPI_Comm comm = (CommMode::CommToken == CommMode::Self) ? MPI_COMM_SELF : MPI_COMM_WORLD;             \
        if (CommMode::CommToken == CommMode::World && mpi::size(comm) == 1) {                                 \
            SKIP("Skipping multi-rank scenario for " #NAME " (world size=1)");                                \
        }                                                                                                     \
        auto simulator = build_zero_cutoff_full_rescue<FixtureType::n_modes>(data, comm);                     \
        const double energy = evaluate_zero_cutoff_full_rescue_energy<FixtureType::n_modes>(simulator, data); \
        CHECK_THAT(energy - data.actual_expval, Catch::Matchers::WithinAbs(0.0, kEnergyAtol));                \
    }

MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact, ExampleDataFix, Self)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact, ExampleDataFix, World)

MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact_lih, LihFixture, Self)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact_lih, LihFixture, World)

#undef MAKE_ZERO_CUTOFF_RESCUE_TEST
#undef STRINGIFY
#undef STRINGIFY_IMPL
