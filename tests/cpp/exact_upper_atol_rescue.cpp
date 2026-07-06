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
#include <cmath>
#include <optional>
#include <thread>
#include <vector>

#include <tbb/global_control.h>

#include "TestUtilities.h"

// upper_atol RESCUE invariant: a structural cutoff of 0 rejects every partner term the evolution
// generates (only the identity has popcount <= 0), so on its own it would truncate the operator down
// to nothing. upper_atol = 0 rescues a rejected partner whenever its sine coefficient magnitude is
// >= 0 — which is ALWAYS true — so every partner is kept and the evolution is exact: the energy must
// match the reference to FP-summation tolerance regardless of cutoff.
//
// The rescue keys off the partner's coefficient, so it only fires on the coefficient-carrying
// (in-place) build path; the structural graph-only build has no coefficients to test and is NOT
// exact at cutoff 0 (it is the complementary, deliberately-NOT-tested case). See
// CutoffContext::is_above_upper and the emit gate in LayerBuilder.h.

namespace {

using namespace test_utils;
using namespace monoprop;

constexpr double kEnergyAtol = 1e-9;

enum class CommMode { Self, World };

inline auto thread_mode_values() -> std::array<int, 2> {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    return {1, std::max(2, hw)};
}

// Build a simulator with a ZERO structural cutoff and upper_atol = 0 (full rescue). Unlike
// build_simulator (which hardcodes cutoff = 2*NumModes), this exercises the rescue path: cutoff 0
// rejects everything, upper_atol 0 keeps everything.
template <size_t NumModes>
auto build_zero_cutoff_full_rescue(const CaseData& data, MPI_Comm comm) -> MonomialPropagator<NumModes> {
    return MonomialPropagator<NumModes>(data.hamiltonian,
                                        /*cutoff=*/0U,
                                        data.hartree_fock,
                                        /*schrodinger_cutoff=*/std::nullopt,
                                        comm,
                                        /*atol=*/std::nullopt,
                                        /*upper_atol=*/std::optional<double>{0.0},
                                        CutoffType::Length,
                                        /*basis_change=*/std::nullopt);
}

// In-place (coefficient-carrying) evolve + energy: this is the path on which upper_atol can rescue.
template <size_t NumModes>
auto evaluate_zero_cutoff_full_rescue_energy(MonomialPropagator<NumModes>& simulator, const CaseData& data) -> double {
    simulator.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
    auto energy_fn = simulator.expectation_value_functional(std::nullopt);
    return energy_fn(VecD{});
}

struct RandomExactFixture {
    static constexpr size_t n_modes = 8;
    CaseData data;
    RandomExactFixture() : data(load_case_data<n_modes>("random_exact.msgpack")) {}
};

struct LihFixture {
    static constexpr size_t n_modes = 12;
    CaseData data;
    LihFixture() : data(load_case_data<n_modes>("lih_fermionic_spin_exact.msgpack")) {}
};

} // namespace

// One test per (fixture, comm, thread-mode) so a failure pinpoints the configuration.
#define MAKE_ZERO_CUTOFF_RESCUE_TEST(NAME, FixtureType, CommToken, ThrIdx)                                    \
    BOOST_FIXTURE_TEST_CASE(NAME##_##CommToken##_thr##ThrIdx, FixtureType) {                                  \
        const auto thread_modes = thread_mode_values();                                                       \
        const auto capped_threads = static_cast<std::size_t>(std::max(1, thread_modes[ThrIdx]));              \
        tbb::global_control thread_guard(tbb::global_control::max_allowed_parallelism, capped_threads);       \
        MPI_Comm comm = (CommMode::CommToken == CommMode::Self) ? MPI_COMM_SELF : MPI_COMM_WORLD;             \
        if (CommMode::CommToken == CommMode::World && mpi::size(comm) == 1) {                                 \
            BOOST_TEST_MESSAGE("Skipping multi-rank scenario for " #NAME " (world size=1)");                  \
            return;                                                                                           \
        }                                                                                                     \
        auto simulator = build_zero_cutoff_full_rescue<FixtureType::n_modes>(data, comm);                     \
        const double energy = evaluate_zero_cutoff_full_rescue_energy<FixtureType::n_modes>(simulator, data); \
        BOOST_CHECK_SMALL(std::abs(energy - data.actual_expval), kEnergyAtol);                                \
    }

MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact, RandomExactFixture, Self, 0)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact, RandomExactFixture, Self, 1)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact, RandomExactFixture, World, 0)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact, RandomExactFixture, World, 1)

MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact_lih, LihFixture, Self, 0)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact_lih, LihFixture, Self, 1)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact_lih, LihFixture, World, 0)
MAKE_ZERO_CUTOFF_RESCUE_TEST(zero_cutoff_upper_atol_zero_is_exact_lih, LihFixture, World, 1)

#undef MAKE_ZERO_CUTOFF_RESCUE_TEST
