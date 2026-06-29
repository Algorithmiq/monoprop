#include <boost/test/unit_test.hpp>

#include <cmath>
#include <optional>
#include <string>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// Verifies that single-rank and multi-rank simulations produce equivalent energy
// values (within floating-point accumulation tolerance). Rounding differences from
// different summation order are expected; they grow with system size but are bounded
// by ~1e-7 relative for all tested configurations.

namespace {

using namespace monoprop;
using namespace test_utils;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 4;
constexpr double kAtol = 1e-9;
constexpr double kFpRtol = 1e-7; // tolerance for n=1 vs n>1 floating-point accumulation

auto near(double lhs, double rhs, double atol = kAtol, double rtol = kFpRtol) -> bool {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= (atol + rtol * scale);
}

struct TestInputs {
    MPData data;
};

auto load_inputs() -> TestInputs {
    return {load_case_data<kNumModes>("random_exact.msgpack")};
}

auto run_energy(const TestInputs& inputs, MPI_Comm comm) -> double {
    MonomialPropagator<kNumModes> sim(inputs.data.fermionic_operator,
                                           kCutoff,
                                           inputs.data.slater_determinant,
                                           std::nullopt,
                                           comm,
                                           std::nullopt,
                                           std::nullopt,
                                           CutoffType::Length,
                                           std::nullopt);
    sim.propagate(inputs.data.majoranas);
    auto fn = sim.expectation_value_functional(inputs.data.param_inds, inputs.data.gen_coeffs);
    return fn(inputs.data.parameters);
}

// ─── Test 1: single-rank (SELF) energy matches multi-rank (WORLD) energy ────────

BOOST_AUTO_TEST_CASE(rank_count_energy_within_fp_tolerance) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping cross-rank-count case: requires at least 2 ranks.");
        return;
    }
    const auto inputs = load_inputs();
    const double e_serial = run_energy(inputs, MPI_COMM_SELF);
    const double e_world  = run_energy(inputs, MPI_COMM_WORLD);
    BOOST_TEST_MESSAGE("serial=" << e_serial << " world=" << e_world
                       << " diff=" << (e_world - e_serial));
    BOOST_TEST(near(e_serial, e_world));
}

// ─── Test 2: gradient is consistent across rank counts ──────────────────────────

BOOST_AUTO_TEST_CASE(gradient_rank_count_within_fp_tolerance) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping gradient cross-rank-count case: requires at least 2 ranks.");
        return;
    }
    const auto& inputs = load_inputs();

    auto run_gradient = [&](MPI_Comm comm) -> VecD {
        MonomialPropagator<kNumModes> sim(inputs.data.fermionic_operator,
                                               kCutoff,
                                               inputs.data.slater_determinant,
                                               std::nullopt,
                                               comm,
                                               std::nullopt,
                                               std::nullopt,
                                               CutoffType::Length,
                                               std::nullopt);
        sim.propagate(inputs.data.majoranas);
        auto fn = sim.expectation_value_and_gradient_functional(inputs.data.param_inds, inputs.data.gen_coeffs);
        return fn(inputs.data.parameters).second;
    };

    const auto g_serial = run_gradient(MPI_COMM_SELF);
    const auto g_world  = run_gradient(MPI_COMM_WORLD);

    BOOST_REQUIRE_EQUAL(g_serial.size(), g_world.size());
    for (size_t i = 0; i < g_serial.size(); ++i) {
        BOOST_TEST_CONTEXT("gradient idx=" << i) {
            BOOST_TEST(near(g_serial[i], g_world[i]));
        }
    }
}

} // namespace
