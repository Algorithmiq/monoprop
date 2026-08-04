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
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "PauliTestOracle.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// Single-rank (self) vs multi-rank (world) equivalence of energy, gradient, native-Pauli energy and
// the MPI x partition hybrid. Oracle: the self run; the only expected difference is summation order,
// covered by near()'s kFpRtol = 1e-7.

namespace {

using namespace monoprop;
using namespace test_utils;
using pauli_oracle::slots_of_string;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 4;

struct TestInputs {
    CaseData data;
};

auto load_inputs() -> TestInputs {
    return {load_case_data<kNumModes>("random_exact.msgpack")};
}

auto run_energy(const TestInputs& inputs, MPI_Comm comm) -> double {
    MonomialPropagator<kNumModes> sim(inputs.data.hamiltonian,
                                      kCutoff,
                                      inputs.data.initial_state,
                                      std::nullopt,
                                      comm,
                                      std::nullopt,
                                      std::nullopt,
                                      CutoffType::Length,
                                      std::nullopt);
    sim.build_graph(inputs.data.majoranas, inputs.data.param_inds, inputs.data.gen_coeffs);
    auto fn = sim.expectation_value_functional();
    return fn(inputs.data.parameters);
}

BOOST_AUTO_TEST_CASE(rank_count_energy_within_fp_tolerance) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping cross-rank-count case: requires at least 2 ranks.");
        return;
    }
    const auto inputs = load_inputs();
    const double e_serial = run_energy(inputs, MPI_COMM_SELF);
    const double e_world = run_energy(inputs, MPI_COMM_WORLD);
    BOOST_TEST_MESSAGE("serial=" << e_serial << " world=" << e_world << " diff=" << (e_world - e_serial));
    BOOST_TEST(near(e_serial, e_world));
}

BOOST_AUTO_TEST_CASE(gradient_rank_count_within_fp_tolerance) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping gradient cross-rank-count case: requires at least 2 ranks.");
        return;
    }
    const auto& inputs = load_inputs();

    auto run_gradient = [&](MPI_Comm comm) -> VecD {
        MonomialPropagator<kNumModes> sim(inputs.data.hamiltonian,
                                          kCutoff,
                                          inputs.data.initial_state,
                                          std::nullopt,
                                          comm,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Length,
                                          std::nullopt);
        sim.build_graph(inputs.data.majoranas, inputs.data.param_inds, inputs.data.gen_coeffs);
        auto fn = sim.expectation_value_and_gradient_functional();
        return fn(inputs.data.parameters).second;
    };

    const auto g_serial = run_gradient(MPI_COMM_SELF);
    const auto g_world = run_gradient(MPI_COMM_WORLD);

    BOOST_REQUIRE_EQUAL(g_serial.size(), g_world.size());
    for (size_t i = 0; i < g_serial.size(); ++i) {
        BOOST_TEST_CONTEXT("gradient idx=" << i) {
            BOOST_TEST(near(g_serial[i], g_world[i]));
        }
    }
}

// Native Pauli engine across rank counts. The same owner hash and cross-rank resolve path drive the
// intra-process partition runtime, so this guards both.

constexpr size_t kPauliQ = 6;

auto run_pauli_energy(MPI_Comm comm) -> double {
    OperatorDict init;
    init[slots_of_string("ZIIIII")] = std::complex<double>(1.0, 0.0);
    init[slots_of_string("IIZZII")] = std::complex<double>(0.5, 0.0);
    MonomialPropagator<kPauliQ> sim(init,
                                    kPauliQ,
                                    VecZ{},
                                    std::nullopt,
                                    comm,
                                    1e-12,
                                    std::nullopt,
                                    CutoffType::Support,
                                    std::nullopt,
                                    kPauliQ,
                                    Basis::Pauli);
    std::vector<VecZ> gens;
    VecZ pmap;
    VecD gcoeffs;
    size_t p = 0;
    for (size_t q = 0; q < kPauliQ; ++q) {
        std::string s(kPauliQ, 'I');
        s[q] = 'X';
        gens.push_back(slots_of_string(s));
        pmap.push_back(p++);
        gcoeffs.push_back(1.0);
    }
    for (size_t q = 0; q + 1 < kPauliQ; ++q) {
        std::string s(kPauliQ, 'I');
        s[q] = 'Z';
        s[q + 1] = 'Z';
        gens.push_back(slots_of_string(s));
        pmap.push_back(p++);
        gcoeffs.push_back(1.0);
    }
    sim.propagate(gens, pmap, gcoeffs, VecD(p, 0.3));
    return sim.expectation_value({});
}

BOOST_AUTO_TEST_CASE(pauli_rank_count_energy_within_fp_tolerance) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping Pauli cross-rank-count case: requires at least 2 ranks.");
        return;
    }
    const double e_serial = run_pauli_energy(MPI_COMM_SELF);
    const double e_world = run_pauli_energy(MPI_COMM_WORLD);
    BOOST_TEST_MESSAGE("pauli serial=" << e_serial << " world=" << e_world);
    BOOST_TEST(near(e_serial, e_world));
}

// MPI x partition hybrid: partitions=S under R ranks builds the HybridComm flat R*S world, which only changes
// allreduce association, so the energy must match serial and the global term count must be exactly
// invariant. Explicit partitions= wins over the suite's monoprop_PARTITIONS=off, so this is the sole case
// exercising the hybrid transport end to end.

auto run_energy_partitioned(const TestInputs& inputs, MPI_Comm comm, size_t partitions) -> std::pair<double, size_t> {
    MonomialPropagator<kNumModes> sim(inputs.data.hamiltonian,
                                      kCutoff,
                                      inputs.data.initial_state,
                                      std::nullopt,
                                      comm,
                                      std::nullopt,
                                      std::nullopt,
                                      CutoffType::Length,
                                      std::nullopt,
                                      kNumModes,
                                      Basis::Majorana,
                                      partitions);
    sim.build_graph(inputs.data.majoranas, inputs.data.param_inds, inputs.data.gen_coeffs);
    auto fn = sim.expectation_value_functional();
    const double e = fn(inputs.data.parameters);
    return {e, sim.size()};
}

BOOST_AUTO_TEST_CASE(hybrid_mpi_partition_energy_and_size_equivalence) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping hybrid case: requires at least 2 ranks.");
        return;
    }
    const auto inputs = load_inputs();
    const auto [e_serial, n_serial] = run_energy_partitioned(inputs, MPI_COMM_SELF, 1);
    const auto [e_hybrid, n_local] = run_energy_partitioned(inputs, MPI_COMM_WORLD, 2);
    // Each rank's facade holds only its local partitions; the global term count is the cross-rank sum.
    const size_t n_hybrid_global = mpi::allreduce_sum<size_t>(n_local, MPI_COMM_WORLD);
    BOOST_TEST_MESSAGE("serial=" << e_serial << " (n=" << n_serial << ") hybrid R*2=" << e_hybrid
                                 << " (global n=" << n_hybrid_global << ")");
    BOOST_TEST(near(e_serial, e_hybrid));
    BOOST_CHECK_EQUAL(n_serial, n_hybrid_global);
}

} // namespace
