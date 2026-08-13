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

// An emit gate that keeps one half of a matched rotation pair and drops the other is what makes the
// cross-rank leader/follower exchange asymmetric, and nothing in the suite exercised that with real
// ranks. Both atol arguments are nullopt in every case above; exact_upper_atol_rescue.cpp and
// mpi_fresh_insert_equivalence.cpp pass upper_atol = 0, which rescues every truncated term and so
// restores the symmetry they appear to test. The only asymmetric-gate case with more than one partition
// is partition_equivalence_tests.cpp's Pauli energy case, which runs over the in-process ShmComm and
// never crosses a rank boundary. A protocol defect that appears only when the leader emits and the
// follower does not would therefore pass the entire suite.
//
// graph_path selects which sink resolves the exchange: build_graph goes through GraphSink, which
// pre-sizes a response slot per incoming query, while propagate goes through ContractSink, which writes
// a half-rotation record instead. They fail differently, so both are covered.
// LihFixture (n=12), not the random_exact fixture the rest of this file uses. That one truncates
// nothing at any setting -- partition_equivalence_tests.cpp already records this -- and the first run of
// this case proved it from the other direction: the ungated term count was **3**, so no coefficient
// threshold could drop anything and the case passed while exercising none of the asymmetry it exists to
// cover. A gate that cannot bite is the same defect as no gate at all.
template <size_t N>
auto run_gated(const CaseData& data,
               MPI_Comm comm,
               std::optional<double> lower_atol,
               std::optional<size_t> only_rotate_len_k,
               bool graph_path) -> std::pair<double, size_t> {
    // only_rotate_len_k is a gate-application argument, not a construction one -- the constructor's
    // ninth parameter is basis_change. Passing it here instead is a compile error, which is the useful
    // kind of mistake to make.
    MonomialPropagator<N> sim(data.hamiltonian,
                              kCutoff,
                              data.initial_state,
                              std::nullopt,
                              comm,
                              lower_atol,
                              std::nullopt,
                              CutoffType::Length);
    double e = 0.0;
    if (graph_path) {
        sim.build_graph(data.majoranas,
                        data.param_inds,
                        data.gen_coeffs,
                        std::nullopt,
                        std::nullopt,
                        only_rotate_len_k);
        auto fn = sim.expectation_value_functional();
        e = fn(data.parameters);
    }
    else {
        sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters, only_rotate_len_k);
        e = sim.expectation_value({});
    }
    // size() is per-rank -- the facade holds only its local partitions -- so the comparable quantity is
    // the sum over the propagator's own communicator. Over MPI_COMM_SELF that is the identity, which is
    // what lets the serial reference and the distributed run share this helper.
    return {e, mpi::allreduce_sum<size_t>(sim.size(), comm)};
}

BOOST_AUTO_TEST_CASE(rank_count_matches_under_an_asymmetric_emit_gate) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping asymmetric-gate case: requires at least 2 ranks.");
        return;
    }
    constexpr size_t kLihModes = LihFixture::n_modes;
    const LihFixture lih;
    const CaseData& data = lih.data;

    // Ungated reference count. Without it this case could pass while every atol below was too small to
    // drop anything -- that is, while exercising none of the asymmetry it exists to cover, which is the
    // hole being closed. Checked once at the end, since one biting value is enough.
    const auto [e_ref, n_ungated] = run_gated<kLihModes>(data, MPI_COMM_SELF, std::nullopt, std::nullopt, true);
    static_cast<void>(e_ref);
    bool gate_bit = false;
    // Tracked separately from gate_bit. A single flag OR-ed across every combination is satisfied by the
    // propagate cases alone, which would leave the GraphSink path -- where a mis-sized or uninitialised
    // response slot would live -- covered in name only. lower_atol turns out not to reduce build_graph's
    // term count at all, so the length cap is what has to bite there.
    bool graph_gate_bit = false;

    for (const double atol : {1e-2, 1e-3, 1e-6}) {
        for (const bool graph_path : {true, false}) {
            const auto [e_serial, n_serial] = run_gated<kLihModes>(data, MPI_COMM_SELF, atol, std::nullopt, graph_path);
            const auto [e_world, n_world] = run_gated<kLihModes>(data, MPI_COMM_WORLD, atol, std::nullopt, graph_path);
            gate_bit = gate_bit || n_serial < n_ungated;
            // Reported, not just asserted: BOOST_TEST_CONTEXT only prints on failure, so without this a
            // passing run cannot show how much asymmetry it actually exercised -- and "the gate bit" is
            // the one fact that decides whether this case is worth anything.
            BOOST_TEST_MESSAGE("lower_atol=" << atol << (graph_path ? " build_graph" : " propagate")
                                             << " ungated=" << n_ungated << " gated=" << n_serial
                                             << " dropped=" << (n_ungated - n_serial));
            BOOST_TEST_CONTEXT("lower_atol=" << atol << (graph_path ? " build_graph" : " propagate")
                                             << " n_serial=" << n_serial << " n_world=" << n_world) {
                // Energy is tolerance-equal across rank counts, never bit-equal: the reduction order
                // differs. The term count, by contrast, must agree exactly -- a rotation dropped or
                // double-counted by the asymmetric path shows up here and essentially nowhere else.
                BOOST_TEST(near(e_serial, e_world));
                BOOST_CHECK_EQUAL(n_serial, n_world);
            }
        }
    }

    // A length cap gates on the generator's length rather than on a coefficient, so it drives the same
    // protocol with a different survivor set.
    for (const size_t cap : {size_t{2}, size_t{3}}) {
        for (const bool graph_path : {true, false}) {
            const auto [e_serial, n_serial] = run_gated<kLihModes>(data, MPI_COMM_SELF, 1e-3, cap, graph_path);
            const auto [e_world, n_world] = run_gated<kLihModes>(data, MPI_COMM_WORLD, 1e-3, cap, graph_path);
            if (graph_path) {
                graph_gate_bit = graph_gate_bit || n_serial < n_ungated;
            }
            BOOST_TEST_MESSAGE("cap=" << cap << (graph_path ? " build_graph" : " propagate") << " ungated=" << n_ungated
                                      << " gated=" << n_serial << " dropped=" << (n_ungated - n_serial));
            BOOST_TEST_CONTEXT("only_rotate_len_k=" << cap << (graph_path ? " build_graph" : " propagate")
                                                    << " n_serial=" << n_serial << " n_world=" << n_world) {
                BOOST_TEST(near(e_serial, e_world));
                BOOST_CHECK_EQUAL(n_serial, n_world);
            }
        }
    }

    BOOST_CHECK_MESSAGE(gate_bit,
                        "no lower_atol reduced the term count below the ungated " << n_ungated
                                                                                  << ": the gate never bit, so this "
                                                                                     "case exercised no asymmetry");
    // Measured on LiH/n=12: lower_atol drops 208-232 of 866 on the propagate path and *nothing* on the
    // build_graph path, while the length cap drops 32 on both. So this second check is what keeps
    // GraphSink covered, and it fails independently of the one above -- if a fixture or default change
    // ever makes the cap stop biting, the loss of coverage is reported rather than silent.
    BOOST_CHECK_MESSAGE(graph_gate_bit,
                        "no gate reduced build_graph's term count below the ungated "
                            << n_ungated << ": the GraphSink path was not exercised asymmetrically");
}

} // namespace
