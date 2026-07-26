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
#include <complex>
#include <map>
#include <stdexcept>
#include <string>

#include "PauliTestOracle.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// Intra-process shard equivalence: a propagator built with shards>1 (S single-threaded shard
// propagators over an in-process ShmComm) must agree with the ordinary single-partition propagator
// within floating-point accumulation tolerance — the same standard the MPI rank-count test uses. This
// is also the first C++ coverage of the native Pauli engine at S>1. Runs in the serial build (pure
// std::thread; no mpiexec).

namespace {

using namespace monoprop;
using namespace test_utils;
using pauli_oracle::slots_of_string;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 4;

// ─── Majorana (fixture-driven) ───────────────────────────────────────────────

auto majorana_sim(const CaseData &data, size_t shards) -> MonomialPropagator<kNumModes> {
    return MonomialPropagator<kNumModes>(data.hamiltonian,
                                         kCutoff,
                                         data.hartree_fock,
                                         std::nullopt,
                                         MPI_COMM_SELF,
                                         std::nullopt,
                                         std::nullopt,
                                         CutoffType::Length,
                                         std::nullopt,
                                         kNumModes,
                                         Basis::Majorana,
                                         shards);
}

BOOST_AUTO_TEST_CASE(shard_majorana_energy_matches_across_shard_counts) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto ref = majorana_sim(data, 1);
    ref.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double e1 = ref.expectation_value(data.parameters);

    for (const size_t S : {size_t{2}, size_t{4}}) {
        auto sim = majorana_sim(data, S);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        const double e = sim.expectation_value(data.parameters);
        BOOST_TEST_CONTEXT("shards=" << S << " e=" << e << " ref=" << e1) {
            BOOST_TEST(near(e1, e));
        }
        // Aggregated operator size is shard-count invariant (terms are hash-partitioned, no overlap).
        BOOST_CHECK_EQUAL(ref.size(), sim.size());
    }
}

BOOST_AUTO_TEST_CASE(shard_majorana_gradient_matches_across_shard_counts) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto ref = majorana_sim(data, 1);
    ref.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const auto g1 = ref.expectation_value_and_gradient(data.parameters).second;

    for (const size_t S : {size_t{2}, size_t{4}}) {
        auto sim = majorana_sim(data, S);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        const auto g = sim.expectation_value_and_gradient(data.parameters).second;
        BOOST_REQUIRE_EQUAL(g1.size(), g.size());
        for (size_t i = 0; i < g1.size(); ++i) {
            BOOST_TEST_CONTEXT("shards=" << S << " grad idx=" << i) {
                BOOST_TEST(near(g1[i], g[i]));
            }
        }
    }
}

// propagate() (contract-immediately, the benchmark path) then expectation_value of the contracted
// operator must match S=1.
BOOST_AUTO_TEST_CASE(shard_majorana_propagate_then_expectation_matches) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto run = [&](size_t S) {
        auto sim = majorana_sim(data, S);
        sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
        return std::pair<double, size_t>{sim.expectation_value({}), sim.size()};
    };
    const auto [e1, n1] = run(1);
    for (const size_t S : {size_t{2}, size_t{4}}) {
        const auto [e, n] = run(S);
        BOOST_TEST_CONTEXT("shards=" << S) {
            BOOST_TEST(near(e1, e));
            BOOST_CHECK_EQUAL(n1, n);
        }
    }
}

// Two independent S=4 runs are bit-identical: ShmComm sums in fixed rank order and each shard is
// deterministic, so a given shard count has no run-to-run jitter.
BOOST_AUTO_TEST_CASE(shard_energy_is_deterministic) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto energy_s4 = [&] {
        auto sim = majorana_sim(data, 4);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        return sim.expectation_value(data.parameters);
    };
    BOOST_CHECK_EQUAL(energy_s4(), energy_s4());
}

// A deep copy of a shard-backed propagator (clones the whole group) evaluates identically.
BOOST_AUTO_TEST_CASE(shard_deep_copy_matches) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = majorana_sim(data, 4);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double e = sim.expectation_value(data.parameters);

    MonomialPropagator<kNumModes> copy(sim); // clones the shard group (fresh threads + ShmComm)
    const double e_copy = copy.expectation_value(data.parameters);
    BOOST_CHECK_EQUAL(e, e_copy);
    BOOST_CHECK_EQUAL(sim.size(), copy.size());
}

// ─── Native Pauli (inline circuit) ───────────────────────────────────────────
// Pauli strings map to Majorana-slot index vectors via pauli_oracle::slots_of_string.

constexpr size_t kNq = 6; // qubits for the Pauli case

auto pauli_sim(const std::map<std::string, double> &obs, size_t shards) -> MonomialPropagator<kNq> {
    FermiOperatorMap init;
    for (const auto &[p, c] : obs) {
        init[slots_of_string(p)] = std::complex<double>(c, 0.0);
    }
    return MonomialPropagator<kNq>(init,
                                   /*cutoff=*/kNq,
                                   /*slater=*/{},
                                   std::nullopt,
                                   MPI_COMM_SELF,
                                   /*lower_atol=*/1e-12,
                                   std::nullopt,
                                   CutoffType::Support,
                                   std::nullopt,
                                   kNq,
                                   Basis::Pauli,
                                   shards);
}

// A kicked-Ising-style layer: transverse X rotations then ZZ couplings, driven at fixed angles.
auto run_pauli_energy(size_t shards) -> std::pair<double, size_t> {
    std::map<std::string, double> obs;
    obs["ZIIIII"] = 1.0;
    obs["IIZZII"] = 0.5;
    auto sim = pauli_sim(obs, shards);

    std::vector<VecZ> gens;
    VecZ pmap;
    VecD gcoeffs;
    size_t p = 0;
    for (size_t q = 0; q < kNq; ++q) { // X on each qubit
        std::string s(kNq, 'I');
        s[q] = 'X';
        gens.push_back(slots_of_string(s));
        pmap.push_back(p++);
        gcoeffs.push_back(1.0);
    }
    for (size_t q = 0; q + 1 < kNq; ++q) { // ZZ on neighbours
        std::string s(kNq, 'I');
        s[q] = 'Z';
        s[q + 1] = 'Z';
        gens.push_back(slots_of_string(s));
        pmap.push_back(p++);
        gcoeffs.push_back(1.0);
    }
    VecD params(p, 0.3);
    sim.propagate(gens, pmap, gcoeffs, params);
    return {sim.expectation_value({}), sim.size()};
}

BOOST_AUTO_TEST_CASE(shard_pauli_energy_matches_across_shard_counts) {
    const auto [e1, n1] = run_pauli_energy(1);
    for (const size_t S : {size_t{2}, size_t{4}}) {
        const auto [e, n] = run_pauli_energy(S);
        BOOST_TEST_CONTEXT("pauli shards=" << S << " e=" << e << " ref=" << e1) {
            BOOST_TEST(near(e1, e));
            BOOST_CHECK_EQUAL(n1, n);
        }
    }
}

} // namespace

// A shard factory that throws must surface the exception, not std::terminate. The ctor starts the
// master threads BEFORE building the shards on them (for first-touch locality), so an escaping
// exception used to unwind past joinable threads without ~ShardGroup ever setting stop_. Every
// MonomialPropagator ctor validation reaches this path, as does an allocation failure.
BOOST_AUTO_TEST_CASE(shard_factory_exception_propagates_without_terminate) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    // logical_num_modes = 0 is rejected by each shard's own constructor, on its own master thread.
    BOOST_CHECK_THROW(MonomialPropagator<kNumModes>(data.hamiltonian,
                                                    kCutoff,
                                                    data.hartree_fock,
                                                    std::nullopt,
                                                    MPI_COMM_SELF,
                                                    std::nullopt,
                                                    std::nullopt,
                                                    CutoffType::Length,
                                                    std::nullopt,
                                                    /*logical_num_modes=*/0,
                                                    Basis::Majorana,
                                                    /*shards=*/4),
                      std::runtime_error);

    // An out-of-range operator index takes the same path, and the group stays usable afterwards.
    auto bad_op = data.hamiltonian;
    bad_op[VecZ{2 * kNumModes}] = std::complex<double>(1.0, 0.0);
    BOOST_CHECK_THROW(MonomialPropagator<kNumModes>(bad_op,
                                                    kCutoff,
                                                    data.hartree_fock,
                                                    std::nullopt,
                                                    MPI_COMM_SELF,
                                                    std::nullopt,
                                                    std::nullopt,
                                                    CutoffType::Length,
                                                    std::nullopt,
                                                    kNumModes,
                                                    Basis::Majorana,
                                                    /*shards=*/4),
                      std::runtime_error);
    BOOST_CHECK_NO_THROW(majorana_sim(data, 4));
}
