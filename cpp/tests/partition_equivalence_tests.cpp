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
#include <cmath>
#include <complex>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "PauliTestOracle.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// Intra-process partition equivalence: a propagator with partitions>1 (S partition propagators over an in-process
// ShmComm) must match the ordinary single-partition propagator within fp accumulation tolerance.
// Oracle: the S=1 run. Pure std::thread, so this runs in the serial build with no mpiexec.

namespace {

using namespace monoprop;
using namespace test_utils;
using pauli_oracle::slots_of_string;

constexpr size_t kNumModes = 8;
constexpr unsigned int kCutoff = 4;

auto majorana_sim(const CaseData &data, size_t partitions, std::optional<double> lower_atol = std::nullopt)
    -> MonomialPropagator {
    return test_utils::make_propagator<kNumModes>(data.hamiltonian,
                                                  kCutoff,
                                                  data.initial_state,
                                                  std::nullopt,
                                                  MPI_COMM_SELF,
                                                  lower_atol,
                                                  std::nullopt,
                                                  CutoffType::Length,
                                                  std::nullopt,
                                                  kNumModes,
                                                  Basis::Majorana,
                                                  partitions);
}

BOOST_AUTO_TEST_CASE(partition_majorana_energy_matches_across_partition_counts) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto ref = majorana_sim(data, 1);
    ref.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double e1 = ref.expectation_value(data.parameters);

    for (const size_t S : {size_t{2}, size_t{4}}) {
        auto sim = majorana_sim(data, S);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        const double e = sim.expectation_value(data.parameters);
        BOOST_TEST_CONTEXT("partitions=" << S << " e=" << e << " ref=" << e1) {
            BOOST_TEST(near(e1, e));
        }
        // Aggregated operator size is partition-count invariant (terms are hash-partitioned, no overlap).
        BOOST_CHECK_EQUAL(ref.size(), sim.size());
    }
}

BOOST_AUTO_TEST_CASE(partition_majorana_gradient_matches_across_partition_counts) {
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
            BOOST_TEST_CONTEXT("partitions=" << S << " grad idx=" << i) {
                BOOST_TEST(near(g1[i], g[i]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(partition_majorana_propagate_then_expectation_matches) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto run = [&](size_t S) {
        auto sim = majorana_sim(data, S);
        sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
        return std::pair<double, size_t>{sim.expectation_value({}), sim.size()};
    };
    const auto [e1, n1] = run(1);
    for (const size_t S : {size_t{2}, size_t{4}}) {
        const auto [e, n] = run(S);
        BOOST_TEST_CONTEXT("partitions=" << S) {
            BOOST_TEST(near(e1, e));
            BOOST_CHECK_EQUAL(n1, n);
        }
    }
}

// Two independent S=4 runs are bit-identical: ShmComm sums in ascending rank order and each partition is
// deterministic, so a given partition count has no run-to-run jitter.
BOOST_AUTO_TEST_CASE(partition_energy_is_deterministic) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto energy_s4 = [&] {
        auto sim = majorana_sim(data, 4);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        return sim.expectation_value(data.parameters);
    };
    BOOST_CHECK_EQUAL(energy_s4(), energy_s4());
}

// contract_partially() hands back raw coefficients positioned by the owning partition's own indexing,
// so a facade's array is the per-partition blocks concatenated in partition order. What IS invariant is
// the multiset: the partitions are disjoint and cover every term. Sorting both sides is the only
// comparison the API's contract supports — see the note on contract_partially().
BOOST_AUTO_TEST_CASE(partition_contract_partially_matches_as_a_multiset) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sorted_coeffs = [&](size_t S) {
        auto sim = majorana_sim(data, S);
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        auto coeffs = sim.contract_partially(data.parameters, /*inplace=*/false);
        std::ranges::sort(coeffs);
        return coeffs;
    };
    const auto ref = sorted_coeffs(1);
    for (const size_t S : {size_t{2}, size_t{4}}) {
        const auto coeffs = sorted_coeffs(S);
        BOOST_REQUIRE_EQUAL(ref.size(), coeffs.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            BOOST_TEST_CONTEXT("partitions=" << S << " sorted idx=" << i) {
                BOOST_TEST(near(ref[i], coeffs[i]));
            }
        }
    }
}

// The raw per-partition accessors have no facade reading: the facade's own graph_/mp_op_ are never
// populated, so returning them would hand a C++ consumer empty state that looks valid.
BOOST_AUTO_TEST_CASE(partition_raw_accessors_reject_a_facade) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = majorana_sim(data, 4);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_CHECK_THROW(static_cast<void>(sim.graph()), std::runtime_error);
    BOOST_CHECK_THROW(static_cast<void>(sim.mp_op()), std::runtime_error);
    BOOST_CHECK_THROW(static_cast<void>(sim.num_local_terms()), std::runtime_error);
    BOOST_CHECK_THROW(static_cast<void>(sim.graph_data()), std::runtime_error);

    auto solo = majorana_sim(data, 1);
    solo.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_CHECK_GT(solo.graph().layers(), 0U);
    BOOST_CHECK_EQUAL(solo.graph_data().size(), solo.graph_layers());
}

// A facade owns no operator, so a setter that stopped there would leave every partition on its old
// configuration and the run would silently ignore the new value. The cutoff truncates structurally
// during the graph build, so it shows up in the aggregated term count. LiH is used rather than the
// random_exact fixture, which is small enough that no setting truncates anything. The tightened
// oracle must itself differ from the wide run, else the last assertion would hold vacuously.
BOOST_AUTO_TEST_CASE(partition_setters_reach_every_partition) {
    constexpr size_t kLihModes = LihFixture::n_modes;
    const auto data = load_case_data<kLihModes>("lih_fermionic_spin_exact.msgpack");

    auto build = [&](unsigned int cutoff, unsigned int updated_cutoff) {
        auto sim = test_utils::make_propagator<kLihModes>(data.hamiltonian,
                                                          cutoff,
                                                          data.initial_state,
                                                          std::nullopt,
                                                          MPI_COMM_SELF,
                                                          std::nullopt,
                                                          std::nullopt,
                                                          CutoffType::Length,
                                                          std::nullopt,
                                                          kLihModes,
                                                          Basis::Majorana,
                                                          /*partitions=*/4);
        if (cutoff != updated_cutoff) {
            sim.update_cutoff(updated_cutoff);
            BOOST_CHECK_EQUAL(sim.cutoff(), updated_cutoff);
        }
        sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
        return sim.size();
    };

    const size_t n_wide = build(2 * kLihModes, 2 * kLihModes);
    const size_t n_tight = build(4, 4);
    BOOST_REQUIRE_NE(n_wide, n_tight);
    // Constructed wide, then tightened: only the partitions' own cutoff can produce the tight count.
    BOOST_CHECK_EQUAL(n_tight, build(2 * kLihModes, 4));
}

BOOST_AUTO_TEST_CASE(partition_deep_copy_matches) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    auto sim = majorana_sim(data, 4);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const double e = sim.expectation_value(data.parameters);

    MonomialPropagator copy(sim); // clones the partition group (fresh threads + ShmComm)
    const double e_copy = copy.expectation_value(data.parameters);
    BOOST_CHECK_EQUAL(e, e_copy);
    BOOST_CHECK_EQUAL(sim.size(), copy.size());
}

constexpr size_t kNq = 6;

auto pauli_sim(const std::map<std::string, double> &obs, size_t partitions) -> MonomialPropagator {
    OperatorDict init;
    for (const auto &[p, c] : obs) {
        init[slots_of_string(p)] = std::complex<double>(c, 0.0);
    }
    return test_utils::make_propagator<kNq>(init,
                                            /*cutoff=*/kNq,
                                            /*initial_state=*/{},
                                            std::nullopt,
                                            MPI_COMM_SELF,
                                            /*lower_atol=*/1e-12,
                                            std::nullopt,
                                            CutoffType::Support,
                                            std::nullopt,
                                            kNq,
                                            Basis::Pauli,
                                            partitions);
}

auto run_pauli_energy(size_t partitions) -> std::pair<double, size_t> {
    std::map<std::string, double> obs;
    obs["ZIIIII"] = 1.0;
    obs["IIZZII"] = 0.5;
    auto sim = pauli_sim(obs, partitions);

    std::vector<VecZ> gens;
    VecZ pmap;
    VecD gcoeffs;
    size_t p = 0;
    for (size_t q = 0; q < kNq; ++q) {
        std::string s(kNq, 'I');
        s[q] = 'X';
        gens.push_back(slots_of_string(s));
        pmap.push_back(p++);
        gcoeffs.push_back(1.0);
    }
    for (size_t q = 0; q + 1 < kNq; ++q) {
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

BOOST_AUTO_TEST_CASE(partition_pauli_energy_matches_across_partition_counts) {
    const auto [e1, n1] = run_pauli_energy(1);
    for (const size_t S : {size_t{2}, size_t{4}}) {
        const auto [e, n] = run_pauli_energy(S);
        BOOST_TEST_CONTEXT("pauli partitions=" << S << " e=" << e << " ref=" << e1) {
            BOOST_TEST(near(e1, e));
            BOOST_CHECK_EQUAL(n1, n);
        }
    }
}

} // namespace

// A partition factory that throws must surface the exception, not std::terminate: the ctor starts the
// master threads before building the partitions on them (first-touch locality), so the unwind has to join
// already-started threads. Every MonomialPropagator ctor validation reaches this path.
BOOST_AUTO_TEST_CASE(partition_factory_exception_propagates_without_terminate) {
    const auto data = load_case_data<kNumModes>("random_exact.msgpack");
    // logical_num_modes = 0 is rejected by each partition's own constructor, on its own master thread.
    BOOST_CHECK_THROW(test_utils::make_propagator<kNumModes>(data.hamiltonian,
                                                             kCutoff,
                                                             data.initial_state,
                                                             std::nullopt,
                                                             MPI_COMM_SELF,
                                                             std::nullopt,
                                                             std::nullopt,
                                                             CutoffType::Length,
                                                             std::nullopt,
                                                             /*logical_num_modes=*/0,
                                                             Basis::Majorana,
                                                             /*partitions=*/4),
                      std::runtime_error);

    // An out-of-range operator index takes the same path, and the group stays usable afterwards.
    auto bad_op = data.hamiltonian;
    bad_op[VecZ{2 * kNumModes}] = std::complex<double>(1.0, 0.0);
    BOOST_CHECK_THROW(test_utils::make_propagator<kNumModes>(bad_op,
                                                             kCutoff,
                                                             data.initial_state,
                                                             std::nullopt,
                                                             MPI_COMM_SELF,
                                                             std::nullopt,
                                                             std::nullopt,
                                                             CutoffType::Length,
                                                             std::nullopt,
                                                             kNumModes,
                                                             Basis::Majorana,
                                                             /*partitions=*/4),
                      std::runtime_error);
    BOOST_CHECK_NO_THROW(majorana_sim(data, 4));
}
