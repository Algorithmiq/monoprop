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

// Multi-rank equivalence for the Schrödinger fresh-insert arm of ContractSink::on_resolved, where a
// fresh partner is state-scored (majorana_state_phase / pauli_state_phase) rather than left at 0. The
// Heisenberg R>1 resolve/apply paths are already covered by exact_upper_atol_rescue and
// mpi_distributed_layer_equivalence. Only runs at world >= 2. Oracle: serial<->world equivalence --
// the deterministic base+j miss-prefix must sum the same terms at any rank count, to near()'s rtol. In a sharded
// prototype build the same public calls drive the sharded root at P ranks x the launch's T (cpp/tests/CMakeLists.txt).

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <bit>
#include <complex>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "KernelTestSupport.h"
#include "PauliTestOracle.h"
#include "PropagatorTestAccess.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace {

using namespace monoprop;
using namespace test_utils;
using pauli_oracle::slots_of_string;

// schrodinger_cutoff engages the picture; a low structural cutoff plus the upper_atol = 0 rescue
// forces most partners to be fresh inserts, so the miss arm runs on nearly every partner.
template <size_t NumModes>
auto run_schrodinger_majorana(const CaseData &data, MPI_Comm comm) -> double {
    MonomialPropagator<NumModes> sim(data.hamiltonian,
                                     /*cutoff=*/2U,
                                     data.initial_state,
                                     /*schrodinger_cutoff=*/std::optional<unsigned int>{4U},
                                     comm,
                                     /*lower_atol=*/std::nullopt,
                                     /*upper_atol=*/std::optional<double>{0.0},
                                     CutoffType::Length,
                                     /*basis_change=*/std::nullopt);
    sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
    auto energy_fn = sim.expectation_value_functional(std::nullopt);
    return energy_fn(VecD{});
}

BOOST_FIXTURE_TEST_CASE(mpi_fresh_insert_schrodinger_majorana_serial_world_equiv, ExampleDataFix) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping Schrödinger Majorana fresh-insert equivalence (world size = 1).");
        return;
    }
    const double e_serial = run_schrodinger_majorana<ExampleDataFix::n_modes>(data, MPI_COMM_SELF);
    const double e_world = run_schrodinger_majorana<ExampleDataFix::n_modes>(data, MPI_COMM_WORLD);
    BOOST_TEST_MESSAGE("schrodinger majorana serial=" << e_serial << " world=" << e_world);
    BOOST_TEST(near(e_serial, e_world));
}

// Drives the pauli_state_phase sub-branch of the same miss arm: a hand Pauli operator with X / ZZ
// generator layers forces fresh paired cross-rank inserts.
constexpr size_t kPauliQ = 6;

auto run_schrodinger_pauli(MPI_Comm comm) -> double {
    OperatorDict init;
    init[slots_of_string("ZIIIII")] = std::complex<double>(1.0, 0.0);
    init[slots_of_string("IIZZII")] = std::complex<double>(0.5, 0.0);
    MonomialPropagator<kPauliQ> sim(init,
                                    /*cutoff=*/2U,
                                    VecZ{},
                                    /*schrodinger_cutoff=*/std::optional<unsigned int>{4U},
                                    comm,
                                    /*lower_atol=*/std::nullopt,
                                    /*upper_atol=*/std::optional<double>{0.0},
                                    CutoffType::Support,
                                    /*basis_change=*/std::nullopt,
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

BOOST_AUTO_TEST_CASE(mpi_fresh_insert_schrodinger_pauli_serial_world_equiv) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping Schrödinger Pauli fresh-insert equivalence (world size = 1).");
        return;
    }
    const double e_serial = run_schrodinger_pauli(MPI_COMM_SELF);
    const double e_world = run_schrodinger_pauli(MPI_COMM_WORLD);
    BOOST_TEST_MESSAGE("schrodinger pauli serial=" << e_serial << " world=" << e_world);
    BOOST_TEST(near(e_serial, e_world));
}

#ifndef monoprop_SHARDED_OPENMP_PROTOTYPE
// The one-store prototype's cases below construct with explicit partitions = 1, which the sharded root rejects.

// ── Threaded resolution at fixed geometry (Task 6) ────────────────────────────────────────────────
//
// The one-store prototype (explicit partitions=1 on an ordinary communicator) at launch budgets 2-4 against
// budget 1, at this launch's world size: every rank's rows in row-ID order and its coefficients bit for bit.
// The Schrodinger fresh-insert arm keeps its scatter serial (state scoring), while the probe, decode and
// publication around it are the threaded ones; Heisenberg graph builds use the parallel scatter.

template <size_t NumModes>
auto same_store(const MonomialPropagator<NumModes> &a, const MonomialPropagator<NumModes> &b) -> bool {
    const auto &sa = *a.mp_op().store;
    const auto &sb = *b.mp_op().store;
    if (sa.size() != sb.size()) {
        return false;
    }
    for (size_t i = 0; i < sa.size(); ++i) {
        if (materialize_row<NumModes>(sa, i) != materialize_row<NumModes>(sb, i)) {
            return false;
        }
    }
    return true;
}

auto same_bits(const VecD &a, const VecD &b) -> bool {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

template <size_t NumModes>
auto prototype(const OperatorDict &ham,
               unsigned cutoff,
               const VecZ &state,
               std::optional<unsigned> schrodinger_cutoff,
               std::optional<double> upper_atol,
               int budget,
               Basis basis = Basis::Majorana,
               CutoffType cutoff_type = CutoffType::Length) -> MonomialPropagator<NumModes> {
    const kernel_test::ScopedBudget scoped(budget);
    MonomialPropagator<NumModes> sim(ham,
                                     cutoff,
                                     state,
                                     schrodinger_cutoff,
                                     MPI_COMM_WORLD,
                                     std::nullopt,
                                     upper_atol,
                                     cutoff_type,
                                     std::nullopt,
                                     NumModes,
                                     basis,
                                     /*partitions=*/1);
    BOOST_TEST_REQUIRE(monoprop::detail::PropagatorTestAccess<NumModes>::options(sim).threads == budget);
    return sim;
}

BOOST_FIXTURE_TEST_CASE(openmp_fresh_insert_threaded_budgets_match_serial_exactly, ExampleDataFix) {
    constexpr size_t N = ExampleDataFix::n_modes;
    using Access = monoprop::detail::PropagatorTestAccess<N>;
    const auto run = [&](int budget, bool schrodinger) {
        auto sim = prototype<N>(data.hamiltonian,
                                schrodinger ? 2U : 2U * N,
                                data.initial_state,
                                schrodinger ? std::optional<unsigned>{4U} : std::nullopt,
                                std::optional<double>{0.0},
                                budget);
        sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters);
        return sim;
    };
    for (const bool schrodinger : {false, true}) {
        auto serial = run(1, schrodinger);
        const double e_serial = serial.expectation_value({});
        for (const int budget : {2, 3, 4}) {
            BOOST_TEST_CONTEXT("schrodinger=" << schrodinger << " budget=" << budget
                                              << " world=" << mpi::size(MPI_COMM_WORLD)) {
                auto threaded = run(budget, schrodinger);
                BOOST_TEST(same_store<N>(serial, threaded));
                BOOST_TEST(same_bits(Access::picture_coeffs(serial), Access::picture_coeffs(threaded)));
                BOOST_TEST(std::bit_cast<uint64_t>(threaded.expectation_value({}))
                           == std::bit_cast<uint64_t>(e_serial));
            }
        }
    }
}

// A 16-mode operator large enough that every rank receives thousands of remote queries per pass.
constexpr size_t kWideModes = 16;

auto wide_hamiltonian() -> OperatorDict {
    OperatorDict ham;
    kernel_test::SplitMix rng{0x7A5C6ULL};
    while (ham.size() < 90000) {
        VecZ idx;
        const size_t weight = 2 * (1 + rng.below(3));
        while (idx.size() < weight) {
            const size_t m = rng.below(2 * kWideModes);
            if (std::ranges::find(idx, m) == idx.end()) {
                idx.push_back(m);
            }
        }
        std::ranges::sort(idx);
        const double x = static_cast<double>(rng.below(1U << 20)) / (1U << 20) - 0.5;
        ham[idx] = x * hermitian_coefficient<kWideModes>(indices_to_bitset<kWideModes>(idx));
    }
    return ham;
}

// Under an MPI launch the real build_layer's incoming decode, frozen probe and GraphSink scatter each run in
// their own region with several workers. Linear routing sends some generators only to self, so the case walks
// generators until one crosses the wire. Serial ctest (world 1) has no incoming queries and skips.
BOOST_AUTO_TEST_CASE(openmp_incoming_resolution_workers_participate_at_world_size,
                     *boost::unit_test::precondition(kernel_test::runtime_offers_two_workers)) {
    if (mpi::size(MPI_COMM_WORLD) < 2) {
        BOOST_TEST_MESSAGE("Skipping incoming-resolution participation (world size = 1).");
        return;
    }
    using Access = monoprop::detail::PropagatorTestAccess<kWideModes>;
    const auto ham = wide_hamiltonian();
    auto sim = prototype<kWideModes>(ham, 10U, VecZ{0, 2, 4, 6}, std::nullopt, std::nullopt, 4);
    const std::vector<VecZ> gens{{0, 5}, {1, 2}, {3, 9, 12, 20}, {4, 7}, {6, 11}, {8, 13, 17, 30}};
    bool observed = false;
    for (const auto &gen : gens) {
        kernel_test::PhaseLog log;
        (void)Access::build_layer_observed(sim, gen, kernel_test::AccumulatingObserver{&log});
        const auto decode = log.of(monoprop::detail::KernelRange::decode);
        const auto probe = log.of(monoprop::detail::KernelRange::incoming_probe);
        const auto scatter = log.of(monoprop::detail::KernelRange::scatter);
        // Both passes run the three phases, in the same order, whenever this generator crosses ranks.
        BOOST_TEST(decode.size() == probe.size());
        BOOST_TEST(decode.size() == scatter.size());
        for (size_t pass = 0; pass < decode.size(); ++pass) {
            if (decode[pass]->worker.size() >= 4 && scatter[pass]->worker.size() >= 4) {
                kernel_test::check_participation(kernel_test::slots_of(*decode[pass]),
                                                 decode[pass]->worker.size(),
                                                 "decode");
                kernel_test::check_participation(kernel_test::slots_of(*probe[pass]),
                                                 probe[pass]->worker.size(),
                                                 "incoming probe");
                kernel_test::check_participation(kernel_test::slots_of(*scatter[pass]),
                                                 scatter[pass]->worker.size(),
                                                 "scatter");
                observed = true;
            }
        }
        if (observed) {
            break;
        }
    }
    BOOST_TEST(observed);
    BOOST_TEST(!Access::is_invalid(sim));
}

#endif // monoprop_SHARDED_OPENMP_PROTOTYPE

} // namespace
