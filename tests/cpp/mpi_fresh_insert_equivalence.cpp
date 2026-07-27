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
// the deterministic base+j miss-prefix must sum the same terms at any rank count, to near()'s rtol.

#include <boost/test/unit_test.hpp>

#include <complex>
#include <optional>
#include <string>
#include <vector>

#include "PauliTestOracle.h"
#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace {

using namespace monoprop;
using namespace test_utils;
using pauli_oracle::slots_of_string;

// ── Majorana, Schrödinger picture, coefficient-carrying (fused) propagate ────────────────────────
// schrodinger_cutoff engages the picture; a low structural cutoff plus the upper_atol = 0 rescue
// forces most partners to be FRESH inserts, so the miss arm runs on nearly every partner.
template <size_t NumModes>
auto run_schrodinger_majorana(const CaseData& data, MPI_Comm comm) -> double {
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

// ── Native Pauli, Schrödinger picture, fused propagate ───────────────────────────────────────────
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

} // namespace
