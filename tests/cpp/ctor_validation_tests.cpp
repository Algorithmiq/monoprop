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

// Coverage of the MonomialPropagator constructor/API guard rails (ctor argument validation, the
// operator-index range check, the propagate()-on-a-stored-graph guard, and MPGraph::get_layer bounds).
// These throw paths define the public contract.

#include <boost/test/unit_test.hpp>

#include <complex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;
using test_utils::build_simulator;
using test_utils::ExampleDataFix;
using test_utils::SimulatorConfig;

namespace {
constexpr size_t N = 8;
using MP = MonomialPropagator<N>;

// Construct with the full argument list; individual cases vary just the field(s) under test.
auto make(const OperatorDict &op,
          unsigned int cutoff = 2 * N,
          std::optional<double> lower_atol = std::nullopt,
          std::optional<double> upper_atol = std::nullopt,
          CutoffType cutoff_type = CutoffType::Length,
          std::optional<std::vector<VecZ>> basis_change = std::nullopt,
          size_t logical_num_modes = N,
          Basis basis = Basis::Majorana) -> MP {
    return MP(op,
              cutoff,
              VecZ{},
              std::nullopt,
              MPI_COMM_SELF,
              lower_atol,
              upper_atol,
              cutoff_type,
              basis_change,
              logical_num_modes,
              basis);
}
} // namespace

// A minimal valid configuration must construct without throwing.
BOOST_AUTO_TEST_CASE(ctor_accepts_valid_config) {
    BOOST_CHECK_NO_THROW(make(OperatorDict{}));
}

BOOST_AUTO_TEST_CASE(ctor_logical_num_modes_out_of_range_throws) {
    BOOST_CHECK_THROW(make(OperatorDict{},
                           2 * N,
                           std::nullopt,
                           std::nullopt,
                           CutoffType::Length,
                           std::nullopt,
                           /*logical=*/0),
                      std::runtime_error);
    BOOST_CHECK_THROW(make(OperatorDict{},
                           2 * N,
                           std::nullopt,
                           std::nullopt,
                           CutoffType::Length,
                           std::nullopt,
                           /*logical=*/N + 1),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(ctor_pauli_requires_support_cutoff_throws) {
    // Pauli basis + Length cutoff is rejected (Length has no Pauli-weight meaning).
    BOOST_CHECK_THROW(
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Length, std::nullopt, N, Basis::Pauli),
        std::invalid_argument);
    // Pauli basis + Support cutoff is fine.
    BOOST_CHECK_NO_THROW(
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Support, std::nullopt, N, Basis::Pauli));
}

BOOST_AUTO_TEST_CASE(ctor_pauli_forbids_basis_change_throws) {
    const std::vector<VecZ> some_basis(2 * N, VecZ{0});
    BOOST_CHECK_THROW(
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Support, some_basis, N, Basis::Pauli),
        std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ctor_upper_atol_below_lower_atol_throws) {
    BOOST_CHECK_THROW(make(OperatorDict{}, 2 * N, /*lower=*/1e-6, /*upper=*/1e-8), std::runtime_error);
    // upper >= lower is accepted.
    BOOST_CHECK_NO_THROW(make(OperatorDict{}, 2 * N, /*lower=*/1e-8, /*upper=*/1e-6));
}

BOOST_AUTO_TEST_CASE(ctor_operator_index_out_of_range_throws) {
    OperatorDict op;
    op[VecZ{20}] = std::complex<double>(1.0, 0.0); // 20 >= 2*logical (=16)
    BOOST_CHECK_THROW(make(op), std::runtime_error);
}

// A gate generator index outside the system is rejected too. Nothing between the public
// build_graph/propagate entry points and indices_to_bitset used to constrain a generator, so an
// out-of-range index underflowed 2*NumModes-1-index and wrote out of bounds through Bitset::set.
BOOST_AUTO_TEST_CASE(build_graph_generator_index_out_of_range_throws) {
    OperatorDict op;
    op[VecZ{0, 1}] = std::complex<double>(0.0, 1.0);
    auto sim = make(op);
    // 2*logical_num_modes == 16, so slot 20 is outside this system.
    BOOST_CHECK_THROW(sim.build_graph({VecZ{20, 21}}, VecZ{0}, VecD{1.0}), std::runtime_error);
    // A generator inside the system still builds.
    BOOST_CHECK_NO_THROW(sim.build_graph({VecZ{0, 3}}, VecZ{0}, VecD{1.0}));
}

// A propagator over fewer LOGICAL modes than its instantiation must reject indices outside its own
// system, not merely outside the storage width.
BOOST_AUTO_TEST_CASE(generator_index_bound_is_logical_not_storage) {
    OperatorDict op;
    op[VecZ{0, 1}] = std::complex<double>(0.0, 1.0);
    auto sim = make(op, 2 * N, std::nullopt, std::nullopt, CutoffType::Length, std::nullopt, /*logical=*/4);
    // 2*logical == 8 <= slot 9 < 2*N == 16: inside the storage, outside the system.
    BOOST_CHECK_THROW(sim.build_graph({VecZ{9}}, VecZ{0}, VecD{1.0}), std::runtime_error);
}

// The update_* setters must enforce the same invariants the constructor does. They wrote straight
// through to regenerate_cutoff_fn_(), so a Pauli propagator could be given a Length cutoff or a
// basis change it rejects at construction, and a short basis_change read out of bounds.
BOOST_AUTO_TEST_CASE(setters_enforce_the_constructor_invariants) {
    auto pauli =
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Support, std::nullopt, N, Basis::Pauli);
    BOOST_CHECK_THROW(pauli.update_cutoff_type(CutoffType::Length), std::invalid_argument);
    BOOST_CHECK_THROW(pauli.update_basis_change(std::vector<VecZ>(2 * N, VecZ{0})), std::invalid_argument);

    auto majorana = make(OperatorDict{});
    // Too few rows: regenerate_cutoff_fn_ indexes [0, 2*logical_num_modes) unconditionally.
    BOOST_CHECK_THROW(majorana.update_basis_change(std::vector<VecZ>{VecZ{0}}), std::invalid_argument);
    // A row naming a slot outside the system is rejected as well.
    BOOST_CHECK_THROW(majorana.update_basis_change(std::vector<VecZ>(2 * N, VecZ{2 * N})), std::runtime_error);
    // A well-formed basis change is accepted.
    std::vector<VecZ> identity(2 * N);
    for (size_t i = 0; i < identity.size(); ++i) {
        identity[i] = VecZ{i};
    }
    BOOST_CHECK_NO_THROW(majorana.update_basis_change(identity));
}

// propagate() must refuse to run on top of a graph already built by build_graph().
BOOST_FIXTURE_TEST_CASE(propagate_on_nonempty_graph_throws, ExampleDataFix) {
    auto sim = build_simulator<n_modes>(data, SimulatorConfig{});
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    BOOST_REQUIRE(sim.graph_layers() > 0);
    BOOST_CHECK_THROW(sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters),
                      std::runtime_error);
}

// MPGraph::get_layer bounds-checks the layer index (checked_layer_offset throw site).
BOOST_FIXTURE_TEST_CASE(graph_get_layer_out_of_range_throws, ExampleDataFix) {
    auto sim = build_simulator<n_modes>(data, SimulatorConfig{});
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const auto &graph = sim.graph();
    const size_t n_layers = graph.layers();
    BOOST_REQUIRE(n_layers > 0);
    BOOST_CHECK_NO_THROW((void)graph.get_layer(0));
    BOOST_CHECK_THROW((void)graph.get_layer(n_layers), std::exception);
}
