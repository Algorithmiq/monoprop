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

// The MonomialPropagator throw paths that define the public contract.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

TEST_CASE("ctor_accepts_valid_config") {
    CHECK_NOTHROW(make(OperatorDict{}));
}

TEST_CASE("ctor_logical_num_modes_out_of_range_throws") {
    CHECK_THROWS_AS(make(OperatorDict{},
                         2 * N,
                         std::nullopt,
                         std::nullopt,
                         CutoffType::Length,
                         std::nullopt,
                         /*logical=*/0),
                    std::runtime_error);
    CHECK_THROWS_AS(make(OperatorDict{},
                         2 * N,
                         std::nullopt,
                         std::nullopt,
                         CutoffType::Length,
                         std::nullopt,
                         /*logical=*/N + 1),
                    std::runtime_error);
}

TEST_CASE("ctor_pauli_requires_support_cutoff_throws") {
    // Length has no Pauli-weight meaning.
    CHECK_THROWS_AS(
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Length, std::nullopt, N, Basis::Pauli),
        std::invalid_argument);
    CHECK_NOTHROW(
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Support, std::nullopt, N, Basis::Pauli));
}

TEST_CASE("ctor_pauli_forbids_basis_change_throws") {
    const std::vector<VecZ> some_basis(2 * N, VecZ{0});
    CHECK_THROWS_AS(
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Support, some_basis, N, Basis::Pauli),
        std::invalid_argument);
}

TEST_CASE("ctor_upper_atol_below_lower_atol_throws") {
    CHECK_THROWS_AS(make(OperatorDict{}, 2 * N, /*lower=*/1e-6, /*upper=*/1e-8), std::runtime_error);
    CHECK_NOTHROW(make(OperatorDict{}, 2 * N, /*lower=*/1e-8, /*upper=*/1e-6));
}

TEST_CASE("ctor_operator_index_out_of_range_throws") {
    OperatorDict op;
    op[VecZ{20}] = std::complex<double>(1.0, 0.0); // 20 >= 2*logical (=16)
    CHECK_THROWS_AS(make(op), std::runtime_error);
}

// A gate generator index outside the system must throw, not underflow 2*NumModes-1-index into an
// out-of-bounds Bitset::set.
TEST_CASE("build_graph_generator_index_out_of_range_throws") {
    OperatorDict op;
    op[VecZ{0, 1}] = std::complex<double>(0.0, 1.0);
    auto sim = make(op);
    // 2*logical_num_modes == 16, so slot 20 is outside this system.
    CHECK_THROWS_AS(sim.build_graph({VecZ{20, 21}}, VecZ{0}, VecD{1.0}), std::runtime_error);
    CHECK_NOTHROW(sim.build_graph({VecZ{0, 3}}, VecZ{0}, VecD{1.0}));
}

TEST_CASE("generator_index_bound_is_logical_not_storage") {
    OperatorDict op;
    op[VecZ{0, 1}] = std::complex<double>(0.0, 1.0);
    auto sim = make(op, 2 * N, std::nullopt, std::nullopt, CutoffType::Length, std::nullopt, /*logical=*/4);
    // 2*logical == 8 <= slot 9 < 2*N == 16: inside the storage, outside the system.
    CHECK_THROWS_AS(sim.build_graph({VecZ{9}}, VecZ{0}, VecD{1.0}), std::runtime_error);
}

TEST_CASE("only_rotate_len_k_build_graph_validation_matches_python_contract") {
    auto sim = make(OperatorDict{});

    CHECK_THROWS_AS(sim.build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, std::nullopt, std::nullopt, /*k=*/0u),
                    std::runtime_error);
    CHECK_NOTHROW(sim.build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, std::nullopt, std::nullopt, std::nullopt));

    auto logical_bound = make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Length, std::nullopt, 4);
    CHECK_THROWS_AS(logical_bound.build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, std::nullopt, std::nullopt, 9u),
                    std::runtime_error);
    CHECK_NOTHROW(logical_bound.build_graph({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, std::nullopt, std::nullopt, 8u));
}

TEST_CASE("only_rotate_len_k_propagate_validation_matches_python_contract") {
    auto sim = make(OperatorDict{});

    CHECK_THROWS_AS(sim.propagate({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, VecD{0.5}, /*k=*/0u), std::runtime_error);
    CHECK_NOTHROW(sim.propagate({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, VecD{0.5}, std::nullopt));

    auto logical_bound = make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Length, std::nullopt, 4);
    CHECK_THROWS_AS(logical_bound.propagate({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, VecD{0.5}, 9u), std::runtime_error);
    CHECK_NOTHROW(logical_bound.propagate({VecZ{0, 1}}, VecZ{0}, VecD{1.0}, VecD{0.5}, 8u));
}

// The update_* setters must not write straight through to regenerate_cutoff_fn_().
TEST_CASE("setters_enforce_the_constructor_invariants") {
    auto pauli =
        make(OperatorDict{}, 2 * N, std::nullopt, std::nullopt, CutoffType::Support, std::nullopt, N, Basis::Pauli);
    CHECK_THROWS_AS(pauli.update_cutoff_type(CutoffType::Length), std::invalid_argument);
    CHECK_THROWS_AS(pauli.update_basis_change(std::vector<VecZ>(2 * N, VecZ{0})), std::invalid_argument);

    auto majorana = make(OperatorDict{});
    // Too few rows: regenerate_cutoff_fn_ indexes [0, 2*logical_num_modes) unconditionally.
    CHECK_THROWS_AS(majorana.update_basis_change(std::vector<VecZ>{VecZ{0}}), std::invalid_argument);
    CHECK_THROWS_AS(majorana.update_basis_change(std::vector<VecZ>(2 * N, VecZ{2 * N})), std::runtime_error);
    std::vector<VecZ> identity(2 * N);
    for (size_t i = 0; i < identity.size(); ++i) {
        identity[i] = VecZ{i};
    }
    CHECK_NOTHROW(majorana.update_basis_change(identity));
}

TEST_CASE_METHOD(ExampleDataFix, "propagate_on_nonempty_graph_throws") {
    auto sim = build_simulator<n_modes>(data, SimulatorConfig{});
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    REQUIRE(sim.graph_layers() > 0);
    CHECK_THROWS_AS(sim.propagate(data.majoranas, data.param_inds, data.gen_coeffs, data.parameters),
                    std::runtime_error);
}

// Pins MPGraph::get_layer's checked_layer_offset throw site.
TEST_CASE_METHOD(ExampleDataFix, "graph_get_layer_out_of_range_throws") {
    auto sim = build_simulator<n_modes>(data, SimulatorConfig{});
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);
    const auto &graph = sim.graph();
    const size_t n_layers = graph.layers();
    REQUIRE(n_layers > 0);
    CHECK_NOTHROW((void)graph.get_layer(0));
    CHECK_THROWS_AS((void)graph.get_layer(n_layers), std::exception);
}
