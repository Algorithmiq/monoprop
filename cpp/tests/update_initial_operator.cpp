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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <map>
#include <optional>
#include <stdexcept>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;
TEST_CASE("update_initial_operator_updates_core_expval") {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{}] = std::complex<double>{1.0, 0.0};

    VecZ initial_state{0, 1};
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          2 * n_modes,
                                          initial_state,
                                          std::nullopt,
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    const VecD empty_params;
    auto expval_fn = simulator.expectation_value_functional(std::nullopt);
    CHECK((expval_fn(empty_params)) == Catch::Approx(1.0).epsilon(1e-12));

    OperatorDict updated;
    updated[VecZ{}] = std::complex<double>{2.75, 0.0};
    simulator.update_initial_operator(updated);

    auto updated_fn = simulator.expectation_value_functional(std::nullopt);
    CHECK((updated_fn(empty_params)) == Catch::Approx(2.75).epsilon(1e-12));

    // The functional built before the re-weight snapshotted the old coefficients, so it must reject
    // the call rather than answer for an operator the propagator no longer holds.
    CHECK_THROWS_AS(expval_fn(empty_params), std::runtime_error);
}

TEST_CASE("update_initial_operator_invalidates_gradient_functional") {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{}] = std::complex<double>{1.0, 0.0};

    VecZ initial_state{0, 1};
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          2 * n_modes,
                                          initial_state,
                                          std::nullopt,
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    const VecD empty_params;
    auto grad_fn = simulator.expectation_value_and_gradient_functional(std::nullopt);
    CHECK((grad_fn(empty_params).first) == Catch::Approx(1.0).epsilon(1e-12));

    OperatorDict updated;
    updated[VecZ{}] = std::complex<double>{2.75, 0.0};
    simulator.update_initial_operator(updated);

    CHECK_THROWS_AS(grad_fn(empty_params), std::runtime_error);
    CHECK((simulator.expectation_value_and_gradient_functional(std::nullopt)(empty_params).first)
          == Catch::Approx(2.75).epsilon(1e-12));
}

TEST_CASE("update_initial_operator_throws_for_unknown_term_in_heisenberg") {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{0, 1}] = std::complex<double>{0, 1.0};

    VecZ initial_state{0, 1};
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          2 * n_modes,
                                          initial_state,
                                          std::nullopt,
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    const VecZ invalid_term{2, 3};
    OperatorDict missing_term;
    missing_term[invalid_term] = std::complex<double>{0.0, 0.5};

    // On a single rank, the owning rank always sees the error.
    CHECK_THROWS_AS(simulator.update_initial_operator(missing_term), std::runtime_error);
}

TEST_CASE("update_initial_operator_accepts_new_terms_in_schrodinger") {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{0, 1}] = std::complex<double>{0, 1.0};

    VecZ initial_state{0, 1};
    const unsigned int cutoff = static_cast<unsigned int>(2 * n_modes);
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          cutoff,
                                          initial_state,
                                          cutoff,
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    OperatorDict new_term;
    new_term[VecZ{2, 3}] = std::complex<double>{0.0, 0.25};
    CHECK_NOTHROW(simulator.update_initial_operator(new_term));
}
