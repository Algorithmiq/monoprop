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

#include <complex>
#include <map>
#include <optional>
#include <stdexcept>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

using namespace monoprop;
namespace tt = boost::test_tools;
namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(update_initial_operator_updates_core_expval) {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{}] = std::complex<double>{1.0, 0.0};

    VecZ initial_state{0, 1};
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          2 * n_modes,
                                          initial_state,
                                          Heisenberg{},
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    const VecD empty_params;
    auto expval_fn = simulator.expectation_value_functional(std::nullopt);
    BOOST_TEST(expval_fn(empty_params) == 1.0, tt::tolerance(1e-12));

    OperatorDict updated;
    updated[VecZ{}] = std::complex<double>{2.75, 0.0};
    simulator.update_initial_operator(updated);

    auto updated_fn = simulator.expectation_value_functional(std::nullopt);
    BOOST_TEST(updated_fn(empty_params) == 2.75, tt::tolerance(1e-12));

    // The functional built before the re-weight snapshotted the old coefficients, so it must reject
    // the call rather than answer for an operator the propagator no longer holds.
    BOOST_CHECK_THROW(expval_fn(empty_params), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(update_initial_operator_invalidates_gradient_functional) {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{}] = std::complex<double>{1.0, 0.0};

    VecZ initial_state{0, 1};
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          2 * n_modes,
                                          initial_state,
                                          Heisenberg{},
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    const VecD empty_params;
    auto grad_fn = simulator.expectation_value_and_gradient_functional(std::nullopt);
    BOOST_TEST(grad_fn(empty_params).first == 1.0, tt::tolerance(1e-12));

    OperatorDict updated;
    updated[VecZ{}] = std::complex<double>{2.75, 0.0};
    simulator.update_initial_operator(updated);

    BOOST_CHECK_THROW(grad_fn(empty_params), std::runtime_error);
    BOOST_TEST(simulator.expectation_value_and_gradient_functional(std::nullopt)(empty_params).first == 2.75,
               tt::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(update_initial_operator_throws_for_unknown_term_in_heisenberg) {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{0, 1}] = std::complex<double>{0, 1.0};

    VecZ initial_state{0, 1};
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          2 * n_modes,
                                          initial_state,
                                          Heisenberg{},
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    const VecZ invalid_term{2, 3};
    OperatorDict missing_term;
    missing_term[invalid_term] = std::complex<double>{0.0, 0.5};

    // On a single rank, the owning rank always sees the error.
    BOOST_CHECK_THROW(simulator.update_initial_operator(missing_term), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(update_initial_operator_accepts_new_terms_in_schrodinger) {
    constexpr size_t n_modes = 2;
    OperatorDict initial_ham;
    initial_ham[VecZ{0, 1}] = std::complex<double>{0, 1.0};

    VecZ initial_state{0, 1};
    const unsigned int cutoff = static_cast<unsigned int>(2 * n_modes);
    MonomialPropagator<n_modes> simulator(initial_ham,
                                          cutoff,
                                          initial_state,
                                          Schrodinger{cutoff},
                                          MPI_COMM_SELF,
                                          std::nullopt,
                                          std::nullopt,
                                          CutoffType::Support,
                                          std::nullopt);

    OperatorDict new_term;
    new_term[VecZ{2, 3}] = std::complex<double>{0.0, 0.25};
    BOOST_CHECK_NO_THROW(simulator.update_initial_operator(new_term));
}
