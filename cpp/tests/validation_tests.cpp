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

// The validators in Validation.cpp that guard the public build/propagate/functional API.

#include <boost/test/unit_test.hpp>

#include <stdexcept>
#include <string_view>

#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"

using namespace monoprop;

BOOST_AUTO_TEST_CASE(validation_coefficient_lengths) {
    BOOST_CHECK_NO_THROW(validate_coefficient_lengths(VecZ{0, 1, 2}, VecD{1.0, 2.0, 3.0}));
    BOOST_CHECK_NO_THROW(validate_coefficient_lengths(VecZ{}, VecD{}));
    BOOST_CHECK_THROW(validate_coefficient_lengths(VecZ{0, 1}, VecD{1.0}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_gate_indices) {
    BOOST_CHECK_NO_THROW(validate_gate_indices(VecZ{0, 0, 1, 1, 2}, 5));
    BOOST_CHECK_NO_THROW(validate_gate_indices(VecZ{}, 0));
    BOOST_CHECK_NO_THROW(validate_gate_indices(VecZ{0, 1, 2}, 3));
    // Length must match the monomial count.
    BOOST_CHECK_THROW(validate_gate_indices(VecZ{0, 1}, 3), std::runtime_error);
    // Must start at 0.
    BOOST_CHECK_THROW(validate_gate_indices(VecZ{1, 2}, 2), std::runtime_error);
    // Must not jump by more than 1.
    BOOST_CHECK_THROW(validate_gate_indices(VecZ{0, 1, 3}, 3), std::runtime_error);
    // Must not decrease.
    BOOST_CHECK_THROW(validate_gate_indices(VecZ{0, 1, 0}, 3), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_parameters_length) {
    BOOST_CHECK_NO_THROW(validate_parameters_length(VecD{0.1, 0.2, 0.3}, VecZ{0, 1, 2}));
    BOOST_CHECK_NO_THROW(validate_parameters_length(VecD{0.1, 0.2}, VecZ{0, 1, 1, 0})); // max=1 -> len 2
    BOOST_CHECK_NO_THROW(validate_parameters_length(VecD{}, VecZ{}));
    BOOST_CHECK_THROW(validate_parameters_length(VecD{0.1, 0.2}, VecZ{0, 1, 2}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_functional_call) {
    BOOST_CHECK_NO_THROW(validate_functional_call(VecD{0.1, 0.2}, 2));
    BOOST_CHECK_NO_THROW(validate_functional_call(VecD{}, 0));
    BOOST_CHECK_THROW(validate_functional_call(VecD{0.1}, 2), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_functional_state) {
    const FunctionalState healthy{.propagator_alive = true,
                                  .current_revision = 3,
                                  .expected_revision = 3,
                                  .operator_layout_unchanged = true,
                                  .last_structural_change = nullptr};
    BOOST_CHECK_NO_THROW(validate_functional_state(healthy));

    auto destroyed = healthy;
    destroyed.propagator_alive = false;
    BOOST_CHECK_THROW(validate_functional_state(destroyed), std::runtime_error);

    // The revision names the mutation that moved the structure, so the message can point at it.
    auto mutated = healthy;
    mutated.current_revision = 4;
    mutated.last_structural_change = "build_graph()";
    BOOST_CHECK_EXCEPTION(validate_functional_state(mutated), std::runtime_error, [](const auto &e) {
        return std::string_view(e.what()).find("build_graph()") != std::string_view::npos;
    });

    // The backstop: the operator moved without a revision bump.
    auto rebuilt = healthy;
    rebuilt.operator_layout_unchanged = false;
    BOOST_CHECK_THROW(validate_functional_state(rebuilt), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_weight_refresh) {
    const WeightRefresh followable{.weights_revision = 3, .expected_revision = 3, .pared_from_operator = false};
    BOOST_CHECK_NO_THROW(validate_weight_refresh(followable));

    // The keep-set came from the coefficients the re-weight replaced, so replaying it would answer for a
    // paring nobody asked for.
    auto pared_from_op = followable;
    pared_from_op.pared_from_operator = true;
    BOOST_CHECK_EXCEPTION(validate_weight_refresh(pared_from_op), std::runtime_error, [](const auto &e) {
        return std::string_view(e.what()).find("cannot follow the new weights") != std::string_view::npos;
    });

    // The backstop: weights from another revision reached a functional whose own revision still matches.
    auto other_revision = followable;
    other_revision.weights_revision = 2;
    BOOST_CHECK_THROW(validate_weight_refresh(other_revision), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_only_rotate_len_k) {
    BOOST_CHECK_NO_THROW(validate_only_rotate_len_k_(std::nullopt, 8));
    BOOST_CHECK_NO_THROW(validate_only_rotate_len_k_(8u, 8));
    BOOST_CHECK_THROW(validate_only_rotate_len_k_(0u, 8), std::runtime_error);
    BOOST_CHECK_THROW(validate_only_rotate_len_k_(9u, 8), std::runtime_error);
}
