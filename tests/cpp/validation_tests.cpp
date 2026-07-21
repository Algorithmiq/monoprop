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

// Direct coverage of the (live) parameter validators in Validation.cpp. These pure throw-or-return
// functions guard the public build/propagate/functional API but were previously exercised only
// indirectly through Python. Each case pins one accept path and one reject path.

#include <boost/test/unit_test.hpp>

#include <stdexcept>

#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"

using namespace monoprop;

BOOST_AUTO_TEST_CASE(validation_coefficient_lengths) {
    BOOST_CHECK_NO_THROW(validate_coefficient_lengths(VecZ{0, 1, 2}, VecD{1.0, 2.0, 3.0}));
    BOOST_CHECK_NO_THROW(validate_coefficient_lengths(VecZ{}, VecD{}));
    BOOST_CHECK_THROW(validate_coefficient_lengths(VecZ{0, 1}, VecD{1.0}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_gate_indices) {
    // Contiguous runs from 0 are accepted; empty is accepted (no monomials).
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
    // Expected length is max(parameter_mapping) + 1.
    BOOST_CHECK_NO_THROW(validate_parameters_length(VecD{0.1, 0.2, 0.3}, VecZ{0, 1, 2}));
    BOOST_CHECK_NO_THROW(validate_parameters_length(VecD{0.1, 0.2}, VecZ{0, 1, 1, 0})); // max=1 -> len 2
    // Empty mapping needs no parameters.
    BOOST_CHECK_NO_THROW(validate_parameters_length(VecD{}, VecZ{}));
    BOOST_CHECK_THROW(validate_parameters_length(VecD{0.1, 0.2}, VecZ{0, 1, 2}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_functional_call) {
    BOOST_CHECK_NO_THROW(validate_functional_call(VecD{0.1, 0.2}, 2));
    BOOST_CHECK_NO_THROW(validate_functional_call(VecD{}, 0));
    BOOST_CHECK_THROW(validate_functional_call(VecD{0.1}, 2), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(validation_expected_graph_layers) {
    BOOST_CHECK_NO_THROW(validate_expected_graph_layers(3, 3));
    BOOST_CHECK_THROW(validate_expected_graph_layers(4, 3), std::runtime_error);
}
