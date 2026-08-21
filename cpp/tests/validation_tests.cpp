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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>

#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"

using namespace monoprop;

TEST_CASE("validation_coefficient_lengths") {
    CHECK_NOTHROW(validate_coefficient_lengths(VecZ{0, 1, 2}, VecD{1.0, 2.0, 3.0}));
    CHECK_NOTHROW(validate_coefficient_lengths(VecZ{}, VecD{}));
    CHECK_THROWS_AS(validate_coefficient_lengths(VecZ{0, 1}, VecD{1.0}), std::runtime_error);
}

TEST_CASE("validation_gate_indices") {
    CHECK_NOTHROW(validate_gate_indices(VecZ{0, 0, 1, 1, 2}, 5));
    CHECK_NOTHROW(validate_gate_indices(VecZ{}, 0));
    CHECK_NOTHROW(validate_gate_indices(VecZ{0, 1, 2}, 3));
    // Length must match the monomial count.
    CHECK_THROWS_AS(validate_gate_indices(VecZ{0, 1}, 3), std::runtime_error);
    // Must start at 0.
    CHECK_THROWS_AS(validate_gate_indices(VecZ{1, 2}, 2), std::runtime_error);
    // Must not jump by more than 1.
    CHECK_THROWS_AS(validate_gate_indices(VecZ{0, 1, 3}, 3), std::runtime_error);
    // Must not decrease.
    CHECK_THROWS_AS(validate_gate_indices(VecZ{0, 1, 0}, 3), std::runtime_error);
}

TEST_CASE("validation_parameters_length") {
    CHECK_NOTHROW(validate_parameters_length(VecD{0.1, 0.2, 0.3}, VecZ{0, 1, 2}));
    CHECK_NOTHROW(validate_parameters_length(VecD{0.1, 0.2}, VecZ{0, 1, 1, 0})); // max=1 -> len 2
    CHECK_NOTHROW(validate_parameters_length(VecD{}, VecZ{}));
    CHECK_THROWS_AS(validate_parameters_length(VecD{0.1, 0.2}, VecZ{0, 1, 2}), std::runtime_error);
}

TEST_CASE("validation_functional_call") {
    CHECK_NOTHROW(validate_functional_call(VecD{0.1, 0.2}, 2));
    CHECK_NOTHROW(validate_functional_call(VecD{}, 0));
    CHECK_THROWS_AS(validate_functional_call(VecD{0.1}, 2), std::runtime_error);
}

TEST_CASE("validation_expected_graph_layers") {
    CHECK_NOTHROW(validate_expected_graph_layers(3, 3));
    CHECK_THROWS_AS(validate_expected_graph_layers(4, 3), std::runtime_error);
}

TEST_CASE("validation_only_rotate_len_k") {
    CHECK_NOTHROW(validate_only_rotate_len_k_(std::nullopt, 8));
    CHECK_NOTHROW(validate_only_rotate_len_k_(8u, 8));
    CHECK_THROWS_AS(validate_only_rotate_len_k_(0u, 8), std::runtime_error);
    CHECK_THROWS_AS(validate_only_rotate_len_k_(9u, 8), std::runtime_error);
}
