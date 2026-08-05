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

#include "monoprop/Validation.h"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace monoprop {

// Every validate_* precondition on a caller-supplied argument reports through this one type. Derived
// from std::runtime_error because nanobind's built-in translation table dispatches on the nearest std
// base: changing the base would change the Python exception these surface as.
class ValidationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The graph was rebuilt after a functional captured its layer count, so the functional's parameter
// mapping no longer describes the graph. Kept apart from ValidationError because the arguments are
// well-formed here -- the object moved under them, and the caller recovers by rebuilding the
// functional rather than by fixing a call. Same std base, so both surface identically in Python.
class StaleFunctionalGraph : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Declared in Validation.h and used across translation units, so internal linkage does not apply.
// NOLINTBEGIN(misc-use-internal-linkage)

namespace {

auto validate_equal_sizes(size_t lhs, size_t rhs, const char *message) -> void {
    if (lhs != rhs) {
        throw ValidationError(message);
    }
}

} // namespace

auto validate_coefficient_lengths(const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void {
    validate_equal_sizes(parameter_mapping.size(),
                         gen_coeffs.size(),
                         "The length of parameter_mapping and gen_coeffs must be the same.");
}

auto validate_gate_indices(const VecZ &gate_indices, size_t num_monomials) -> void {
    if (gate_indices.size() != num_monomials) {
        throw ValidationError(
            std::format("gate_indices has {} entries but there are {} monomials.", gate_indices.size(), num_monomials));
    }
    if (gate_indices.empty()) {
        return;
    }
    if (gate_indices.front() != 0) {
        throw ValidationError(std::format("gate_indices must start at 0; got {}.", gate_indices.front()));
    }
    for (size_t i = 1; i < gate_indices.size(); ++i) {
        if (gate_indices[i] != gate_indices[i - 1] && gate_indices[i] != gate_indices[i - 1] + 1) {
            throw ValidationError(std::format("gate_indices must be contiguous runs from 0 (each entry "
                                              "equal to the previous or previous+1); got a jump from {} "
                                              "to {} at position {}.",
                                              gate_indices[i - 1],
                                              gate_indices[i],
                                              i));
        }
    }
}

auto validate_parameters_length(const VecD &params, const VecZ &parameter_mapping) -> void {
    if (parameter_mapping.empty()) {
        return;
    }

    const auto expected_param_length = *std::ranges::max_element(parameter_mapping) + 1;
    if (params.size() != expected_param_length) {
        throw ValidationError(
            std::format("The length of parameters ({}) must be the same as max(parameter_mapping)+1 ({}).",
                        params.size(),
                        expected_param_length));
    }
}

auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void {
    if (parameters.size() != expected_num_params) {
        throw ValidationError(std::format("Invalid functional call. Parameter length {} does not "
                                          "match the expected number of parameters {}.",
                                          parameters.size(),
                                          expected_num_params));
    }
}

auto validate_expected_graph_layers(size_t current_layers, size_t expected_layers) -> void {
    if (current_layers != expected_layers) {
        throw StaleFunctionalGraph(std::format("MP object has been modified since the functional was created. "
                                               "Previous number of graph layers was {} and now is {}.",
                                               expected_layers,
                                               current_layers));
    }
}

// NOLINTEND(misc-use-internal-linkage)

} // namespace monoprop
