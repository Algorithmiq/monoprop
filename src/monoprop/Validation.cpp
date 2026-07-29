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

// Declared in Validation.h and used across translation units, so internal linkage does not apply.
// NOLINTBEGIN(misc-use-internal-linkage)

namespace {

auto validate_equal_sizes(size_t lhs, size_t rhs, const char *message) -> void {
    if (lhs != rhs) {
        throw std::runtime_error(message);
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
        throw std::runtime_error(
            std::format("gate_indices has {} entries but there are {} monomials.", gate_indices.size(), num_monomials));
    }
    if (gate_indices.empty()) {
        return;
    }
    if (gate_indices.front() != 0) {
        throw std::runtime_error(std::format("gate_indices must start at 0; got {}.", gate_indices.front()));
    }
    for (size_t i = 1; i < gate_indices.size(); ++i) {
        if (gate_indices[i] != gate_indices[i - 1] && gate_indices[i] != gate_indices[i - 1] + 1) {
            throw std::runtime_error(std::format("gate_indices must be contiguous runs from 0 (each entry "
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

    size_t expected_param_length = *std::max_element(parameter_mapping.begin(), parameter_mapping.end()) + 1;
    if (params.size() != expected_param_length) {
        throw std::runtime_error(
            std::format("The length of parameters ({}) must be the same as max(parameter_mapping)+1 ({}).",
                        params.size(),
                        expected_param_length));
    }
}

auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void {
    if (parameters.size() != expected_num_params) {
        throw std::runtime_error(std::format("Invalid functional call. Parameter length {} does not "
                                             "match the expected number of parameters {}.",
                                             parameters.size(),
                                             expected_num_params));
    }
}

auto validate_expected_graph_layers(size_t current_layers, size_t expected_layers) -> void {
    if (current_layers != expected_layers) {
        throw std::runtime_error(std::format("MP object has been modified since the functional was created. "
                                             "Previous number of graph layers was {} and now is {}.",
                                             expected_layers,
                                             current_layers));
    }
}

// NOLINTEND(misc-use-internal-linkage)

} // namespace monoprop
