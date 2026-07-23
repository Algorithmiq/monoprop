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

// These helpers are declared in Validation.h and used across translation units,
// so clang-tidy's internal-linkage suggestion does not apply here.
// NOLINTBEGIN(misc-use-internal-linkage)

namespace {

auto validate_equal_sizes(size_t lhs, size_t rhs, const char *message) -> void {
    if (lhs != rhs) {
        throw std::runtime_error(message);
    }
}

auto has_complete_evolution_parameters(const std::optional<VecZ> &parameter_mapping,
                                       const std::optional<VecD> &gen_coeffs,
                                       const std::optional<VecD> &parameters) -> bool {
    return parameter_mapping && gen_coeffs && parameters;
}

} // namespace

auto determine_evolution_mode(const std::optional<VecZ> &parameter_mapping,
                              const std::optional<VecD> &gen_coeffs,
                              const std::optional<VecD> &parameters,
                              const std::optional<VecD> &operator_coeffs) -> EvolutionMode {
    const auto has_all_params = has_complete_evolution_parameters(parameter_mapping, gen_coeffs, parameters);

    if (!has_all_params && !operator_coeffs.has_value()) {
        return EvolutionMode::GraphOnly;
    }
    if (has_all_params && operator_coeffs.has_value()) {
        return EvolutionMode::GraphWithCoeffs;
    }
    if (has_all_params && !operator_coeffs.has_value()) {
        return EvolutionMode::ContractImmediately;
    }
    throw std::runtime_error(
        "Invalid evolution mode detected. This function supports three main evolution strategies:\n"
        "\n"
        "1. Building the evolution graph only.\n"
        "    - To use this mode, only provide the majoranas parameter.\n"
        "\n"
        "2. Building the evolution graph with coefficient information.\n"
        "    - To use this mode, provide majoranas, parameter_mapping, gen_coeffs, parameters and "
        "operator_coeffs. Operator coefficients can be obtained from a prior call to contract_partially() with "
        "inplace set to false if you want to preserve the graph.\n"
        "\n"
        "3. Evolving and contracting immediately without building a graph.\n"
        "    - To use this mode, provide majoranas, parameter_mapping, gen_coeffs, and parameters. Do not "
        "provide operator_coeffs. This mode is memory efficient as it does not store the evolution graph.");
}

auto validate_evolution_parameters(const std::optional<VecZ> &parameter_mapping,
                                   const std::optional<VecD> &gen_coeffs,
                                   const std::optional<VecD> &parameters) -> void {
    const auto has_all_params = has_complete_evolution_parameters(parameter_mapping, gen_coeffs, parameters);
    const auto has_some_params = parameter_mapping || gen_coeffs || parameters;

    if (has_some_params && !has_all_params) {
        throw std::runtime_error(
            "Either all of parameters, parameter_mapping, and gen_coeffs must be None, or all must be provided.");
    }

    if (has_all_params) {
        validate_coefficient_lengths(parameter_mapping.value(), gen_coeffs.value());
        validate_parameters_length(parameters.value(), parameter_mapping.value());
    }
}

auto validate_graph_state_for_mode(EvolutionMode mode,
                                   const std::optional<VecZ> &parameter_mapping,
                                   const std::optional<VecD> &gen_coeffs,
                                   const std::optional<VecD> &parameters,
                                   size_t graph_size) -> void {
    const auto has_all_params = has_complete_evolution_parameters(parameter_mapping, gen_coeffs, parameters);
    const auto graph_is_empty = graph_size == 0;

    if (has_all_params && !graph_is_empty) {
        if (mode == EvolutionMode::ContractImmediately) {
            throw std::runtime_error(
                std::format("Cannot evolve inplace as there is a previous evolution of {} Majoranas. "
                            "Please call 'contract_partially' to contract the graph first.",
                            graph_size));
        }
    }
}

auto validate_params(const VecD &params, const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void {
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);

    for (const auto &ind : parameter_mapping) {
        if (ind >= params.size()) {
            throw std::runtime_error(std::format("Index {} in parameter_mapping is out of range.", ind));
        }
    }
}

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

auto validate_param_map_gen_coeffs_majoranas_match(size_t parameter_mapping_size,
                                                   size_t gen_coeffs_size,
                                                   size_t majoranas_size) -> void {
    validate_equal_sizes(parameter_mapping_size,
                         gen_coeffs_size,
                         "The length of parameter_mapping and gen_coeffs must be the same.");
    if (parameter_mapping_size != majoranas_size) {
        throw std::runtime_error(std::format(
            "The length of parameter_mapping and gen_coeffs must match the number of evolved Majoranas ({}).",
            majoranas_size));
    }
}

auto validate_parameters_length(const VecD &params, const VecZ &parameter_mapping) -> void {
    if (parameter_mapping.empty()) {
        return; // No validation needed for empty parameter_mapping
    }

    // Find the maximum index in parameter_mapping
    size_t expected_param_length = *std::max_element(parameter_mapping.begin(), parameter_mapping.end()) + 1;
    if (params.size() != expected_param_length) {
        throw std::runtime_error(
            std::format("The length of parameters ({}) must be the same as max(parameter_mapping)+1 ({}).",
                        params.size(),
                        expected_param_length));
    }
}

auto validate_propagation_params(size_t parameter_mapping_size, size_t num_evolved) -> void {
    if (parameter_mapping_size != num_evolved) {
        throw std::runtime_error(std::format("The length of parameter_mapping and gen_coeffs must be the same as the "
                                             "number of propagated Majoranas {}.",
                                             num_evolved));
    }
}

auto validate_propagation_contraction(size_t parameter_mapping_size, size_t num_evolved) -> void {
    if (parameter_mapping_size > num_evolved) {
        throw std::runtime_error(std::format("The length of parameter_mapping must be less than or equal to the "
                                             "number of propagated Majoranas {}.",
                                             num_evolved));
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
