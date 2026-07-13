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

#pragma once

#include <optional>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

/**
 * @brief Evolution mode enumeration for different evolution strategies
 */
enum class EvolutionMode {
    GraphOnly,          ///< Build evolution graph only
    GraphWithCoeffs,    ///< Build evolution graph with coefficient information
    ContractImmediately ///< Evolve and contract immediately without building a graph
};

/**
 * @brief Determines the evolution mode based on provided parameters
 *
 * Analyzes the combination of provided parameters to determine which of the three
 * evolution strategies should be used.
 *
 * @param parameter_mapping Optional mapping from variational parameters to generator indices
 * @param gen_coeffs Optional generator coefficients corresponding to each parameter mapping
 * @param parameters Optional parameter values for evolution
 * @param operator_coeffs Optional operator coefficients for the current state or Hamiltonian
 * @return EvolutionMode indicating which evolution strategy to use
 * @throws std::runtime_error If parameter combinations are invalid
 */
monoprop_EXPORT auto determine_evolution_mode(const std::optional<VecZ> &parameter_mapping,
                                              const std::optional<VecD> &gen_coeffs,
                                              const std::optional<VecD> &parameters,
                                              const std::optional<VecD> &operator_coeffs) -> EvolutionMode;

/**
 * @brief Validates the consistency of evolution parameters
 *
 * Ensures that if any of the evolution parameters are provided, all must be provided.
 * Also validates that the parameter mapping and generator coefficients have matching lengths,
 * and that the parameters array has the correct length.
 *
 * @param parameter_mapping Optional mapping from variational parameters to generator indices
 * @param gen_coeffs Optional generator coefficients corresponding to each parameter mapping
 * @param parameters Optional parameter values for evolution
 * @throws std::runtime_error If parameter combinations are invalid
 */
monoprop_EXPORT auto validate_evolution_parameters(const std::optional<VecZ> &parameter_mapping,
                                                   const std::optional<VecD> &gen_coeffs,
                                                   const std::optional<VecD> &parameters) -> void;

/**
 * @brief Validates the graph state for the selected evolution mode.
 *
 * @param mode The selected evolution mode
 * @param parameter_mapping Optional mapping from variational parameters to generator indices
 * @param gen_coeffs Optional generator coefficients corresponding to each parameter mapping
 * @param parameters Optional parameter values for evolution
 * @param graph_size The current size of the evolution graph
 * @throws std::runtime_error If the graph state is inconsistent with the evolution mode
 */
monoprop_EXPORT auto validate_graph_state_for_mode(EvolutionMode mode,
                                                   const std::optional<VecZ> &parameter_mapping,
                                                   const std::optional<VecD> &gen_coeffs,
                                                   const std::optional<VecD> &parameters,
                                                   size_t graph_size) -> void;

/**
 * @brief Validate parameters against parameter mapping and generator coefficients
 *
 * @param params Parameter values
 * @param parameter_mapping Mapping from parameters to generator indices
 * @param gen_coeffs Generator coefficients
 * @throws std::runtime_error If validation fails
 */
monoprop_EXPORT auto validate_params(const VecD &params, const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void;

/**
 * @brief Validate that parameter_mapping and gen_coeffs have equal lengths.
 *
 * @param parameter_mapping Mapping from parameters to generator indices
 * @param gen_coeffs Generator coefficients
 * @throws std::runtime_error if lengths don't match
 */
monoprop_EXPORT auto validate_coefficient_lengths(const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void;

/**
 * @brief Validate per-monomial gate indices supplied to build_graph.
 *
 * Gate indices label which ingested gate each monomial belongs to. They must have one entry
 * per monomial and form contiguous runs starting at 0 (each element equals the previous or
 * previous+1), matching the shape produced by expanding gates into monomials.
 *
 * @param gate_indices Per-monomial gate index (local, 0-based).
 * @param num_monomials Expected number of entries.
 * @throws std::runtime_error if the length differs or the indices are not contiguous from 0.
 */
monoprop_EXPORT auto validate_gate_indices(const VecZ &gate_indices, size_t num_monomials) -> void;

/**
 * @brief Validate that parameters length matches what's expected from parameter_mapping.
 *
 * @param params Parameter values
 * @param parameter_mapping Mapping from parameters to generator indices
 * @throws std::runtime_error if parameter length is insufficient
 */
monoprop_EXPORT auto validate_parameters_length(const VecD &params, const VecZ &parameter_mapping) -> void;

/**
 * @brief Validate that parameter_mapping length matches the number of propagated Majoranas.
 *
 * @param parameter_mapping_size Size of parameter mapping
 * @param num_evolved Number of propagated Majoranas
 * @throws std::runtime_error if lengths don't match
 */
monoprop_EXPORT auto validate_propagation_params(size_t parameter_mapping_size, size_t num_evolved) -> void;

/**
 * @brief Validate that parameter_mapping length is valid for contraction.
 *
 * @param parameter_mapping_size Size of parameter mapping
 * @param num_evolved Number of propagated Majoranas
 * @throws std::runtime_error if parameter_mapping is too long
 */
monoprop_EXPORT auto validate_propagation_contraction(size_t parameter_mapping_size, size_t num_evolved) -> void;

/**
 * @brief Validate that a functional call is valid.
 *
 * @param parameters Parameters provided to functional
 * @param expected_num_params Expected number of parameters
 * @throws std::runtime_error if validation fails
 */
monoprop_EXPORT auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void;

/**
 * @brief Validate that parameter_mapping, gen_coeffs and majoranas have matching lengths.
 *
 * @param parameter_mapping_size Size of parameter mapping
 * @param gen_coeffs_size Size of generator coefficients
 * @param majoranas_size Number of majorana operators
 * @throws std::runtime_error if lengths don't match
 */
monoprop_EXPORT auto validate_param_map_gen_coeffs_majoranas_match(size_t parameter_mapping_size,
                                                                   size_t gen_coeffs_size,
                                                                   size_t majoranas_size) -> void;

/**
 * @brief Validate that the current graph layers match expected.
 *
 * @param current_layers Current number of graph layers
 * @param expected_layers Expected number of graph layers
 * @throws std::runtime_error if layers don't match
 */
monoprop_EXPORT auto validate_expected_graph_layers(size_t current_layers, size_t expected_layers) -> void;

} // namespace monoprop
