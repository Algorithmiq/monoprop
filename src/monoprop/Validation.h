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

#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

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
 * @brief Validate that a functional call is valid.
 *
 * @param parameters Parameters provided to functional
 * @param expected_num_params Expected number of parameters
 * @throws std::runtime_error if validation fails
 */
monoprop_EXPORT auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void;

/**
 * @brief Validate that the current graph layers match expected.
 *
 * @param current_layers Current number of graph layers
 * @param expected_layers Expected number of graph layers
 * @throws std::runtime_error if layers don't match
 */
monoprop_EXPORT auto validate_expected_graph_layers(size_t current_layers, size_t expected_layers) -> void;

} // namespace monoprop
