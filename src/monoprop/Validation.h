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

// Each of these throws std::runtime_error when the stated condition does not hold.

// parameter_mapping and gen_coeffs must have equal lengths.
monoprop_EXPORT auto validate_coefficient_lengths(const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void;

// gate_indices (which ingested gate each monomial came from) must have one entry per monomial and form
// contiguous runs from 0 — each entry equal to the previous or previous+1.
monoprop_EXPORT auto validate_gate_indices(const VecZ &gate_indices, size_t num_monomials) -> void;

// params must have max(parameter_mapping)+1 entries.
monoprop_EXPORT auto validate_parameters_length(const VecD &params, const VecZ &parameter_mapping) -> void;

// A functional call must supply exactly expected_num_params parameters.
monoprop_EXPORT auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void;

// The graph must still have the layer count the functional was built against.
monoprop_EXPORT auto validate_expected_graph_layers(size_t current_layers, size_t expected_layers) -> void;

} // namespace monoprop
