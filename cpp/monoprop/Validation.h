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

#include <cstddef>
#include <optional>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

// Validators throw ValidationError for invalid arguments or StaleFunctionalGraph for stale plans.

monoprop_EXPORT auto validate_coefficient_lengths(const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void;

// gate_indices has one contiguous-run index per ingested monomial.
monoprop_EXPORT auto validate_gate_indices(const VecZ &gate_indices, size_t num_monomials) -> void;

// params must have max(parameter_mapping)+1 entries.
monoprop_EXPORT auto validate_parameters_length(const VecD &params, const VecZ &parameter_mapping) -> void;

monoprop_EXPORT auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void;

/// Propagator state required before reading a functional's borrowed data.
struct FunctionalState {
    bool propagator_alive;              ///< False after destruction begins.
    size_t current_revision;            ///< Current structure revision.
    size_t expected_revision;           ///< Build-time structure revision.
    bool operator_layout_unchanged;     ///< Borrowed index still matches its store and rows.
    const char *last_structural_change; ///< Last mutation, or nullptr.
};

monoprop_EXPORT auto validate_functional_state(const FunctionalState &state) -> void;

/// Inputs for checking whether a functional may follow new weights.
struct WeightRefresh {
    size_t weights_revision;  ///< Revision at publication.
    size_t expected_revision; ///< Build-time revision.
    bool may_follow_weights;  ///< Functional weight-following policy.
};

monoprop_EXPORT auto validate_weight_refresh(const WeightRefresh &refresh) -> void;

// only_rotate_len_k is optional; when set it must satisfy 0 < k <= max_k.
monoprop_EXPORT auto validate_only_rotate_len_k_(std::optional<size_t> only_rotate_len_k, size_t max_k) -> void;

monoprop_EXPORT auto expected_num_params(const VecZ &parameter_mapping) -> size_t;

} // namespace monoprop
