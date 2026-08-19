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

// Each of these throws when the stated condition does not hold: ValidationError for inconsistent
// arguments, StaleFunctionalGraph when the propagator was mutated after a functional captured the
// graph and operator it replays. Both derive from std::runtime_error, so catching that still catches
// either.

monoprop_EXPORT auto validate_coefficient_lengths(const VecZ &parameter_mapping, const VecD &gen_coeffs) -> void;

// gate_indices records which ingested gate each monomial came from: one entry per monomial, forming
// contiguous runs from 0.
monoprop_EXPORT auto validate_gate_indices(const VecZ &gate_indices, size_t num_monomials) -> void;

// params must have max(parameter_mapping)+1 entries.
monoprop_EXPORT auto validate_parameters_length(const VecD &params, const VecZ &parameter_mapping) -> void;

monoprop_EXPORT auto validate_functional_call(const VecD &parameters, size_t expected_num_params) -> void;

/// What a functional must be able to say about its propagator before it reads anything it borrowed.
// Assembled from detail::FunctionalControl plus, for a single-partition plan, two facts derived from the
// operator itself. The derived pair is the backstop: it holds even for a mutation that forgot to bump.
struct FunctionalState {
    bool propagator_alive;              ///< false once ~MonomialPropagator has run
    size_t current_revision;            ///< the propagator's structure revision now
    size_t expected_revision;           ///< the revision the functional was built at
    bool operator_layout_unchanged;     ///< the borrowed inverted index still spans the same store and rows
    const char *last_structural_change; ///< the method that last bumped the revision, or nullptr
};

monoprop_EXPORT auto validate_functional_state(const FunctionalState &state) -> void;

/// The inputs to the check that a functional may follow a newer set of initial-operator weights.
struct WeightRefresh {
    size_t weights_revision;  ///< the structure revision the newer weights were published at
    size_t expected_revision; ///< the revision the functional was built at
    bool pared_from_operator; ///< the functional's keep-set was thresholded from the operator itself
};

monoprop_EXPORT auto validate_weight_refresh(const WeightRefresh &refresh) -> void;

// only_rotate_len_k is optional; when set it must satisfy 0 < k <= max_k.
monoprop_EXPORT auto validate_only_rotate_len_k_(std::optional<size_t> only_rotate_len_k, size_t max_k) -> void;

monoprop_EXPORT auto expected_num_params(const VecZ &parameter_mapping) -> size_t;

} // namespace monoprop
