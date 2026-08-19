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

// Every validate_* precondition on a caller-supplied argument reports through this one type.
class ValidationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The propagator was mutated after a functional captured what it replays: a rebuilt graph leaves the
// functional's parameter mapping describing a graph that is gone, a re-weight leaves its snapshotted
// operator coefficients stale.
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

auto validate_functional_state(const FunctionalState &state) -> void {
    // Aliveness first: every other handle a functional holds points into the propagator, so there is
    // nothing else it may legally read once this is false.
    if (!state.propagator_alive) {
        throw StaleFunctionalGraph("The propagator this functional was built from has been destroyed. "
                                   "A functional reads the propagator's operator index directly, so it "
                                   "cannot outlive it; keep the propagator alive, or build the functional "
                                   "again from a live one.");
    }
    if (state.current_revision != state.expected_revision) {
        throw StaleFunctionalGraph(std::format(
            "MP object has been modified since the functional was created: {} changed the "
            "graph or operator the functional replays. Create a new functional.",
            state.last_structural_change != nullptr ? state.last_structural_change : "a structural mutation"));
    }
    // No bump, but the operator moved anyway: some mutation is missing its bump_structure_ call. Report
    // it as staleness rather than read a rebuilt inverted index through a pointer to the old one.
    if (!state.operator_layout_unchanged) {
        throw StaleFunctionalGraph("MP object has been modified since the functional was created: the "
                                   "operator index it reads was rebuilt or grown. Create a new functional.");
    }
}

auto validate_weight_refresh(const WeightRefresh &refresh) -> void {
    // A pared plan holds the layers pare_graph kept, and Schrodinger thresholds that keep-set from the
    // operator coefficients themselves -- so new coefficients would need a different keep-set, and
    // replaying this one would silently answer for a paring nobody asked for. Heisenberg thresholds the
    // state, which a re-weight does not touch, so it follows exactly.
    if (refresh.pared_from_operator) {
        throw StaleFunctionalGraph("MP object has been modified since the functional was created: the "
                                   "initial operator was re-weighted, and this functional pares its graph "
                                   "against the operator coefficients (a Schrodinger picture functional "
                                   "with a pare_threshold), so it cannot follow the new weights. Create a "
                                   "new functional.");
    }
    // Unreachable while every structural mutation bumps and every publication carries the revision it
    // was made at: a functional whose revision still matches the propagator's cannot see weights from
    // another revision. Kept as the backstop for the day one of those two stops being true.
    if (refresh.weights_revision != refresh.expected_revision) {
        throw StaleFunctionalGraph("MP object has been modified since the functional was created: the "
                                   "initial-operator weights it reads belong to a different graph. Create "
                                   "a new functional.");
    }
}

auto validate_only_rotate_len_k_(std::optional<size_t> only_rotate_len_k, size_t max_k) -> void {
    if (!only_rotate_len_k.has_value()) {
        return;
    }

    const auto k = *only_rotate_len_k;
    if (k == 0 || static_cast<size_t>(k) > max_k) {
        throw ValidationError(std::format("only_rotate_len_k={} is out of range; must be 0 < k <= 2*num_qubits", k));
    }
}

auto expected_num_params(const VecZ &parameter_mapping) -> size_t {
    return parameter_mapping.empty() ? 0 : *std::ranges::max_element(parameter_mapping) + 1;
}

// NOLINTEND(misc-use-internal-linkage)

} // namespace monoprop
