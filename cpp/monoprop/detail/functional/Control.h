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

#include <atomic>
#include <cstddef>
#include <memory>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// Immutable initial-operator weights, published together on re-weight.
//
// Re-weighting preserves the store, inverted index and graph. A single atomic load keeps `op` and
// `core_term` from the same publication.
struct OperatorWeights {
    VecD op;                      // One coefficient per store row.
    double core_term{0.0};        // Identity contribution to the expectation value.
    size_t structure_revision{0}; // Revision at publication.
};

// Shared validity state for a propagator and its functionals.
// A copy has a fresh block because it has no functionals.
struct FunctionalControl {
    // Bumped when a replayed graph layer, parameter label, or operator row changes.
    std::atomic<size_t> structure_revision{0};

    // Cleared before propagator members are destroyed.
    std::atomic<bool> propagator_alive{true};

    // String literal naming the last structural change.
    std::atomic<const char *> last_structural_change{nullptr};

    // Current weights. Null until the first functional plan is built.
    // The propagator writes; functional calls read.
    std::atomic<std::shared_ptr<const OperatorWeights>> weights{};
};

} // namespace monoprop::detail
