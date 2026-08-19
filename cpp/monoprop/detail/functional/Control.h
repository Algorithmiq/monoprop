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

// The initial-operator weights an evaluation runs against, published as one immutable set.
//
// A re-weight writes only these two fields: MPOperator::update_initial_operator cannot add a store row,
// and apply_initial_operator_ writes core_term_ beside it, so the store, the inverted index and the graph
// all stay put. That is what lets a functional follow a re-weight instead of going stale. A published set
// is never edited, so one atomic load gives a caller a pair that belong together -- `op` and `core_term`
// read separately could come from two re-weights.
struct OperatorWeights {
    VecD op;                      // un-evolved operator coefficients, one per store row
    double core_term{0.0};        // the identity term, added to the summed expectation value
    size_t structure_revision{0}; // the FunctionalControl revision these weights were published at
};

// The validity block a propagator shares with every functional plan it makes.
//
// A plan borrows from its propagator, so before it reads any of those handles it needs two facts its
// own snapshot cannot supply: whether the propagator is still there, and whether the structure the
// snapshot describes is still the propagator's. Both live here, behind one shared_ptr, so a plan
// answers them without dereferencing the propagator at all. The propagator holds the only mutating
// handle; plans hold shared_ptr<const>.
//
// A copied propagator gets its own block: a copy carries no functionals, so it starts at revision 0.
struct FunctionalControl {
    // Bumped by every change to what a plan replays -- the graph's layers, their parameter labels, or
    // the operator's rows. Deliberately NOT bumped by the settings that only gate the next build (the
    // atols, the cutoff, the cutoff type, the basis change): none of them touches a plan's snapshot.
    std::atomic<size_t> structure_revision{0};

    // Cleared by ~MonomialPropagator, which runs before its members go away. A plan that outlives its
    // propagator must report that instead of reading through handles into freed memory.
    std::atomic<bool> propagator_alive{true};

    // The method that last bumped structure_revision, so the error can name it. Always a string
    // literal, whose lifetime outlives every propagator.
    std::atomic<const char *> last_structural_change{nullptr};

    // The live initial-operator weights, republished by every re-weight and by the first plan built at a
    // given revision. Null until the first plan is built: a propagator with no functionals has nobody to
    // publish for, and publishing is a copy of `op`.
    //
    // Written only by the propagator, on its own thread (a facade publishes through for_each_partition_,
    // so each partition publishes on its own pinned master). Read by a call through the plan.
    std::atomic<std::shared_ptr<const OperatorWeights>> weights{};
};

} // namespace monoprop::detail
