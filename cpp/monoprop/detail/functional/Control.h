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
#include <exception>
#include <memory>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// Immutable initial-operator weights, atomically published on re-weight.
// A single load keeps `op` and `core_term` from one publication.
struct OperatorWeights {
    VecD op;                      // One coefficient per store row.
    double core_term{0.0};        // Identity contribution to the expectation value.
    size_t structure_revision{0}; // Revision at publication.
};

// Validity state shared by a propagator and its functionals.
// Copies start with a fresh block.
struct FunctionalControl {
    // Bumped when replayed structure changes.
    std::atomic<size_t> structure_revision{0};

    // Cleared before propagator destruction.
    std::atomic<bool> propagator_alive{true};

    // Last structural change, as a string literal.
    std::atomic<const char *> last_structural_change{nullptr};

    // Current weights, null until the first functional plan. Written by the propagator and read by
    // functional calls.
    std::atomic<std::shared_ptr<const OperatorWeights>> weights{};

    // Record a replayed-structure change. `site` must outlive the propagator.
    auto bump(const char *site) -> void {
        last_structural_change.store(site);
        structure_revision.fetch_add(1);
    }
};

// Bumps the control block when a mutation throws after it may have changed replayed state.
// Construct after validation and before the first write. Set `armed` to false for known no-ops;
// the caller records successful changes.
class [[nodiscard]] BumpOnUnwind {
public:
    // `site` must outlive the propagator.
    BumpOnUnwind(FunctionalControl &control, const char *site, bool armed = true)
        : control_(control),
          site_(site),
          armed_(armed),
          uncaught_(std::uncaught_exceptions()) {}

    BumpOnUnwind(const BumpOnUnwind &) = delete;
    BumpOnUnwind(BumpOnUnwind &&) = delete;
    auto operator=(const BumpOnUnwind &) -> BumpOnUnwind & = delete;
    auto operator=(BumpOnUnwind &&) -> BumpOnUnwind & = delete;

    ~BumpOnUnwind() {
        // This also works when called during another unwind.
        if (armed_ && std::uncaught_exceptions() > uncaught_) {
            control_.bump(site_);
        }
    }

private:
    FunctionalControl &control_;
    const char *site_;
    bool armed_;
    int uncaught_;
};

} // namespace monoprop::detail
