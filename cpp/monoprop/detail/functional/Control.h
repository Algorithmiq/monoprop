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
#include <mutex>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// Initial-operator weights published together.
struct OperatorWeights {
    VecD op;                      // One coefficient per store row.
    double core_term{0.0};        // Identity contribution.
    size_t structure_revision{0}; // Publication revision.
};

// Thread-safe slot for the current weights.
class WeightsSlot {
public:
    auto load() const -> std::shared_ptr<const OperatorWeights> {
        const std::lock_guard lock(mutex_);
        return weights_;
    }

    auto store(std::shared_ptr<const OperatorWeights> weights) -> void {
        const std::lock_guard lock(mutex_);
        weights_ = std::move(weights);
    }

private:
    mutable std::mutex mutex_; // Allows reads through const FunctionalControl.
    std::shared_ptr<const OperatorWeights> weights_;
};

// State shared by a propagator and its functionals.
struct FunctionalControl {
    // Changes when replayed structure changes.
    std::atomic<size_t> structure_revision{0};

    // Cleared before destruction.
    std::atomic<bool> propagator_alive{true};

    // Last structural change.
    std::atomic<const char *> last_structural_change{nullptr};

    // Current weights, null until the first plan.
    WeightsSlot weights;

    // Names the fan-out mutation that unwound part-way, or null while the facade is intact.
    std::atomic<const char *> partition_fault{nullptr};

    // Record a structure change.
    auto bump(const char *site) -> void {
        last_structural_change.store(site);
        structure_revision.fetch_add(1);
    }
};

// Bumps the control block if a mutation throws after a possible state change.
class [[nodiscard]] BumpOnUnwind {
public:
    BumpOnUnwind(FunctionalControl &control, const char *site, bool armed = true, const bool *facade_diverged = nullptr)
        : control_(control),
          site_(site),
          armed_(armed),
          facade_diverged_(facade_diverged),
          uncaught_(std::uncaught_exceptions()) {}

    BumpOnUnwind(const BumpOnUnwind &) = delete;
    BumpOnUnwind(BumpOnUnwind &&) = delete;
    auto operator=(const BumpOnUnwind &) -> BumpOnUnwind & = delete;
    auto operator=(BumpOnUnwind &&) -> BumpOnUnwind & = delete;

    ~BumpOnUnwind() {
        if (armed_ && std::uncaught_exceptions() > uncaught_) {
            control_.bump(site_);
            if (facade_diverged_ != nullptr && *facade_diverged_) {
                control_.partition_fault.store(site_);
            }
        }
    }

private:
    FunctionalControl &control_;
    const char *site_;
    bool armed_;
    const bool *facade_diverged_;
    int uncaught_;
};

} // namespace monoprop::detail
