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

    auto bump(const char *site) -> void {
        last_structural_change.store(site);
        structure_revision.fetch_add(1);
    }
};

// Records one mutation on the control block when its scope exits, so a mutator states its site name
// and its arming predicate once instead of once per exit path.
class [[nodiscard]] MutationGuard {
public:
    // `on_success` false leaves a completed mutation unrecorded, which is what a re-weight needs: a
    // functional follows it, and only one that unwound part-way invalidates.
    MutationGuard(FunctionalControl &control,
                  const char *site,
                  bool armed,
                  bool on_success,
                  const bool *facade_diverged)
        : control_(control),
          site_(site),
          armed_(armed),
          on_success_(on_success),
          facade_diverged_(facade_diverged),
          uncaught_(std::uncaught_exceptions()) {}

    MutationGuard(const MutationGuard &) = delete;
    MutationGuard(MutationGuard &&) = delete;
    auto operator=(const MutationGuard &) -> MutationGuard & = delete;
    auto operator=(MutationGuard &&) -> MutationGuard & = delete;

    ~MutationGuard() {
        if (!armed_) {
            return;
        }
        const bool unwinding = std::uncaught_exceptions() > uncaught_;
        if (!unwinding && !on_success_) {
            return;
        }
        control_.bump(site_);
        // Only a mutation that stopped part-way can have left the partitions disagreeing.
        if (unwinding && facade_diverged_ != nullptr && *facade_diverged_) {
            control_.partition_fault.store(site_);
        }
    }

private:
    FunctionalControl &control_;
    const char *site_;
    bool armed_;
    bool on_success_;
    const bool *facade_diverged_;
    int uncaught_;
};

} // namespace monoprop::detail
