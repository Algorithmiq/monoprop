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
#include <cstdint>

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/sharded/Construction.h"
#include "monoprop/detail/sharded/Evaluation.h"
#include "monoprop/detail/sharded/State.h"

/*
 * Test-only observation of the sharded prototype root (monoprop_SHARDED_OPENMP_PROTOTYPE).
 *
 * The rank-level MonomialPropagator holds a pointer to a RootObserver that is null in production and can be set only
 * through the test suite's PropagatorTestAccess. Every seam the root calls -- seeding, copying, construction,
 * evaluation, replay, retained preparation and the root's own owner teams -- reports to it through the adapters
 * below, which compile to a null check when no observer is set. An observer method runs inside the protected phase
 * of the work it reports, on the thread performing that work, so recording the executing worker observes actual
 * participation, and throwing injects a real failure at that point. Observations are evidence collected by tests,
 * never a production geometry check.
 */

namespace monoprop::detail::sharded {

//! Work the root performs in its own owner teams or on the caller, as reported to RootObserver::root().
enum class RootWork : std::uint8_t {
    initial_operator, //!< Owner phase: apply this shard's share of an initial-operator update.
    remap,            //!< Owner phase: relabel this shard's layers for a new parameter mapping.
    picture,          //!< Owner phase: copy this shard's current picture (contraction at empty parameters).
    export_block,     //!< Caller: decode this shard's evolved block for evolved_operator_terms().
    combine,          //!< Caller, after the evaluation team joined: fold and reduce the shards' results.
};

/*!
 * \brief Test-only observer of the prototype root; every method defaults to doing nothing.
 *
 * Inherits the evaluation seam's visit(), so one object observes evaluation, replay and retained preparation too.
 */
class RootObserver : public EvaluationObserver {
public:
    //! Evaluation, replay and retained-preparation work (see EvaluationObserver); may throw.
    auto visit(EvaluationWork /*work*/, size_t /*step*/, size_t /*shard*/) const -> void override {}

    //! Seed or copy work of `shard` (see NoShardObserver): `end` is false before, true after; may throw.
    virtual auto shard_work(ShardWork /*work*/, bool /*end*/, size_t /*shard*/) const -> void {}

    //! Construction work (see NoConstructionObserver); may throw.
    virtual auto construction(ConstructionWork /*work*/, size_t /*step*/, size_t /*shard*/) const -> void {}

    //! The root's own work; `shard` is the shard concerned. May throw.
    virtual auto root(RootWork /*work*/, size_t /*shard*/) const -> void {}
};

//! The seed_shards() / copy_shards() observer the root passes: forwards to a RootObserver when one is set.
struct RootShardObserver {
    const RootObserver *observer = nullptr; //!< Null in production.

    //! Before the owner's substantial allocation.
    auto begin(ShardWork work, size_t shard) const -> void {
        if (observer != nullptr) {
            observer->shard_work(work, false, shard);
        }
    }

    //! After the owner's state is complete, before it is published.
    auto end(ShardWork work, size_t shard) const -> void {
        if (observer != nullptr) {
            observer->shard_work(work, true, shard);
        }
    }
};

//! The construction observer the root passes: forwards visits; kernels report to no range observer.
struct RootConstructionObserver {
    const RootObserver *observer = nullptr; //!< Null in production.

    //! Before `work` of `step` for `shard`.
    auto visit(ConstructionWork work, size_t step, size_t shard) const -> void {
        if (observer != nullptr) {
            observer->construction(work, step, shard);
        }
    }

    //! Within-shard kernels and exchange phases report to the production default.
    [[nodiscard]] auto kernels(size_t /*shard*/) const noexcept -> NoRangeObserver { return {}; }
};

/*!
 * \brief Report the root's own work to `observer` when one is set.
 */
inline auto notify_root(const RootObserver *observer, RootWork work, size_t shard) -> void {
    if (observer != nullptr) {
        observer->root(work, shard);
    }
}

} // namespace monoprop::detail::sharded
