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

// White-box access for tests only. MonomialPropagator befriends this template but the library never
// defines it, so it adds no production API: tests use it to observe the captured thread budget, to
// inject failures at real mutation and evaluation boundaries through existing members, to inspect the
// root's actual shards and to attach the test-only RootObserver.

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "monoprop/Evolution.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/parallel/Options.h"
#include "monoprop/detail/sharded/RootObserver.h"
#include "monoprop/detail/sharded/State.h"

namespace monoprop::detail {

template <size_t NumModes>
struct PropagatorTestAccess {
    using Propagator = MonomialPropagator<NumModes>;

    static auto options(const Propagator &p) -> parallel::Options { return p.parallel_; }

    static auto is_invalid(const Propagator &p) -> bool { return p.invalid_; }

    // The existing protected cutoff predicate: every scan evaluates it, so a throwing one fails a real
    // build after the operation has started mutating.
    static auto set_cutoff_fn(Propagator &p, CutoffFn<NumModes> fn) -> void { p.cutoff_fn_ = std::move(fn); }

    static auto clone(const Propagator &p) -> std::unique_ptr<Propagator> { return p.clone_(); }

    using Observer = sharded::RootObserver;

    // The root's actual shard states, for private inspection; never a production accessor.
    static auto shards(const Propagator &p) -> const sharded::Shards<NumModes> & { return p.shards_; }

    // The root's physical rounds, reused across its multi-rank operations.
    static auto rounds(const Propagator &p) -> const sharded::PhysicalRounds * { return p.rounds_.get(); }

    // The (P, T) router the root prepared at construction.
    static auto router(const Propagator &p) -> const routing::Router & { return *p.router_; }

    static auto core_term(const Propagator &p) -> double { return p.core_term_; }

    static auto epoch(const Propagator &p) -> size_t { return p.initial_operator_epoch_; }

    // Attach (or, with null, detach) a test observer; copies made while attached inherit it.
    static auto observe(Propagator &p, const Observer *observer) -> void { p.observer_ = observer; }

    // Construct through the public constructor's body with `observer` attached, so seeding is observed too.
    template <typename... Args>
    static auto construct_observed(const Observer *observer, Args &&...args) -> std::unique_ptr<Propagator> {
        return std::unique_ptr<Propagator>(
            new Propagator(typename Propagator::ObservedTag{}, observer, std::forward<Args>(args)...));
    }

    // One energy evaluation through the evaluation seam with a caller-local physical round (rounds = nullptr), as an
    // external caller of the exported evaluate_shards() runs it, at the root's own geometry and world; the root's
    // persistent rounds are not used. The outcome, error included, is returned after the team has joined.
    static auto evaluate_with_local_round(Propagator &p,
                                          const VecD &params,
                                          const sharded::EvaluationObserver *observer) -> sharded::EvaluationOutcome {
        bool mutation_started = false;
        const auto retained = p.prepare_retained_(std::nullopt, mutation_started);
        const auto requests = retained.requests(params);
        return sharded::evaluate_shards(requests, retained.callbacks, retained.options, false, observer, p.world_);
    }

    // One graph-only layer (no coefficients, not appended to the graph) built by the low-level one-owner-per-rank
    // engine (build_layer) on the sole shard of a T = 1 root, over the root's communicator, with an explicit kernel
    // budget and a test observer on the kernels' ranges, under the root's own operation guard: a failure after the
    // layer started counts as a post-mutation failure (invalidation, then mpi::operation_failed). The shard may grow,
    // so the object is for inspection only.
    template <class KernelObserver>
    static auto sole_shard_layer_observed(Propagator &p,
                                          const VecZ &gate,
                                          parallel::Options kernels,
                                          const KernelObserver &observer) -> std::shared_ptr<LayerCore> {
        return p.run_operation_(true, [&](bool &mutation_started) {
            auto &state = p.sole_shard_("sole_shard_layer_observed()");
            const auto gen = indices_to_bitset_checked<NumModes>(gate, 2 * p.logical_num_modes_);
            mutation_started = true;
            return build_layer<NumModes>(state.op,
                                         gen,
                                         p.cutoff_fn_,
                                         p.lower_atol_,
                                         std::nullopt,
                                         p.upper_atol_,
                                         std::nullopt,
                                         std::nullopt,
                                         state.matched,
                                         p.comm_,
                                         nullptr,
                                         nullptr,
                                         p.schrodinger_,
                                         nullptr,
                                         nullptr,
                                         p.basis_,
                                         kernels,
                                         observer);
        });
    }
};

} // namespace monoprop::detail
