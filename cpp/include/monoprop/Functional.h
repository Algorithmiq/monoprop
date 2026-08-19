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
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Validation.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/functional/Control.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/operator/MPOperator.h"
#include "monoprop/detail/partition/PartitionGroup.h"

namespace monoprop {

template <size_t NumModes>
class MonomialPropagator;

namespace detail {

/// One propagator snapshot a functional replays, plus the checks that say the snapshot is still that
/// propagator's own.
///
/// Immutable once built and held by `shared_ptr<const>`, so the value and the gradient functional over
/// the same snapshot share one plan. Every field is either owned or, where the comment says so,
/// borrowed from the propagator — which is why a functional must not outlive it.
template <size_t NumModes>
class FunctionalPlan {
public:
    /// A single-partition propagator's snapshot: one replay of its graph against its operator.
    struct Local {
        double core_term{0.0}; ///< the identity term, added to the summed expectation value
        // Owns its rows and snapshots the term count: the operator's sparse rows grow by push_back as
        // terms are appended, so a view would both dangle and outrun `op`.
        EvalState state;        ///< the contraction partner, sparse (Heisenberg) or dense (Schrodinger)
        VecD op;                ///< un-evolved operator coefficients, copied out of the propagator
        VecZ parameter_mapping; ///< optimizer order: which parameter drives graph layer i
        VecD gen_coeffs;        ///< optimizer order, parallel to parameter_mapping
        // One owning handle either way: pare hands back a heap-owned MPGraph the plan must keep alive
        // (`cos` holds pointers into its layers' stored cos); non-pare aliases the propagator's graph_
        // through a shared_ptr with an empty owner block, so it stays live only while the propagator does.
        std::shared_ptr<const MPGraph> graph;
        // The folds keep raw column pointers into the propagator's inverted index, so this plan must not
        // outlive the propagator either.
        CosCallbacks cos;
        mpi::Comm comm{}; ///< real MPI across nodes, or the in-process comm across partitions

        // The operator-layout backstop. `mp_op` is borrowed and read only after the alive flag says the
        // propagator is still there; the other two are what the borrowed inverted index was built over.
        // Compared before any use of that index, so a mutation that forgot its revision bump still
        // reports staleness rather than folding a rebuilt index through a pointer to the old one.
        const MPOperator<NumModes> *mp_op{nullptr};
        const OperatorIndex<NumModes> *op_store{nullptr};
        size_t inverted_index_rows{0};
    };

    /// A partition facade's snapshot: one child plan per partition, replayed together.
    struct Fanout {
        // Borrowed, like every other field: the group belongs to the facade propagator.
        partition::PartitionGroup<NumModes> *group{nullptr};
        std::vector<std::shared_ptr<const FunctionalPlan>> partitions; ///< in partition order
    };

    /// `control` is the propagator's own block; the plan pins the revision it is built at.
    FunctionalPlan(size_t num_params, std::shared_ptr<const FunctionalControl> control, Local local)
        : num_params_(num_params),
          control_(std::move(control)),
          expected_revision_(control_->structure_revision.load()),
          shape_(std::move(local)) {}

    FunctionalPlan(size_t num_params, std::shared_ptr<const FunctionalControl> control, Fanout fanout)
        : num_params_(num_params),
          control_(std::move(control)),
          expected_revision_(control_->structure_revision.load()),
          shape_(std::move(fanout)) {}

    /// The parameter-axis length the plan was built against; a call must supply exactly this many.
    auto num_params() const -> size_t { return num_params_; }

    /// Throw unless `params` fits and the propagator still holds what the plan replays.
    // A facade checks its own control block here -- the group it fans out over belongs to the facade --
    // and each child plan then checks its own partition's, on that partition's master thread. Only the
    // single-partition shape has an operator to run the layout backstop against.
    auto validate(const VecD &params) const -> void {
        // Aliveness is settled here rather than left to validate_functional_state: the layout backstop
        // reads the propagator's operator, and every argument is evaluated before the callee runs, so a
        // dead propagator has to drop out of the argument list itself. The control block is shared, so it
        // stays readable after the propagator is gone -- nothing else the plan holds does.
        const bool alive = control_->propagator_alive.load();
        const auto *local = alive ? std::get_if<Local>(&shape_) : nullptr;
        validate_functional_state({.propagator_alive = alive,
                                   .current_revision = control_->structure_revision.load(),
                                   .expected_revision = expected_revision_,
                                   .operator_layout_unchanged = local == nullptr || operator_layout_unchanged(*local),
                                   .last_structural_change = control_->last_structural_change.load()});
        validate_functional_call(params, num_params_);
    }

    /// Replay the snapshot: `fn(request, comm, cos)` locally, or partition 0's answer on a facade.
    template <typename Fn,
              typename R = std::invoke_result_t<Fn &, const EvalRequest &, mpi::Comm, const CosCallbacks &>>
    auto evaluate(Fn &&fn, const VecD &params) const -> R {
        validate(params);
        if (const auto *fanout = std::get_if<Fanout>(&shape_)) {
            // Each partition allreduces internally, so partition 0 already carries the global answer.
            // The fan-out must reach every master: the partitions' collectives are barrier-synced.
            return partition::collect_on_all(*fanout->group, [&](int r) -> R {
                return fanout->partitions[static_cast<size_t>(r)]->evaluate(fn, params);
            })[0];
        }
        const auto &local = std::get<Local>(shape_);
        return fn(EvalRequest{.e_core = local.core_term,
                              .state = local.state,
                              .op = local.op,
                              .parameter_mapping = local.parameter_mapping,
                              .gen_coeffs = local.gen_coeffs,
                              .graph = local.graph->replay_view(),
                              .params = params},
                  local.comm,
                  local.cos);
    }

private:
    // Read straight off the borrowed operator, never through inverted_index(): that accessor rebuilds a
    // stale index, which is a write, and a plan must not write to its propagator.
    static auto operator_layout_unchanged(const Local &local) -> bool {
        return local.mp_op->store.get() == local.op_store && local.mp_op->inverted_index_.has_value()
               && local.mp_op->inverted_index_->rows() == local.inverted_index_rows;
    }

    size_t num_params_{0};
    std::shared_ptr<const FunctionalControl> control_;
    size_t expected_revision_{0};
    std::variant<Local, Fanout> shape_;
};

} // namespace detail

/// A reusable expectation value over one propagator snapshot: `fn(parameters) -> double`.
///
/// Built by MonomialPropagator::expectation_value_functional(). It borrows from the propagator that
/// made it (see detail::FunctionalPlan), so it must not outlive it, and a structural change to the
/// propagator makes a call throw rather than answer.
template <size_t NumModes>
class ExpectationValueFunctional {
public:
    /// The parameter-axis length this functional was built against.
    auto num_params() const -> size_t { return plan_->num_params(); }

    auto operator()(const VecD &parameters) const -> double {
        return plan_->evaluate(
            [](const EvalRequest &request, mpi::Comm comm, const detail::CosCallbacks &cos) -> double {
                return ev(request, comm, cos);
            },
            parameters);
    }

private:
    friend class MonomialPropagator<NumModes>;

    explicit ExpectationValueFunctional(std::shared_ptr<const detail::FunctionalPlan<NumModes>> plan)
        : plan_(std::move(plan)) {}

    std::shared_ptr<const detail::FunctionalPlan<NumModes>> plan_;
};

/// As ExpectationValueFunctional, plus the gradient from the same backward pass:
/// `fn(parameters) -> (value, gradient)`, the gradient in parameter-axis order.
template <size_t NumModes>
class ExpectationValueAndGradientFunctional {
public:
    /// The parameter-axis length this functional was built against.
    auto num_params() const -> size_t { return plan_->num_params(); }

    auto operator()(const VecD &parameters) const -> std::pair<double, VecD> {
        return plan_->evaluate(
            [](const EvalRequest &request, mpi::Comm comm, const detail::CosCallbacks &cos) -> std::pair<double, VecD> {
                return ev_and_grad(request, comm, cos);
            },
            parameters);
    }

private:
    friend class MonomialPropagator<NumModes>;

    explicit ExpectationValueAndGradientFunctional(std::shared_ptr<const detail::FunctionalPlan<NumModes>> plan)
        : plan_(std::move(plan)) {}

    std::shared_ptr<const detail::FunctionalPlan<NumModes>> plan_;
};

} // namespace monoprop
