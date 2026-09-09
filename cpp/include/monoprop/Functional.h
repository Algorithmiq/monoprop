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

/// Immutable, shared functional replay plan.
/// Borrowed fields require the functional to outlive neither its propagator nor its partitions.
template <size_t NumModes>
class FunctionalPlan {
public:
    /// A single-partition propagator's snapshot: one replay of its graph against its operator.
    struct Local {
        // Build-time weights; also used until a re-weight publishes new weights.
        std::shared_ptr<const OperatorWeights> weights;
        // Owned snapshot: operator rows can grow, but `op` cannot.
        EvalState state;        ///< the contraction partner, sparse (Heisenberg) or dense (Schrodinger)
        VecZ parameter_mapping; ///< optimizer order: which parameter drives graph layer i
        VecD gen_coeffs;        ///< optimizer order, parallel to parameter_mapping
        // Owned because `cos` holds raw pointers into graph layers.
        std::shared_ptr<const MPGraph> graph;
        // `cos` borrows columns from the propagator's inverted index.
        CosCallbacks cos;
        mpi::Comm comm{}; ///< real MPI across nodes, or the in-process comm across partitions

        // Borrowed operator and inverted-index identity check.
        // This catches a missing revision bump before the stale index is used.
        const MPOperator<NumModes> *mp_op{nullptr};
        const OperatorIndex<NumModes> *op_store{nullptr};
        size_t inverted_index_rows{0};

        // A coefficient-pared Schrodinger graph cannot follow a re-weight.
        bool pared_from_operator{false};
    };

    /// A partition facade's snapshot: one child plan per partition, replayed together.
    struct Fanout {
        // Borrowed from the facade propagator.
        partition::PartitionGroup<NumModes> *group{nullptr};
        std::vector<std::shared_ptr<const FunctionalPlan>> partitions; ///< in partition order
    };

    /// Pins the propagator control block and its current revision.
    FunctionalPlan(std::shared_ptr<const FunctionalControl> control, std::variant<Local, Fanout> shape)
        : control_(std::move(control)),
          expected_revision_(control_->structure_revision.load()),
          shape_(std::move(shape)),
          num_params_(derive_num_params(shape_)) {
        // A re-weight publishes only while a plan is reading; see FunctionalControl.
        control_->add_plan();
    }

    FunctionalPlan(const FunctionalPlan &) = delete;
    FunctionalPlan(FunctionalPlan &&) = delete;
    auto operator=(const FunctionalPlan &) -> FunctionalPlan & = delete;
    auto operator=(FunctionalPlan &&) -> FunctionalPlan & = delete;

    ~FunctionalPlan() { control_->drop_plan(); }

    /// Required parameter-axis length.
    auto num_params() const -> size_t { return num_params_; }

    /// Whether calls may follow re-weighted coefficients.
    auto follows_weights() const -> bool {
        if (const auto *fanout = std::get_if<Fanout>(&shape_)) {
            // Child plans share picture and threshold.
            return fanout->partitions.front()->follows_weights();
        }
        return !std::get<Local>(shape_).pared_from_operator;
    }

    /// Throw unless `params` and the propagator still match this plan.
    // The facade validates its group; child plans validate on their partition masters.
    auto validate(const VecD &params) const -> void {
        // Check liveness before reading the borrowed operator for the layout check.
        const bool alive = control_->propagator_alive.load();
        const auto *local = alive ? std::get_if<Local>(&shape_) : nullptr;
        validate_functional_state({.propagator_alive = alive,
                                   .current_revision = control_->structure_revision.load(),
                                   .expected_revision = expected_revision_,
                                   .operator_layout_unchanged = local == nullptr || operator_layout_unchanged(*local),
                                   .last_structural_change = control_->last_structural_change.load()});
        if (std::holds_alternative<Fanout>(shape_)) {
            // evaluate() reaches the group directly, not through MonomialPropagator::partitions_(), so
            // the door's check is repeated here -- after the revision, which every latch to date bumps too.
            validate_partition_facade_intact(control_->partition_fault.load());
        }
        validate_functional_call(params, num_params_);
    }

    /// Replay locally, or return partition 0's facade result.
    template <typename Fn,
              typename R = std::invoke_result_t<Fn &, const EvalRequest &, mpi::Comm, const CosCallbacks &>>
    auto evaluate(Fn &&fn, const VecD &params) const -> R {
        validate(params);
        if (const auto *fanout = std::get_if<Fanout>(&shape_)) {
            // Every partition must join its synchronized collective; partition 0 has the result.
            return std::move(partition::collect_on_all(*fanout->group, [&](int r) -> R {
                return fanout->partitions[static_cast<size_t>(r)]->evaluate(fn, params);
            })[0]);
        }
        const auto &local = std::get<Local>(shape_);
        // Keeps `request.op` alive for the call.
        const auto weights = resolve_weights(local);
        return fn(EvalRequest{.e_core = weights->core_term,
                              .state = local.state,
                              .op = weights->op,
                              .parameter_mapping = local.parameter_mapping,
                              .gen_coeffs = local.gen_coeffs,
                              .graph = local.graph->replay_view(),
                              .params = params},
                  local.comm,
                  local.cos);
    }

private:
    // Load matching `op` and `core_term` from the current weight publication.
    auto resolve_weights(const Local &local) const -> std::shared_ptr<const OperatorWeights> {
        auto published = control_->weights.load();
        // No publication or no re-weight: build-time weights are current.
        if (published == nullptr || published == local.weights) {
            return local.weights;
        }
        validate_weight_refresh({.weights_revision = published->structure_revision,
                                 .expected_revision = expected_revision_,
                                 .may_follow_weights = follows_weights()});
        return published;
    }

    // Do not call inverted_index(): it could rebuild the borrowed index.
    static auto operator_layout_unchanged(const Local &local) -> bool {
        return local.mp_op->store.get() == local.op_store && local.mp_op->inverted_index_.has_value()
               && local.mp_op->inverted_index_->rows() == local.inverted_index_rows;
    }

    // Read off the shape rather than passed in, so a plan cannot declare a parameter axis its own
    // mapping disagrees with.
    static auto derive_num_params(const std::variant<Local, Fanout> &shape) -> size_t {
        if (const auto *fanout = std::get_if<Fanout>(&shape)) {
            // Child plans share graph structure, so any of them gives the axis.
            return fanout->partitions.front()->num_params();
        }
        return expected_num_params(std::get<Local>(shape).parameter_mapping);
    }

    std::shared_ptr<const FunctionalControl> control_;
    size_t expected_revision_{0};
    std::variant<Local, Fanout> shape_;
    size_t num_params_{0};
};

/// Shared functional handle; derived types differ only in `operator()`.
template <size_t NumModes>
class FunctionalHandle {
public:
    /// Required parameter-axis length.
    auto num_params() const -> size_t { return plan_->num_params(); }

    /// Whether calls may follow a re-weight.
    auto follows_weights() const -> bool { return plan_->follows_weights(); }

protected:
    explicit FunctionalHandle(std::shared_ptr<const FunctionalPlan<NumModes>> plan) : plan_(std::move(plan)) {}

    std::shared_ptr<const FunctionalPlan<NumModes>> plan_;
};

} // namespace detail

/// Reusable expectation value: `fn(parameters) -> double`.
///
/// Borrows its propagator and throws after structural mutation. It follows re-weighted initial-operator
/// coefficients unless its graph was coefficient-pared.
template <size_t NumModes>
class ExpectationValueFunctional : public detail::FunctionalHandle<NumModes> {
public:
    auto operator()(const VecD &parameters) const -> double { return this->plan_->evaluate(ev, parameters); }

private:
    friend class MonomialPropagator<NumModes>;

    explicit ExpectationValueFunctional(std::shared_ptr<const detail::FunctionalPlan<NumModes>> plan)
        : detail::FunctionalHandle<NumModes>(std::move(plan)) {}
};

/// Expectation value and gradient from one backward pass.
/// `fn(parameters) -> (value, gradient)`, with the gradient in parameter-axis order.
template <size_t NumModes>
class ExpectationValueAndGradientFunctional : public detail::FunctionalHandle<NumModes> {
public:
    auto operator()(const VecD &parameters) const -> std::pair<double, VecD> {
        return this->plan_->evaluate(ev_and_grad, parameters);
    }

private:
    friend class MonomialPropagator<NumModes>;

    explicit ExpectationValueAndGradientFunctional(std::shared_ptr<const detail::FunctionalPlan<NumModes>> plan)
        : detail::FunctionalHandle<NumModes>(std::move(plan)) {}
};

} // namespace monoprop
