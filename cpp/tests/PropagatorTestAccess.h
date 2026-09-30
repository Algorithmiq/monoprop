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
// defines it, so it adds no production API: tests use it to observe the captured thread budget and to
// inject failures at real mutation and evaluation boundaries through existing protected members.

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
#include "monoprop/detail/parallel/Options.h"
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

    // The existing protected functional factory, with a caller-supplied evaluation body.
    template <typename Fn>
    static auto make_functional(Propagator &p, Fn &&fn, std::optional<double> pare_threshold = std::nullopt) {
        return p.make_functional_(std::forward<Fn>(fn), pare_threshold);
    }

    static auto clone(const Propagator &p) -> std::unique_ptr<Propagator> { return p.clone_(); }

    // A deep copy of this single store's operator, graph and matched marks as one shard state, behind the
    // propagator's own validity guard: an invalid owner is rejected before anything is copied. A test bridge from
    // legacy stores to sharded state, not a production conversion.
    static auto shard_state(const Propagator &p) -> std::unique_ptr<sharded::ShardState<NumModes>> {
        p.require_valid_();
        auto state = std::make_unique<sharded::ShardState<NumModes>>(p.mp_op_, p.schrodinger_);
        state->graph = p.graph_;
        state->matched = p.matched_scratch_;
        return state;
    }

    // The live coefficients of the propagator's picture: the operator (Heisenberg) or the state (Schrödinger).
    static auto picture_coeffs(const Propagator &p) -> const VecD & {
        return p.schrodinger_ ? p.mp_op_.state_coeffs : p.mp_op_.op_coeffs;
    }

    // One gate of the ContractImmediately path (evolve_mode_contract_immediately_) up to, not including,
    // apply_fused_contract: the real build and coefficient extension, so a test can inspect the records the
    // build produced and run the apply itself on the returned coefficient vector.
    struct ContractedGate {
        FusedContract fc;
        CosMask cos;
        bool fused_scale = false;
        VecD *coeffs = nullptr; // the picture's live coefficients, already extended
        double apply_angle = 0.0;
    };
    static auto contract_gate(Propagator &p,
                              const VecZ &gate,
                              std::optional<size_t> only_rotate_len_k,
                              double build_angle) -> ContractedGate {
        (void)p.current_picture_coeffs_();
        ContractedGate g;
        g.coeffs = p.schrodinger_ ? &p.mp_op_.state_coeffs : &p.mp_op_.op_coeffs;
        (void)p.build_evolve_result_(gate,
                                     only_rotate_len_k,
                                     std::cref(*g.coeffs),
                                     build_angle,
                                     &g.cos,
                                     &g.fc,
                                     g.coeffs,
                                     &g.fused_scale);
        p.extend_coeffs_from_current_picture_if_needed_(*g.coeffs);
        g.apply_angle = p.schrodinger_ ? -build_angle : build_angle;
        return g;
    }

    // One graph-only layer (no coefficients, not appended to the graph) built by the real build_layer on the
    // propagator's operator, budget and communicator, under the propagator's own operation guard, with a
    // test observer on the scan's ranges. The operator may grow, so the object is for inspection only.
    template <class Observer>
    static auto build_layer_observed(Propagator &p, const VecZ &gate, const Observer &observer)
        -> std::shared_ptr<LayerCore> {
        return p.run_operation_(true, [&](bool &mutation_started) {
            const auto gen = indices_to_bitset_checked<NumModes>(gate, 2 * p.logical_num_modes_);
            mutation_started = true;
            return build_layer<NumModes>(p.mp_op_,
                                         gen,
                                         p.cutoff_fn_,
                                         p.lower_atol_,
                                         std::nullopt,
                                         p.upper_atol_,
                                         std::nullopt,
                                         std::nullopt,
                                         p.matched_scratch_,
                                         p.comm_,
                                         nullptr,
                                         nullptr,
                                         p.schrodinger_,
                                         nullptr,
                                         nullptr,
                                         p.basis_,
                                         p.parallel_,
                                         observer);
        });
    }

    // Runs `fn(r, store)` on every legacy store of `p`: each partition child r of a facade, concurrently on the
    // partition group's own workers, or `p` itself as store 0 when it is a single store. A reference oracle for
    // sharded tests only.
    template <class Fn>
    static auto for_each_store(Propagator &p, Fn &&fn) -> void {
        if (p.partition_group_) {
            p.partition_group_->run_on_all(
                [&](int r) { fn(static_cast<size_t>(r), p.partition_group_->partition(r)); });
        }
        else {
            fn(size_t{0}, p);
        }
    }

    // One gate of the legacy graph build (propagate_one_ without the append): the real build_layer on the store's
    // own operator, cutoff, bound and communicator, with a test observer on its kernels and exchange phases.
    template <class Observer>
    static auto graph_gate_observed(Propagator &p,
                                    const VecZ &gate,
                                    std::optional<size_t> only_rotate_len_k,
                                    const Observer &observer) -> std::shared_ptr<LayerCore> {
        const auto gen = indices_to_bitset_checked<NumModes>(gate, 2 * p.logical_num_modes_);
        return build_layer<NumModes>(p.mp_op_,
                                     gen,
                                     p.cutoff_fn_,
                                     p.lower_atol_,
                                     std::nullopt,
                                     p.upper_atol_,
                                     std::nullopt,
                                     only_rotate_len_k,
                                     p.matched_scratch_,
                                     p.comm_,
                                     nullptr,
                                     nullptr,
                                     p.schrodinger_,
                                     nullptr,
                                     nullptr,
                                     p.basis_,
                                     p.parallel_,
                                     observer);
    }

    // One whole gate of the legacy ContractImmediately path (evolve_mode_contract_immediately_), build, extension
    // and apply, with a test observer on its kernels and exchange phases.
    template <class Observer>
    static auto contract_gate_observed(Propagator &p,
                                       const VecZ &gate,
                                       std::optional<size_t> only_rotate_len_k,
                                       double build_angle,
                                       const Observer &observer) -> void {
        (void)p.current_picture_coeffs_();
        VecD *coeffs = p.schrodinger_ ? &p.mp_op_.state_coeffs : &p.mp_op_.op_coeffs;
        const auto gen = indices_to_bitset_checked<NumModes>(gate, 2 * p.logical_num_modes_);
        CosMask cos;
        FusedContract fc;
        bool fused_scale = false;
        (void)build_layer<NumModes>(p.mp_op_,
                                    gen,
                                    p.cutoff_fn_,
                                    p.lower_atol_,
                                    std::cref(*coeffs),
                                    p.upper_atol_,
                                    build_angle,
                                    only_rotate_len_k,
                                    p.matched_scratch_,
                                    p.comm_,
                                    &cos,
                                    &fc,
                                    p.schrodinger_,
                                    coeffs,
                                    &fused_scale,
                                    p.basis_,
                                    p.parallel_,
                                    observer);
        p.extend_coeffs_from_current_picture_if_needed_(*coeffs);
        apply_fused_contract(fc,
                             *coeffs,
                             cos,
                             p.schrodinger_ ? -build_angle : build_angle,
                             p.schrodinger_,
                             fused_scale,
                             p.parallel_,
                             observer);
    }

    // The legacy gate loop's closing cache warm-up (run_gate_loop_).
    static auto finish_gate_loop(Propagator &p) -> void { p.initialize_operator_caches_(); }

    // The legacy evaluator's inputs for this single store, as make_functional_ snapshots them: the state (sparse
    // Heisenberg scores or the dense Schrödinger state), the un-evolved operator and the optimizer-order gate arrays.
    static auto evaluation_state(Propagator &p) -> EvalState {
        if (p.schrodinger_) {
            return EvalState::dense(p.mp_op_.dense_state());
        }
        const auto sparse = p.mp_op_.sparse_state();
        return EvalState::sparse(p.mp_op_.size(), sparse.rows, sparse.values);
    }

    static auto gate_arrays(const Propagator &p) -> std::pair<VecZ, VecD> { return p.graph_gate_arrays_(); }

    // The legacy energy evaluation's forward replay of this store at `params`: ev()'s evolved operator, through the
    // store's own communicator and budget. On a partition child it communicates, so every child must run it together
    // (for_each_store).
    static auto evaluation_operator(Propagator &p, const VecD &params) -> VecD {
        const auto [mapping, gen_coeffs] = p.graph_gate_arrays_();
        const auto view = p.graph_.replay_view();
        const auto cos = build_cos_callbacks<NumModes>(p.mp_op_.inverted_index(), view, p.basis_, p.parallel_);
        VecD op = p.mp_op_.get_operator();
        return evolve_operator(std::move(op),
                               view,
                               map_params(params, mapping, gen_coeffs, 1.0, true),
                               p.comm_,
                               cos.scale,
                               p.parallel_);
    }

    static auto partition_count(const Propagator &p) -> int {
        return p.partition_group_ ? p.partition_group_->partition_count() : 0;
    }

    static auto partition(const Propagator &p, int r) -> const Propagator & { return p.partition_group_->partition(r); }
};

} // namespace monoprop::detail
