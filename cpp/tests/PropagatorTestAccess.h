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

#include <memory>
#include <optional>
#include <utility>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/parallel/Options.h"

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

    static auto partition_count(const Propagator &p) -> int {
        return p.partition_group_ ? p.partition_group_->partition_count() : 0;
    }

    static auto partition(const Propagator &p, int r) -> const Propagator & { return p.partition_group_->partition(r); }
};

} // namespace monoprop::detail
