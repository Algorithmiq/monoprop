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

// Each model answers the same fixed set of questions about a simulation picture: which way the gate
// loop walks the circuit, the sign an angle carries when it is applied, which coefficient vector the
// gates mutate, and which vector is the contraction partner. Sibling models, as MajoranaAlgebra and
// PauliAlgebra are -- see algebra/Algebra.h, whose with_algebra bridge this file mirrors.
//
// The models are not templates: NumModes is deduced per member, so with_picture needs no width and a
// call site reads P::live_coeffs(mp_op_).
//
// There is no runtime-dispatching helper layer on purpose. Each public entry point of MonomialPropagator
// binds the policy once with with_picture(); its whole private layer is then written against one picture
// and never re-tests which one it is.

#include <concepts>
#include <cstddef>
#include <utility>

#include "monoprop/MPFunctions.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/core/Picture.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop {

struct HeisenbergPicture {
    static constexpr Picture picture = Picture::Heisenberg;
    static constexpr bool is_schrodinger = false;

    // Simulation step i consumes optimizer slot n-1-i: the observable walks the circuit backwards.
    static auto gate_slot(size_t i, size_t n) -> size_t { return n - 1 - i; }
    static constexpr double apply_sign = 1.0; // the applied angle is the build angle

    // map_params() arguments for contract_partially: forward phase, written in reverse.
    static constexpr double contract_phase = 1.0;
    static constexpr bool contract_reverse = true;

    // The live vector the gates mutate, and the slot it lives in.
    template <size_t NumModes>
    static auto live_coeffs(detail::MPOperator<NumModes> &op) -> const VecD & {
        return op.get_operator();
    }
    template <size_t NumModes>
    static auto live_coeffs_slot(detail::MPOperator<NumModes> &op) -> VecD & {
        return op.op_coeffs;
    }

    // Warms the sparse state only; densifying here would defeat it.
    template <size_t NumModes>
    static auto warm_state(detail::MPOperator<NumModes> &op) -> void {
        (void)op.sparse_state();
    }

    // Energy only dots the state against the evolved operator, and the gradient scatters it into its
    // own scratch, so the sparse scores are enough.
    template <size_t NumModes>
    static auto eval_state(detail::MPOperator<NumModes> &op, size_t num_terms) -> EvalState {
        const auto sparse = op.sparse_state();
        return EvalState::sparse(num_terms, sparse.rows, sparse.values);
    }

    // The paring keep-set, thresholded on the picture's driving vector: (keep-set, local index count).
    static auto pare_seed(const EvalState &state, const VecD & /*op*/, double threshold) -> std::pair<VecZ, size_t> {
        return {state.indices_above(threshold), state.length()};
    }

    // A perf hint, never a correctness constraint: overflow spills losslessly. The bound is already in
    // physical slots (CutoffEvaluator::max_slot_bound), so nothing to scale. NumModes is explicit
    // everywhere it names a CutoffFn: 2*NumModes inside Monomial is a non-deduced context.
    template <size_t NumModes>
    static auto packed_inline_width(const CutoffFn<NumModes> &cutoff_fn) -> size_t {
        constexpr size_t kMax = detail::OperatorIndex<NumModes>::kMaxInlinePositions;
        constexpr size_t kDefault = detail::OperatorIndex<NumModes>::kDefaultInlinePositions;
        const auto bound = detail::CutoffEvaluator<NumModes>(cutoff_fn).max_slot_bound();
        if (!bound) {
            return kDefault;
        }
        return std::min<size_t>(*bound, kMax);
    }
};

struct SchrodingerPicture {
    static constexpr Picture picture = Picture::Schrodinger;
    static constexpr bool is_schrodinger = true;

    // Simulation step i consumes optimizer slot i: the state walks the circuit front-to-back.
    static auto gate_slot(size_t i, size_t /*n*/) -> size_t { return i; }
    static constexpr double apply_sign = -1.0; // the applied angle is the negated build angle

    static constexpr double contract_phase = -1.0;
    static constexpr bool contract_reverse = false;

    // The dense state IS the live evolved vector here, so it is both the source and the slot.
    template <size_t NumModes>
    static auto live_coeffs(detail::MPOperator<NumModes> &op) -> const VecD & {
        return op.dense_state();
    }
    template <size_t NumModes>
    static auto live_coeffs_slot(detail::MPOperator<NumModes> &op) -> VecD & {
        return op.state_coeffs;
    }

    template <size_t NumModes>
    static auto warm_state(detail::MPOperator<NumModes> &op) -> void {
        (void)op.dense_state();
    }

    // Snapshotted whole: dense_state() returns the live vector, which evolution then mutates.
    template <size_t NumModes>
    static auto eval_state(detail::MPOperator<NumModes> &op, size_t /*num_terms*/) -> EvalState {
        return EvalState::dense(op.dense_state());
    }

    static auto pare_seed(const EvalState & /*state*/, const VecD &op, double threshold) -> std::pair<VecZ, size_t> {
        return {indices_above(op, threshold), op.size()};
    }

    // The state's monomials come from generate_paired_op(), not from cutoff_fn_, so the cutoff carries
    // no structural bound on them.
    template <size_t NumModes>
    static auto packed_inline_width(const CutoffFn<NumModes> & /*cutoff_fn*/) -> size_t {
        return detail::OperatorIndex<NumModes>::kDefaultInlinePositions;
    }
};

// Shape check only: the members the propagator actually calls are enforced by use, not by this concept.
template <typename P>
concept PicturePolicy = requires {
    { P::picture } -> std::convertible_to<Picture>;
    { P::is_schrodinger } -> std::convertible_to<bool>;
    { P::apply_sign } -> std::convertible_to<double>;
    { P::contract_phase } -> std::convertible_to<double>;
    { P::contract_reverse } -> std::convertible_to<bool>;
};

static_assert(PicturePolicy<HeisenbergPicture>);
static_assert(PicturePolicy<SchrodingerPicture>);

// The only place a picture becomes a graph layer order. MPGraph deliberately knows nothing about pictures,
// and the translation belongs on this side: Heisenberg gives each arriving gate a lower optimizer slot than
// the last, Schrödinger a higher one.
inline auto layer_growth_of(Picture picture) -> LayerGrowth {
    return picture == Picture::Schrodinger ? LayerGrowth::Front : LayerGrowth::Back;
}

// The one runtime->policy branch, taken once per public call. decltype(auto), not auto, so a policy that
// hands back a reference into the operator does not decay to a copy; both arms must then deduce the same
// type.
template <typename F>
auto with_picture(Picture picture, F &&f) -> decltype(auto) {
    if (picture == Picture::Schrodinger) {
        return std::forward<F>(f).template operator()<SchrodingerPicture>();
    }
    return std::forward<F>(f).template operator()<HeisenbergPicture>();
}

} // namespace monoprop
