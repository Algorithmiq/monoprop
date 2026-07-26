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

#include <complex>
#include <concepts>
#include <cstddef>
#include <utility>

#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/core/Monomial.h"

/*!
 * @file algebra/Algebra.h
 * @brief The @c Algebra policy: what the propagation backbone needs from an algebra.
 *
 * The backbone (anticommutation scan, cosine fold, Givens split) is GENERIC over the algebra. Each
 * algebra is a compile-time policy model (@c MajoranaAlgebra, @c PauliAlgebra) answering a fixed set
 * of questions about how a @ref Monomial evolves under a rotation exp(iθ·G): which columns to fold
 * (and whether an odd-|G| parity correction is needed), the per-term rotation sign and emitted sine
 * phase, the coeff codec, and the diagonal (HF) score. @c with_algebra binds the runtime @ref Basis
 * to one model exactly once; each model only forwards to the sibling-header kernels.
 */

namespace monoprop {

/// @brief The Majorana algebra model: a @ref Monomial read as a product of Majorana operators.
template <size_t NumModes>
struct MajoranaAlgebra {
    static constexpr Basis basis = Basis::Majorana;
    static constexpr bool requires_support_cutoff = false; ///< length OR support cutoff both valid
    static constexpr bool allows_basis_change = true;      ///< Majorana basis changes are supported

    /// Per-generator context, built once per layer: the generator G and the fixed interleave mask W
    /// with interleave_phase(M,G) == (M.parity_and(W) ? -1 : 1).
    struct GenContext {
        const Monomial<NumModes> &gen;
        Monomial<NumModes> interleave_mask;
    };
    static auto make_gen_context(const Monomial<NumModes> &gen) -> GenContext {
        return GenContext{gen, interleave_phase_mask<NumModes>(gen)};
    }
    static auto generator(const GenContext &ctx) -> const Monomial<NumModes> & { return ctx.gen; }

    /// Ordering sign (-1)^x of maj·G via the per-layer mask (branch/scan-free). new_maj unused.
    static auto rotation_sign(const GenContext &ctx,
                              const Monomial<NumModes> &maj,
                              const Monomial<NumModes> & /*new_maj*/) -> int {
        return maj.parity_and(ctx.interleave_mask) ? -1 : 1;
    }
    /// Emitted sine phase = ordering sign folded with the Hermitian phase of the product.
    static auto emit_phase(int rotation_sign, size_t maj_pop, size_t gen_pop, size_t overlap) -> int {
        return rotation_sign * hermitian_phase(maj_pop, gen_pop, overlap);
    }

    /// Anticommutation fold columns = G itself; odd |G| needs the per-row parity(|M|) correction.
    static auto fold_generator(const Monomial<NumModes> &gen) -> Monomial<NumModes> { return gen; }
    static auto fold_needs_odd_correction(const Monomial<NumModes> &gen) -> bool { return gen.count() % 2 != 0; }

    static auto encode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> double {
        return monoprop::encode_coeff<NumModes>(coeff, maj);
    }
    static auto decode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> std::complex<double> {
        return monoprop::decode_coeff<NumModes>(coeff, maj);
    }
    static auto hf_phase(const Monomial<NumModes> &maj, const Monomial<NumModes> &hf_mask) -> double {
        return monoprop::hf_phase<NumModes>(maj, hf_mask);
    }
};

/// @brief The Pauli algebra model: a @ref Monomial read as a Pauli string (native JW-image encoding).
template <size_t NumModes>
struct PauliAlgebra {
    static constexpr Basis basis = Basis::Pauli;
    static constexpr bool requires_support_cutoff = true; ///< the support cutoff measures Pauli weight
    static constexpr bool allows_basis_change = false;    ///< the native encoding forbids a basis change

    /// Per-generator context = the precomputed Pauli rotation-sign kernel context (holds G and |G|).
    struct GenContext {
        PauliGenContext<NumModes> pauli_ctx;
    };
    static auto make_gen_context(const Monomial<NumModes> &gen) -> GenContext {
        return GenContext{make_pauli_gen_context<NumModes>(gen)};
    }
    static auto generator(const GenContext &ctx) -> const Monomial<NumModes> & { return ctx.pauli_ctx.gen; }

    /// Rotation-ready sign (already the negated raw product sign) from the hot Pauli kernel.
    static auto rotation_sign(const GenContext &ctx, const Monomial<NumModes> &maj, const Monomial<NumModes> &new_maj)
        -> int {
        return pauli_rotation_sign<NumModes>(ctx.pauli_ctx, maj, new_maj);
    }
    /// Pauli's rotation sign is already the emitted sine phase -- no Hermitian fold.
    static auto emit_phase(int rotation_sign, size_t /*maj_pop*/, size_t /*gen_pop*/, size_t /*overlap*/) -> int {
        return rotation_sign;
    }

    /// Anticommutation fold columns = J(G) = pair_swap(G); the self-commutation invariant makes the
    /// odd-|G| row-parity correction unnecessary for Pauli (always false).
    static auto fold_generator(const Monomial<NumModes> &gen) -> Monomial<NumModes> { return pair_swap<NumModes>(gen); }
    static auto fold_needs_odd_correction(const Monomial<NumModes> & /*gen*/) -> bool { return false; }

    static auto encode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> & /*maj*/) -> double {
        return encode_pauli_coeff(coeff);
    }
    static auto decode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> & /*maj*/)
        -> std::complex<double> {
        return decode_pauli_coeff(coeff.real());
    }
    static auto hf_phase(const Monomial<NumModes> &maj, const Monomial<NumModes> &hf_mask) -> double {
        return pauli_hf_phase<NumModes>(maj, hf_mask);
    }
};

/*!
 * @brief The minimal surface the propagation backbone requires of an algebra model.
 *
 * Deliberately lightweight: checks the shape, not every return type. @c MajoranaAlgebra and
 * @c PauliAlgebra both satisfy it.
 */
template <typename A>
concept Algebra = requires {
    typename A::GenContext;
    { A::basis } -> std::convertible_to<Basis>;
    { A::requires_support_cutoff } -> std::convertible_to<bool>;
    { A::allows_basis_change } -> std::convertible_to<bool>;
};

static_assert(Algebra<MajoranaAlgebra<1>>);
static_assert(Algebra<PauliAlgebra<1>>);

/*!
 * @brief Bind a runtime @ref Basis to its compile-time algebra model, once.
 *
 * The single runtime->policy branch: the hot backbone passes a generic lambda and is then fully
 * specialized on the chosen algebra. Both arms must return the same type.
 */
template <size_t NumModes, class F>
auto with_algebra(Basis basis, F &&f) {
    if (basis == Basis::Pauli) {
        return std::forward<F>(f).template operator()<PauliAlgebra<NumModes>>();
    }
    return std::forward<F>(f).template operator()<MajoranaAlgebra<NumModes>>();
}

// Point-dispatch helpers for cold sites (per-layer / per-materialization) that carry a runtime Basis:
// each forwards to the matching model; the runtime dispatch cost is irrelevant at these cold sites.

template <size_t NumModes>
auto algebra_fold_generator(Basis basis, const Monomial<NumModes> &gen) -> Monomial<NumModes> {
    return with_algebra<NumModes>(basis, [&]<class A>() { return A::fold_generator(gen); });
}
template <size_t NumModes>
auto algebra_fold_needs_odd_correction(Basis basis, const Monomial<NumModes> &gen) -> bool {
    return with_algebra<NumModes>(basis, [&]<class A>() { return A::fold_needs_odd_correction(gen); });
}
template <size_t NumModes>
auto algebra_encode_coeff(Basis basis, const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> double {
    return with_algebra<NumModes>(basis, [&]<class A>() { return A::encode_coeff(coeff, maj); });
}
template <size_t NumModes>
auto algebra_decode_coeff(Basis basis, const std::complex<double> &coeff, const Monomial<NumModes> &maj)
    -> std::complex<double> {
    return with_algebra<NumModes>(basis, [&]<class A>() { return A::decode_coeff(coeff, maj); });
}
template <size_t NumModes>
auto algebra_hf_phase(Basis basis, const Monomial<NumModes> &maj, const Monomial<NumModes> &hf_mask) -> double {
    return with_algebra<NumModes>(basis, [&]<class A>() { return A::hf_phase(maj, hf_mask); });
}

/*!
 * @brief Score the diagonal (Hartree-Fock) coefficient of each fully-paired term.
 *
 * Emits `sink(row, phase)` per entry, in ascending @p paired_inds order. A sink (rather than a dense
 * `out[row] = ...`) keeps the caller free to store the result sparsely: the scored set is a vanishing
 * fraction of the rows, so a dense destination would be almost entirely zeros.
 *
 * @c with_algebra hoists the runtime->policy branch OUT of the per-term loop, so the loop is
 * monomorphic in A (a Z-only Pauli scores (-1)^{|Z n occ|}; a Majorana term folds in the pairing sign).
 */
template <size_t NumModes, typename Rows, typename Sink>
auto algebra_score_hf(Basis basis, const VecZ &paired_inds, const VecZ &hf, const Rows &store, Sink &&sink) -> void {
    with_algebra<NumModes>(basis, [&]<class A>() {
        const auto hf_mask = get_hf_mask<NumModes>(hf);
        for (size_t i = 0; i < paired_inds.size(); ++i) {
            const auto &row = materialize_row<NumModes>(store, paired_inds[i]);
            sink(paired_inds[i], A::hf_phase(row, hf_mask));
        }
    });
}

} // namespace monoprop
