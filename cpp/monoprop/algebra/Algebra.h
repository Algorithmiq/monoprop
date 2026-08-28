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

// Each model answers the same fixed set of questions about a rotation exp(i*theta*G): which columns to
// fold, a term's rotation sign and emitted sine phase, the coefficient codec, and the diagonal
// initial-state score. The arithmetic behind each answer lives in MajoranaAlgebra.h / PauliAlgebra.h.

#include <complex>
#include <concepts>
#include <cstddef>
#include <utility>

#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/CodesAlgebra.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop {

struct MajoranaAlgebra {
    static constexpr Basis basis = Basis::Majorana;
    static constexpr bool requires_support_cutoff = false; // length OR support cutoff both valid
    static constexpr bool allows_basis_change = true;

    // Built once per layer: the generator G and its fixed interleave mask W (see interleave_phase_mask).
    // G is stored by value so the context can outlive a caller's temporary; the cost is one bitset copy
    // per layer, cheaper than a lifetime contract on every call site.
    // Both members are assigned by make_gen_context, so they carry the generator's width; a
    // default-constructed GenContext would hold width-0 bitsets.
    struct GenContext {
        Bitset gen;
        Bitset interleave_mask;
    };
    static auto make_gen_context(const Bitset &gen) -> GenContext {
        return GenContext{gen, interleave_phase_mask(gen)};
    }
    static auto generator(const GenContext &ctx) -> const Bitset & { return ctx.gen; }

    // Ordering sign of mono·G via the per-layer mask (branch/scan-free).
    static auto rotation_sign(const GenContext &ctx, const Bitset &mono, const Bitset & /*new_mono*/) -> int {
        return mono.parity_and(ctx.interleave_mask) ? -1 : 1;
    }
    // The same sign off word pointers the caller resolved once, with the word count bound by the
    // caller rather than read off the operand. Same fold, same parity; see detail::WordKernel.
    template <size_t W>
    static auto rotation_sign_words(const GenContext &ctx,
                                    const Bitset::word_type *mono,
                                    const Bitset::word_type * /*new_mono*/) -> int {
        return detail::WordKernel<W>::parity_and(mono, ctx.interleave_mask.data()) ? -1 : 1;
    }
    // The same sign in support form. No GenContext: the interleave mask is dense by construction
    // (roughly half the register), so the sparse form walks the two rows instead of carrying a mask, and
    // the product row is not an argument either -- see codes_interleave_phase.
    static auto codes_rotation_sign(const detail::SparseRow &mono, const detail::SparseRow &gen) -> int {
        return detail::codes_interleave_phase(mono, gen);
    }
    static auto emit_phase(int rotation_sign, size_t mono_pop, size_t gen_pop, size_t overlap) -> int {
        return rotation_sign * hermitian_phase(mono_pop, gen_pop, overlap);
    }

    // Anticommutation fold columns = G itself; odd |G| needs the per-row parity(|M|) correction.
    static auto fold_generator(const Bitset &gen) -> Bitset { return gen; }
    static auto fold_needs_odd_correction(const Bitset &gen) -> bool { return gen.count() % 2 != 0; }
};

struct PauliAlgebra {
    static constexpr Basis basis = Basis::Pauli;
    static constexpr bool requires_support_cutoff = true; // the support cutoff measures Pauli weight
    static constexpr bool allows_basis_change = false;    // the native encoding has no basis change

    struct GenContext {
        PauliGenContext pauli_ctx;
    };
    static auto make_gen_context(const Bitset &gen) -> GenContext { return GenContext{make_pauli_gen_context(gen)}; }
    static auto generator(const GenContext &ctx) -> const Bitset & { return ctx.pauli_ctx.gen; }

    // Rotation-ready sign: already the negated raw product sign (see pauli_rotation_sign).
    static auto rotation_sign(const GenContext &ctx, const Bitset &mono, const Bitset &new_mono) -> int {
        return pauli_rotation_sign(ctx.pauli_ctx, mono, new_mono);
    }
    // W is unused here and that is the point: this sign already loops over the generator's non-zero
    // words only, so there is no trip count to bind -- what the word form removes is the storage-pointer
    // select that mono.word(w) repeats on every access.
    template <size_t W>
    static auto rotation_sign_words(const GenContext &ctx,
                                    const Bitset::word_type *mono,
                                    const Bitset::word_type *new_mono) -> int {
        return pauli_rotation_sign_words(ctx.pauli_ctx, mono, new_mono);
    }
    // Same exponent as above off the two rows; new_mono never has to exist, since a mode the generator
    // misses contributes nothing (see codes_pauli_rotation_sign).
    static auto codes_rotation_sign(const detail::SparseRow &mono, const detail::SparseRow &gen) -> int {
        return detail::codes_pauli_rotation_sign(mono, gen);
    }
    // Pauli's rotation sign is already the emitted sine phase -- no Hermitian fold.
    static auto emit_phase(int rotation_sign, size_t /*mono_pop*/, size_t /*gen_pop*/, size_t /*overlap*/) -> int {
        return rotation_sign;
    }

    // Anticommutation fold columns = J(G) = pair_swap(G); Pauli needs no odd-|G| row-parity correction.
    static auto fold_generator(const Bitset &gen) -> Bitset { return pair_swap(gen); }
    static auto fold_needs_odd_correction(const Bitset & /*gen*/) -> bool { return false; }
};

// Shape check only: the members the backbone actually calls are enforced by use, not by this concept.
template <typename A>
concept Algebra = requires {
    typename A::GenContext;
    { A::basis } -> std::convertible_to<Basis>;
    { A::requires_support_cutoff } -> std::convertible_to<bool>;
    { A::allows_basis_change } -> std::convertible_to<bool>;
};

static_assert(Algebra<MajoranaAlgebra>);
static_assert(Algebra<PauliAlgebra>);

// The single runtime->policy branch: the hot backbone passes a generic lambda and is then fully
// specialized on the chosen algebra. Both arms must return the same type.
template <typename F>
auto with_algebra(Basis basis, F &&f) {
    if (basis == Basis::Pauli) {
        return std::forward<F>(f).template operator()<PauliAlgebra>();
    }
    return std::forward<F>(f).template operator()<MajoranaAlgebra>();
}

// Point-dispatch helpers for cold sites (per-layer / per-materialization) that carry a runtime Basis.

template <MonomialLike T>
auto algebra_fold_generator(Basis basis, const T &gen) -> T {
    return with_algebra(basis, [&gen]<typename A>() { return A::fold_generator(gen); });
}
template <MonomialLike T>
auto algebra_fold_needs_odd_correction(Basis basis, const T &gen) -> bool {
    return with_algebra(basis, [&gen]<typename A>() { return A::fold_needs_odd_correction(gen); });
}
// These three branch on Basis directly instead of going through with_algebra, and the policy classes
// carry no encode_coeff/decode_coeff/state_phase member as a result -- both would have been a
// width-agnostic passthrough to the free function called here, so an algebra *class* bought nothing and
// left the mapping written twice, with only one of the two reachable. The branch below is the whole
// mapping; see the note on state_phase_rows for why a per-scored-row branch is the right altitude here.
auto algebra_encode_coeff(Basis basis, const std::complex<double> &coeff, const MonomialLike auto &mono) -> double {
    return basis == Basis::Pauli ? encode_pauli_coeff(coeff) : monoprop::encode_coeff(coeff, mono);
}
auto algebra_decode_coeff(Basis basis, const std::complex<double> &coeff, const MonomialLike auto &mono)
    -> std::complex<double> {
    return basis == Basis::Pauli ? decode_pauli_coeff(coeff.real()) : monoprop::decode_coeff(coeff, mono);
}
auto algebra_state_phase(Basis basis, const MonomialLike auto &mono, const auto &state_mask) -> double {
    return basis == Basis::Pauli ? pauli_state_phase(mono, state_mask) : majorana_state_phase(mono, state_mask);
}

// Score each fully-paired term's diagonal element against the initial product state, emitting
// sink(row, phase). A sink rather than a dense out[row] because the scored set is a vanishing
// fraction of the rows.
//
// num_bits is the width of the rows in `store`, which the state mask must match. No width template
// parameter and no with_algebra: the only thing the algebra policy supplied here was A::state_phase,
// and algebra_state_phase above is the same branch without a compile-time width. The branch does move
// inside the loop, which is why this is spelled out rather than left implicit -- it is a per-*scored*-row
// branch on a value fixed for the propagator's lifetime, on a path that runs over the fully-paired
// terms only (~0.07% of rows) and not per term in the scan.
template <typename Rows, typename Sink>
auto algebra_score_state(Basis basis,
                         const VecZ &paired_inds,
                         const VecZ &initial_state,
                         const Rows &store,
                         size_t num_bits,
                         Sink &&sink) -> void {
    const auto state_mask = initial_state_mask(initial_state, num_bits);
    for (const auto &idx : paired_inds) {
        const auto &row = materialize_row(store, idx);
        sink(idx, algebra_state_phase(basis, row, state_mask));
    }
}

} // namespace monoprop
