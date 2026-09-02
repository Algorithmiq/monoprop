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

#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <utility>

#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/algebra/PauliAlgebra.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop {

template <size_t NumModes>
struct MajoranaAlgebra {
    static constexpr Basis basis = Basis::Majorana;
    static constexpr bool requires_support_cutoff = false; // length OR support cutoff both valid
    static constexpr bool allows_basis_change = true;

    // Built once per layer: the generator G and its fixed interleave mask W (see interleave_phase_mask).
    // G is stored by value so the context can outlive a caller's temporary; the cost is one bitset copy
    // per layer, cheaper than a lifetime contract on every call site.
    struct GenContext {
        Monomial<NumModes> gen;
        Monomial<NumModes> interleave_mask;
    };
    static auto make_gen_context(const Monomial<NumModes> &gen) -> GenContext {
        return GenContext{gen, interleave_phase_mask<NumModes>(gen)};
    }
    static auto generator(const GenContext &ctx) -> const Monomial<NumModes> & { return ctx.gen; }

    // Ordering sign of mono·G via the per-layer mask (branch/scan-free).
    static auto rotation_sign(const GenContext &ctx,
                              const Monomial<NumModes> &mono,
                              const Monomial<NumModes> & /*new_mono*/) -> int {
        return mono.parity_and(ctx.interleave_mask) ? -1 : 1;
    }
    static auto emit_phase(int rotation_sign, size_t mono_pop, size_t gen_pop, size_t overlap) -> int {
        return rotation_sign * hermitian_phase(mono_pop, gen_pop, overlap);
    }

    // The same sign from M's ascending positions instead of its bitset: parity_and(W) is the parity of
    // |M ∩ W|, i.e. the XOR of W's bits AT those positions, so a packed row is signed without expansion.
    // Always available for this algebra (the mask covers every position).
    static auto sign_from_positions_ok(const GenContext & /*ctx*/) -> bool { return true; }
    template <typename PosT>
    [[gnu::always_inline]] static auto rotation_sign_positions(const GenContext &ctx, const PosT *pos, size_t count)
        -> int {
        bool parity = false;
        for (size_t j = 0; j < count; ++j) {
            parity ^= ctx.interleave_mask.test(static_cast<size_t>(pos[j]));
        }
        return parity ? -1 : 1;
    }

    // Anticommutation fold columns = G itself; odd |G| needs the per-row parity(|M|) correction.
    static auto fold_generator(const Monomial<NumModes> &gen) -> Monomial<NumModes> { return gen; }
    static auto fold_needs_odd_correction(const Monomial<NumModes> &gen) -> bool { return gen.count() % 2 != 0; }

    static auto encode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &mono) -> double {
        return monoprop::encode_coeff<NumModes>(coeff, mono);
    }
    static auto decode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &mono)
        -> std::complex<double> {
        return monoprop::decode_coeff<NumModes>(coeff, mono);
    }
    static auto state_phase(const Monomial<NumModes> &mono, const Monomial<NumModes> &state_mask) -> double {
        return monoprop::majorana_state_phase<NumModes>(mono, state_mask);
    }
    // The same score from the term's ascending positions: (-1)^(|M ∩ mask| + |M|/2), with |M ∩ mask| a
    // count of positions the mask has set. Meaningful only for a fully paired M, like state_phase.
    template <typename PosT>
    static auto state_phase_positions(const PosT *pos, size_t count, const Monomial<NumModes> &state_mask) -> double {
        size_t in_mask = 0;
        for (size_t j = 0; j < count; ++j) {
            in_mask += static_cast<size_t>(state_mask.test(static_cast<size_t>(pos[j])));
        }
        return POWERS_OF_MINUS_ONE[(in_mask + count / 2) % 2];
    }
};

template <size_t NumModes>
struct PauliAlgebra {
    static constexpr Basis basis = Basis::Pauli;
    static constexpr bool requires_support_cutoff = true; // the support cutoff measures Pauli weight
    static constexpr bool allows_basis_change = false;    // the native encoding has no basis change

    struct GenContext {
        PauliGenContext<NumModes> pauli_ctx;
    };
    static auto make_gen_context(const Monomial<NumModes> &gen) -> GenContext {
        return GenContext{make_pauli_gen_context<NumModes>(gen)};
    }
    static auto generator(const GenContext &ctx) -> const Monomial<NumModes> & { return ctx.pauli_ctx.gen; }

    // Rotation-ready sign: already the negated raw product sign (see pauli_rotation_sign).
    static auto rotation_sign(const GenContext &ctx, const Monomial<NumModes> &mono, const Monomial<NumModes> &new_mono)
        -> int {
        return pauli_rotation_sign<NumModes>(ctx.pauli_ctx, mono, new_mono);
    }
    // Pauli's rotation sign is already the emitted sine phase -- no Hermitian fold.
    static auto emit_phase(int rotation_sign, size_t /*mono_pop*/, size_t /*gen_pop*/, size_t /*overlap*/) -> int {
        return rotation_sign;
    }

    // The same sign from M's positions: gather the bits at G's qubits into the per-gate compact word and
    // run the kernel there (pauli_rotation_sign_compact). Unavailable when G acts on more than 32 qubits.
    static auto sign_from_positions_ok(const GenContext &ctx) -> bool { return ctx.pauli_ctx.compact_ok; }
    template <typename PosT>
    [[gnu::always_inline]] static auto rotation_sign_positions(const GenContext &ctx, const PosT *pos, size_t count)
        -> int {
        assert(ctx.pauli_ctx.compact_ok);
        constexpr uint8_t kNone = PauliGenContext<NumModes>::kNotInGen;
        uint64_t m_compact = 0;
        for (size_t j = 0; j < count; ++j) {
            const uint8_t cs = ctx.pauli_ctx.compact_slot[static_cast<size_t>(pos[j])];
            // cs == kNone contributes nothing: shift by a masked amount and AND with the predicate.
            m_compact |= (uint64_t{cs != kNone}) << (cs & 63U);
        }
        return pauli_rotation_sign_compact<NumModes>(ctx.pauli_ctx, m_compact);
    }

    // Anticommutation fold columns = J(G) = pair_swap(G); Pauli needs no odd-|G| row-parity correction.
    static auto fold_generator(const Monomial<NumModes> &gen) -> Monomial<NumModes> { return pair_swap<NumModes>(gen); }
    static auto fold_needs_odd_correction(const Monomial<NumModes> & /*gen*/) -> bool { return false; }

    static auto encode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> & /*mono*/) -> double {
        return encode_pauli_coeff(coeff);
    }
    static auto decode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> & /*mono*/)
        -> std::complex<double> {
        return decode_pauli_coeff(coeff.real());
    }
    static auto state_phase(const Monomial<NumModes> &mono, const Monomial<NumModes> &state_mask) -> double {
        return pauli_state_phase<NumModes>(mono, state_mask);
    }
    // The same score from the term's ascending positions: (-1)^(|Z ∩ occupied|), the mask count alone.
    template <typename PosT>
    static auto state_phase_positions(const PosT *pos, size_t count, const Monomial<NumModes> &state_mask) -> double {
        size_t in_mask = 0;
        for (size_t j = 0; j < count; ++j) {
            in_mask += static_cast<size_t>(state_mask.test(static_cast<size_t>(pos[j])));
        }
        return (in_mask & 1U) ? -1.0 : 1.0;
    }
};

// Shape check only: the members the backbone actually calls are enforced by use, not by this concept.
template <typename A>
concept Algebra = requires {
    typename A::GenContext;
    { A::basis } -> std::convertible_to<Basis>;
    { A::requires_support_cutoff } -> std::convertible_to<bool>;
    { A::allows_basis_change } -> std::convertible_to<bool>;
};

static_assert(Algebra<MajoranaAlgebra<1>>);
static_assert(Algebra<PauliAlgebra<1>>);

// The single runtime->policy branch: the hot backbone passes a generic lambda and is then fully
// specialized on the chosen algebra. Both arms must return the same type.
template <size_t NumModes, typename F>
auto with_algebra(Basis basis, F &&f) {
    if (basis == Basis::Pauli) {
        return std::forward<F>(f).template operator()<PauliAlgebra<NumModes>>();
    }
    return std::forward<F>(f).template operator()<MajoranaAlgebra<NumModes>>();
}

// Point-dispatch helpers for cold sites (per-layer / per-materialization) that carry a runtime Basis.

template <size_t NumModes>
auto algebra_fold_generator(Basis basis, const Monomial<NumModes> &gen) -> Monomial<NumModes> {
    return with_algebra<NumModes>(basis, [&]<typename A>() { return A::fold_generator(gen); });
}
template <size_t NumModes>
auto algebra_fold_needs_odd_correction(Basis basis, const Monomial<NumModes> &gen) -> bool {
    return with_algebra<NumModes>(basis, [&]<typename A>() { return A::fold_needs_odd_correction(gen); });
}
template <size_t NumModes>
auto algebra_encode_coeff(Basis basis, const std::complex<double> &coeff, const Monomial<NumModes> &mono) -> double {
    return with_algebra<NumModes>(basis, [&]<typename A>() { return A::encode_coeff(coeff, mono); });
}
template <size_t NumModes>
auto algebra_decode_coeff(Basis basis, const std::complex<double> &coeff, const Monomial<NumModes> &mono)
    -> std::complex<double> {
    return with_algebra<NumModes>(basis, [&]<typename A>() { return A::decode_coeff(coeff, mono); });
}
template <size_t NumModes>
auto algebra_state_phase(Basis basis, const Monomial<NumModes> &mono, const Monomial<NumModes> &state_mask) -> double {
    return with_algebra<NumModes>(basis, [&]<typename A>() { return A::state_phase(mono, state_mask); });
}

// Score each fully-paired term's diagonal element against the initial product state, emitting
// sink(row, phase). A sink rather than a dense out[row] because the scored set is a vanishing
// fraction of the rows.
template <size_t NumModes, typename Rows, typename Sink>
auto algebra_score_state(Basis basis,
                         const VecZ &paired_inds,
                         const VecZ &initial_state,
                         const Rows &store,
                         Sink &&sink) -> void {
    with_algebra<NumModes>(basis, [&]<typename A>() {
        const auto state_mask = initial_state_mask<NumModes>(initial_state);
        for (size_t i = 0; i < paired_inds.size(); ++i) {
            const auto &row = materialize_row<NumModes>(store, paired_inds[i]);
            sink(paired_inds[i], A::state_phase(row, state_mask));
        }
    });
}

} // namespace monoprop
