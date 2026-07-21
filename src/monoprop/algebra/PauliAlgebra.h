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

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "monoprop/Bitset.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/algebra/AlgebraCommon.h" // is_paired / support_cutoff (basis-agnostic structure)

/*!
 * @file PauliAlgebra.h
 * @brief Pauli-native operator algebra over the Majorana bitset container.
 *
 * A qubit Pauli string P = i^{#Y} X^x Z^z is stored in the SAME container as a Majorana
 * monomial (`Monomial<NumModes> = Bitset<2*NumModes>`), under the per-qubit image of
 * `change_basis(JordanWigner(P))`. For qubit q (NumModes = number of qubits N):
 *   - X_q sets gamma-slot 2q, Y_q sets slot 2q+1, Z_q sets both slots {2q, 2q+1}.
 * `indices_to_bitset` maps slot k -> physical bit 2N-1-k (MSb0), so with m = N-1-q qubit q
 * owns the physically aligned pair {2m, 2m+1}. In the (u,v) symplectic split:
 *   - v (z-plane) lives on the EVEN physical bit (slot 2q+1):  v = w & E
 *   - u           lives on the ODD  physical bit (slot 2q):    u = (w >> 1) & E
 *   - x (x-plane) = u ^ v;   a qubit is Y iff (v=1, u=0);   Z-only iff x-plane is empty.
 * where E = even_bits<2*NumModes, LSb0>() is the physical-even-bit mask (the same mask
 * support_cutoff uses). Under this encoding support_cutoff's xor_sum = popcount(x-plane) and
 * or_sum = popcount(x|z) = qubit Pauli weight, and is_paired(P) holds iff P is Z-only -- which
 * is why the fully-paired keep-exception protects exactly the diagonal (expectation-carrying)
 * Paulis, matching the existing Jordan-Wigner path.
 */

namespace monoprop {

/// @brief Physical-even-bit mask E (z-plane / v-plane selector) for the Pauli encoding.
/// @tparam NumModes Number of qubits N; the Majorana container holds 2N bits.
template <size_t NumModes>
[[nodiscard]] inline constexpr auto pauli_even_mask() -> Monomial<NumModes> {
    return even_bits<2 * NumModes, LSb0>();
}

namespace detail {
/// The (u,v) symplectic planes of one physical word of a Pauli bitset, with `e` the
/// physical-even-bit mask (pauli_even_mask's word). `v` is the z-plane (even physical bits);
/// `u` is the odd-bit plane shifted onto the even lane; the x-plane is `u ^ v`. Factors the
/// per-word split shared by pauli_y_count and pauli_rotation_sign.
struct PauliUv {
    uint64_t v; ///< z-plane (even physical bits)
    uint64_t u; ///< odd-bit plane, aligned onto the even lane
};
[[nodiscard]] inline auto pauli_uv(uint64_t word, uint64_t e) -> PauliUv { return {word & e, (word >> 1) & e}; }
} // namespace detail

/*!
 * @brief The pair-swap involution J: swap the two physical bits of every qubit pair.
 *
 * Swaps u <-> v per qubit, i.e. J(w) = ((w & E) << 1) | ((w >> 1) & E) per physical word.
 * Each qubit pair is {2m, 2m+1}, so the swap stays inside its word (no cross-word carry) and
 * never sets a bit above 2N-1. J is an involution: J(J(P)) == P.
 */
template <size_t NumModes>
[[nodiscard]] auto pair_swap(const Monomial<NumModes> &p) -> Monomial<NumModes> {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    Monomial<NumModes> result;
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        const uint64_t word = p.word(w);
        const uint64_t e = e_mask.word(w);
        result.data()[w] = ((word & e) << 1) | ((word >> 1) & e);
    }
    return result;
}

/*!
 * @brief Total number of Y letters over all qubits (a Y has v=1, u=0).
 */
template <size_t NumModes>
[[nodiscard]] auto pauli_y_count(const Monomial<NumModes> &p) -> size_t {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    size_t y = 0;
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        const auto [v, u] = detail::pauli_uv(p.word(w), e_mask.word(w));
        y += static_cast<size_t>(std::popcount(v & ~u));
    }
    return y;
}

/*!
 * @brief Whether two Pauli strings anticommute (symplectic inner product is odd).
 *
 * Reference form: P.parity_and(pair_swap(G)) == (u_P . v_G + v_P . u_G) mod 2
 *                                            == (x_P . z_G + z_P . x_G) mod 2.
 */
template <size_t NumModes>
[[nodiscard]] auto pauli_anticommutes(const Monomial<NumModes> &p, const Monomial<NumModes> &g) -> bool {
    return p.parity_and(pair_swap<NumModes>(g));
}

namespace detail {
/// Reduce a (possibly negative) i-power exponent to [0, 4) for POWERS_OF_I indexing.
[[nodiscard]] inline constexpr auto mod4(long e) -> int { return static_cast<int>(((e % 4) + 4) % 4); }
} // namespace detail

/*!
 * @brief Precomputed per-generator context for the hot emit-sign kernel.
 *
 * Caches the generator, its popcount, its Y count, and the indices of its nonzero physical
 * words so pauli_rotation_sign() can skip words outside the generator's support.
 */
template <size_t NumModes>
struct PauliGenContext final {
    Monomial<NumModes> gen{};
    size_t gen_pop = 0; ///< gen.count()
    size_t g_y = 0;     ///< pauli_y_count(gen)
    std::array<size_t, Monomial<NumModes>::num_words()> nz_words{}; ///< indices of gen's nonzero words
    size_t nz_count = 0;                                              ///< number of valid entries in nz_words
};

/*!
 * @brief Build the per-generator context (call once per layer, not per term).
 */
template <size_t NumModes>
[[nodiscard]] auto make_pauli_gen_context(const Monomial<NumModes> &gen) -> PauliGenContext<NumModes> {
    PauliGenContext<NumModes> ctx;
    ctx.gen = gen;
    ctx.gen_pop = gen.count();
    ctx.g_y = pauli_y_count<NumModes>(gen);
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        if (gen.word(w) != 0) {
            ctx.nz_words[ctx.nz_count++] = w;
        }
    }
    return ctx;
}

/*!
 * @brief HOT kernel: the ROTATION sign +/-1 for the anticommuting product maj * gen, given
 * new_maj = maj ^ gen.
 *
 * Returns the sign the rotation O' = U†OU (U = exp(iθ·gen)) needs on the off-diagonal partner
 * term, i.e. the NEGATED raw product sign: pauli_rotation_sign == -pauli_emit_sign_antic(maj, gen)
 * (pinned by pauli_build_layer_dense_matrix_ground_truth / T7), so the emit site needs no extra
 * negation. Loops ONLY over the generator's nonzero words: outside gen's support maj and new_maj
 * agree (their per-word Y counts cancel in yMaj - yNew) and x_gen = 0 (the cross term contributes
 * nothing), leaving the exponent unchanged. e = g_y + sum_w(yMaj(w) - yNew(w)) +
 * 2*sum_w(v_maj(w) & x_gen(w)); the raw sign is (e mod 4 == 1 ? +1 : -1), so the rotation sign is
 * its negation, (e mod 4 == 1 ? -1 : +1).
 */
template <size_t NumModes>
[[gnu::always_inline]] inline auto pauli_rotation_sign(const PauliGenContext<NumModes> &ctx,
                                                       const Monomial<NumModes> &maj,
                                                       const Monomial<NumModes> &new_maj) -> int {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    long delta = static_cast<long>(ctx.g_y); // + yGen
    long cross = 0;                           // zMaj . xGen
    for (size_t k = 0; k < ctx.nz_count; ++k) {
        const size_t w = ctx.nz_words[k];
        const uint64_t e = e_mask.word(w);
        const auto [v_m, u_m] = detail::pauli_uv(maj.word(w), e);
        const auto [v_n, u_n] = detail::pauli_uv(new_maj.word(w), e);
        const auto [v_g, u_g] = detail::pauli_uv(ctx.gen.word(w), e);
        delta += std::popcount(v_m & ~u_m); // + yMaj(w)
        delta -= std::popcount(v_n & ~u_n); // - yNew(w)
        const uint64_t x_g = u_g ^ v_g;
        cross += std::popcount(v_m & x_g);
    }
    return detail::mod4(delta + 2 * cross) == 1 ? -1 : 1;
}

/*!
 * @brief Hartree-Fock phase (-1)^{|Z ∩ occupied|} for a Z-only (diagonal) Pauli.
 *
 * hf_mask marks the z-plane (even physical) bits of the occupied qubits, so
 * maj.count_and(hf_mask) counts the occupied qubits carrying a Z. Only meaningful for Z-only
 * terms (is_paired holds); for a non-diagonal Pauli <b|P|b> = 0 and the caller must not use this.
 */
template <size_t NumModes>
[[nodiscard]] auto pauli_hf_phase(const Monomial<NumModes> &maj, const Monomial<NumModes> &hf_mask) -> double {
    return (maj.count_and(hf_mask) & 1) ? -1.0 : 1.0;
}

/*!
 * @brief Encode a Pauli coefficient (Hermitian representative) into its real storage value.
 *
 * Pauli strings are Hermitian, so their coefficients are already real; this is the identity on
 * the real part and rejects any stray imaginary component (mirrors encode_coeff's guard).
 */
[[nodiscard]] inline auto encode_pauli_coeff(const std::complex<double> &coeff) -> double {
    if (std::abs(coeff.imag()) > 1e-10) {
        throw std::runtime_error("Non-real Pauli coeffs detected");
    }
    return coeff.real();
}

/*!
 * @brief Decode a real Pauli coefficient back to complex (identity, zero imaginary part).
 */
[[nodiscard]] inline auto decode_pauli_coeff(double coeff) -> std::complex<double> { return {coeff, 0.0}; }

} // namespace monoprop
