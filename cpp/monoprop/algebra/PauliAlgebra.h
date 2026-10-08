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

// A qubit Pauli string is stored in the same bitset as a Majorana monomial, under the per-qubit JW
// image -- qubit q owns the physical pair {2m, 2m+1}, m = N-1-q.
// (u,v) symplectic split with E = pauli_even_mask (physical even bits):
// v (z-plane) = w & E, u = (w >> 1) & E, x-plane = u ^ v; a qubit is Y iff (v=1, u=0), Z-only iff the
// x-plane is empty. So xor_sum = popcount(x-plane), or_sum = Pauli weight, is_paired(P) iff P Z-only.

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "monoprop/Bitset.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/algebra/AlgebraCommon.h"

namespace monoprop {

template <size_t NumModes>
[[nodiscard]] inline constexpr auto pauli_even_mask() -> Monomial<NumModes> {
    return even_bits<2 * NumModes, LSb0>();
}

namespace detail {
struct PauliUv {
    uint64_t u; // odd-bit plane, aligned onto the even lane
    uint64_t v; // z-plane (even physical bits)
};
[[nodiscard]] inline auto pauli_uv(uint64_t word, uint64_t e) -> PauliUv {
    return {.u = (word >> 1) & e, .v = word & e};
}
} // namespace detail

// The pair-swap involution J: swap the two physical bits of every qubit pair (u <-> v). Stays inside
// each word -- pairs are {2m, 2m+1}, so there is no cross-word carry.
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

// A Y letter has v=1, u=0.
template <size_t NumModes>
[[nodiscard]] auto pauli_y_count(const Monomial<NumModes> &p) -> size_t {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    size_t y = 0;
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        const auto [u, v] = detail::pauli_uv(p.word(w), e_mask.word(w));
        y += static_cast<size_t>(std::popcount(v & ~u));
    }
    return y;
}

// Whether two Pauli strings anticommute (symplectic inner product is odd):
// P.parity_and(pair_swap(G)) == (x_P . z_G + z_P . x_G) mod 2.
template <size_t NumModes>
[[nodiscard]] auto pauli_anticommutes(const Monomial<NumModes> &p, const Monomial<NumModes> &g) -> bool {
    return p.parity_and(pair_swap<NumModes>(g));
}

namespace detail {
// Reduce a (possibly negative) i-power exponent to [0, 4).
[[nodiscard]] inline constexpr auto mod4(long e) -> int {
    return static_cast<int>(((e % 4) + 4) % 4);
}

/*! @brief One word's share of the rotation-sign exponent: y(mono) - y(new_mono) + 2·|v_mono & x_gen|.
 *
 *  `mono`, `new_mono` and `gen` are the same word of each operand, and `e` is that word's even mask.
 */
[[nodiscard]] inline auto pauli_sign_exponent_word(uint64_t mono, uint64_t new_mono, uint64_t gen, uint64_t e) -> long {
    const auto [u_m, v_m] = pauli_uv(mono, e);
    const auto [u_n, v_n] = pauli_uv(new_mono, e);
    const auto [u_g, v_g] = pauli_uv(gen, e);
    const uint64_t x_g = u_g ^ v_g;
    return static_cast<long>(std::popcount(v_m & ~u_m)) - static_cast<long>(std::popcount(v_n & ~u_n))
           + (2 * static_cast<long>(std::popcount(v_m & x_g)));
}
} // namespace detail

// Per-generator context for the hot emit-sign kernel: nz_words lets pauli_rotation_sign() skip words
// outside G's support.
template <size_t NumModes>
struct PauliGenContext final {
    Monomial<NumModes> gen{};
    size_t g_y = 0;
    std::array<size_t, Monomial<NumModes>::num_words()> nz_words{};
    size_t nz_count = 0;
};

// Call once per layer, not per term.
template <size_t NumModes>
[[nodiscard]] auto make_pauli_gen_context(const Monomial<NumModes> &gen) -> PauliGenContext<NumModes> {
    PauliGenContext<NumModes> ctx;
    ctx.gen = gen;
    ctx.g_y = pauli_y_count<NumModes>(gen);
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        if (gen.word(w) != 0) {
            ctx.nz_words[ctx.nz_count++] = w;
        }
    }
    return ctx;
}

// Hot kernel: the rotation sign +/-1 for the anticommuting product mono*gen (new_mono = mono^gen).
// Returns the sign O' = U†OU (U = exp(iθ·gen)) needs on the off-diagonal partner term: the negated
// raw product sign, so the emit site needs no extra negation (pinned by pauli_algebra_tests.cpp).
// Loops only over gen's nonzero words (elsewhere mono/new_mono Y counts cancel and x_gen = 0). Exponent
// e = g_y + Σ_w(yMono - yNew) + 2·Σ_w(v_mono & x_gen); raw sign = (e mod 4 == 1 ? +1 : -1), negated here.
template <size_t NumModes>
[[gnu::always_inline]] inline auto pauli_rotation_sign(const PauliGenContext<NumModes> &ctx,
                                                       const Monomial<NumModes> &mono,
                                                       const Monomial<NumModes> &new_mono) -> int {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    auto exponent = static_cast<long>(ctx.g_y);
    for (size_t k = 0; k < ctx.nz_count; ++k) {
        const size_t w = ctx.nz_words[k];
        exponent += detail::pauli_sign_exponent_word(mono.word(w), new_mono.word(w), ctx.gen.word(w), e_mask.word(w));
    }
    return detail::mod4(exponent) == 1 ? -1 : 1;
}

/*! @brief pauli_rotation_sign from the source's ascending positions.
 *
 *  Rebuilds only the source words where the generator is nonzero; the partner word is source ^ generator.
 */
template <size_t NumModes, typename PosT>
[[gnu::always_inline]] inline auto pauli_rotation_sign_positions(const PauliGenContext<NumModes> &ctx,
                                                                 std::span<const PosT> src) -> int {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    auto exponent = static_cast<long>(ctx.g_y);
    size_t k = 0;
    uint64_t m = 0;
    for (const PosT p : src) {
        const auto q = static_cast<size_t>(p);
        while (k < ctx.nz_count && (q >> 6) > ctx.nz_words[k]) {
            const size_t w = ctx.nz_words[k++];
            exponent += detail::pauli_sign_exponent_word(m, m ^ ctx.gen.word(w), ctx.gen.word(w), e_mask.word(w));
            m = 0;
        }
        if (k == ctx.nz_count) {
            break;
        }
        m |= static_cast<uint64_t>((q >> 6) == ctx.nz_words[k]) << (q & 63);
    }
    for (; k < ctx.nz_count; ++k) {
        const size_t w = ctx.nz_words[k];
        exponent += detail::pauli_sign_exponent_word(m, m ^ ctx.gen.word(w), ctx.gen.word(w), e_mask.word(w));
        m = 0;
    }
    return detail::mod4(exponent) == 1 ? -1 : 1;
}

// Diagonal element <b|P|b> = (-1)^{|Z ∩ occupied|} of a Z-only Pauli against the initial product
// state. Only meaningful where is_paired holds; for a non-diagonal Pauli <b|P|b> = 0.
template <size_t NumModes>
[[nodiscard]] auto pauli_state_phase(const Monomial<NumModes> &mono, const Monomial<NumModes> &state_mask) -> double {
    return (mono.count_and(state_mask) & 1) ? -1.0 : 1.0;
}

// Pauli strings are Hermitian, so coefficients are already real -- no phase to normalize away.
[[nodiscard]] inline auto encode_pauli_coeff(const std::complex<double> &coeff) -> double {
    if (std::abs(coeff.imag()) > 1e-10) {
        throw NonEncodableCoefficient("Non-real Pauli coeffs detected");
    }
    return coeff.real();
}

[[nodiscard]] inline auto decode_pauli_coeff(double coeff) -> std::complex<double> {
    return {coeff, 0.0};
}

} // namespace monoprop
