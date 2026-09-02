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
#include <limits>
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
    uint64_t v; // z-plane (even physical bits)
    uint64_t u; // odd-bit plane, aligned onto the even lane
};
[[nodiscard]] inline auto pauli_uv(uint64_t word, uint64_t e) -> PauliUv {
    return {word & e, (word >> 1) & e};
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
        const auto [v, u] = detail::pauli_uv(p.word(w), e_mask.word(w));
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
} // namespace detail

// Per-generator context for the hot emit-sign kernel: nz_words lets pauli_rotation_sign() skip words
// outside G's support.
template <size_t NumModes>
struct PauliGenContext final {
    Monomial<NumModes> gen{};
    size_t g_y = 0;
    std::array<size_t, Monomial<NumModes>::num_words()> nz_words{};
    size_t nz_count = 0;

    // The compacted form for signing from a term's positions alone. Only the qubits G acts on can change
    // the sign (elsewhere mono == new_mono and x_gen == 0), and those are at most the cutoff's weight, so
    // their 2 bits each are packed into ONE 64-bit word preserving the (even, odd) lane structure:
    // compact_slot[p] is the compact bit of physical position p, kNotInGen for every other position.
    // compact_ok is false when G acts on more than 32 qubits, and the dense kernel must be used.
    static constexpr uint8_t kNotInGen = std::numeric_limits<uint8_t>::max();
    std::array<uint8_t, 2 * NumModes> compact_slot{};
    uint64_t g_compact = 0;
    bool compact_ok = false;
};

// Sign kernel over the compacted word (see PauliGenContext::compact_slot): the same exponent
// pauli_rotation_sign folds over the whole bitset, restricted to the qubits that can contribute, so the
// two agree bit for bit (pinned by pauli_algebra_tests.cpp). `m_compact` gathers the source term's bits at
// the compact slots; the partner is m_compact ^ g_compact.
template <size_t NumModes>
[[gnu::always_inline]] inline auto pauli_rotation_sign_compact(const PauliGenContext<NumModes> &ctx, uint64_t m_compact)
    -> int {
    constexpr uint64_t e = 0x5555'5555'5555'5555ULL;
    const uint64_t n_compact = m_compact ^ ctx.g_compact;
    const auto [v_m, u_m] = detail::pauli_uv(m_compact, e);
    const auto [v_n, u_n] = detail::pauli_uv(n_compact, e);
    const auto [v_g, u_g] = detail::pauli_uv(ctx.g_compact, e);
    long delta = static_cast<long>(ctx.g_y);
    delta += std::popcount(v_m & ~u_m);
    delta -= std::popcount(v_n & ~u_n);
    const long cross = std::popcount(v_m & (u_g ^ v_g));
    return detail::mod4(delta + 2 * cross) == 1 ? -1 : 1;
}

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
    ctx.compact_slot.fill(PauliGenContext<NumModes>::kNotInGen);
    size_t qubits = 0;
    for (size_t q = 0; q < NumModes; ++q) {
        const bool even = gen.test(2 * q);
        const bool odd = gen.test((2 * q) + 1);
        if (!even && !odd) {
            continue;
        }
        if (qubits < 32) {
            ctx.compact_slot[2 * q] = static_cast<uint8_t>(2 * qubits);
            ctx.compact_slot[(2 * q) + 1] = static_cast<uint8_t>((2 * qubits) + 1);
            ctx.g_compact |=
                (static_cast<uint64_t>(even) << (2 * qubits)) | (static_cast<uint64_t>(odd) << ((2 * qubits) + 1));
        }
        ++qubits;
    }
    ctx.compact_ok = qubits <= 32;
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
    long delta = static_cast<long>(ctx.g_y);
    long cross = 0;
    for (size_t k = 0; k < ctx.nz_count; ++k) {
        const size_t w = ctx.nz_words[k];
        const uint64_t e = e_mask.word(w);
        const auto [v_m, u_m] = detail::pauli_uv(mono.word(w), e);
        const auto [v_n, u_n] = detail::pauli_uv(new_mono.word(w), e);
        const auto [v_g, u_g] = detail::pauli_uv(ctx.gen.word(w), e);
        delta += std::popcount(v_m & ~u_m);
        delta -= std::popcount(v_n & ~u_n);
        const uint64_t x_g = u_g ^ v_g;
        cross += std::popcount(v_m & x_g);
    }
    return detail::mod4(delta + 2 * cross) == 1 ? -1 : 1;
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
