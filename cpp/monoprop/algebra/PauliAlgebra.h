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
#include <iterator>
#include <stdexcept>
#include <vector>

#include "monoprop/Bitset.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/algebra/AlgebraCommon.h"

namespace monoprop {

[[nodiscard]] inline auto pauli_even_mask(size_t num_bits) -> Bitset {
    return even_bits<LSb0>(num_bits);
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
// e_mask/result build from p.size()/p.num_words() (instance calls), not the qualified
// decltype(p)::size() other functions in this file use: p is only constrained MonomialLike, and a
// caller may hand this a plain Bitset (e.g. from a ^ b), whose width is data and so has no static
// size() to qualify-call.
template <MonomialLike T>
[[nodiscard]] auto pair_swap(const T &p) -> T {
    const auto &e_mask = cached_even_bits<LSb0>(p.size());
    Bitset result(p.size());
    const size_t nw = p.num_words();
    for (size_t w = 0; w < nw; ++w) {
        const uint64_t word = p.word(w);
        const uint64_t e = e_mask.word(w);
        result.data()[w] = ((word & e) << 1) | ((word >> 1) & e);
    }
    return result;
}

// A Y letter has v=1, u=0.
[[nodiscard]] auto pauli_y_count(const MonomialLike auto &p) -> size_t {
    const auto &e_mask = cached_even_bits<LSb0>(p.size());
    size_t y = 0;
    const size_t nw = p.num_words();
    for (size_t w = 0; w < nw; ++w) {
        const auto [v, u] = detail::pauli_uv(p.word(w), e_mask.word(w));
        y += static_cast<size_t>(std::popcount(v & ~u));
    }
    return y;
}

// Whether two Pauli strings anticommute (symplectic inner product is odd):
// P.parity_and(pair_swap(G)) == (x_P . z_G + z_P . x_G) mod 2.
[[nodiscard]] auto pauli_anticommutes(const MonomialLike auto &p, const auto &g) -> bool {
    return p.parity_and(pair_swap(g));
}

namespace detail {
// Reduce a (possibly negative) i-power exponent to [0, 4).
[[nodiscard]] constexpr auto mod4(long e) -> int {
    return static_cast<int>(((e % 4) + 4) % 4);
}
} // namespace detail

// One entry per word G occupies -- the only words the sign kernel below visits, since elsewhere the
// mono/new_mono Y counts cancel and x_gen is 0. All three fields are fixed for the layer, so they are
// derived once here rather than in the per-term loop, which used to rebuild G's x-plane from G's own
// word on every term and needed the even mask parked in the context to do it.
//
// Deriving them here is a simplification and not a speedup: measured pinned single-threaded, it moves
// the instruction count on either shipping model by under 0.05%, because the optimizer was already
// hoisting the derivation out of the inlined scan loop.
struct PauliGenWord {
    size_t w;     // storage word index, ascending
    uint64_t e;   // the even-bit mask for word w
    uint64_t x_g; // G's x-plane in word w, aligned onto the even lane
};

// Per-generator context for the hot emit-sign kernel. Three members, not five: `words` carries its own
// length, and the even mask no longer has to be held here because nothing rebuilds it per term.
//
// `words` is a vector, not the std::array<..., num_words()> the word list was: with no compile-time
// width there is no bound to size an array by. It holds at most num_words() entries and is built once
// per layer, so the allocation is per layer while the reads are per term -- the same trade the retained
// LazyFold already makes for its columns.
struct PauliGenContext final {
    Bitset gen{};
    size_t g_y = 0;
    std::vector<PauliGenWord> words{};
};

// Call once per layer, not per term.
auto make_pauli_gen_context(const MonomialLike auto &gen) -> PauliGenContext {
    PauliGenContext ctx;
    const size_t nw = gen.num_words();
    ctx.gen = gen;
    ctx.g_y = pauli_y_count(gen);
    const auto &e_mask = cached_even_bits<LSb0>(gen.size());
    ctx.words.reserve(nw);
    for (size_t w = 0; w < nw; ++w) {
        const uint64_t word = gen.word(w);
        if (word == 0) {
            continue;
        }
        const uint64_t e = e_mask.word(w);
        const auto [v_g, u_g] = detail::pauli_uv(word, e);
        ctx.words.emplace_back(w, e, u_g ^ v_g);
    }
    return ctx;
}

// Hot kernel: the rotation sign +/-1 for the anticommuting product mono*gen (new_mono = mono^gen).
// Returns the sign O' = U†OU (U = exp(iθ·gen)) needs on the off-diagonal partner term: the negated
// raw product sign, so the emit site needs no extra negation (pinned by pauli_algebra_tests.cpp).
// Loops only over gen's nonzero words (elsewhere mono/new_mono Y counts cancel and x_gen = 0). Exponent
// e = g_y + Σ_w(yMono - yNew) + 2·Σ_w(v_mono & x_gen); raw sign = (e mod 4 == 1 ? +1 : -1), negated here.
//
// Takes the two operands as word pointers, which is the form the per-gate kernel already has: it
// resolved them once, where mono.word(w) / new_mono.word(w) re-select a storage pointer on every one
// of the ctx.words accesses. Both must point at ctx.gen's width.
[[gnu::always_inline]] inline auto pauli_rotation_sign_words(const PauliGenContext &ctx,
                                                             const uint64_t *mono,
                                                             const uint64_t *new_mono) -> int {
    auto delta = static_cast<long>(ctx.g_y);
    long cross = 0;
    const size_t n = ctx.words.size();
    for (size_t k = 0; k < n; ++k) {
        const auto [w, e, x_g] = ctx.words[k];
        const auto [v_m, u_m] = detail::pauli_uv(mono[w], e);
        const auto [v_n, u_n] = detail::pauli_uv(new_mono[w], e);
        delta += std::popcount(v_m & ~u_m);
        delta -= std::popcount(v_n & ~u_n);
        cross += std::popcount(v_m & x_g);
    }
    return detail::mod4(delta + 2 * cross) == 1 ? -1 : 1;
}

// The monomial form of the above, for callers that hold bitsets rather than words. data() is where
// word(w) reads from, so this is the same computation and not a second one.
[[gnu::always_inline]] inline auto pauli_rotation_sign(const auto &ctx,
                                                       const MonomialLike auto &mono,
                                                       const auto &new_mono) -> int {
    return pauli_rotation_sign_words(ctx, std::data(mono), std::data(new_mono));
}

// Diagonal element <b|P|b> = (-1)^{|Z ∩ occupied|} of a Z-only Pauli against the initial product
// state. Only meaningful where is_paired holds; for a non-diagonal Pauli <b|P|b> = 0.
[[nodiscard]] auto pauli_state_phase(const MonomialLike auto &mono, const auto &state_mask) -> double {
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
