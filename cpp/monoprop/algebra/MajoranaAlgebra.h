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

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/operator/RowAccess.h"

namespace monoprop {

inline constexpr auto POWERS_OF_I =
    std::array<std::complex<double>, 4>{{1.0, std::complex<double>(0.0, 1.0), -1.0, std::complex<double>(0.0, -1.0)}};
inline constexpr auto POWERS_OF_MINUS_ONE = std::array{1, -1};
inline constexpr auto REAL_PARTS = std::array{1, 0, -1, 0};

// The i^C(|maj|,2) factor that makes a Majorana product Hermitian.
auto hermitian_coefficient(const MonomialLike auto &maj) -> std::complex<double> {
    const auto pop = maj.count();
    return POWERS_OF_I[n_choose_2(pop) % 4];
}

// A Majorana product of L indices is antihermitian iff L/2 is odd.
inline auto is_antihermitian(const VecZ &indices) -> bool {
    return ((indices.size() / 2) % 2) != 0;
}

inline auto antihermitian_generator_correction(const VecZ &indices) -> std::complex<double> {
    return POWERS_OF_I[(n_choose_2(indices.size()) + 1) % 4];
}

// Diagonal element <b|M|b> against the initial product state, whose occupation mask is state_mask
// (initial_state_mask): (-1)^(|maj & state_mask| + |maj|/2) -- the pairing sign folds in on top of the
// occupation parity. Only meaningful for fully-paired terms.
auto majorana_state_phase(const MonomialLike auto &maj, const auto &state_mask) -> double {
    const auto num_pairs = maj.count_and(state_mask);
    return POWERS_OF_MINUS_ONE[(num_pairs + maj.count() / 2) % 2];
}

constexpr auto prefix_xor_64(uint64_t x) -> uint64_t {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    return x;
}

// Ordering sign (-1)^S of maj·gen, S = #{set bits of maj strictly below each set bit of gen} mod 2.
// Reference spec: the hot path uses the equivalent per-layer mask form (see interleave_phase_mask).
auto interleave_phase(const MonomialLike auto &maj_bs, const auto &gen_bs) -> int {
    const size_t n_words = maj_bs.num_words();
    size_t parity = 0;
    uint64_t carry = 0;

    for (size_t i = 0; i < n_words; ++i) {
        const uint64_t maj_word = maj_bs.word(i);
        const uint64_t gen_word = gen_bs.word(i);
        if (gen_word == 0) {
            carry ^= static_cast<uint64_t>(std::popcount(maj_word)) & 1;
            continue;
        }

        const uint64_t prefix_xor = prefix_xor_64(maj_word);
        // Shift left by 1 to exclude the bit itself; carry_mask is 0 or all-ones, broadcasting the
        // previous words' parity across the word. Written as 0 - carry, not -carry: same result,
        // without a unary minus on an unsigned operand.
        const uint64_t carry_mask = 0ULL - carry;
        const uint64_t running_parity = (prefix_xor << 1) ^ carry_mask;
        parity ^= static_cast<size_t>(std::popcount(running_parity & gen_word));
        carry ^= prefix_xor >> 63;
    }

    return (parity & 1) == 0 ? 1 : -1;
}

// Per-generator mask W collapsing the per-term interleave sign to one masked parity.
// Identity: interleave_phase(M,G) = (−1)^{parity(M ∩ W)} with W = {c : #{g∈G : g>c} odd}, fixed for
// the layer; the per-term sign is then one maj.parity_and(W) instead of the prefix-XOR scan.
auto interleave_phase_mask(const MonomialLike auto &gen) -> std::remove_cvref_t<decltype(gen)> {
    // Copy-then-reset for the same reason as change_basis: this needs a zero bitset at gen's width, and
    // MonomialLike constrains operations, not constructors, so there is no width-argument construction
    // to call on a deduced type.
    std::remove_cvref_t<decltype(gen)> w = gen;
    w.reset();
    size_t above = 0; // #{g∈G : g>c}, maintained as c descends
    for (size_t c = gen.size(); c-- > 0;) {
        if ((above & 1U) != 0U) {
            w.set(c);
        }
        above += gen.test(c) ? 1U : 0U;
    }
    return w;
}

inline auto hermitian_phase(size_t maj_count, size_t gen_count, size_t overlap) -> int {
    const auto intersection = maj_count + gen_count - 2 * overlap;
    const auto power =
        (n_choose_2(maj_count) + n_choose_2(gen_count) - n_choose_2(intersection) + 3) % 4; // +3 for 1j denominator
    return REAL_PARTS[power];
};

// Selected slot i of the logical range owns the bit pair 2*(prefix+i), 2*(prefix+i)+1 — a paired
// Majorana term sets both bits of every selected mode, so the two bits always travel together.
// inline: a plain function in a header, where being a template used to supply the linkage.
inline auto monomial_from_selector(const std::vector<bool> &selector, size_t inactive_mode_prefix, size_t num_bits)
    -> Bitset {
    Bitset current(num_bits);
    for (size_t i = 0; i < selector.size(); ++i) {
        if (selector[i]) {
            const size_t bit_pair_offset = inactive_mode_prefix + i;
            current.set(2 * bit_pair_offset);
            current.set(2 * bit_pair_offset + 1);
        }
    }
    return current;
}

// How many monomials for_each_paired_op() yields: Sum_{k<=max_ones} C(logical_num_modes, k). Lets a
// caller size storage without generating anything first. Saturates rather than reports nonsense only
// at mode counts whose monomial list could not fit in memory anyway.
[[nodiscard]] inline auto count_paired_op(size_t max_ones, size_t logical_num_modes) -> size_t {
    max_ones = std::min(max_ones, logical_num_modes);
    size_t total = 0;
    for (size_t k = 0, binomial = 1; k <= max_ones; ++k) {
        total += binomial;
        binomial = binomial * (logical_num_modes - k) / (k + 1);
    }
    return total;
}

// Every fully paired Majorana monomial with up to max_ones pairs over the active logical modes, one
// at a time, in the same order generate_paired_op() lists them.
//
// A caller that keeps only a subset must use this and not generate_paired_op: the full list is
// count_paired_op() monomials, which in the Schrodinger picture is the entire term count, and a
// propagator with S partitions constructs S propagators that would each hold a complete copy at the
// same moment. Insertion order is load-bearing -- it fixes term indices and hence float accumulation
// order -- so this yields in exactly the list's order.
// num_bits is the storage width the monomials are built at; the logical modes occupy its *top* slots,
// so the inactive prefix is the difference between the two widths and not something logical_num_modes
// can supply on its own.
auto for_each_paired_op(size_t max_ones, size_t logical_num_modes, size_t num_bits, auto &&fn) -> void {
    // Clamp in pairs, not bits: max_ones counts pairs and bounds the fill over `selector`, one slot per mode.
    max_ones = std::min(max_ones, logical_num_modes);

    auto selector = std::vector(logical_num_modes, false);
    const size_t inactive_mode_prefix = num_bits / 2 - logical_num_modes;

    for (size_t num_ones = 0; num_ones <= max_ones; ++num_ones) {
        std::fill(selector.begin(), selector.begin() + num_ones, true);

        do {
            fn(monomial_from_selector(selector, inactive_mode_prefix, num_bits));
        }
        while (std::ranges::prev_permutation(selector).found);

        std::ranges::fill(selector, false);
    }
}

// All fully paired Majorana monomials with up to max_ones pairs, over the active logical modes only.
// Prefer for_each_paired_op() unless the whole list is genuinely needed at once.
inline auto generate_paired_op(size_t max_ones, size_t logical_num_modes, size_t num_bits) -> MonomialList {
    MonomialList combinations;
    combinations.reserve(count_paired_op(max_ones, logical_num_modes));
    for_each_paired_op(max_ones, logical_num_modes, num_bits, [&combinations](const auto &mono) {
        combinations.push_back(mono);
    });
    return combinations;
}

auto encode_coeff(const std::complex<double> &coeff, const MonomialLike auto &maj) -> double {
    const auto encoded = coeff / hermitian_coefficient(maj);

    if (std::abs(encoded.imag()) > 1e-10) {
        throw NonEncodableCoefficient("Non-Hermitian coeffs detected");
    }

    return encoded.real();
}

auto decode_coeff(const std::complex<double> &coeff, const MonomialLike auto &maj) -> std::complex<double> {
    return coeff * hermitian_coefficient(maj);
}

// `basis` stays a plain (unconstrained) auto: it is a MonomialList, a container of monomials rather
// than a monomial, so it is not itself MonomialLike. Its elements' width always matches maj's at
// every call site -- required, since the XOR below asserts matching widths.
auto change_basis(const MonomialLike auto &maj, const auto &basis) -> std::remove_cvref_t<decltype(maj)> {
    using Mono = std::remove_cvref_t<decltype(maj)>;
    const size_t width = maj.size();
    const size_t num_modes = width / 2;
    // Copy-then-reset, rather than `Mono new_maj;` (width 0) or a width-argument constructor:
    // MonomialLike constrains operations, not constructors, so a deduced Mono is not known to have
    // one. The copied words are immediately overwritten; this path only runs when a basis change is
    // configured.
    Mono new_maj = maj;
    new_maj.reset();

    size_t pos = maj.find_first();
    while (pos < width) {
        new_maj ^= materialize_row(basis, 2 * num_modes - pos - 1);
        pos = maj.find_next(pos);
    }

    return new_maj;
}

} // namespace monoprop
