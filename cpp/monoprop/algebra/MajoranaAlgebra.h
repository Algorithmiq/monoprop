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

namespace monoprop {

inline constexpr auto POWERS_OF_I =
    std::array<std::complex<double>, 4>{{1.0, std::complex<double>(0.0, 1.0), -1.0, std::complex<double>(0.0, -1.0)}};
inline constexpr auto POWERS_OF_MINUS_ONE = std::array{1, -1};
inline constexpr auto REAL_PARTS = std::array{1, 0, -1, 0};

// The i^C(|maj|,2) factor that makes a Majorana product Hermitian.
template <size_t NumModes>
auto hermitian_coefficient(const Monomial<NumModes> &maj) -> std::complex<double> {
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
template <size_t NumModes>
auto majorana_state_phase(const Monomial<NumModes> &maj, const Monomial<NumModes> &state_mask) -> double {
    const auto num_pairs = maj.count_and(state_mask);
    return POWERS_OF_MINUS_ONE[(num_pairs + maj.count() / 2) % 2];
}

inline constexpr auto prefix_xor_64(uint64_t x) -> uint64_t {
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
template <size_t NumModes>
auto interleave_phase(const Monomial<NumModes> &maj_bs, const Monomial<NumModes> &gen_bs) -> int {
    constexpr size_t n_words = Monomial<NumModes>::num_words();
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
template <size_t NumModes>
auto interleave_phase_mask(const Monomial<NumModes> &gen) -> Monomial<NumModes> {
    Monomial<NumModes> w;
    size_t above = 0; // #{g∈G : g>c}, maintained as c descends
    for (size_t c = Monomial<NumModes>::size(); c-- > 0;) {
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

// All fully paired Majorana monomials with up to max_ones pairs, over the active logical modes only.
template <size_t NumModes>
auto generate_paired_op(size_t max_ones, size_t logical_num_modes) -> MonomialList<NumModes> {
    MonomialList<NumModes> combinations;
    // Clamp in pairs, not bits: max_ones counts pairs and bounds the fill over `selector`, one slot per mode.
    max_ones = std::min(max_ones, logical_num_modes);
    auto selector = std::vector(logical_num_modes, false);
    const size_t inactive_mode_prefix = NumModes - logical_num_modes;

    for (size_t num_ones = 0; num_ones <= max_ones; ++num_ones) {
        std::fill(selector.begin(), selector.begin() + num_ones, true);

        do {
            Monomial<NumModes> current;
            for (size_t i = 0; i < logical_num_modes; ++i) {
                if (selector[i]) {
                    const size_t bit_pair_offset = inactive_mode_prefix + i;
                    current.set(2 * bit_pair_offset);
                    current.set(2 * bit_pair_offset + 1);
                }
            }
            combinations.push_back(current);
        }
        while (std::prev_permutation(selector.begin(), selector.end()));

        std::fill(selector.begin(), selector.end(), false);
    }

    return combinations;
}

template <size_t NumModes>
auto encode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> double {
    const auto encoded = coeff / hermitian_coefficient<NumModes>(maj);

    if (std::abs(encoded.imag()) > 1e-10) {
        throw std::runtime_error("Non-Hermitian coeffs detected");
    }

    return encoded.real();
}

template <size_t NumModes>
auto decode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> std::complex<double> {
    return coeff * hermitian_coefficient<NumModes>(maj);
}

template <size_t NumModes>
auto change_basis(const Monomial<NumModes> &maj, const MonomialList<NumModes> &basis) -> Monomial<NumModes> {
    Monomial<NumModes> new_maj;

    size_t pos = maj.find_first();
    while (pos < maj.size()) {
        new_maj ^= materialize_row<NumModes>(basis, 2 * NumModes - pos - 1);
        pos = maj.find_next(pos);
    }

    return new_maj;
}

} // namespace monoprop
