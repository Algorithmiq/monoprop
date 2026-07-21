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

/*!
 * @file algebra/MajoranaAlgebra.h
 * @brief The Majorana algebra: how a @ref Monomial is read as a product of Majorana operators.
 *
 * Sibling of algebra/PauliAlgebra.h; both build on the basis-agnostic structural primitives in
 * algebra/AlgebraCommon.h (pairing, cutoffs, index<->bit conversions). This header carries the
 * Majorana-specific algebra: the Hermitian coefficient normalization i^(C(|maj|,2)), the ordering
 * (interleave) sign and its per-layer mask form, the Hartree-Fock phase, the real<->complex
 * coefficient codec, and Majorana basis changes. The generic propagation backbone reaches these
 * through the @c MajoranaAlgebra policy model in algebra/Algebra.h.
 */

namespace monoprop {

inline constexpr auto POWERS_OF_I =
    std::array<std::complex<double>, 4>{{1.0, std::complex<double>(0.0, 1.0), -1.0, std::complex<double>(0.0, -1.0)}};
inline constexpr auto POWERS_OF_MINUS_ONE = std::array{1, -1};
inline constexpr auto REAL_PARTS = std::array{1, 0, -1, 0};

/**
 * @brief Maps a Majorana operator to its hermitian coefficient
 */
template <size_t NumModes>
auto hermitian_coefficient(const Monomial<NumModes> &maj) -> std::complex<double> {
    const auto pop = maj.count();
    // Calculate i^(|maj| choose 2) = |maj|(|maj|-1)/2
    return POWERS_OF_I[n_choose_2(pop) % 4];
}

/**
 * @brief Check if a Majorana operator (represented by indices) is antihermitian
 * @note Test-support only (tests/cpp/mpfunctions.cpp); not called by the shipped library.
 */
inline auto is_antihermitian(const VecZ &indices) -> bool {
    // Check if the number of Majorana operators is odd/even
    return ((indices.size() / 2) % 2) != 0;
}

/**
 * @brief Get the generator correction for a Majorana product represented by indices.
 * @note Test-support only (tests/cpp/mpfunctions.cpp); not called by the shipped library.
 */
inline auto antihermitian_generator_correction(const VecZ &indices) -> std::complex<double> {
    return POWERS_OF_I[(n_choose_2(indices.size()) + 1) % 4];
}

/**
 * @brief Calculates the Hartree-Fock phase contribution for a single Majorana term
 */
template <size_t NumModes>
auto hf_phase(const Monomial<NumModes> &maj, const Monomial<NumModes> &hf_mask) -> double {
    const auto num_pairs = maj.count_and(hf_mask);
    return POWERS_OF_MINUS_ONE[(num_pairs + maj.count() / 2) % 2];
}


/**
 * @brief Computes the ordering sign of the Majorana product maj * gen.
 *
 * Reference implementation. The build hot path does NOT call this per term: it precomputes the
 * fixed-per-layer interleave mask W once and evaluates the identical sign as `maj.parity_and(W)`
 * (see interleave_phase_mask + its use in Scan.h). Keep this as the branch-clear spec that the mask
 * form is proven against; don't reintroduce it into the per-term scan.
 *
 * For each set bit in @p gen, the sign flips once for each set bit in @p maj
 * at strictly lower bit positions. The returned value is therefore
 * (-1)^S where S is that crossing count modulo 2.
 *
 * This implementation is word-based:
 * - prefix_xor_64 gives per-bit prefix parity inside each 64-bit word,
 * - carry tracks prefix parity from previous words,
 * - popcount(running_parity & gen_word) accumulates the odd-crossing bits.
 */
inline constexpr auto prefix_xor_64(uint64_t x) -> uint64_t {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    return x;
}

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
        // Strict-lower-position parity: shift left by 1 to exclude the bit itself, fold in carry
        // (-carry broadcasts the previous words' parity to all 64 bits).
        const uint64_t running_parity = (prefix_xor << 1) ^ (-carry);
        parity ^= static_cast<size_t>(std::popcount(running_parity & gen_word));
        carry ^= prefix_xor >> 63;
    }

    return (parity & 1) == 0 ? 1 : -1;
}

/**
 * @brief Per-generator mask W that collapses the per-term interleave sign to one masked parity.
 *
 * IDENTITY (exact): with x = #{(m∈M, g∈G) : m<g} = Σ_{g∈G} rank_M(g),
 *   interleave_phase(M,G) = (−1)^x  and  x ≡ |{m∈M : w(m)}| (mod 2),  w(c) = #{g∈G : g>c} (mod 2).
 * Hence interleave_phase(M,G) = (−1)^{parity(M ∩ W)} with W = {c : w(c) odd}, FIXED for the layer.
 * Building W is O(2N); the per-term sign then costs one `maj.parity_and(W)` instead of the
 * latency-bound prefix-XOR scan of interleave_phase(). w(c) is computed by sweeping c high→low,
 * tracking #{g>c} (each generator bit at position c contributes to all strictly-lower columns).
 */
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

/**
 * @brief Generates all paired Majorana operators up to a maximum weight for the active logical modes.
 */
template <size_t NumModes>
auto generate_paired_op(size_t max_ones, size_t logical_num_modes) -> MonomialList<NumModes> {
    MonomialList<NumModes> combinations;
    max_ones = std::min(max_ones, 2 * logical_num_modes);
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

/**
 * @brief Encode a single Majorana coefficient into its real representation
 */
template <size_t NumModes>
auto encode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> double {
    const auto encoded = coeff / hermitian_coefficient<NumModes>(maj);

    if (std::abs(encoded.imag()) > 1e-10) {
        throw std::runtime_error("Non-Hermitian coeffs detected");
    }

    return encoded.real();
}

/**
 * @brief Decode a single real coefficient back to its complex representation
 */
template <size_t NumModes>
auto decode_coeff(const std::complex<double> &coeff, const Monomial<NumModes> &maj) -> std::complex<double> {
    return coeff * hermitian_coefficient<NumModes>(maj);
}

/**
 * @brief Changes Majorana basis using a provided transformation
 */
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
