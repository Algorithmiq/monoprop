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
#include <cstdint>
#include <optional>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"

/*!
 * @file algebra/AlgebraCommon.h
 * @brief Basis-agnostic structural primitives (pairing, cutoffs, index<->bit conversions) meaningful
 * in either basis; the basis-specific algebra lives in the sibling MajoranaAlgebra.h / PauliAlgebra.h.
 */

namespace monoprop {

/**
 * @brief Converts a vector of Majorana indices to a bitset representation
 */
template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> Monomial<NumModes> {
    Monomial<NumModes> bs;
    for (const auto &bit_loc : arr) {
        bs.set(2 * NumModes - 1 - bit_loc); // MSb0 convention: index 0 maps to the top bit
    }
    return bs;
}

/**
 * @brief Converts a bitset to a vector of indices where bits are set to 1.
 *        Uses find_first/find_next for O(popcount) scanning instead of O(NumModes).
 */
template <size_t NumModes>
auto bitset_to_indices(const Monomial<NumModes> &bs) -> VecZ {
    const auto pop = bs.count();
    VecZ indices(pop);
    size_t idx = pop;
    for (size_t pos = bs.find_first(); pos < bs.size(); pos = bs.find_next(pos)) {
        indices[--idx] = bs.size() - 1 - pos;
    }
    return indices;
}

/**
 * @brief Checks if a single Majorana operator is fully paired
 */
template <size_t NumModes>
auto is_paired(const Monomial<NumModes> &maj, const Monomial<NumModes> &even_mask) -> bool {
    // Paired = each mode's even bit and its odd partner agree (both set or both clear).
    const auto even_bits_masked = maj & even_mask;
    const auto odd_bits_masked = (maj >> 1) & even_mask;
    return (even_bits_masked ^ odd_bits_masked).none();
}

/**
 * @brief Convenience overload that builds the pairing mask internally
 */
template <size_t NumModes>
auto is_paired(const Monomial<NumModes> &maj) -> bool {
    const auto even_mask = even_bits<2 * NumModes, LSb0>();
    return is_paired<NumModes>(maj, even_mask);
}

template <size_t NumModes>
auto is_paired(const VecZ &maj) -> bool {
    return is_paired<NumModes>(indices_to_bitset<NumModes>(maj));
}

/**
 * @brief Checks if a collection of Majorana operators are fully paired
 */
template <size_t NumModes, typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op) -> VecZ {
    VecZ result;
    const auto mask = even_bits<2 * NumModes, LSb0>();
    // Appended in ascending `inds` order; only the SET is observable to callers, but order is deterministic.
    for (const auto index : inds) {
        const auto &op_row = materialize_row<NumModes>(op, index);
        if (is_paired<NumModes>(op_row, mask)) {
            result.push_back(index);
        }
    }
    return result;
}

/**
 * @brief Builds a Hartree-Fock mask from occupied fermionic modes
 */
template <size_t NumModes>
auto get_hf_mask(const VecZ &hf) -> Monomial<NumModes> {
    VecZ hf_bits;
    hf_bits.reserve(hf.size());
    for (const auto &mode : hf) {
        hf_bits.push_back(2 * mode);
    }
    return indices_to_bitset<NumModes>(hf_bits);
}

/**
 * @brief Length cutoff: keep a monomial iff its length is within @p cutoff, OR it is fully paired.
 *
 * Fully paired monomials (xor_sum == 0) are kept unconditionally: they are the only terms that
 * contribute to an expectation value against a computational-basis state / Slater determinant, so
 * dropping them by length would discard signal. Otherwise keep iff Majorana count <= @p cutoff.
 */
template <size_t NumModes>
auto length_cutoff(const Monomial<NumModes> &maj, unsigned int cutoff, size_t logical_num_modes) -> bool {
    const size_t inactive_mode_prefix = NumModes - logical_num_modes;
    const size_t active_bit_offset = 2 * inactive_mode_prefix;

    if constexpr (Monomial<NumModes>::num_words() == 1) {
        constexpr size_t num_bits = Monomial<NumModes>::size();
        constexpr uint64_t valid_mask = num_bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << num_bits) - 1);
        constexpr uint64_t even_mask = even_bits<2 * NumModes, LSb0>().word(0);
        const uint64_t active_mask =
            active_bit_offset == 0 ? valid_mask : (valid_mask & ~((uint64_t{1} << active_bit_offset) - 1));
        const uint64_t active_word = maj.word(0) & active_mask;
        const uint64_t pair_mask = even_mask & active_mask;
        const auto xor_sum = std::popcount((active_word & pair_mask) ^ ((active_word >> 1) & pair_mask));
        const auto popcount_sum = std::popcount(active_word);
        return xor_sum == 0 || popcount_sum <= cutoff;
    }

    const auto active_maj = logical_num_modes == NumModes ? maj : (maj >> active_bit_offset);
    const auto mask = even_bits<2 * NumModes, LSb0>();
    const auto first_pair = active_maj & mask;
    const auto second_pair = (active_maj >> 1) & mask;
    const auto xor_sum = (first_pair ^ second_pair).count();
    const auto popcount_sum = active_maj.count();
    return xor_sum == 0 || popcount_sum <= cutoff;
}

template <size_t NumModes>
auto length_cutoff(const Monomial<NumModes> &maj, unsigned int cutoff) -> bool {
    return length_cutoff<NumModes>(maj, cutoff, NumModes);
}

/**
 * @brief Support cutoff: keep a monomial iff its orbital support is within @p cutoff, OR it is fully paired.
 *
 * Fully paired terms are kept unconditionally (same expectation-value reason as length_cutoff).
 * Support (or_sum: orbital j counts once if either of its Majoranas is present) is coarser than
 * length; under Jordan-Wigner it equals the qubit Pauli weight, so this bounds the X/Y/Z factor count.
 */
template <size_t NumModes>
auto support_cutoff(const Monomial<NumModes> &maj, unsigned int cutoff, size_t logical_num_modes) -> bool {
    const size_t inactive_mode_prefix = NumModes - logical_num_modes;
    const size_t active_bit_offset = 2 * inactive_mode_prefix;

    if constexpr (Monomial<NumModes>::num_words() == 1) {
        constexpr size_t num_bits = Monomial<NumModes>::size();
        constexpr uint64_t valid_mask = num_bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << num_bits) - 1);
        constexpr uint64_t even_mask = even_bits<2 * NumModes, LSb0>().word(0);
        const uint64_t active_mask =
            active_bit_offset == 0 ? valid_mask : (valid_mask & ~((uint64_t{1} << active_bit_offset) - 1));
        const uint64_t active_word = maj.word(0) & active_mask;
        const uint64_t pair_mask = even_mask & active_mask;
        const auto first_pair = active_word & pair_mask;
        const auto second_pair = (active_word >> 1) & pair_mask;
        const auto xor_sum = std::popcount(first_pair ^ second_pair);
        const auto or_sum = std::popcount(first_pair | second_pair);
        return xor_sum == 0 || or_sum <= cutoff;
    }

    const auto active_maj = logical_num_modes == NumModes ? maj : (maj >> active_bit_offset);
    const auto mask = even_bits<2 * NumModes, LSb0>();
    const auto first_pair = active_maj & mask;
    const auto second_pair = (active_maj >> 1) & mask;
    const auto xor_sum = (first_pair ^ second_pair).count();
    const auto or_sum = (first_pair | second_pair).count();
    return xor_sum == 0 || or_sum <= cutoff;
}

template <size_t NumModes>
auto support_cutoff(const Monomial<NumModes> &maj, unsigned int cutoff) -> bool {
    return support_cutoff<NumModes>(maj, cutoff, NumModes);
}

namespace detail {

template <size_t NumModes>
struct LengthCutoff {
    unsigned int cutoff = 0;
    size_t logical_num_modes = NumModes;

    auto operator()(const Monomial<NumModes> &maj) const -> bool {
        return length_cutoff<NumModes>(maj, cutoff, logical_num_modes);
    }
};

template <size_t NumModes>
struct SupportCutoff {
    unsigned int cutoff = 0;
    size_t logical_num_modes = NumModes;

    auto operator()(const Monomial<NumModes> &maj) const -> bool {
        return support_cutoff<NumModes>(maj, cutoff, logical_num_modes);
    }
};

template <size_t NumModes>
class CutoffEvaluator {
public:
    explicit CutoffEvaluator(const CutoffFn<NumModes> &cutoff_fn)
        : cutoff_fn_(cutoff_fn),
          length_cutoff_(cutoff_fn.template target<LengthCutoff<NumModes>>()),
          support_cutoff_(cutoff_fn.template target<SupportCutoff<NumModes>>()) {}

    auto length_cutoff() const -> const LengthCutoff<NumModes> * { return length_cutoff_; }

    auto support_cutoff() const -> const SupportCutoff<NumModes> * { return support_cutoff_; }

    auto operator()(const Monomial<NumModes> &maj) const -> bool {
        if (length_cutoff_ != nullptr) {
            return (*length_cutoff_)(maj);
        }
        if (support_cutoff_ != nullptr) {
            return (*support_cutoff_)(maj);
        }
        return cutoff_fn_(maj);
    }

    // Fast path when popcount(maj) is known: the predicate is `xor_sum==0 || (popcount/or_sum)<=cutoff`,
    // so popcount<=cutoff alone proves keep without reading the bitset (or_sum<=popcount makes support safe).
    auto passes_with_popcount(const Monomial<NumModes> &maj, size_t popcount_sum) const -> bool {
        if (length_cutoff_ != nullptr) {
            if (popcount_sum <= length_cutoff_->cutoff) {
                return true;
            }
            return (*length_cutoff_)(maj);
        }
        if (support_cutoff_ != nullptr) {
            if (popcount_sum <= support_cutoff_->cutoff) {
                return true;
            }
            return (*support_cutoff_)(maj);
        }
        return cutoff_fn_(maj);
    }

    // Upper bound on the Majorana positions a surviving term can carry, for the structural cutoffs
    // (nullopt for an arbitrary user cutoff_fn). Lets the store size its packed inline rows from the cutoff.
    auto max_positions_bound() const -> std::optional<size_t> {
        if (length_cutoff_ != nullptr) {
            return length_cutoff_->cutoff;
        }
        if (support_cutoff_ != nullptr) {
            return support_cutoff_->cutoff;
        }
        return std::nullopt;
    }

private:
    const CutoffFn<NumModes> &cutoff_fn_;
    const LengthCutoff<NumModes> *length_cutoff_;
    const SupportCutoff<NumModes> *support_cutoff_;
};

} // namespace detail

} // namespace monoprop
