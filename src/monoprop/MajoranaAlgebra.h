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

#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"

namespace monoprop {

inline constexpr auto POWERS_OF_I =
    std::array<std::complex<double>, 4>{{1.0, std::complex<double>(0.0, 1.0), -1.0, std::complex<double>(0.0, -1.0)}};
inline constexpr auto POWERS_OF_MINUS_ONE = std::array{1, -1};
inline constexpr auto REAL_PARTS = std::array{1, 0, -1, 0};

/**
 * @brief Maps a Majorana operator to its hermitian coefficient
 */
template <size_t NumModes>
auto hermitian_coefficient(const MajoranaSet<NumModes> &maj) -> std::complex<double> {
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
 * @brief Converts a vector of Majorana indices to a bitset representation
 */
template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> MajoranaSet<NumModes> {
    MajoranaSet<NumModes> bs;
    for (const auto &bit_loc : arr) {
        bs.set(2 * NumModes - 1 - bit_loc); // MSb0 index convention: index 0 maps to the top bit
    }
    return bs;
}

/**
 * @brief Converts a bitset to a vector of indices where bits are set to 1.
 *        Uses find_first/find_next for O(popcount) scanning instead of O(NumModes).
 */
template <size_t NumModes>
auto bitset_to_indices(const MajoranaSet<NumModes> &bs) -> VecZ {
    const auto pop = bs.count();
    VecZ indices(pop);
    size_t idx = pop;
    for (size_t pos = bs.find_first(); pos < bs.size(); pos = bs.find_next(pos)) {
        indices[--idx] = bs.size() - 1 - pos;
    }
    return indices;
}

/**
 * @brief Converts a fermionic operator from index representation to binary (bitset) representation
 * @note Test-support only (tests/cpp/mpfunctions.cpp); not called by the shipped library.
 */
template <size_t NumModes>
auto fermionic_to_binary_operator(const std::vector<VecZ> &op) -> MajoranaVector<NumModes> {
    auto majorana_operator = MajoranaVector<NumModes>(op.size());
    std::transform(op.cbegin(), op.cend(), majorana_operator.begin(), indices_to_bitset<NumModes>);
    return majorana_operator;
}

/**
 * @brief Checks if a single Majorana operator is fully paired
 */
template <size_t NumModes>
auto is_paired(const MajoranaSet<NumModes> &maj, const MajoranaSet<NumModes> &even_mask) -> bool {
    // Paired = each mode's two Majoranas (adjacent bits) are both set or both clear, i.e. the
    // even bit and its odd partner agree for every mode.
    const auto even_bits_masked = maj & even_mask;
    const auto odd_bits_masked = (maj >> 1) & even_mask;
    return (even_bits_masked ^ odd_bits_masked).none();
}

/**
 * @brief Convenience overload that builds the pairing mask internally
 */
template <size_t NumModes>
auto is_paired(const MajoranaSet<NumModes> &maj) -> bool {
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

    if (inds.empty()) {
        return result;
    }

    // Hits are staged per CHUNK and concatenated in chunk order, so the result preserves the input
    // order of `inds` deterministically at any thread count (the former per-thread merge returned a
    // scheduling-dependent order; every caller scatters through the returned indices, so only the SET
    // was ever observable).
    constexpr size_t grain_size = 512;
    const size_t n = inds.size();
    const size_t chunks = (n + grain_size - 1) / grain_size;
    std::vector<VecZ> parts(chunks);
    threading::run_static(chunks, [&](size_t c) {
        auto &local = parts[c];
        const size_t lo = c * grain_size;
        const size_t hi = std::min(n, lo + grain_size);
        for (size_t i = lo; i < hi; ++i) {
            const auto index = inds[i];
            const auto &op_row = materialize_row<NumModes>(op, index);
            if (is_paired<NumModes>(op_row, mask)) {
                local.push_back(index);
            }
        }
    });

    for (const auto &local : parts) {
        result.insert(result.end(), local.begin(), local.end());
    }

    return result;
}

/**
 * @brief Builds a Hartree-Fock mask from occupied fermionic modes
 */
template <size_t NumModes>
auto get_hf_mask(const VecZ &hf) -> MajoranaSet<NumModes> {
    VecZ hf_bits;
    hf_bits.reserve(hf.size());
    for (const auto &mode : hf) {
        hf_bits.push_back(2 * mode);
    }
    return indices_to_bitset<NumModes>(hf_bits);
}

/**
 * @brief Calculates the Hartree-Fock phase contribution for a single Majorana term
 */
template <size_t NumModes>
auto hf_phase(const MajoranaSet<NumModes> &maj, const MajoranaSet<NumModes> &hf_mask) -> double {
    const auto num_pairs = maj.count_and(hf_mask);
    return POWERS_OF_MINUS_ONE[(num_pairs + maj.count() / 2) % 2];
}

/**
 * @brief Calculates phases for paired Majorana operators with respect to a Hartree-Fock state
 */
template <size_t NumModes, typename Rows>
auto get_hf_phases(const VecZ &paired_inds, const VecZ &hf, const Rows &op) -> VecD {
    const auto hf_mask = get_hf_mask<NumModes>(hf);
    const auto size = paired_inds.size();
    auto result = std::vector(size, 0.0);

    if (size > 0) {
        constexpr size_t grain_size = 512;
        threading::parallel_for_indices(
            size,
            [&paired_inds, &result, &hf_mask, &op](size_t idx) {
                const auto op_idx = paired_inds[idx];
                const auto &op_row = materialize_row<NumModes>(op, op_idx);
                result[idx] = hf_phase<NumModes>(op_row, hf_mask);
            },
            grain_size);
    }
    return result;
}

/**
 * @brief Length cutoff: keep a monomial iff its length is within @p cutoff, OR it is fully paired.
 *
 * Returns true (keep) when either:
 *   - the monomial is *fully paired* (`xor_sum == 0`): every Majorana operator it
 *     contains belongs to a complete pair m_{2j-1} m_{2j} on a mode, so no mode
 *     carries a lone Majorana; or
 *   - its length (`popcount_sum`, the number of Majorana operators) is <= @p cutoff.
 *
 * Fully paired monomials are kept unconditionally because they are the only terms
 * that can overlap a computational-basis state or Slater determinant under the
 * trace, and so are the only terms that contribute to an expectation value;
 * discarding them would discard signal regardless of their length.
 */
template <size_t NumModes>
auto length_cutoff(const MajoranaSet<NumModes> &maj, unsigned int cutoff, size_t logical_num_modes)
    -> bool {
    const size_t inactive_mode_prefix = NumModes - logical_num_modes;
    const size_t active_bit_offset = 2 * inactive_mode_prefix;

    if constexpr (MajoranaSet<NumModes>::num_words() == 1) {
        constexpr size_t num_bits = MajoranaSet<NumModes>::size();
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
auto length_cutoff(const MajoranaSet<NumModes> &maj, unsigned int cutoff) -> bool {
    return length_cutoff<NumModes>(maj, cutoff, NumModes);
}

/**
 * @brief Support cutoff: keep a monomial iff its orbital support is within @p cutoff, OR it is fully paired.
 *
 * Returns true (keep) when either:
 *   - the monomial is *fully paired* (`xor_sum == 0`), kept unconditionally for the
 *     same reason as in length_cutoff() -- only paired monomials contribute to an
 *     expectation value against a computational-basis state or Slater determinant; or
 *   - the number of distinct orbitals it touches (`or_sum`, the orbital support --
 *     orbital j counts once if either m_{2j-1} or m_{2j} is present) is <= @p cutoff.
 *
 * The support is a coarser measure than length, since one orbital can carry two
 * Majorana operators. Under the Jordan-Wigner mapping each occupied orbital
 * contributes exactly one single-qubit Pauli (X, Y or Z), so the support equals the
 * qubit Pauli weight and this cutoff bounds the number of X/Y/Z factors.
 */
template <size_t NumModes>
auto support_cutoff(const MajoranaSet<NumModes> &maj, unsigned int cutoff, size_t logical_num_modes) -> bool {
    const size_t inactive_mode_prefix = NumModes - logical_num_modes;
    const size_t active_bit_offset = 2 * inactive_mode_prefix;

    if constexpr (MajoranaSet<NumModes>::num_words() == 1) {
        constexpr size_t num_bits = MajoranaSet<NumModes>::size();
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
auto support_cutoff(const MajoranaSet<NumModes> &maj, unsigned int cutoff) -> bool {
    return support_cutoff<NumModes>(maj, cutoff, NumModes);
}

namespace detail {

template <size_t NumModes>
struct LengthCutoff {
    unsigned int cutoff = 0;
    size_t logical_num_modes = NumModes;

    auto operator()(const MajoranaSet<NumModes> &maj) const -> bool {
        return length_cutoff<NumModes>(maj, cutoff, logical_num_modes);
    }
};

template <size_t NumModes>
struct SupportCutoff {
    unsigned int cutoff = 0;
    size_t logical_num_modes = NumModes;

    auto operator()(const MajoranaSet<NumModes> &maj) const -> bool {
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

    auto length_cutoff() const -> const LengthCutoff<NumModes> * {
        return length_cutoff_;
    }

    auto support_cutoff() const -> const SupportCutoff<NumModes> * { return support_cutoff_; }

    auto operator()(const MajoranaSet<NumModes> &maj) const -> bool {
        if (length_cutoff_ != nullptr) {
            return (*length_cutoff_)(maj);
        }
        if (support_cutoff_ != nullptr) {
            return (*support_cutoff_)(maj);
        }
        return cutoff_fn_(maj);
    }

    // Fast path: caller already knows popcount(maj). For length_pairing and mode cutoffs
    // the predicate is `xor_sum == 0 || (popcount or or_sum) <= cutoff`; if the popcount
    // alone is already <= cutoff we can return true without touching the bitset.
    // For support_cutoff, or_sum <= popcount_sum so the same shortcut is safe.
    auto passes_with_popcount(const MajoranaSet<NumModes> &maj, size_t popcount_sum) const -> bool {
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

    // Upper bound on the number of Majorana positions a surviving term can carry, when the
    // cutoff is one of the structural kinds whose predicate fails once popcount exceeds the
    // cutoff (length-pairing-distance, mode). For an arbitrary user-supplied cutoff_fn no such
    // bound exists and this returns std::nullopt. This is the same threshold passes_with_popcount
    // short-circuits on; it lets the operator store size its packed inline rows from the cutoff
    // instead of always reserving the maximum width.
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
auto interleave_phase(const MajoranaSet<NumModes> &maj_bs, const MajoranaSet<NumModes> &gen_bs) -> int {
    constexpr size_t n_words = MajoranaSet<NumModes>::num_words();
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
auto interleave_phase_mask(const MajoranaSet<NumModes> &gen) -> MajoranaSet<NumModes> {
    MajoranaSet<NumModes> w;
    size_t above = 0; // #{g∈G : g>c}, maintained as c descends
    for (size_t c = MajoranaSet<NumModes>::size(); c-- > 0;) {
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
 * @brief Calculates the multiplicative phase factor for Majorana operator evolution
 * @note Test-support only (tests/cpp/mpfunctions.cpp); not called by the shipped library.
 */
template <size_t NumModes>
auto get_multiplicative_phase(const MajoranaSet<NumModes> &maj,
                              const MajoranaSet<NumModes> &gen_maj,
                              size_t maj_count,
                              size_t gen_count,
                              size_t overlap) -> int {
    return interleave_phase<NumModes>(maj, gen_maj) * hermitian_phase(maj_count, gen_count, overlap);
}

/**
 * @brief Generates all paired Majorana operators up to a maximum weight for the active logical modes.
 */
template <size_t NumModes>
auto generate_paired_op(size_t max_ones, size_t logical_num_modes) -> MajoranaVector<NumModes> {
    MajoranaVector<NumModes> combinations;
    max_ones = std::min(max_ones, 2 * logical_num_modes);
    auto selector = std::vector(logical_num_modes, false);
    const size_t inactive_mode_prefix = NumModes - logical_num_modes;

    for (size_t num_ones = 0; num_ones <= max_ones; ++num_ones) {
        std::fill(selector.begin(), selector.begin() + num_ones, true);

        do {
            MajoranaSet<NumModes> current;
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
 * @brief Generates all paired Majorana operators up to a maximum weight
 */
template <size_t NumModes>
auto generate_paired_op(size_t max_ones) -> MajoranaVector<NumModes> {
    return generate_paired_op<NumModes>(max_ones, NumModes);
}

/**
 * @brief Encode a single Majorana coefficient into its real representation
 */
template <size_t NumModes>
auto encode_coeff(const std::complex<double> &coeff, const MajoranaSet<NumModes> &maj) -> double {
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
auto decode_coeff(const std::complex<double> &coeff, const MajoranaSet<NumModes> &maj) -> std::complex<double> {
    return coeff * hermitian_coefficient<NumModes>(maj);
}

/**
 * @brief Changes Majorana basis using a provided transformation
 */
template <size_t NumModes>
auto change_basis(const MajoranaSet<NumModes> &maj, const MajoranaVector<NumModes> &basis) -> MajoranaSet<NumModes> {
    MajoranaSet<NumModes> new_maj;

    size_t pos = maj.find_first();
    while (pos < maj.size()) {
        new_maj ^= materialize_row<NumModes>(basis, 2 * NumModes - pos - 1);
        pos = maj.find_next(pos);
    }

    return new_maj;
}

} // namespace monoprop
