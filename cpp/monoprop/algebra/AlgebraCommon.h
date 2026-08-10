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
#include <format>
#include <optional>
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"

namespace monoprop {

// A Majorana/Pauli index at or past the width of the system it is being applied to.
class AlgebraIndexOutOfRange : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A coefficient with no real encoding under the algebra model: non-Hermitian for Majorana products,
// non-real for Pauli strings.
class NonEncodableCoefficient : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Unchecked: `2 * NumModes - 1 - bit_loc` underflows for an out-of-range index and Monomial::set is
// noexcept, so the result is an out-of-bounds write. Use indices_to_bitset_checked() for user input.
template <size_t NumModes>
auto indices_to_bitset(const VecZ &arr) -> Monomial<NumModes> {
    Monomial<NumModes> bs;
    for (const auto &bit_loc : arr) {
        bs.set(2 * NumModes - 1 - bit_loc); // MSb0 convention: index 0 maps to the top bit
    }
    return bs;
}

// The bound is the logical width (2 * logical_num_modes), not the storage width 2 * NumModes: a
// propagator over fewer modes than its instantiation must still reject indices outside its own system.
template <size_t NumModes>
auto indices_to_bitset_checked(const VecZ &arr, size_t max_index) -> Monomial<NumModes> {
    for (const auto &bit_loc : arr) {
        if (bit_loc >= max_index) {
            throw AlgebraIndexOutOfRange(
                std::format("Majorana/Pauli index {} is out of range; must be less than {}.", bit_loc, max_index));
        }
    }
    return indices_to_bitset<NumModes>(arr);
}

// O(popcount) via find_first/find_next rather than an O(NumModes) scan.
auto bitset_to_indices(const MonomialLike auto &bs) -> VecZ {
    const auto pop = bs.count();
    VecZ indices(pop);
    size_t idx = pop;
    for (size_t pos = bs.find_first(); pos < bs.size(); pos = bs.find_next(pos)) {
        indices[--idx] = bs.size() - 1 - pos;
    }
    return indices;
}

auto is_paired(const MonomialLike auto &mono, const auto &even_mask) -> bool {
    // Paired = each mode's even bit and its odd partner agree (both set or both clear).
    const auto even_bits_masked = mono & even_mask;
    const auto odd_bits_masked = (mono >> 1) & even_mask;
    return (even_bits_masked ^ odd_bits_masked).none();
}

auto is_paired(const MonomialLike auto &mono) -> bool {
    const auto even_mask = even_bits<std::remove_cvref_t<decltype(mono)>::size(), LSb0>();
    return is_paired(mono, even_mask);
}

// No monomial argument to deduce a width from -- NumModes stays explicit (it sizes the
// Monomial<NumModes> this constructs from `mono`, an index list, not a monomial itself).
template <size_t NumModes>
auto is_paired(const VecZ &mono) -> bool {
    return is_paired(indices_to_bitset<NumModes>(mono));
}

// Rows carries no structural width of its own (unlike a MonomialLike argument), so NumModes stays
// explicit here too -- it only reaches materialize_row.
template <size_t NumModes, typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op) -> VecZ {
    VecZ result;
    const auto mask = even_bits<2 * NumModes, LSb0>();
    for (const auto index : inds) {
        const auto &op_row = materialize_row(op, index);
        if (is_paired(op_row, mask)) {
            result.push_back(index);
        }
    }
    return result;
}

// Occupation mask of the initial product state: the even index 2*i of each listed mode (Majorana) or
// qubit (Pauli) that starts in state 1. Both algebras read the same mask and differ only in the phase
// they score against it (majorana_state_phase / pauli_state_phase).
template <size_t NumModes>
auto initial_state_mask(const VecZ &initial_state) -> Monomial<NumModes> {
    VecZ bits;
    bits.reserve(initial_state.size());
    for (const auto &mode : initial_state) {
        bits.push_back(2 * mode);
    }
    return indices_to_bitset<NumModes>(bits);
}

// The per-mode sums the structural cutoffs measure, over the active modes only.
struct CutoffSums {
    size_t xor_sum;      // modes with exactly one of their two Majoranas set; 0 == fully paired
    size_t popcount_sum; // Majorana operators present -- the length measure
    size_t or_sum;       // modes with either Majorana present -- the support measure (JW Pauli weight)
};

[[gnu::always_inline]] inline auto cutoff_sums(const MonomialLike auto &mono, size_t logical_num_modes) -> CutoffSums {
    using Mono = std::remove_cvref_t<decltype(mono)>;
    constexpr size_t num_modes = Mono::size() / 2;
    const size_t active_bit_offset = 2 * (num_modes - logical_num_modes);

    if constexpr (Mono::num_words() == 1) {
        constexpr size_t num_bits = Mono::size();
        constexpr uint64_t valid_mask = num_bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << num_bits) - 1);
        const uint64_t even_mask = even_bits<Mono::size(), LSb0>().word(0);
        const uint64_t active_mask =
            active_bit_offset == 0 ? valid_mask : (valid_mask & ~((uint64_t{1} << active_bit_offset) - 1));
        const uint64_t active_word = mono.word(0) & active_mask;
        const uint64_t pair_mask = even_mask & active_mask;
        const uint64_t first_pair = active_word & pair_mask;
        const uint64_t second_pair = (active_word >> 1) & pair_mask;
        return {static_cast<size_t>(std::popcount(first_pair ^ second_pair)),
                static_cast<size_t>(std::popcount(active_word)),
                static_cast<size_t>(std::popcount(first_pair | second_pair))};
    }

    // Both branches must agree on type for the ternary: mono is Mono (a Monomial<NumModes>), but
    // `mono >> ...` is plain Bitset (operators live on the base -- see the Stage 2b wrapper note in
    // core/Monomial.h), so mono needs the same explicit upcast to avoid an ambiguous common type.
    const auto active_mono =
        logical_num_modes == num_modes ? static_cast<const Bitset &>(mono) : (mono >> active_bit_offset);
    const auto mask = even_bits<Mono::size(), LSb0>();
    const auto first_pair = active_mono & mask;
    const auto second_pair = (active_mono >> 1) & mask;
    return {(first_pair ^ second_pair).count(), active_mono.count(), (first_pair | second_pair).count()};
}

// Both cutoffs below keep a fully paired monomial (xor_sum == 0) unconditionally: those are the only
// terms contributing to an expectation value against a product reference state, so bounding them by
// length or support would discard signal.

auto length_cutoff(const MonomialLike auto &mono, unsigned int cutoff, size_t logical_num_modes) -> bool {
    const auto sums = cutoff_sums(mono, logical_num_modes);
    return sums.xor_sum == 0 || sums.popcount_sum <= cutoff;
}

auto length_cutoff(const MonomialLike auto &mono, unsigned int cutoff) -> bool {
    return length_cutoff(mono, cutoff, std::remove_cvref_t<decltype(mono)>::size() / 2);
}

auto support_cutoff(const MonomialLike auto &mono, unsigned int cutoff, size_t logical_num_modes) -> bool {
    const auto sums = cutoff_sums(mono, logical_num_modes);
    return sums.xor_sum == 0 || sums.or_sum <= cutoff;
}

auto support_cutoff(const MonomialLike auto &mono, unsigned int cutoff) -> bool {
    return support_cutoff(mono, cutoff, std::remove_cvref_t<decltype(mono)>::size() / 2);
}

namespace detail {

template <size_t NumModes>
struct LengthCutoff {
    unsigned int cutoff = 0;
    size_t logical_num_modes = NumModes;

    auto operator()(const Monomial<NumModes> &mono) const -> bool {
        return length_cutoff(mono, cutoff, logical_num_modes);
    }
};

template <size_t NumModes>
struct SupportCutoff {
    unsigned int cutoff = 0;
    size_t logical_num_modes = NumModes;

    auto operator()(const Monomial<NumModes> &mono) const -> bool {
        return support_cutoff(mono, cutoff, logical_num_modes);
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

    auto operator()(const Monomial<NumModes> &mono) const -> bool {
        if (length_cutoff_ != nullptr) {
            return (*length_cutoff_)(mono);
        }
        if (support_cutoff_ != nullptr) {
            return (*support_cutoff_)(mono);
        }
        return cutoff_fn_(mono);
    }

    // Fast path when popcount(mono) is known: the predicate is `xor_sum==0 || (popcount/or_sum)<=cutoff`,
    // so popcount<=cutoff alone proves keep without reading the bitset (or_sum<=popcount makes support safe).
    auto passes_with_popcount(const Monomial<NumModes> &mono, size_t popcount_sum) const -> bool {
        if (length_cutoff_ != nullptr) {
            if (popcount_sum <= length_cutoff_->cutoff) {
                return true;
            }
            return (*length_cutoff_)(mono);
        }
        if (support_cutoff_ != nullptr) {
            if (popcount_sum <= support_cutoff_->cutoff) {
                return true;
            }
            return (*support_cutoff_)(mono);
        }
        return cutoff_fn_(mono);
    }

    // Upper bound on the set bits (physical slots) a surviving term can carry, so the store can size
    // its packed inline rows. A length cutoff counts set bits directly; a support cutoff counts
    // modes/qubits, each spanning two slots, hence the x2.
    auto max_slot_bound() const -> std::optional<size_t> {
        if (length_cutoff_ != nullptr) {
            return length_cutoff_->cutoff;
        }
        if (support_cutoff_ != nullptr) {
            return 2 * support_cutoff_->cutoff;
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
