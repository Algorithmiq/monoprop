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

#include <concepts>
#include <cstddef>
#include <format>
#include <ranges>
#include <stdexcept>
#include <type_traits>

#include "monoprop/Bitset.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop {
struct BitOrdering {};

struct MSb0 : BitOrdering {};
struct LSb0 : BitOrdering {};

namespace detail {
// n is an ordinary runtime argument: Bitset carries its width as data, so there is nothing for a
// template parameter of its own to supply.
inline auto make_repeating_bitset(size_t n, uint64_t pattern) -> Bitset {
    Bitset bits(n);
    const size_t num_words = bits.num_words();
    for (size_t i = 0; i < num_words; ++i)
        bits.data()[i] = pattern;
    if (const size_t top_bits = n % 64; top_bits != 0) {
        const uint64_t top_mask = (uint64_t(1) << top_bits) - 1;
        bits.data()[num_words - 1] &= top_mask;
    }
    return bits;
}

inline auto even_bits(size_t n) -> Bitset {
    return make_repeating_bitset(n, 0x5555555555555555ULL);
}
inline auto odd_bits(size_t n) -> Bitset {
    return make_repeating_bitset(n, 0xAAAAAAAAAAAAAAAAULL);
}
} // namespace detail

// Under MSb0 the logical even positions are physically odd, so the pattern is swapped vs LSb0. The
// width is an ordinary argument: only the Ordering has to be a template parameter, since it selects
// which pattern at compile time and is never data.
template <typename Ordering>
auto even_bits(size_t n) -> Bitset {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::odd_bits(n);
    }
    else {
        return detail::even_bits(n);
    }
};
// Same MSb0/LSb0 swap as even_bits().
template <typename Ordering>
auto odd_bits(size_t n) -> Bitset {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::even_bits(n);
    }
    else {
        return detail::odd_bits(n);
    }
};

// Memoized even-bit mask for per-term code. A mask depends only on the storage width, which is fixed
// for a propagator's lifetime, so rebuilding one per term is pure waste -- and with Bitset
// runtime-width, building one is a full object construction, not a compile-time constant.
//
// thread_local rather than shared: the scan runs concurrently on the partitions' pinned masters, and
// a shared cache would need synchronisation on the hottest path in the library. The width only ever
// changes between propagators, so the miss branch is taken once per thread in practice.
//
// The reference is valid until the next call *on this thread* with a different width. Callers use it
// within a single expression or loop; do not store it across a width change.
template <typename Ordering>
[[nodiscard]] inline auto cached_even_bits(size_t n) -> const Bitset & {
    thread_local Bitset cached;
    if (thread_local auto cached_n = static_cast<size_t>(-1); cached_n != n) [[unlikely]] {
        cached = even_bits<Ordering>(n);
        cached_n = n;
    }
    return cached;
}

inline auto n_choose_2(std::integral auto n) -> size_t {
    return static_cast<size_t>(n * (n - 1) / 2);
}

auto join_with_separator(std::ranges::range auto const &values, std::string_view separator) -> std::string {
    std::string joined;
    bool first = true;
    for (const auto &value : values) {
        if (!first) {
            joined.append(separator);
        }
        first = false;
        joined.append(std::format("{}", value));
    }
    return joined;
}

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

// Unchecked: `num_bits - 1 - bit_loc` underflows for an out-of-range index and Bitset::set is
// noexcept, so the result is an out-of-bounds write. Use indices_to_bitset_checked() for user input.
inline auto indices_to_bitset(const VecZ &arr, size_t num_bits) -> Bitset {
    Bitset bs(num_bits);
    for (const auto &bit_loc : arr) {
        bs.set(num_bits - 1 - bit_loc); // MSb0 convention: index 0 maps to the top bit
    }
    return bs;
}

// The two bounds are different quantities and neither implies the other, so they are separate
// arguments: max_index is the *logical* width (2 * logical_num_modes), which is what a caller's
// indices must fall inside, while num_bits is the *storage* width the result is built at. A
// propagator running fewer modes than its storage holds must still reject indices outside its own
// system, and storage rounds up.
inline auto indices_to_bitset_checked(const VecZ &arr, size_t max_index, size_t num_bits) -> Bitset {
    for (const auto &bit_loc : arr) {
        if (bit_loc >= max_index) {
            throw AlgebraIndexOutOfRange(
                std::format("Majorana/Pauli index {} is out of range; must be less than {}.", bit_loc, max_index));
        }
    }
    return indices_to_bitset(arr, num_bits);
}

// O(popcount) via find_first/find_next rather than an O(num_bits) scan.
auto bitset_to_indices(const MonomialLike auto &bs) -> VecZ {
    const auto pop = bs.count();
    VecZ indices(pop);
    size_t idx = pop;
    const size_t n = bs.size();
    for (size_t pos = bs.find_first(); pos < n; pos = bs.find_next(pos)) {
        indices[--idx] = n - 1 - pos;
    }
    return indices;
}

auto is_paired(const MonomialLike auto &mono, const auto &even_mask) -> bool {
    // Paired = each mode's even bit and its odd partner agree (both set or both clear). Word loop
    // rather than `(mono & m) ^ ((mono >> 1) & m)`, which built three runtime-width temporaries per
    // call; (word >> 1) & m is within-word for the same reason as in cutoff_sums().
    const size_t nw = mono.num_words();
    for (size_t w = 0; w < nw; ++w) {
        const uint64_t word = mono.word(w);
        const uint64_t m = even_mask.word(w);
        if (((word & m) ^ ((word >> 1) & m)) != 0) {
            return false;
        }
    }
    return true;
}

auto is_paired(const MonomialLike auto &mono) -> bool {
    return is_paired(mono, cached_even_bits<LSb0>(mono.size()));
}

// `mono` is an index list, not a monomial, so there is no argument to deduce a width from -- hence the
// explicit num_bits, which sizes the bitset this builds.
inline auto is_paired(const VecZ &mono, size_t num_bits) -> bool {
    return is_paired(indices_to_bitset(mono, num_bits));
}

// Rows carries no structural width of its own (unlike a MonomialLike argument), so the width is
// explicit here too. It must be the width of the rows themselves: the mask is compared against them
// pairwise, and a mismatch trips Bitset's width assertions.
template <typename Rows>
auto is_fully_paired(const VecZ &inds, const Rows &op, size_t num_bits) -> VecZ {
    VecZ result;
    // Memoized rather than rebuilt: the reference stays valid across the loop because is_paired's
    // two-argument form takes the mask and so never re-enters the cache with another width.
    const auto &mask = cached_even_bits<LSb0>(num_bits);
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
inline auto initial_state_mask(const VecZ &initial_state, size_t num_bits) -> Bitset {
    // Set straight into the result rather than through indices_to_bitset: the index vector that would
    // build costs an allocation and a second walk, for a mapping that is one multiply per mode. Same
    // MSb0 convention indices_to_bitset applies.
    Bitset mask(num_bits);
    for (const auto &mode : initial_state) {
        mask.set(num_bits - 1 - (2 * mode));
    }
    return mask;
}
} // namespace monoprop
