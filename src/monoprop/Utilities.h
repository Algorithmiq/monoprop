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
#include <type_traits>

#include "monoprop/Bitset.h"
#include "monoprop/monopropExport.h"

namespace monoprop {
struct BitOrdering {};

struct MSb0 : BitOrdering {};
struct LSb0 : BitOrdering {};

namespace detail {
template <size_t N>
constexpr auto make_repeating_bitset(uint64_t pattern) -> Bitset<N> {
    constexpr size_t kNumWords = Bitset<N>::num_words();
    Bitset<N> bits;
    for (size_t i = 0; i < kNumWords; ++i)
        bits.data()[i] = pattern;
    constexpr size_t kTopBits = N % 64;
    if constexpr (kTopBits != 0) {
        constexpr uint64_t kTopMask = (uint64_t(1) << kTopBits) - 1;
        bits.data()[kNumWords - 1] &= kTopMask;
    }
    return bits;
}

/// Repeating 0x5555...5555 pattern (even bits set) truncated to N bits.
template <size_t N>
constexpr auto even_bits() -> Bitset<N> {
    return make_repeating_bitset<N>(0x5555555555555555ULL);
}

/// Repeating 0xAAAA...AAAA pattern (odd bits set) truncated to N bits.
template <size_t N>
constexpr auto odd_bits() -> Bitset<N> {
    return make_repeating_bitset<N>(0xAAAAAAAAAAAAAAAAULL);
}
} // namespace detail

/// @brief Bitset with the even logical positions (0, 2, 4, …) set, truncated to N bits.
/// @tparam N        Bit width.
/// @tparam Ordering Bit-numbering convention. Under MSb0 (bit 0 = most significant) the logical even
///                  positions are the physically odd bits, so the underlying pattern is swapped
///                  relative to LSb0.
template <size_t N, typename Ordering>
constexpr auto even_bits() -> Bitset<N> {
    if constexpr (std::is_same_v<Ordering, MSb0>) { // MSb0
        return detail::odd_bits<N>();
    }
    else { // LSb0
        return detail::even_bits<N>();
    }
};
/// @brief Bitset with the odd logical positions (1, 3, 5, …) set, truncated to N bits.
/// @tparam N        Bit width.
/// @tparam Ordering Bit-numbering convention; see even_bits() for the MSb0/LSb0 swap.
template <size_t N, typename Ordering>
constexpr auto odd_bits() -> Bitset<N> {
    if constexpr (std::is_same_v<Ordering, MSb0>) { // MSb0
        return detail::even_bits<N>();
    }
    else { // LSb0
        return detail::odd_bits<N>();
    }
};

/*!
 * @brief Compute n-choose-2
 * @param n
 * @return size_t
 */
inline auto n_choose_2(std::integral auto n) -> size_t {
    return static_cast<size_t>(n * (n - 1) / 2);
}

/// @brief Join a range's elements into a string, inserting `separator` between consecutive elements.
/// @param values    Range whose elements are formatted with std::format("{}", value).
/// @param separator Text placed between elements (not before the first or after the last).
/// @return The concatenated string; empty when `values` is empty.
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

/*!
 * @brief Get maximum resident set size in KiB.
 * @return double maximum RSS in KiB.
 */
monoprop_EXPORT auto get_memory_usage() -> double;
} // namespace monoprop
