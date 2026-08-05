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
    if constexpr (constexpr size_t kTopBits = N % 64; kTopBits != 0) {
        constexpr uint64_t kTopMask = (uint64_t(1) << kTopBits) - 1;
        bits.data()[kNumWords - 1] &= kTopMask;
    }
    return bits;
}

template <size_t N>
constexpr auto even_bits() -> Bitset<N> {
    return make_repeating_bitset<N>(0x5555555555555555ULL);
}

template <size_t N>
constexpr auto odd_bits() -> Bitset<N> {
    return make_repeating_bitset<N>(0xAAAAAAAAAAAAAAAAULL);
}
} // namespace detail

// Under MSb0 the logical even positions are physically odd, so the pattern is swapped vs LSb0.
template <size_t N, typename Ordering>
constexpr auto even_bits() -> Bitset<N> {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::odd_bits<N>();
    }
    else {
        return detail::even_bits<N>();
    }
};
// Same MSb0/LSb0 swap as even_bits().
template <size_t N, typename Ordering>
constexpr auto odd_bits() -> Bitset<N> {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::even_bits<N>();
    }
    else {
        return detail::odd_bits<N>();
    }
};

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
} // namespace monoprop
