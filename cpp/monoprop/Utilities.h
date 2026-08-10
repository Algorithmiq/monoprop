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
// n is an ordinary runtime argument -- the return type is the runtime-width Bitset (see the
// NumModes-NTTP-removal plan's Stage 2b) -- so this needs no template parameter of its own.
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

// Under MSb0 the logical even positions are physically odd, so the pattern is swapped vs LSb0.
//
// Two overloads: most call sites have a compile-time N (even_bits<Mono::size(), LSb0>(),
// even_bits<2 * NumModes, LSb0>()) and keep spelling it that way unchanged. Some -- anything that
// only has a Bitset in hand, e.g. an operator's result (a ^ b), which is not Monomial<NumModes>-typed
// and so cannot recover a compile-time N via ::size() -- have only a *runtime* width, so N moves from
// the template-argument list to an ordinary function argument instead: even_bits<LSb0>(some_bitset.size()).
template <size_t N, typename Ordering>
auto even_bits() -> Bitset {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::odd_bits(N);
    }
    else {
        return detail::even_bits(N);
    }
};
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
template <size_t N, typename Ordering>
auto odd_bits() -> Bitset {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::even_bits(N);
    }
    else {
        return detail::odd_bits(N);
    }
};
template <typename Ordering>
auto odd_bits(size_t n) -> Bitset {
    if constexpr (std::is_same_v<Ordering, MSb0>) {
        return detail::even_bits(n);
    }
    else {
        return detail::odd_bits(n);
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
