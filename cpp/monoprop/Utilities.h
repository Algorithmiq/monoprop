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

// Under MSb0 the logical even positions are physically odd, so the pattern is swapped vs LSb0.
//
// Two overloads, because a caller's width may be either. Call sites with a compile-time N spell it as
// a template argument (even_bits<Mono::size(), LSb0>()); anything holding only a Bitset -- an
// operator's result (a ^ b), say, whose width is data and not recoverable via a static ::size() --
// passes it as an ordinary argument instead: even_bits<LSb0>(some_bitset.size()).
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
    thread_local size_t cached_n = static_cast<size_t>(-1);
    if (cached_n != n) [[unlikely]] {
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
} // namespace monoprop
