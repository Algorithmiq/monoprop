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
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include "monoprop/Bitset.h"

namespace monoprop {

template <size_t NumModes>
using Monomial = Bitset<2 * NumModes>;

// Not the evolved operator's row storage -- that is detail::OperatorIndex (see detail/operator/OperatorIndex.h).
template <size_t NumModes>
using MonomialList = std::vector<Monomial<NumModes>>;

template <size_t NumModes>
struct MonomialHash final {
    using is_transparent = void;

    auto operator()(const Monomial<NumModes> &arr) const noexcept -> size_t {
        return SplitmixHash<Monomial<NumModes>>{}(arr);
    }
};

template <size_t NumModes>
struct MonomialEqual final {
    using is_transparent = void;

    auto operator()(const Monomial<NumModes> &lhs, const Monomial<NumModes> &rhs) const noexcept -> bool {
        return lhs == rhs;
    }
};

template <size_t NumModes>
using MonomialMap =
    boost::unordered_flat_map<Monomial<NumModes>, double, MonomialHash<NumModes>, MonomialEqual<NumModes>>;

template <size_t NumModes>
inline auto monomial_hash(const Monomial<NumModes> &mono) noexcept -> size_t {
    if constexpr (Monomial<NumModes>::num_words() == 1) {
        return static_cast<size_t>(SplitmixHash<Monomial<NumModes>>::mix(mono.word(0)));
    }
    else {
        return MonomialHash<NumModes>{}(mono);
    }
}

namespace detail {
// Hash term of an all-zero word i.
template <size_t NumModes>
inline constexpr auto kZeroWordMix = [] {
    std::array<uint64_t, Monomial<NumModes>::num_words()> mixes{};
    for (uint64_t i = 0; auto &mix : mixes) {
        mix = SplitmixHash<Monomial<NumModes>>::mix(i++);
    }
    return mixes;
}();

// Hash of the all-zero monomial (W > 1).
template <size_t NumModes>
inline constexpr uint64_t kZeroWordFold = [] {
    uint64_t h = 0;
    for (const uint64_t m : kZeroWordMix<NumModes>) {
        h ^= m;
    }
    return h;
}();

// Up to this many words, monomial_hash_positions scatters densely rather than walking occupied words.
inline constexpr size_t kDenseHashMaxWords = 4;
} // namespace detail

// monomial_hash, bit for bit, from ascending positions below 2 * NumModes.
template <size_t NumModes, typename PosT>
[[gnu::always_inline]] inline auto monomial_hash_positions(std::span<const PosT> pos) noexcept -> size_t {
    using Hash = SplitmixHash<Monomial<NumModes>>;
    if constexpr (Monomial<NumModes>::num_words() == 1) {
        uint64_t word = 0;
        for (const PosT p : pos) {
            word |= uint64_t{1} << static_cast<size_t>(p);
        }
        return static_cast<size_t>(Hash::mix(word));
    }
    else if constexpr (Monomial<NumModes>::num_words() <= detail::kDenseHashMaxWords) {
        std::array<uint64_t, Monomial<NumModes>::num_words()> words{};
        for (const PosT p : pos) {
            words[static_cast<size_t>(p) >> 6] |= uint64_t{1} << (static_cast<size_t>(p) & 63);
        }
        uint64_t h = 0;
        for (size_t i = 0; i < words.size(); ++i) {
            h ^= Hash::mix(words[i] + static_cast<uint64_t>(i));
        }
        return static_cast<size_t>(h);
    }
    else {
        uint64_t h = detail::kZeroWordFold<NumModes>;
        const size_t n = pos.size();
        size_t j = 0;
        while (j < n) {
            const size_t w = static_cast<size_t>(pos[j]) >> 6;
            uint64_t word = 0;
            for (; j < n && (static_cast<size_t>(pos[j]) >> 6) == w; ++j) {
                word |= uint64_t{1} << (static_cast<size_t>(pos[j]) & 63);
            }
            h ^= Hash::mix(word + static_cast<uint64_t>(w)) ^ detail::kZeroWordMix<NumModes>[w];
        }
        return static_cast<size_t>(h);
    }
}

// Structural keep/drop predicate applied to a monomial after each gate.
template <size_t NumModes>
using CutoffFn = std::function<bool(const Monomial<NumModes> &)>;

enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

enum class Basis : uint8_t { Majorana, Pauli };

} // namespace monoprop
