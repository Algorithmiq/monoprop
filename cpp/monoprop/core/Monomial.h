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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// Structural keep/drop predicate applied to a monomial after each gate.
template <size_t NumModes>
using CutoffFn = std::function<bool(const Monomial<NumModes> &)>;

enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

enum class Basis : uint8_t { Majorana, Pauli };

} // namespace monoprop
