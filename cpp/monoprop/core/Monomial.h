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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include "monoprop/Bitset.h"

namespace monoprop {

// A monomial is a Bitset of width 2 * num_modes, two bits per mode/qubit -- the width is data, so
// there is no monomial *type* to name and no header-level alias for one: spell Bitset, and carry the
// width with the value. Sites that need a NumModes deduce it from a monomial argument's own width
// (mono.size() / 2), never from a template parameter.

// Structural stand-in for "a monomial, width unspecified", for the free functions in the algebra
// headers and elsewhere that read a monomial parameter generically. Instance (not qualified) calls,
// since a width only exists per value.
template <typename T>
concept MonomialLike = requires(const T &t) {
    { t.size() } -> std::convertible_to<size_t>;
    { t.num_words() } -> std::convertible_to<size_t>;
    { t.count() } -> std::convertible_to<size_t>;
    { t.find_first() } -> std::convertible_to<size_t>;
};

// Not the evolved operator's row storage -- that is detail::OperatorIndex (see TypeAliases.h).
//
// Element-width caveat, from Bitset being runtime-width: a sized construction `MonomialList l(n)`
// fills with *width-0* bitsets, since nothing in the element type carries a width. Any site
// that sizes up front and assigns into slots afterwards must pass a fill value of the intended
// width, `MonomialList l(n, Bitset(num_bits))`; push_back-only sites need nothing. A width-0 element
// reaching a binary op trips the width assertions in Bitset.h.
using MonomialList = std::vector<Bitset>;

struct MonomialHash final {
    using is_transparent = void;

    auto operator()(const Bitset &arr) const noexcept -> size_t { return SplitmixHash{}(arr); }
};

struct MonomialEqual final {
    using is_transparent = void;

    auto operator()(const Bitset &lhs, const Bitset &rhs) const noexcept -> bool { return lhs == rhs; }
};

using MonomialMap = boost::unordered_flat_map<Bitset, double, MonomialHash, MonomialEqual>;

// MPI owner routing (find_rank) hashes through here, so the value must not change: it decides which
// rank owns a term and, with it, probe order. It does not. The single-word fast path this used to
// select with `if constexpr` is the same one SplitmixHash::operator() now takes at runtime, and the
// wide arm was already a call to SplitmixHash via MonomialHash -- so both arms collapse into the one
// call below. Kept as a named function rather than inlined at the call sites: the name is what marks a
// hash as owner-routing (pinned) rather than an ordinary container hash.
inline auto monomial_hash(const Bitset &mono) noexcept -> size_t {
    return SplitmixHash{}(mono);
}

// Structural keep/drop predicate applied to a monomial after each gate.
using CutoffFn = std::function<bool(const Bitset &)>;

enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

enum class Basis : uint8_t { Majorana, Pauli };

} // namespace monoprop
