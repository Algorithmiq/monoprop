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

// Bitset (Stage 2b of the NumModes-NTTP-removal plan) is now runtime-width, so Monomial<NumModes>
// can no longer be a bare alias -- there would be nothing left to tell a default-constructed
// `Monomial<NumModes> m;` its width. It is instead a thin, empty (no extra data members) subclass
// that supplies 2*NumModes to Bitset's runtime constructor and shadows size()/num_words() with
// compile-time versions, so every existing call site that spells Monomial<NumModes> -- default
// construction, and array-sizing/template-argument uses like
// `std::array<size_t, Monomial<NumModes>::size()>` -- keeps compiling unchanged. This wrapper is a
// transitional shim: once Stages 2c/2d/2f de-template the classes that still need a compile-time
// NumModes, it can be deleted and callers can use Bitset directly, per the plan's original
// "Monomial stops being an alias" wording.
//
// An operator (^, &, fused_xor's result, ...) still returns plain Bitset by value -- rewrapping
// would mean redefining every operator on the subclass instead of inheriting them. This is safe:
// nothing that recovers NumModes from a value (`decltype(x)::size()`, always a *qualified* call)
// does so on an operator's result, only on a parameter or an explicitly Monomial<NumModes>-typed
// local (see Stage 2a's conversions) -- so the type used there is always the wrapper. Assigning an
// operator's plain-Bitset result back into a Monomial<NumModes>-typed variable is unaffected: that
// variable's static type, and hence which size()/num_words() it resolves to, never changes.
template <size_t NumModes>
class Monomial : public Bitset {
public:
    static constexpr size_t kNumBits = 2 * NumModes;

    Monomial() noexcept : Bitset(kNumBits) {}
    // Implicit, matching the old Bitset<NumBits>(uint64_t) this replaces: unlike Bitset's own
    // (width, value) constructor, NumModes already supplies the width, so only the value is missing
    // here -- test code (and BOOST_TEST comparisons like `mono == 0b1010`) relies on this converting.
    explicit(false) Monomial(uint64_t val) noexcept : Bitset(kNumBits, val) {}
    // Reinterprets a same-width Bitset (e.g. an operator's result) as a Monomial<NumModes>; the
    // caller is responsible for the width actually matching -- this does not resize.
    explicit(false) Monomial(const Bitset &b) noexcept : Bitset(b) {}
    auto operator=(const Bitset &b) noexcept -> Monomial & {
        Bitset::operator=(b);
        return *this;
    }

    [[nodiscard]] static constexpr auto size() noexcept -> size_t { return kNumBits; }
    [[nodiscard]] static constexpr auto num_words() noexcept -> size_t { return (kNumBits + 63) / 64; }

    // `mono == 0b1010`-style comparisons (BOOST_TEST literals) relied on int -> Bitset<NumBits> being
    // a single implicit user-defined conversion. int -> Monomial<NumModes> still is one, but the
    // inherited Bitset::operator==(const Bitset&) needs a *second* (derived-to-base) step on top of
    // it, and overload resolution does not chain a derived class's converting constructor to satisfy
    // a base-class parameter -- so this needs its own overload, not just the constructor above.
    [[nodiscard]] friend auto operator==(const Monomial &lhs, uint64_t rhs) noexcept -> bool {
        return static_cast<const Bitset &>(lhs) == Monomial(rhs);
    }
    [[nodiscard]] friend auto operator==(uint64_t lhs, const Monomial &rhs) noexcept -> bool { return rhs == lhs; }
};

// Structural stand-in for "some Monomial<N>/Bitset, width unspecified": the free functions in
// algebra/*.h and elsewhere that only ever use a monomial parameter's width (never a caller-chosen
// NumModes unrelated to any parameter) deduce it from the argument instead of naming it as a
// template parameter -- `decltype(mono)::size()` recovers the storage bit width (2 * NumModes for a
// Monomial), so a NumModes value where one is still needed is `decltype(mono)::size() / 2`. Instance
// (not qualified) calls here so a plain Bitset satisfies this too, not just the Monomial<NumModes>
// wrapper. See the NumModes-NTTP-removal plan, Stage 2a/2b.
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
// fills with *width-0* bitsets, where `std::vector<Monomial<NumModes>>(n)` self-widened. Any site
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

template <size_t NumModes>
inline auto monomial_hash(const Monomial<NumModes> &mono) noexcept -> size_t {
    if constexpr (Monomial<NumModes>::num_words() == 1) {
        return static_cast<size_t>(SplitmixHash::mix(mono.word(0)));
    }
    else {
        return MonomialHash{}(mono);
    }
}

// Structural keep/drop predicate applied to a monomial after each gate.
using CutoffFn = std::function<bool(const Bitset &)>;

enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

enum class Basis : uint8_t { Majorana, Pauli };

} // namespace monoprop
