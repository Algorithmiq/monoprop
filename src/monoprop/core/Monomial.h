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

/*!
 * @file core/Monomial.h
 * @brief The one basis-agnostic monomial container and its vocabulary.
 *
 * A monomial is ONE basis operator (a product of generators) stored as a fixed `Bitset<2*NumModes>`,
 * two bits per fermionic mode / qubit. The SAME container serves EITHER algebra (a Majorana product,
 * or its Jordan-Wigner Pauli-string image); the choice is a @ref Basis over the container (see
 * algebra/Algebra.h), never a distinct type. Collections: @ref MonomialList (no coefficients) and
 * @ref MonomialMap (monomial -> real coefficient). The evolved operator's own row storage is instead
 * the entropy-packed detail::OperatorIndex (see TypeAliases.h).
 */

namespace monoprop {

/*!
 * @brief One monomial: a single basis operator (product of generators), basis-agnostic.
 */
template <size_t NumModes>
using Monomial = Bitset<2 * NumModes>;

/*!
 * @brief A plain dense list of monomials (no coefficients): `std::vector<Monomial>`.
 *
 * For plain term lists (gradient pairs, basis-change vectors, commutator operands). NOT the evolved
 * operator's row storage -- that is the entropy-packed detail::OperatorIndex, reached via the
 * backend-agnostic row accessors (see TypeAliases.h).
 */
template <size_t NumModes>
using MonomialList = std::vector<Monomial<NumModes>>;

/*!
 * @brief Transparent hash for Monomial (is_transparent enables heterogeneous map lookup).
 */
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

/*!
 * @brief An operator as a weighted sum of monomials: monomial -> real coefficient.
 */
template <size_t NumModes>
using MonomialMap =
    boost::unordered_flat_map<Monomial<NumModes>, double, MonomialHash<NumModes>, MonomialEqual<NumModes>>;

template <size_t NumModes>
inline auto monomial_hash(const Monomial<NumModes> &maj) noexcept -> size_t {
    if constexpr (Monomial<NumModes>::num_words() == 1) {
        return static_cast<size_t>(SplitmixHash<Monomial<NumModes>>::mix(maj.word(0)));
    }
    else {
        return MonomialHash<NumModes>{}(maj);
    }
}

/// @brief Structural keep/drop predicate applied to a monomial after each gate.
template <size_t NumModes>
using CutoffFn = std::function<bool(const Monomial<NumModes> &)>;

/**
 * @brief Structural truncation criterion applied to monomials after each gate.
 *
 * Both criteria always keep a fully paired monomial (the terms that contribute to an expectation
 * value); they differ only in how they measure the remaining partially paired monomials.
 */
enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

/// @brief Operator basis: the algebra a monomial is read in -- Majorana (default) or Pauli (JW-image).
/// Selects an @c Algebra model (see algebra/Algebra.h).
enum class Basis : uint8_t { Majorana, Pauli };

} // namespace monoprop
