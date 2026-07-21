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
 * ESSENCE. monoprop is a backbone for propagating an operator expanded as a *sum of monomials*
 * through a circuit in the Heisenberg picture. A **monomial** is ONE basis operator -- a product
 * of generators -- stored as a fixed bitset `Bitset<2*NumModes>`: two bits per fermionic mode /
 * qubit. The SAME container represents a term in EITHER algebra; the Majorana/Pauli choice is an
 * *algebra over this container* (a @ref Basis, see algebra/Algebra.h), never a different type:
 *   - Majorana basis: each set bit is a Majorana operator gamma_k present in the product;
 *   - Pauli basis:    the Jordan-Wigner image of the product -- a Pauli string
 *                     (encoding spelled out in algebra/PauliAlgebra.h).
 *
 * Collections of monomials:
 *   - @ref MonomialList : a plain ordered list of monomials, no coefficients;
 *   - @ref MonomialMap  : monomial -> real coefficient, i.e. an operator as a weighted sum.
 * (The evolved operator's own row storage is the entropy-packed detail::OperatorIndex, reached
 * through the backend-agnostic row accessors declared alongside it; see TypeAliases.h.)
 */

namespace monoprop {

/*!
 * @brief One monomial: a single basis operator (product of generators), basis-agnostic.
 * @tparam NumModes Number of fermionic modes / qubits; the container holds 2*NumModes bits
 *         (two per mode). Read as a Majorana product, or as a Pauli string under the JW image.
 */
template <size_t NumModes>
using Monomial = Bitset<2 * NumModes>;

/*!
 * @brief A plain dense list of monomials (no coefficients): `std::vector<Monomial>`.
 * @tparam NumModes Number of fermionic modes / qubits.
 *
 * Used for plain term lists (gradient ham/state pairs, basis-change vectors, commutator
 * pipeline operands). The evolved operator's row storage is NOT this alias -- it is the
 * entropy-packed detail::OperatorIndex (position-list rows plus hash index). Functions that
 * must accept both go through the backend-agnostic row accessors (see TypeAliases.h) and
 * template their rows parameter.
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
 * @tparam NumModes Number of fermionic modes / qubits.
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
 * Both criteria share one rule: a *fully paired* monomial -- one whose support
 * consists entirely of complete pairs m_{2j-1} m_{2j} on a mode -- is always kept,
 * regardless of the cutoff. Fully paired monomials are exactly the terms that can
 * contribute to an expectation value against a computational-basis state or Slater
 * determinant, so discarding them would throw away signal. The criteria differ only
 * in how they measure the remaining, partially paired monomials.
 */
enum class CutoffType {
    Length, // Keep if the monomial length (number of Majorana operators) <= cutoff (or fully paired)
    Support // Keep if the orbital support (number of distinct orbitals) <= cutoff (or fully paired)
};

/// @brief Operator basis: the algebra a monomial is read in -- Majorana monomials (default) or
/// Pauli strings (native JW-image encoding). Selects an @c Algebra model (see algebra/Algebra.h).
enum class Basis : uint8_t { Majorana, Pauli };

} // namespace monoprop
