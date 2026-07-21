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

#include <algorithm>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"

/*!
 * @file AlgebraReference.h
 * @brief Test-only reference helpers for the Majorana algebra.
 *
 * These are exercised only by tests/cpp/mpfunctions.cpp and are not called by the shipped
 * library, so they live here rather than in the shipped algebra headers. They provide
 * straightforward reference forms the production kernels are checked against.
 */

namespace monoprop {

/*!
 * @brief Converts a fermionic operator from index representation to binary (bitset) representation.
 */
template <size_t NumModes>
auto fermionic_to_binary_operator(const std::vector<VecZ> &op) -> MonomialList<NumModes> {
    auto majorana_operator = MonomialList<NumModes>(op.size());
    std::transform(op.cbegin(), op.cend(), majorana_operator.begin(), indices_to_bitset<NumModes>);
    return majorana_operator;
}

/*!
 * @brief Reference multiplicative phase factor for Majorana operator evolution:
 *        the ordering (interleave) sign times the Hermitian phase.
 */
template <size_t NumModes>
auto get_multiplicative_phase(const Monomial<NumModes> &maj,
                              const Monomial<NumModes> &gen_maj,
                              size_t maj_count,
                              size_t gen_count,
                              size_t overlap) -> int {
    return interleave_phase<NumModes>(maj, gen_maj) * hermitian_phase(maj_count, gen_count, overlap);
}

} // namespace monoprop
