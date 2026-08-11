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

// Majorana helpers the shipped library no longer calls, kept alive for tests/cpp/mpfunctions.cpp.

#include <algorithm>
#include <iterator>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"

namespace monoprop {

template <size_t NumModes>
auto fermionic_to_binary_operator(const std::vector<VecZ> &op) -> MonomialList {
    MonomialList majorana_operator;
    majorana_operator.reserve(op.size());
    // push_back, not a sized construction plus transform: sizing up front would fill with width-0
    // bitsets, and every slot is written here anyway.
    std::ranges::transform(op, std::back_inserter(majorana_operator), indices_to_bitset<NumModes>);
    return majorana_operator;
}

template <size_t NumModes>
auto get_multiplicative_phase(const Monomial<NumModes> &mono,
                              const Monomial<NumModes> &gen_mono,
                              size_t mono_count,
                              size_t gen_count,
                              size_t overlap) -> int {
    return interleave_phase(mono, gen_mono) * hermitian_phase(mono_count, gen_count, overlap);
}

} // namespace monoprop
