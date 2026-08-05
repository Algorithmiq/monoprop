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

#include <optional>
#include <stdexcept>

#include "monoprop/Evolution.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"

namespace monoprop::detail {

// A CutoffType enumerator neither cutoff factory knows.
class UnknownCutoffType : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <size_t NumModes>
auto cutoff_function(CutoffType cutoff_type, unsigned int cutoff, size_t logical_num_modes = NumModes)
    -> CutoffFn<NumModes> {
    switch (cutoff_type) {
        case CutoffType::Length:
            return detail::LengthCutoff<NumModes>{cutoff, logical_num_modes};
        case CutoffType::Support:
            return detail::SupportCutoff<NumModes>{cutoff, logical_num_modes};
        default:
            throw UnknownCutoffType("Unknown cutoff type");
    }
}

template <size_t NumModes>
auto cutoff_function_basis_change(CutoffType cutoff_type,
                                  unsigned int cutoff,
                                  const MonomialList<NumModes> &basis,
                                  size_t logical_num_modes = NumModes) -> CutoffFn<NumModes> {
    switch (cutoff_type) {
        case CutoffType::Length:
            return [cutoff, logical_num_modes, basis_copy = basis](const Monomial<NumModes> &mono) {
                const auto mapped_mono = change_basis<NumModes>(mono, basis_copy);
                return length_cutoff<NumModes>(mapped_mono, cutoff, logical_num_modes);
            };
        case CutoffType::Support:
            return [cutoff, logical_num_modes, basis_copy = basis](const Monomial<NumModes> &mono) {
                const auto mapped_mono = change_basis<NumModes>(mono, basis_copy);
                return support_cutoff<NumModes>(mapped_mono, cutoff, logical_num_modes);
            };
        default:
            throw UnknownCutoffType("Unknown cutoff type");
    }
}

} // namespace monoprop::detail
