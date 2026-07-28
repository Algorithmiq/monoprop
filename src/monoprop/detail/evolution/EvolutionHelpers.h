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

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"

namespace monoprop::detail {

inline constexpr size_t kMissingIndex = std::numeric_limits<size_t>::max();

inline auto empty_coeffs() -> const VecD & {
    static const VecD coeffs;
    return coeffs;
}

struct CutoffContext {
    bool check_atol = false;
    bool check_upper_atol = false;
    double atol_value = 0.0;
    double upper_atol_value = 0.0;
    double abs_sin_val = 1.0;
    bool use_coeff_checks = false;

    auto abs_coeff_for(size_t i, const VecD &coeffs) const -> double {
        return use_coeff_checks ? std::abs(i < coeffs.size() ? coeffs[i] : 0.0) : 0.0;
    }
    // upper_atol rescue predicate: a sine-partner term dropped by the structural cutoff is kept alive if
    // its magnitude (the partner's sine coefficient |sin(2θ)|·|c|, not the source) is >= upper_atol.
    auto is_above_upper(double abs_coeff) const -> bool {
        return check_upper_atol && (abs_sin_val * abs_coeff >= upper_atol_value);
    }
    auto is_below_sin(double abs_coeff) const -> bool { return check_atol && (abs_sin_val * abs_coeff <= atol_value); }
};

} // namespace monoprop::detail
