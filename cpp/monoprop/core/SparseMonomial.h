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

// The structural cutoffs over a monomial's (k, d) digest: k = popcount, d = modes carrying BOTH
// Majoranas. Lets CutoffEvaluator decide from integers the emit site already has, without cutoff_sums.

#include <cstddef>

namespace monoprop {

// xor_sum = k - 2d, popcount_sum = k, or_sum = k - d; a fully paired monomial is kept unconditionally.
[[nodiscard]] inline constexpr auto is_paired(size_t k, size_t d) noexcept -> bool {
    return k == 2 * d;
}
[[nodiscard]] inline constexpr auto length_keeps(size_t k, size_t d, size_t cutoff) noexcept -> bool {
    return k == 2 * d || k <= cutoff;
}
[[nodiscard]] inline constexpr auto support_keeps(size_t k, size_t d, size_t cutoff) noexcept -> bool {
    return k == 2 * d || k - d <= cutoff;
}

} // namespace monoprop
