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
#include <string_view>

#include "monoprop/detail/parallel/Options.h"
#include "monoprop/monopropExport.h"

namespace monoprop::detail::parallel {

/*!
 * \brief Resolve a per-object thread budget from a configured value and the OpenMP runtime default.
 *
 * Pure: no environment read, hardware query or runtime mutation. The value is never clamped to the
 * hardware or the allocation.
 *
 * \param configured      The `monoprop_NUM_THREADS` text, or `std::nullopt` when the variable is unset. A
 *                        present value must be nonempty ASCII decimal digits whose value lies in
 *                        `[1, INT_MAX]`; leading zeros are accepted. A valid value wins over any fallback.
 * \param runtime_default The OpenMP default (`omp_get_max_threads()`), used only when `configured` is
 *                        absent; it must then be positive.
 * \return The resolved budget.
 * \throws std::invalid_argument for an empty, signed, padded, list-valued, non-digit, zero or overflowing
 *         configured value, or for an absent value with a non-positive runtime default.
 */
monoprop_EXPORT auto resolve_thread_budget(std::optional<std::string_view> configured, int runtime_default) -> Options;

/*!
 * \brief Read `monoprop_NUM_THREADS` once and resolve it against `omp_get_max_threads()`.
 *
 * Called while constructing a propagator, never during an operation; each call rereads the
 * environment and nothing is cached. An unset variable selects the OpenMP runtime default, while a
 * present but empty one is an error.
 *
 * \return The resolved budget.
 * \throws std::invalid_argument as resolve_thread_budget() does.
 */
monoprop_EXPORT auto capture_thread_budget() -> Options;

} // namespace monoprop::detail::parallel
