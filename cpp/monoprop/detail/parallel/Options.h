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

// Deliberately free of <omp.h>: signatures can carry Options without exposing OpenMP.

namespace monoprop::detail::parallel {

/*!
 * \brief Thread budget for one worksharing region.
 *
 * The budget is an upper bound on the workers a region requests, never a global OpenMP runtime setting.
 * The runtime may provide fewer workers; correctness never depends on receiving the full budget.
 */
struct Options {
    int threads = 1; //!< Validated positive per-object budget; not a global runtime setting.
};

} // namespace monoprop::detail::parallel
