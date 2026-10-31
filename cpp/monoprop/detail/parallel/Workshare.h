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

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

#include "monoprop/detail/parallel/Options.h"

namespace monoprop::detail::parallel {

/*!
 * \brief Invoke `body(block)` once for every logical block in `[0, count)`, then return.
 *
 * Runs serially, in ascending block order, when the budget or `count` is one, or when called from inside
 * any OpenMP region (active or not). Otherwise opens one region requesting `min(options.threads, count)`
 * workers with a static schedule; the runtime may provide fewer, and every block still runs exactly once.
 *
 * Worker exceptions never leave the region. Each worker keeps its first exception and stops taking
 * blocks; after all workers join, the exception of the lowest-numbered failing worker is rethrown on
 * the caller. The serial path lets an exception propagate from the failing block directly. After a
 * failure, which other blocks ran is unspecified, and nothing is rolled back.
 *
 * The helper owns no threads and changes no OpenMP runtime setting.
 *
 * \param count   Number of logical blocks; must not exceed `PTRDIFF_MAX`.
 * \param options Thread budget; `options.threads` must be positive.
 * \param body    Callable taking a `std::size_t` block ID. Distinct blocks may run concurrently, so it
 *                must only write state owned by its block and must not call MPI.
 * \throws std::invalid_argument if `options.threads` is not positive, before any block runs.
 * \throws std::length_error if `count` exceeds `PTRDIFF_MAX`, before any block runs.
 */
template <class Fn>
auto for_blocks(std::size_t count, Options options, Fn &&body) -> void {
    if (options.threads < 1) {
        throw std::invalid_argument(
            std::format("parallel::for_blocks: the thread budget must be positive, got {}", options.threads));
    }
    // The region iterates a signed index, as the most portable OpenMP loop form requires.
    constexpr auto max_count = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (count > max_count) {
        throw std::length_error(
            std::format("parallel::for_blocks: {} blocks exceed the loop bound {}", count, max_count));
    }

    const auto team = std::min(static_cast<std::size_t>(options.threads), count);
    if (team <= 1 || omp_get_level() > 0) {
        for (std::size_t block = 0; block < count; ++block) {
            body(block);
        }
        return;
    }

    // Allocated before the region, so allocation failure propagates normally. The actual team never
    // exceeds the num_threads request, so every worker ID indexes a slot.
    std::vector<std::exception_ptr> errors(team);
    const auto blocks = static_cast<std::ptrdiff_t>(count);

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(team))
    for (std::ptrdiff_t block = 0; block < blocks; ++block) {
        auto &error = errors[static_cast<std::size_t>(omp_get_thread_num())];
        if (error) {
            continue;
        }
        try {
            body(static_cast<std::size_t>(block));
        }
        catch (...) {
            error = std::current_exception();
        }
    }

    for (const auto &error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

} // namespace monoprop::detail::parallel
