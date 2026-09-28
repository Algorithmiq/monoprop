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

#include "monoprop/detail/parallel/ThreadBudget.h"

#include <omp.h>

#include <algorithm>
#include <charconv>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <stdexcept>
#include <string>

namespace monoprop::detail::parallel {
namespace {

constexpr const char *kVariable = "monoprop_NUM_THREADS";

[[noreturn]] auto reject(std::string_view text) -> void {
    throw std::invalid_argument(std::format("{}=\"{}\" is not a thread count: it must be a decimal integer from 1 "
                                            "to {} with no sign or spaces. Set it to the threads per process, or "
                                            "unset it to use the OpenMP runtime default.",
                                            kVariable,
                                            text,
                                            INT_MAX));
}

} // namespace

auto resolve_thread_budget(std::optional<std::string_view> configured, int runtime_default) -> Options {
    if (!configured.has_value()) {
        if (runtime_default < 1) {
            throw std::invalid_argument(std::format("{} is unset and the OpenMP runtime default thread count is {}, "
                                                    "not positive; set {} explicitly.",
                                                    kVariable,
                                                    runtime_default,
                                                    kVariable));
        }
        return {.threads = runtime_default};
    }
    const std::string_view text = *configured;
    // from_chars alone would accept a partial parse; requiring digits throughout also rejects signs,
    // whitespace and lists before any narrowing.
    if (text.empty() || !std::ranges::all_of(text, [](char c) { return c >= '0' && c <= '9'; })) {
        reject(text);
    }
    std::uint64_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (ec != std::errc{} || end != text.data() + text.size() || value < 1
        || value > static_cast<std::uint64_t>(INT_MAX)) {
        reject(text);
    }
    return {.threads = static_cast<int>(value)};
}

auto capture_thread_budget() -> Options {
    const char *configured = std::getenv(kVariable);
    return resolve_thread_budget(configured != nullptr ? std::optional<std::string_view>(configured) : std::nullopt,
                                 omp_get_max_threads());
}

} // namespace monoprop::detail::parallel
