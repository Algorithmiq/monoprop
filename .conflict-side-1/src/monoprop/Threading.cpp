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

#include "monoprop/Threading.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>

#include <tbb/global_control.h>

namespace monoprop::threading {
namespace {

auto parse_positive_int(const char* text) -> std::optional<int> {
    if (text == nullptr) {
        return std::nullopt;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return std::nullopt;
    }
    if (value <= 0 || value > 1'000'000) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

auto get_env_threads() -> std::optional<int> {
    if (const char* v = std::getenv("monoprop_NUM_THREADS")) {
        if (auto parsed = parse_positive_int(v)) {
            return parsed;
        }
    }
    return std::nullopt;
}

std::mutex g_mutex;
std::unique_ptr<tbb::global_control> g_tbb_control;
std::optional<int> g_configured_threads;
std::once_flag g_init_once;

} // namespace

auto init_from_env() -> void {
    std::call_once(g_init_once, []() {
        const auto threads = get_env_threads();
        if (threads.has_value()) {
            set_num_threads(*threads);
        }
    });
}

auto set_num_threads(int threads) -> void {
    if (threads <= 0) {
        return;
    }
    std::lock_guard lock(g_mutex);
    g_tbb_control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism,
                                                          static_cast<std::size_t>(threads));
    g_configured_threads = threads;
}

auto configured_num_threads() -> std::optional<int> {
    std::lock_guard lock(g_mutex);
    return g_configured_threads;
}

auto range_grain_size(size_t count, size_t min_grain) -> size_t {
    const size_t workers = effective_parallelism();
    const size_t scaled = count / std::max<size_t>(1, workers * 4);
    return std::max(min_grain, scaled);
}

} // namespace monoprop::threading
