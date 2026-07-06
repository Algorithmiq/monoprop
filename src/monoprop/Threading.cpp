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
#include <memory>
#include <mutex>
#include <optional>

#include <tbb/global_control.h>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::threading {
namespace {

auto get_env_threads() -> std::optional<int> {
    return config::get().num_threads;
}

std::mutex g_mutex;
std::unique_ptr<tbb::global_control> g_tbb_control;
std::once_flag g_init_once;

// Set oneTBB's maximum parallelism for the current process. File-local: the sole entry point is
// init_from_env reading monoprop_NUM_THREADS. Thread-safe; the last call wins; threads <= 0 ignored.
auto set_num_threads(int threads) -> void {
    if (threads <= 0) {
        return;
    }
    std::lock_guard lock(g_mutex);
    g_tbb_control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism,
                                                          static_cast<std::size_t>(threads));
}

} // namespace

auto init_from_env() -> void {
    std::call_once(g_init_once, []() {
        const auto threads = get_env_threads();
        if (threads.has_value()) {
            set_num_threads(*threads);
        }
    });
}

auto range_grain_size(size_t count, size_t min_grain) -> size_t {
    const size_t workers = effective_parallelism();
    const size_t scaled = count / std::max<size_t>(1, workers * 4);
    return std::max(min_grain, scaled);
}

} // namespace monoprop::threading
