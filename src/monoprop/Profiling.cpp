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

// Single definition of the RegionProfiler's process-wide mutable state (see RegionProfiler.h). Kept
// in one TU, compiled into libmonoprop.so and exported, so the core and the nanobind extension share
// one copy of the enable flag, the accumulators, the per-thread current region, and the atexit dump.

#include "monoprop/detail/profiling/RegionProfiler.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::profiling {
namespace {

auto env_enabled() -> bool {
    return config::get().phase_timers;
}

std::array<RegionAcc, kRegionCount> g_accs{};

auto rank_from_env() -> int {
    for (const char *k : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"}) {
        if (const char *v = std::getenv(k)) {
            return std::atoi(v);
        }
    }
    return 0;
}

// One-shot per-process stderr dump (registered via profiling_ensure_atexit on first region entry).
auto dump() -> void {
    const int rank = rank_from_env();
    for (int i = 0; i < kRegionCount; ++i) {
        const auto &a = g_accs[static_cast<size_t>(i)];
        const auto calls = a.calls.load(std::memory_order_relaxed);
        const auto wall = a.wall_ns.load(std::memory_order_relaxed);
        const auto busy = a.busy_ns.load(std::memory_order_relaxed);
        const auto tasks = a.tasks.load(std::memory_order_relaxed);
        if (calls == 0 && busy == 0) {
            continue;
        }
        const auto name = kRegionNames[static_cast<size_t>(i)];
        std::fprintf(stderr,
                     "monoprop_PHASE rank=%d region=%.*s wall_ms=%.3f busy_ms=%.3f calls=%llu tasks=%llu\n",
                     rank,
                     static_cast<int>(name.size()),
                     name.data(),
                     static_cast<double>(wall) / 1.0e6,
                     static_cast<double>(busy) / 1.0e6,
                     static_cast<unsigned long long>(calls),
                     static_cast<unsigned long long>(tasks));
    }
    std::fflush(stderr);
}

} // namespace

bool g_profiling_enabled = env_enabled();

auto profiling_accs() -> RegionAcc * { return g_accs.data(); }

auto profiling_current() -> Region & {
    static thread_local Region current = Region::Other;
    return current;
}

auto profiling_ensure_atexit() -> void {
    static std::atomic<bool> registered{false};
    bool expected = false;
    if (registered.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        std::atexit(dump);
    }
}

} // namespace monoprop::profiling
