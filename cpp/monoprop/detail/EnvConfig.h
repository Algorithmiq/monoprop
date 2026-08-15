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

#include <cstddef>
#include <cstdlib>
#include <optional>

// Single home for runtime environment configuration. Kept dependency-free by design, because it is
// pulled into hot-path headers.
//
//   monoprop_NUM_THREADS    positive int (1..1e6), else ignored                → num_threads
//   monoprop_PARTITION_PINNING  bool, default ON; 0/false disables per-core pinning → partition_pinning
//   monoprop_PARTITIONS         int N | "auto" | "off"; parsed where it is used (resolve_partition_count_)
//   monoprop_COMM_PROFILE       bool, default OFF; per-partition collective accounting to stderr → comm_profile
//   monoprop_SPIN_BUDGET_US     positive int us, default kDefaultSpinBudgetUs; barrier on-core spin before
//                               yielding. Exists to be swept against a real workload → spin_budget_us
//   monoprop_BARRIER_GROUPING   bool, default ON; 0/false forces the flat barrier while LEAVING PINNING ON
//                               → barrier_grouping. Without this the two-level barrier cannot be measured:
//                               its domains come from the cpusets, so turning pinning off to get a flat
//                               barrier also unpins, and every before/after confounds the two. Exists so
//                               "grouped vs flat, both pinned" is a run rather than an argument.

namespace monoprop::config {

namespace detail {

inline auto parse_flag(const char *value, bool default_value) -> bool {
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    const char c = value[0];
    return !(c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N');
}

inline auto parse_positive_int(const char *text) -> std::optional<int> {
    if (text == nullptr) {
        return std::nullopt;
    }
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return std::nullopt;
    }
    if (value <= 0 || value > 1'000'000) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

} // namespace detail

struct Settings {
    std::optional<int> num_threads;
    bool partition_pinning = true;
    bool comm_profile = false;
    std::optional<int> spin_budget_us; // nullopt ⇒ the barrier's own default
    bool barrier_grouping = true;
};

// Parse the environment once; the Settings are cached and shared across TUs.
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        s.partition_pinning = detail::parse_flag(std::getenv("monoprop_PARTITION_PINNING"), true);
        s.comm_profile = detail::parse_flag(std::getenv("monoprop_COMM_PROFILE"), false);
        s.spin_budget_us = detail::parse_positive_int(std::getenv("monoprop_SPIN_BUDGET_US"));
        s.barrier_grouping = detail::parse_flag(std::getenv("monoprop_BARRIER_GROUPING"), true);
        return s;
    }();
    return settings;
}

} // namespace monoprop::config
