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
//   monoprop_COMMPLACE          bool, default OFF; one COMMPLACE line per rank on stderr → commplace
//   monoprop_PARTITIONS         int N | "auto" | "off"; parsed where it is used (resolve_partition_count_)
//
// monoprop_PARTITION_PINNING is deleted and pinning is unconditional. Its parser matched only the
// first CHARACTER, so `off`, `OFF` and `disabled` all parsed as ON; parse_env_flag compares whole
// words instead, which is the one thing that bug was about.

namespace monoprop::config {

namespace detail {

// Case-insensitive whole-string compare; `lower` must already be lowercase.
inline auto iequals(const char *value, const char *lower) -> bool {
    for (; *value != '\0' && *lower != '\0'; ++value, ++lower) {
        const char c = (*value >= 'A' && *value <= 'Z') ? static_cast<char>(*value - 'A' + 'a') : *value;
        if (c != *lower) {
            return false;
        }
    }
    return *value == *lower;
}

// Off when unset, empty, or a WHOLE-WORD match on 0/false/no/off; on otherwise. Whole words because
// first-character matching is what read `off` as ON.
inline auto parse_env_flag(const char *value) -> bool {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(iequals(value, "0") || iequals(value, "false") || iequals(value, "no") || iequals(value, "off"));
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
    bool commplace = false;
};

// Parse the environment once; the Settings are cached and shared across TUs.
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        s.commplace = detail::parse_env_flag(std::getenv("monoprop_COMMPLACE"));
        return s;
    }();
    return settings;
}

} // namespace monoprop::config
