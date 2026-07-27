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

// Single home for runtime environment configuration: every `monoprop_*` env var is parsed once here
// (function-local static in config::get()) and exposed as a config::Settings field. Dependency-free by
// design (only <cstdlib>) because it is pulled into hot-path headers.
//
//   monoprop_NUM_THREADS    positive int (1..1e6), else ignored                → num_threads
//   monoprop_SHARD_PINNING  bool, default ON; 0/false disables per-core pinning → shard_pinning
//   monoprop_SHARDS         int N | "auto" | "off"; parsed where it is used (resolve_shard_count_)

namespace monoprop::config {

namespace detail {

// Truthy parse: unset/empty ⇒ default; else false iff first char is one of {0,f,F,n,N}.
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
    std::optional<int> num_threads; // monoprop_NUM_THREADS
    bool shard_pinning = true;      // monoprop_SHARD_PINNING
};

// Parse the environment once; the Settings are cached and shared across TUs.
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        s.shard_pinning = detail::parse_flag(std::getenv("monoprop_SHARD_PINNING"), true);
        return s;
    }();
    return settings;
}

} // namespace monoprop::config
