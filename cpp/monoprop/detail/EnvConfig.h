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

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

// Single home for runtime environment configuration. Kept dependency-free by design, because it is
// pulled into hot-path headers.
//
//   monoprop_NUM_THREADS        positive int (1..1e6), else ignored → num_threads
//   monoprop_PARTITIONS         int N | "auto" | "off"; parsed where it is used (resolve_partition_count_)
//   monoprop_ROUTING            "splitmix" | "linear" → routing_mode
//   monoprop_ROUTE_SEED         decimal uint64 basis seed → route_seed
//   monoprop_COMMPROF           0|1 (off by default) → comm_profile
//
// The routing knobs and the profile switch all THROW on a malformed value instead of falling back: each
// silently changes the transport or what is reported, so a typo that defaulted would stay invisible
// until a performance postmortem.

namespace monoprop::config {

class EnvConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class RoutingMode : std::uint8_t { Splitmix, Linear };

namespace detail {

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

[[noreturn]] inline auto reject_env(std::string_view name, const char *text, std::string_view expected) -> void {
    throw EnvConfigError(std::string{name} + "=\"" + text + "\" is not " + std::string{expected}
                         + "; correct it or leave the variable unset.");
}

// Unset and empty are both nullopt; a value that is present but unparseable throws.
inline auto parse_uint64(std::string_view name, const char *text) -> std::optional<std::uint64_t> {
    if (text == nullptr || *text == '\0') {
        return std::nullopt;
    }
    // strtoull WRAPS a negative literal to a huge unsigned rather than failing, so '-' is rejected here.
    if (std::string_view{text}.find('-') != std::string_view::npos) {
        reject_env(name, text, "a decimal uint64");
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        reject_env(name, text, "a decimal uint64");
    }
    return static_cast<std::uint64_t>(value);
}

// 0/1 only, so the variable reads the same way in a job script as in the docs.
inline auto parse_bool(std::string_view name, const char *text) -> std::optional<bool> {
    if (text == nullptr || *text == '\0') {
        return std::nullopt;
    }
    const std::string_view value{text};
    if (value == "0") {
        return false;
    }
    if (value == "1") {
        return true;
    }
    reject_env(name, text, "0 or 1");
}

inline auto parse_routing_mode(std::string_view name, const char *text) -> std::optional<RoutingMode> {
    if (text == nullptr || *text == '\0') {
        return std::nullopt;
    }
    const std::string_view value{text};
    if (value == "splitmix") {
        return RoutingMode::Splitmix;
    }
    if (value == "linear") {
        return RoutingMode::Linear;
    }
    reject_env(name, text, "one of \"splitmix\" or \"linear\"");
}

} // namespace detail

struct Settings {
    std::optional<int> num_threads;
    std::optional<RoutingMode> routing_mode;
    std::optional<std::uint64_t> route_seed;
    // Report-only: one COMMPROF line per slot per propagate/build_graph call, naming the gate exchange's
    // wire volume (MonomialPropagator::run_gate_loop_). Nothing reads it back.
    bool comm_profile = false;
};

// Parse the environment once; the Settings are cached and shared across TUs.
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        s.routing_mode = detail::parse_routing_mode("monoprop_ROUTING", std::getenv("monoprop_ROUTING"));
        s.route_seed = detail::parse_uint64("monoprop_ROUTE_SEED", std::getenv("monoprop_ROUTE_SEED"));
        s.comm_profile = detail::parse_bool("monoprop_COMMPROF", std::getenv("monoprop_COMMPROF")).value_or(false);
        return s;
    }();
    return settings;
}

} // namespace monoprop::config
