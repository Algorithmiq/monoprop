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

#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

// Single home for runtime environment configuration, kept dependency-free because it is pulled into
// hot-path headers: monoprop_NUM_THREADS (positive int 1..1e6, else ignored), monoprop_PARTITION_PINNING
// (bool, default ON), monoprop_PARTITIONS (int N | "auto" | "off", parsed in resolve_partition_count_)
// and monoprop_PROFILE (comma list of diagnostic regions on stderr, default OFF). An UNRECOGNISED
// region throws rather than falling back to off, and unset or empty is the only way to turn everything
// off: a diagnostic that silently does nothing looks exactly like one whose subject did nothing.

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

// A struct of bools rather than a bitmask so a call site reads `profile.layer`.
struct Profile {
    bool layer = false;
    bool comm = false;
    bool replay = false;
    bool mem = false;

    // True when nothing was requested: the only state in which no slot is allocated and no clock read.
    [[nodiscard]] auto any() const -> bool;
};

namespace detail {

// The ONE home for the field list: `all`, any() and the error text all walk it, never the fields.
struct Region {
    std::string_view name; // the token accepted in monoprop_PROFILE
    bool Profile::*bit;    // the field that token sets
};

inline constexpr std::array<Region, 4> kRegions{{
    {"layer", &Profile::layer},
    {"comm", &Profile::comm},
    {"replay", &Profile::replay},
    {"mem", &Profile::mem},
}};

// Profile is an aggregate of bools and nothing else, so its size IS its field count.
static_assert(sizeof(Profile) == kRegions.size() * sizeof(bool),
              "kRegions must have exactly one row per Profile field");

// One row per field is not enough alone -- four rows could point at three; distinctness makes it exact.
constexpr auto regions_are_distinct() -> bool {
    for (std::size_t i = 0; i < kRegions.size(); ++i) {
        for (std::size_t j = i + 1; j < kRegions.size(); ++j) {
            if (kRegions[i].bit == kRegions[j].bit || kRegions[i].name == kRegions[j].name) {
                return false;
            }
        }
    }
    return true;
}
static_assert(regions_are_distinct(), "two kRegions rows share a name or a Profile field");

} // namespace detail

inline auto Profile::any() const -> bool {
    for (const auto &region : detail::kRegions) {
        if (this->*region.bit) {
            return true;
        }
    }
    return false;
}

// `profile` keeps its shape either way: without the instrument nothing parses it, so every bit is false.
struct Settings {
    std::optional<int> num_threads;
    bool partition_pinning = true;
    Profile profile;
};

namespace detail {

#ifdef monoprop_ENABLE_PROFILE

// The accepted values, spelled from kRegions so a new region cannot leave the error text stale.
inline auto region_list_text() -> std::string {
    std::string text;
    for (const auto &region : kRegions) {
        text += region.name;
        text += ", ";
    }
    text += "all"; // every row at once; not a row itself
    return text;
}

// Split on ',' and set one bit per named region; throws on an unrecognised name.
inline auto parse_profile(const char *text) -> Profile {
    Profile p;
    if (text == nullptr || text[0] == '\0') {
        return p;
    }
    std::string_view rest(text);
    while (!rest.empty()) {
        const auto comma = rest.find(',');
        std::string_view tok = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        // Tolerate ' ' and '\t' around a name so `layer, comm` behaves; nothing else is tolerated.
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) {
            tok.remove_prefix(1);
        }
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) {
            tok.remove_suffix(1);
        }
        if (tok.empty()) {
            continue; // a trailing or doubled comma is a typo, not an instruction
        }
        if (tok == "all") {
            // Through the table, one write each: a whole-struct assignment would clear earlier tokens.
            for (const auto &region : kRegions) {
                p.*region.bit = true;
            }
            continue;
        }
        const Region *match = nullptr;
        for (const auto &region : kRegions) {
            if (tok == region.name) {
                match = &region;
                break;
            }
        }
        if (match == nullptr) {
            throw std::invalid_argument("monoprop_PROFILE: unknown region '" + std::string(tok)
                                        + "'; expected a comma list of " + region_list_text()
                                        + ". There is no boolean form: 0, off and none are not regions. To"
                                          " turn every region off, leave monoprop_PROFILE unset or set it"
                                          " to the empty string.");
        }
        p.*match->bit = true;
    }
    return p;
}

#endif // monoprop_ENABLE_PROFILE

} // namespace detail

// Parse the environment once; the Settings are cached and shared across TUs.
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        s.partition_pinning = detail::parse_flag(std::getenv("monoprop_PARTITION_PINNING"), true);
#ifdef monoprop_ENABLE_PROFILE
        s.profile = detail::parse_profile(std::getenv("monoprop_PROFILE"));
#endif
        return s;
    }();
    return settings;
}

// Force the parse now, for its side effect of throwing on a bad monoprop_PROFILE. Once per process,
// EARLY, on every rank, before any collective: a rank throwing mid-collective hangs its peers.
inline auto validate() -> void {
#ifndef monoprop_ENABLE_PROFILE
    // Otherwise a build without the instrument answers monoprop_PROFILE with an empty log -- the "did
    // nothing" ambiguity the strict parse exists to avoid. Set it empty to run an unprofiled cell.
    if (const char *text = std::getenv("monoprop_PROFILE"); text != nullptr && text[0] != '\0') {
        throw std::invalid_argument("monoprop_PROFILE is set to '" + std::string(text)
                                    + "', but this build was compiled without the profiling instrument."
                                      " Rebuild with -Dmonoprop_ENABLE_PROFILE=ON, or unset the variable"
                                      " (or set it to the empty string).");
    }
#endif
    (void)get();
}

} // namespace monoprop::config
