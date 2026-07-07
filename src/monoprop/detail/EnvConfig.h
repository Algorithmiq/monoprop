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

// Single home for all runtime environment configuration. Every `monoprop_*` env var the library reads
// is parsed here, exactly once (function-local static in config::get()), and exposed as a field of
// config::Settings. Callers keep their own named accessors (e.g. threading::get_env_threads)
// and delegate to config::get() so behaviour is unchanged.
//
// Dependency-free by design (only <cstdlib>): this header is pulled into low-level, hot-path headers
// (e.g. cosine recompute), so it must not depend on TBB, MPI, or any monoprop type.
//
// Recognised env vars:
//   monoprop_NUM_THREADS          positive int (1..1e6); else ignored          → num_threads
//   monoprop_RECOMPUTE_CACHE_MAX_MB  size in MB, default 2048 (0 ⇒ recompute)  → recompute_cache_max_mb
//   monoprop_PHASE_TIMERS         bool, default OFF                            → phase_timers
// Shard-runtime vars are parsed at their point of use (they need string forms beyond a plain field),
// not cached here, but are listed for a single inventory:
//   monoprop_SHARDS               int N | "auto" | "off"; overrides the shard-count policy
//                                 (MonomialPropagator::resolve_shard_count_)
//   monoprop_SHARD_PINNING        bool, default ON; 0/false disables per-core pinning (CpuTopology)
// NOTE: the profiler's rank discovery (OMPI_COMM_WORLD_RANK/PMI_RANK/PMIX_RANK) is intentionally NOT
// here — it is launcher-provided, not a monoprop knob.

namespace monoprop::config {

namespace detail {

// Truthy parse shared by the two boolean flags: unset or empty ⇒ default; otherwise false iff the
// first character is one of {0,f,F,n,N}. Matches the historical per-site semantics exactly.
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
    std::optional<int> num_threads;            // monoprop_NUM_THREADS
    std::size_t recompute_cache_max_mb = 2048; // monoprop_RECOMPUTE_CACHE_MAX_MB
    bool phase_timers = false;                 // monoprop_PHASE_TIMERS
};

/// Parse the environment once and return the shared, immutable Settings. The first call reads every
/// env var; later calls return the cached result (inline function ⇒ one instance across TUs).
inline auto get() -> const Settings & {
    static const Settings settings = [] {
        Settings s;
        s.num_threads = detail::parse_positive_int(std::getenv("monoprop_NUM_THREADS"));
        if (const char *e = std::getenv("monoprop_RECOMPUTE_CACHE_MAX_MB")) {
            s.recompute_cache_max_mb = static_cast<std::size_t>(std::strtoull(e, nullptr, 10));
        }
        s.phase_timers = detail::parse_flag(std::getenv("monoprop_PHASE_TIMERS"), false);
        return s;
    }();
    return settings;
}

} // namespace monoprop::config
