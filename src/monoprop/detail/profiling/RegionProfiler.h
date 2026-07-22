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

// Zero-overhead-when-disabled per-region wall-time timer for shard/MPI-scaling analysis. Enabled by
// monoprop_PHASE_TIMERS (read once at load); each hook is one predictable branch when disabled. Within a
// shard a region's walls are additive (serial on one core). Mutable state (flag + accumulators) lives
// EXPORTED in Profiling.cpp so both .so's (core + nanobind extension) share one copy.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <utility>

#include "monoprop/monopropExport.h"

namespace monoprop::profiling {

enum class Region : int {
    Find = 0,     // fused_find_and_collect — anticommutation scan + cutoff + query emit
    SelfResolve,  // resolve_self_queries — resolve this rank's own query stream
    MpiExchange,  // layer-build alltoallv + resolve_incoming_queries + response fold (R>1 only)
    DeferInsert,  // insert_deferred_self_misses — grow op, scatter, index bulk_insert, inverted index resync
    Gather,       // assemble_partners + build_layer_storage_unified
    Evolve,       // evolve_step — cosine scale callback + cross-rank evolution exchange + self-apply
    CosRecompute, // make_fold_cache — inverted index fold at replay / expectation
    CosScale,     // apply_fused_contract — the eager scale_cos_mask bandwidth pass (redesign Piece 2 target)
    FusedApply,   // apply_fused_contract — the per-rotation sine-add pass
    Extend,       // extend_coeffs_from_current_picture_if_needed_ — per-gate coeff tail extension
    COUNT,        // region count / array size
};

inline constexpr int kRegionCount = std::to_underlying(Region::COUNT);

inline constexpr std::array<std::string_view, kRegionCount> kRegionNames{
    "find",
    "self_resolve",
    "mpi_exchange",
    "defer_insert",
    "gather",
    "evolve",
    "cos_recompute",
    "cos_scale",
    "fused_apply",
    "extend",
};

using prof_clock = std::chrono::steady_clock;

struct RegionAcc {
    std::atomic<uint64_t> wall_ns{0};
    std::atomic<uint64_t> calls{0}; // ScopedRegion entries
};

// Single, process-wide state (defined in Profiling.cpp, exported so both .so's share it).
monoprop_EXPORT extern bool g_profiling_enabled;
monoprop_EXPORT auto profiling_accs() -> RegionAcc *;   // base of the kRegionCount-element array
monoprop_EXPORT auto profiling_ensure_atexit() -> void; // register the one-shot stderr dump

// Fold statistics (monoprop_FOLD_STATS): per-gate anticommutation-scan sizing data from
// fused_find_and_collect, one relaxed-atomic publish per (gate, shard); dumped by the same atexit dump.
monoprop_EXPORT extern bool g_fold_stats_enabled;
monoprop_EXPORT auto record_fold_stats(bool all_sparse,
                                       bool skipped,
                                       size_t postings,
                                       size_t word_count,
                                       size_t n_anti,
                                       size_t struct_rejects) -> void;

inline auto acc(Region r) -> RegionAcc & {
    return profiling_accs()[std::to_underlying(r)];
}

// ScopedRegion: mark a named phase and accumulate its wall time.
class ScopedRegion {
public:
    explicit ScopedRegion(Region r) noexcept : r_(r) {
        if (!g_profiling_enabled) {
            return;
        }
        profiling_ensure_atexit();
        active_ = true;
        t0_ = prof_clock::now();
    }
    ScopedRegion(const ScopedRegion &) = delete;
    auto operator=(const ScopedRegion &) -> ScopedRegion & = delete;
    ~ScopedRegion() {
        if (!active_) {
            return;
        }
        const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(prof_clock::now() - t0_).count();
        auto &a = acc(r_);
        a.wall_ns.fetch_add(static_cast<uint64_t>(dt), std::memory_order_relaxed);
        a.calls.fetch_add(1, std::memory_order_relaxed);
    }

private:
    Region r_;
    bool active_ = false;
    prof_clock::time_point t0_{};
};

} // namespace monoprop::profiling
