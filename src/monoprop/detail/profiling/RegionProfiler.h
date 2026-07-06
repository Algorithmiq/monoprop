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

// ─── RegionProfiler ────────────────────────────────────────────────────────────
// A zero-overhead-when-disabled per-region wall/busy timer used for the thread- and MPI-scaling
// analysis. Enabled by the environment variable `monoprop_PHASE_TIMERS` (read once at load); every
// hook is a single predictable branch (an exported bool load) when disabled.
//
// Two metrics per named region:
//   • wall_ns — sum of ScopedRegion lifetimes (measured on the dispatching thread; the layer-build
//     phases run sequentially on the main thread inside the propagate() task_arena, so a region's
//     walls do not overlap and are additive).
//   • busy_ns — sum over parallel tasks (and serial fallbacks) of their body execution time. The
//     region is CAPTURED at the dispatch site (main thread) via capture() and threaded into each task
//     as a TaskScope, so TBB worker threads — which do not inherit the dispatcher's thread-local —
//     attribute their busy-time to the correct region without a shared hot-path region lookup.
//
// The parser derives per-region utilization = busy_ns / (wall_ns · T) and share = wall_ns / Σ wall_ns.
//
// The MUTABLE state (enable flag, accumulators, the per-thread current region) lives in Profiling.cpp
// and is EXPORTED, so the process holds exactly one copy even though the header is compiled into both
// libmonoprop.so (core) and _core.*.so (the nanobind extension). This is load-bearing: without a
// single shared state, busy-time whose dispatch marker and TBB primitive land in different modules
// (e.g. the cosine-scale fold called during Evolve) would be misattributed. This header stays light
// (std + the export macro): it is included by the low-level threading primitives, so it must NOT
// include Threading.h (no cycle).

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>

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
    Other,        // any parallel work dispatched outside a marked region
    COUNT,        // (also the "profiling disabled" sentinel for capture())
};

inline constexpr int kRegionCount = static_cast<int>(Region::COUNT);

inline constexpr std::array<std::string_view, kRegionCount> kRegionNames{
    "find",   "self_resolve", "mpi_exchange", "defer_insert", "gather",  "evolve",
    "cos_recompute", "cos_scale", "fused_apply", "extend",     "other",
};

using prof_clock = std::chrono::steady_clock;

struct RegionAcc {
    std::atomic<uint64_t> wall_ns{0};
    std::atomic<uint64_t> busy_ns{0};
    std::atomic<uint64_t> calls{0}; // ScopedRegion entries
    std::atomic<uint64_t> tasks{0}; // TaskScope bodies (parallel tasks + serial fallbacks)
};

// ── Single, process-wide state (defined in Profiling.cpp, exported so both .so's share it) ──
// The enable flag is read once at load; the accessor for the accumulator array and the per-thread
// current-region reference are only invoked on hot paths when profiling is enabled.
monoprop_EXPORT extern bool g_profiling_enabled;
monoprop_EXPORT auto profiling_accs() -> RegionAcc *;    // base of the kRegionCount-element array
monoprop_EXPORT auto profiling_current() -> Region &;    // the calling thread's current region
monoprop_EXPORT auto profiling_ensure_atexit() -> void;  // register the one-shot stderr dump

inline auto acc(Region r) -> RegionAcc & { return profiling_accs()[static_cast<int>(r)]; }

// ── ScopedRegion: mark a named phase on the dispatching thread (accumulates wall time). ──
class ScopedRegion {
public:
    explicit ScopedRegion(Region r) noexcept : r_(r) {
        if (!g_profiling_enabled) {
            return;
        }
        profiling_ensure_atexit();
        active_ = true;
        Region &cur = profiling_current();
        prev_ = cur;
        cur = r;
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
        profiling_current() = prev_;
    }

private:
    Region r_;
    bool active_ = false;
    Region prev_ = Region::Other;
    prof_clock::time_point t0_{};
};

// ── capture(): read the active region on the dispatching thread before a parallel region. ──
// Returns Region::COUNT when profiling is off — a sentinel that makes TaskScope a no-op.
inline auto capture() noexcept -> Region { return g_profiling_enabled ? profiling_current() : Region::COUNT; }

// ── TaskScope: time one task body (or serial fallback) and attribute busy-time to the captured
// region. Also sets the worker's current region so any NESTED parallel dispatch inherits it. ──
class TaskScope {
public:
    explicit TaskScope(Region captured) noexcept : r_(captured) {
        if (r_ == Region::COUNT) {
            return;
        }
        Region &cur = profiling_current();
        prev_ = cur;
        cur = r_;
        t0_ = prof_clock::now();
    }
    TaskScope(const TaskScope &) = delete;
    auto operator=(const TaskScope &) -> TaskScope & = delete;
    ~TaskScope() {
        if (r_ == Region::COUNT) {
            return;
        }
        const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(prof_clock::now() - t0_).count();
        auto &a = acc(r_);
        a.busy_ns.fetch_add(static_cast<uint64_t>(dt), std::memory_order_relaxed);
        a.tasks.fetch_add(1, std::memory_order_relaxed);
        profiling_current() = prev_;
    }

private:
    Region r_;
    Region prev_ = Region::Other;
    prof_clock::time_point t0_{};
};

} // namespace monoprop::profiling
