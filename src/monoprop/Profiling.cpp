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

// Single definition of the RegionProfiler's process-wide mutable state (see RegionProfiler.h), kept in
// one TU so the core and the nanobind extension share one copy of the flags, accumulators, and dump.

#include "monoprop/detail/profiling/RegionProfiler.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <print>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::profiling {
namespace {

auto env_enabled() -> bool {
    return config::get().phase_timers;
}

std::array<RegionAcc, kRegionCount> g_accs{};

// Fold-stats accumulators (monoprop_FOLD_STATS), one publish per (gate, shard). The ratio histogram
// buckets ~8+log2(postings/word_count) for all-sparse gates: bucket 0 = zero postings, bucket 8 ≈
// ratio 1 (candidate-merge break-even), clamped to 1..15 at the ends.
inline constexpr size_t kFoldRatioBuckets = 16;
struct FoldStats {
    std::atomic<uint64_t> gates{0};          // fused scans recorded
    std::atomic<uint64_t> skipped{0};        // zero-postings early-outs (pass 1 not run)
    std::atomic<uint64_t> all_sparse{0};     // gates whose fold columns are all sparse-tier
    std::atomic<uint64_t> sum_postings{0};   // Σ postings over all-sparse gates only
    std::atomic<uint64_t> sum_words{0};      // Σ fold word_count (K_shard/64) over all gates
    std::atomic<uint64_t> n_anti{0};         // Σ anticommuting terms
    std::atomic<uint64_t> struct_rejects{0}; // Σ structural-cutoff rejections (no upper-atol rescue)
    std::array<std::atomic<uint64_t>, kFoldRatioBuckets> ratio_hist{};
};
FoldStats g_fold_stats{};

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
        if (calls == 0) {
            continue;
        }
        const auto name = kRegionNames[static_cast<size_t>(i)];
        std::print(stderr,
                   "monoprop_PHASE rank={} region={} wall_ms={:.3f} calls={}\n",
                   rank,
                   name,
                   static_cast<double>(wall) / 1.0e6,
                   calls);
    }
    if (const auto gates = g_fold_stats.gates.load(std::memory_order_relaxed); gates != 0) {
        const auto v = [](const std::atomic<uint64_t> &a) {
            return static_cast<unsigned long long>(a.load(std::memory_order_relaxed));
        };
        std::print(stderr,
                   "monoprop_FOLDSTATS rank={} gates={} skipped={} all_sparse={} sum_postings={} "
                   "sum_words={} n_anti={} struct_rejects={}\n",
                   rank,
                   gates,
                   v(g_fold_stats.skipped),
                   v(g_fold_stats.all_sparse),
                   v(g_fold_stats.sum_postings),
                   v(g_fold_stats.sum_words),
                   v(g_fold_stats.n_anti),
                   v(g_fold_stats.struct_rejects));
        std::fprintf(stderr, "monoprop_FOLDSTATS rank=%d ratio_hist=", rank);
        for (size_t b = 0; b < kFoldRatioBuckets; ++b) {
            std::fprintf(stderr, "%s%llu", b == 0 ? "" : ",", v(g_fold_stats.ratio_hist[b]));
        }
        std::fprintf(stderr, " # bucket0: P==0; b ~= 8+log2(P/(K/64)); bucket 8 ~= break-even\n");
    }
    std::fflush(stderr);
}

} // namespace

bool g_profiling_enabled = env_enabled();
bool g_fold_stats_enabled = config::get().fold_stats;

auto profiling_accs() -> RegionAcc * {
    return g_accs.data();
}

auto record_fold_stats(bool all_sparse,
                       bool skipped,
                       size_t postings,
                       size_t word_count,
                       size_t n_anti,
                       size_t struct_rejects) -> void {
    profiling_ensure_atexit();
    constexpr auto relaxed = std::memory_order_relaxed;
    g_fold_stats.gates.fetch_add(1, relaxed);
    g_fold_stats.sum_words.fetch_add(word_count, relaxed);
    g_fold_stats.n_anti.fetch_add(n_anti, relaxed);
    g_fold_stats.struct_rejects.fetch_add(struct_rejects, relaxed);
    if (skipped) {
        g_fold_stats.skipped.fetch_add(1, relaxed);
    }
    if (!all_sparse) {
        return; // the ratio histogram sizes the candidate-merge path, which needs all-sparse columns
    }
    g_fold_stats.all_sparse.fetch_add(1, relaxed);
    g_fold_stats.sum_postings.fetch_add(postings, relaxed);
    size_t bucket = 0;
    if (postings != 0) {
        // b ≈ 8 + log2(postings/word_count), from bit widths (±1 bucket), clamped to 1..15.
        const int diff = std::bit_width(postings) - std::bit_width(word_count | 1);
        const int b = 8 + diff;
        bucket = static_cast<size_t>(std::clamp(b, 1, static_cast<int>(kFoldRatioBuckets) - 1));
    }
    g_fold_stats.ratio_hist[bucket].fetch_add(1, relaxed);
}

auto profiling_ensure_atexit() -> void {
    static std::atomic<bool> registered{false};
    bool expected = false;
    if (registered.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        std::atexit(dump);
    }
}

} // namespace monoprop::profiling
