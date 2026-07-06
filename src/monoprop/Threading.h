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

#include <algorithm>
#include <cstddef>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>

#include "monoprop/detail/profiling/RegionProfiler.h"
#include "monoprop/monopropExport.h"

// These grain-scheduled parallel_for_* / parallel_reduce_* primitives (profiler-wired) are for loops
// that write into PRE-SIZED disjoint slots or reduce commutatively — where output order does not affect
// the result. When a parallel loop must BUILD an ordered output whose byte layout is thread-count
// invariant (the build path's bit-exactness guarantee), use the order-preserving chunk helpers in
// detail/evolution/layer_build/Parallel.h (for_each_chunk / append_gathered_chunks) instead.
namespace monoprop::threading {

inline constexpr size_t kSmallLoopThreshold = 1024;
inline constexpr size_t kDefaultGrainSize = 256;

// ─── Configuration ──────────────────────────────────────────────────────────

/// @brief Configure oneTBB's maximum parallelism from the `monoprop_NUM_THREADS` environment variable.
/// Runs at most once per process (later calls are no-ops); also a no-op if the variable is unset or
/// invalid.
monoprop_EXPORT auto init_from_env() -> void;

/// @brief The current oneTBB maximum parallelism, clamped to at least 1.
inline auto effective_parallelism() -> size_t {
    const auto active = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    return std::max<size_t>(1, static_cast<size_t>(active));
}

/// @brief Grain size (elements per task) for partitioning a `count`-element range: more workers yield
/// finer tasks (count / (4·workers)), but the grain never drops below `min_grain`.
/// @param count     Total elements to partition.
/// @param min_grain Floor on the returned grain size.
/// @return The grain size, at least `min_grain`.
monoprop_EXPORT auto range_grain_size(size_t count, size_t min_grain = kDefaultGrainSize) -> size_t;

// ─── 1-D parallel primitives ────────────────────────────────────────────────

// Parallel-for over [0, count) with small-loop fallback and affinity partitioner.
template <typename Func>
inline auto parallel_for_indices(size_t count, Func &&func, size_t grain_size = kDefaultGrainSize) -> void {
    if (count == 0) {
        return;
    }
    const profiling::Region prof_r = profiling::capture();
    if (count < kSmallLoopThreshold) {
        profiling::TaskScope prof_ts(prof_r);
        for (size_t idx = 0; idx < count; ++idx) {
            func(idx);
        }
        return;
    }
    tbb::affinity_partitioner ap;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, count, grain_size),
        [&func, prof_r](const tbb::blocked_range<size_t> &range) {
            profiling::TaskScope prof_ts(prof_r);
            for (size_t idx = range.begin(); idx < range.end(); ++idx) {
                func(idx);
            }
        },
        ap);
}

// Parallel-reduce over [0, count) with small-loop fallback and affinity partitioner.
template <typename Value, typename Body, typename ReduceOp>
inline Value parallel_reduce_indices(size_t count,
                                     Value identity,
                                     Body &&body,
                                     ReduceOp &&reduce,
                                     size_t grain_size = kDefaultGrainSize) {
    if (count == 0) {
        return identity;
    }
    const size_t effective_grain = std::max<size_t>(1, grain_size);
    const profiling::Region prof_r = profiling::capture();
    if (count < std::max(kSmallLoopThreshold, effective_grain)) {
        profiling::TaskScope prof_ts(prof_r);
        Value local = identity;
        for (size_t idx = 0; idx < count; ++idx) {
            body(idx, local);
        }
        return local;
    }
    tbb::affinity_partitioner ap;
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, count, effective_grain),
        identity,
        [&body, prof_r](const tbb::blocked_range<size_t> &range, Value local) {
            profiling::TaskScope prof_ts(prof_r);
            for (size_t idx = range.begin(); idx < range.end(); ++idx) {
                body(idx, local);
            }
            return local;
        },
        std::forward<ReduceOp>(reduce),
        ap);
}

template <typename Func>
inline auto parallel_for_ranges(size_t count, Func &&func, size_t grain_size = 0) -> void {
    if (count == 0) {
        return;
    }

    const size_t grain = grain_size == 0 ? range_grain_size(count) : std::max<size_t>(1, grain_size);
    const profiling::Region prof_r = profiling::capture();
    if (count < std::max(kSmallLoopThreshold, grain)) {
        profiling::TaskScope prof_ts(prof_r);
        func(0, count);
        return;
    }

    tbb::affinity_partitioner ap;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, count, grain),
        [&func, prof_r](const tbb::blocked_range<size_t> &range) {
            profiling::TaskScope prof_ts(prof_r);
            func(range.begin(), range.end());
        },
        ap);
}

template <typename Value, typename Body, typename ReduceOp>
inline Value parallel_reduce_ranges(size_t count,
                                    Value identity,
                                    Body &&body,
                                    ReduceOp &&reduce,
                                    size_t grain_size = 0) {
    if (count == 0) {
        return identity;
    }

    const size_t grain = grain_size == 0 ? range_grain_size(count) : std::max<size_t>(1, grain_size);
    const profiling::Region prof_r = profiling::capture();
    if (count < std::max(kSmallLoopThreshold, grain)) {
        profiling::TaskScope prof_ts(prof_r);
        return body(0, count, std::move(identity));
    }

    tbb::affinity_partitioner ap;
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, count, grain),
        std::move(identity),
        [&body, prof_r](const tbb::blocked_range<size_t> &range, Value local) {
            profiling::TaskScope prof_ts(prof_r);
            return body(range.begin(), range.end(), std::move(local));
        },
        std::forward<ReduceOp>(reduce),
        ap);
}

// ─── 2-D rank-range parallel primitives ────────────────────────────────────
// Iterate or reduce over (rank, begin, end) spans where each rank has a
// different extent. Automatically falls back to sequential for small workloads.

namespace detail_threading {

template <typename SizeFunc>
inline auto max_extent_skipping(size_t num_ranks, int skip_rank, SizeFunc &sf) -> size_t {
    size_t max_extent = 0;
    for (size_t rank = 0; rank < num_ranks; ++rank) {
        if (static_cast<int>(rank) == skip_rank) {
            continue;
        }
        max_extent = std::max(max_extent, sf(rank));
    }
    return max_extent;
}

template <typename SizeFunc, typename Body>
inline auto for_each_rank_range_window(size_t row_begin,
                                       size_t row_end,
                                       size_t col_begin,
                                       size_t col_end,
                                       int skip_rank,
                                       SizeFunc &size_func,
                                       Body &&body) -> void {
    for (size_t rank = row_begin; rank < row_end; ++rank) {
        if (static_cast<int>(rank) == skip_rank) {
            continue;
        }
        const size_t begin = std::min(col_begin, size_func(rank));
        const size_t end = std::min(col_end, size_func(rank));
        if (begin < end) {
            body(rank, begin, end);
        }
    }
}

} // namespace detail_threading

template <typename SizeFunc, typename Body>
inline auto parallel_for_rank_ranges(size_t num_ranks, int my_rank, SizeFunc &&size_for_rank, Body &&body) -> void {
    auto sf = std::forward<SizeFunc>(size_for_rank);
    auto fn = std::forward<Body>(body);
    const size_t max_extent = detail_threading::max_extent_skipping(num_ranks, my_rank, sf);
    if (max_extent == 0) {
        return;
    }
    const profiling::Region prof_r = profiling::capture();
    if (num_ranks * max_extent < kSmallLoopThreshold) {
        profiling::TaskScope prof_ts(prof_r);
        detail_threading::for_each_rank_range_window(0, num_ranks, 0, max_extent, my_rank, sf, fn);
        return;
    }

    const size_t grain = range_grain_size(max_extent);
    tbb::affinity_partitioner ap;
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, num_ranks, 1, 0, max_extent, grain),
        [&sf, &fn, my_rank, prof_r](const auto &ranges) {
            profiling::TaskScope prof_ts(prof_r);
            detail_threading::for_each_rank_range_window(ranges.rows().begin(),
                                                         ranges.rows().end(),
                                                         ranges.cols().begin(),
                                                         ranges.cols().end(),
                                                         my_rank,
                                                         sf,
                                                         fn);
        },
        ap);
}

template <typename Value, typename SizeFunc, typename Body, typename ReduceOp>
inline Value parallel_reduce_rank_ranges(size_t num_ranks,
                                         int my_rank,
                                         SizeFunc &&size_for_rank,
                                         Value identity,
                                         Body &&body,
                                         ReduceOp &&reduce) {
    auto sf = std::forward<SizeFunc>(size_for_rank);
    auto fn = std::forward<Body>(body);
    const size_t max_extent = detail_threading::max_extent_skipping(num_ranks, my_rank, sf);
    if (max_extent == 0) {
        return identity;
    }
    const profiling::Region prof_r = profiling::capture();
    if (num_ranks * max_extent < kSmallLoopThreshold) {
        profiling::TaskScope prof_ts(prof_r);
        Value local = std::move(identity);
        detail_threading::for_each_rank_range_window(
            0,
            num_ranks,
            0,
            max_extent,
            my_rank,
            sf,
            [&local, &fn](size_t rank, size_t begin, size_t end) { local = fn(rank, begin, end, std::move(local)); });
        return local;
    }

    const size_t grain = range_grain_size(max_extent);
    tbb::affinity_partitioner ap;
    return tbb::parallel_reduce(
        tbb::blocked_range2d<size_t>(0, num_ranks, 1, 0, max_extent, grain),
        std::move(identity),
        [&sf, &fn, my_rank, prof_r](const auto &ranges, Value local) {
            profiling::TaskScope prof_ts(prof_r);
            detail_threading::for_each_rank_range_window(ranges.rows().begin(),
                                                         ranges.rows().end(),
                                                         ranges.cols().begin(),
                                                         ranges.cols().end(),
                                                         my_rank,
                                                         sf,
                                                         [&local, &fn](size_t rank, size_t begin, size_t end) {
                                                             local = fn(rank, begin, end, std::move(local));
                                                         });
            return local;
        },
        std::forward<ReduceOp>(reduce),
        ap);
}

template <typename LayerLike, typename Body>
inline void parallel_for_cross_rank_sin_send_ranges(const LayerLike &layer, int my_rank, Body &&body) {
    parallel_for_rank_ranges(
        layer.cross_rank_rank_count(),
        my_rank,
        [&](size_t rank) { return layer.cross_rank_sin_send_size(rank); },
        std::forward<Body>(body));
}

template <typename LayerLike, typename Body>
inline void parallel_for_cross_rank_sin_recv_ranges(const LayerLike &layer, int my_rank, Body &&body) {
    parallel_for_rank_ranges(
        layer.cross_rank_rank_count(),
        my_rank,
        [&](size_t rank) { return layer.cross_rank_sin_recv_size(rank); },
        std::forward<Body>(body));
}

template <typename LayerLike, typename Value, typename Body, typename ReduceOp>
inline Value parallel_reduce_cross_rank_sin_recv_ranges(const LayerLike &layer,
                                                 int my_rank,
                                                 Value identity,
                                                 Body &&body,
                                                 ReduceOp &&reduce) {
    return parallel_reduce_rank_ranges(
        layer.cross_rank_rank_count(),
        my_rank,
        [&](size_t rank) { return layer.cross_rank_sin_recv_size(rank); },
        std::move(identity),
        std::forward<Body>(body),
        std::forward<ReduceOp>(reduce));
}

} // namespace monoprop::threading
