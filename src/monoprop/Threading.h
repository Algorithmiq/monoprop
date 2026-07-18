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
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/detail/profiling/RegionProfiler.h"
#include "monoprop/monopropExport.h"

// monoprop's shared-memory threading layer: a minimal in-repo persistent thread pool exposing ONE
// primitive — run_static(n_tasks, fn) — plus grain-scheduled parallel_for_* / parallel_reduce_*
// wrappers (profiler-wired) built on it. The wrappers are for loops that write into PRE-SIZED
// disjoint slots or reduce with a deterministic ordered fold — where output order does not affect
// the result. When a parallel loop must BUILD an ordered output whose byte layout is thread-count
// invariant (the build path's bit-exactness guarantee), use the order-preserving chunk helpers in
// detail/evolution/layer_build/Parallel.h (for_each_chunk / append_gathered_chunks) instead.
namespace monoprop::threading {

inline constexpr size_t kSmallLoopThreshold = 1024;
inline constexpr size_t kDefaultGrainSize = 256;
// Per-worker cap on the chunk count of the grain-scheduled wrappers below: enough over-decomposition
// that the atomic-claim countdown loop absorbs per-chunk cost imbalance, without descending to
// per-element tasks on huge ranges. Matches the measured find-scan optimum (Parallel.h,
// kScanChunkCapPerWorker).
inline constexpr size_t kMaxChunksPerWorker = 16;

// ─── Configuration ──────────────────────────────────────────────────────────

/// @brief Configure the pool's maximum parallelism from the `monoprop_NUM_THREADS` environment
/// variable. Runs at most once per process (later calls are no-ops); also a no-op if the variable is
/// unset or invalid (the pool then defaults to the process CPU budget — the affinity mask count).
monoprop_EXPORT auto init_from_env() -> void;

/// @brief Thread-local whole-gate serial override, set by each shard master thread (see
/// detail::shard::ShardGroup) for the lifetime of its work. While true, every dispatch decision made
/// on this thread — effective_parallelism(), the chunk-count policies, and the
/// parallel_for_*/parallel_reduce_* small-loop fallbacks below — stays serial, keeping the whole
/// build+apply pipeline on the calling core. This is what makes a shard run its partition entirely on
/// its pinned core. Serial vs parallel only changes chunking, never results (order-preserving merges
/// are chunk-count invariant), so it is bit-exact.
inline auto gate_serial_override() -> bool & {
    thread_local bool serial = false;
    return serial;
}

/// @brief The current process-wide maximum parallelism (configured thread count, lowered by any live
/// ScopedParallelismCap), ignoring the calling thread's gate_serial_override.
monoprop_EXPORT auto current_max_parallelism() -> size_t;

/// @brief The current maximum parallelism, clamped to at least 1. Reports 1 while the calling
/// thread's gate is in serial mode (see gate_serial_override).
inline auto effective_parallelism() -> size_t {
    if (gate_serial_override()) {
        return 1;
    }
    return current_max_parallelism();
}

/// @brief RAII process-wide parallelism cap (the successor of the former scoped global control): while
/// alive, effective_parallelism() and every pool job are capped at `cap` participants. Caps nest by
/// save/restore; they lower the configured parallelism but never raise it.
class monoprop_EXPORT ScopedParallelismCap final {
public:
    explicit ScopedParallelismCap(size_t cap);
    ~ScopedParallelismCap();
    ScopedParallelismCap(const ScopedParallelismCap &) = delete;
    ScopedParallelismCap(ScopedParallelismCap &&) = delete;
    auto operator=(const ScopedParallelismCap &) -> ScopedParallelismCap & = delete;
    auto operator=(ScopedParallelismCap &&) -> ScopedParallelismCap & = delete;

private:
    size_t previous_;
};

/// @brief Grain size (elements per task) for partitioning a `count`-element range: more workers yield
/// finer tasks (count / (4·workers)), but the grain never drops below `min_grain`.
/// @param count     Total elements to partition.
/// @param min_grain Floor on the returned grain size.
/// @return The grain size, at least `min_grain`.
monoprop_EXPORT auto range_grain_size(size_t count, size_t min_grain = kDefaultGrainSize) -> size_t;

// ─── The pool primitive ─────────────────────────────────────────────────────

// Type-erased core of run_static (the pool lives in Threading.cpp). Runs inline when called from
// inside a pool task (the nesting guard), when n_tasks <= 1, or at effective parallelism 1.
monoprop_EXPORT auto run_static_impl(size_t n_tasks, void (*task)(void *, size_t), void *ctx) -> void;

/// @brief Execute fn(0) … fn(n_tasks-1) on the persistent pool. The caller participates as a worker;
/// completion is an atomic task countdown; tasks are claimed off a shared counter, so per-task cost
/// imbalance is absorbed without work stealing. Nested calls (from inside a task) run inline and
/// serial. Tasks must not assume an execution order; determinism comes from tasks writing disjoint
/// slots (or from an ordered merge after the join). All side effects of every task
/// happen-before the return.
template <typename Fn>
inline auto run_static(size_t n_tasks, Fn &&fn) -> void {
    if (n_tasks == 0) {
        return;
    }
    if (n_tasks == 1 || effective_parallelism() <= 1) {
        for (size_t i = 0; i < n_tasks; ++i) {
            fn(i);
        }
        return;
    }
    using F = std::remove_reference_t<Fn>;
    run_static_impl(
        n_tasks, [](void *ctx, size_t i) { (*static_cast<F *>(ctx))(i); },
        const_cast<void *>(static_cast<const void *>(std::addressof(fn))));
}

namespace detail_threading {

// Chunk count for the grain-scheduled wrappers: at least `grain` elements per chunk, at most
// kMaxChunksPerWorker chunks per worker. Depends on thread count only through
// effective_parallelism(), so the decomposition — and any ordered fold built on it — is
// deterministic for a fixed configuration.
inline auto grain_chunk_count(size_t count, size_t grain) -> size_t {
    const size_t by_grain = count / std::max<size_t>(1, grain);
    const size_t cap = effective_parallelism() * kMaxChunksPerWorker;
    return std::clamp<size_t>(by_grain, 1, std::max<size_t>(1, cap));
}

} // namespace detail_threading

// ─── 1-D parallel primitives ────────────────────────────────────────────────

// Parallel-for over [0, count) with small-loop fallback; chunked at >= grain_size elements per task.
template <typename Func>
inline auto parallel_for_indices(size_t count, Func &&func, size_t grain_size = kDefaultGrainSize) -> void {
    if (count == 0) {
        return;
    }
    const profiling::Region prof_r = profiling::capture();
    if (count < kSmallLoopThreshold || gate_serial_override()) {
        profiling::TaskScope prof_ts(prof_r);
        for (size_t idx = 0; idx < count; ++idx) {
            func(idx);
        }
        return;
    }
    const size_t chunks = detail_threading::grain_chunk_count(count, grain_size);
    const size_t per = (count + chunks - 1) / chunks;
    run_static(chunks, [&func, prof_r, per, count](size_t c) {
        profiling::TaskScope prof_ts(prof_r);
        const size_t lo = c * per;
        const size_t hi = std::min(count, lo + per);
        for (size_t idx = lo; idx < hi; ++idx) {
            func(idx);
        }
    });
}

// Parallel-reduce over [0, count) with small-loop fallback. Each chunk reduces from a copy of
// `identity`; the partials are folded in ascending chunk order, so the result is deterministic for a
// fixed thread configuration (the fold's floating-point association differs from the serial loop's).
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
    const size_t chunks = detail_threading::grain_chunk_count(count, effective_grain);
    if (count < std::max(kSmallLoopThreshold, effective_grain) || chunks <= 1 || gate_serial_override()) {
        profiling::TaskScope prof_ts(prof_r);
        Value local = identity;
        for (size_t idx = 0; idx < count; ++idx) {
            body(idx, local);
        }
        return local;
    }
    // Byte-addressed slot per chunk (a plain std::vector<Value> would bit-pack Value = bool).
    struct Slot {
        Value v;
    };
    std::vector<Slot> partials(chunks, Slot{identity});
    const size_t per = (count + chunks - 1) / chunks;
    run_static(chunks, [&, prof_r, per, count](size_t c) {
        profiling::TaskScope prof_ts(prof_r);
        Value local = identity;
        const size_t lo = c * per;
        const size_t hi = std::min(count, lo + per);
        for (size_t idx = lo; idx < hi; ++idx) {
            body(idx, local);
        }
        partials[c].v = std::move(local);
    });
    Value result = std::move(partials[0].v);
    for (size_t c = 1; c < chunks; ++c) {
        result = reduce(std::move(result), std::move(partials[c].v));
    }
    return result;
}

template <typename Func>
inline auto parallel_for_ranges(size_t count, Func &&func, size_t grain_size = 0) -> void {
    if (count == 0) {
        return;
    }

    const size_t grain = grain_size == 0 ? range_grain_size(count) : std::max<size_t>(1, grain_size);
    const profiling::Region prof_r = profiling::capture();
    if (count < std::max(kSmallLoopThreshold, grain) || gate_serial_override()) {
        profiling::TaskScope prof_ts(prof_r);
        func(0, count);
        return;
    }

    const size_t chunks = detail_threading::grain_chunk_count(count, grain);
    const size_t per = (count + chunks - 1) / chunks;
    run_static(chunks, [&func, prof_r, per, count](size_t c) {
        profiling::TaskScope prof_ts(prof_r);
        const size_t lo = c * per;
        func(lo, std::min(count, lo + per));
    });
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
    const size_t chunks = detail_threading::grain_chunk_count(count, grain);
    if (count < std::max(kSmallLoopThreshold, grain) || chunks <= 1 || gate_serial_override()) {
        profiling::TaskScope prof_ts(prof_r);
        return body(0, count, std::move(identity));
    }

    struct Slot {
        Value v;
    };
    std::vector<Slot> partials(chunks, Slot{identity});
    const size_t per = (count + chunks - 1) / chunks;
    run_static(chunks, [&, prof_r, per, count](size_t c) {
        profiling::TaskScope prof_ts(prof_r);
        const size_t lo = c * per;
        Value local = identity;
        partials[c].v = body(lo, std::min(count, lo + per), std::move(local));
    });
    Value result = std::move(partials[0].v);
    for (size_t c = 1; c < chunks; ++c) {
        result = reduce(std::move(result), std::move(partials[c].v));
    }
    return result;
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

// Column-chunk count for the 2-D wrappers: >= range_grain_size(max_extent) columns per task.
inline auto column_chunk_count(size_t max_extent) -> size_t {
    return grain_chunk_count(max_extent, range_grain_size(max_extent));
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
    if (num_ranks * max_extent < kSmallLoopThreshold || gate_serial_override()) {
        profiling::TaskScope prof_ts(prof_r);
        detail_threading::for_each_rank_range_window(0, num_ranks, 0, max_extent, my_rank, sf, fn);
        return;
    }

    const size_t col_chunks = detail_threading::column_chunk_count(max_extent);
    const size_t per = (max_extent + col_chunks - 1) / col_chunks;
    run_static(num_ranks * col_chunks, [&, prof_r, per, col_chunks, max_extent, my_rank](size_t t) {
        profiling::TaskScope prof_ts(prof_r);
        const size_t rank = t / col_chunks;
        const size_t lo = (t % col_chunks) * per;
        detail_threading::for_each_rank_range_window(
            rank, rank + 1, lo, std::min(max_extent, lo + per), my_rank, sf, fn);
    });
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
    if (num_ranks * max_extent < kSmallLoopThreshold || gate_serial_override()) {
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

    const size_t col_chunks = detail_threading::column_chunk_count(max_extent);
    const size_t per = (max_extent + col_chunks - 1) / col_chunks;
    const size_t n_tasks = num_ranks * col_chunks;
    struct Slot {
        Value v;
    };
    // Partial per (rank, column-chunk) task, folded in task order — rank-major, columns ascending —
    // which matches the serial visit order, so the reduce is deterministic at any thread count.
    std::vector<Slot> partials(n_tasks, Slot{identity});
    run_static(n_tasks, [&, prof_r, per, col_chunks, max_extent, my_rank](size_t t) {
        profiling::TaskScope prof_ts(prof_r);
        const size_t rank = t / col_chunks;
        const size_t lo = (t % col_chunks) * per;
        Value local = identity;
        detail_threading::for_each_rank_range_window(
            rank,
            rank + 1,
            lo,
            std::min(max_extent, lo + per),
            my_rank,
            sf,
            [&local, &fn](size_t rnk, size_t begin, size_t end) { local = fn(rnk, begin, end, std::move(local)); });
        partials[t].v = std::move(local);
    });
    Value result = std::move(partials[0].v);
    for (size_t t = 1; t < n_tasks; ++t) {
        result = reduce(std::move(result), std::move(partials[t].v));
    }
    return result;
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
