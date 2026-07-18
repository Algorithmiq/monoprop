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
#include <vector>

#include "monoprop/Threading.h"
#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/profiling/RegionProfiler.h"

namespace monoprop::detail {

// ─── Chunked-parallel helpers (order-preserving, sort-free) ────────────────────
// Split [0, n) into disjoint ascending chunks, process each into its own output slot in parallel,
// then concatenate in chunk order. Operator uniqueness + the XOR involution put each term in exactly
// one chunk (and one of leaders/followers), so the concatenation is globally sorted with neither
// dedup nor a comparison sort, and is deterministic regardless of thread scheduling.
//
// TWO PARALLEL TOOLKITS, DIFFERENT CONTRACTS — pick by whether output ORDER must be thread-invariant:
//   • These chunked helpers (for_each_chunk / append_gathered_chunks / append_parts_in_order): produce
//     a result whose byte layout is INDEPENDENT of thread count — the bit-exactness / thread-invariance
//     guarantee the build path (and the deterministic replay) rely on. Use whenever a parallel loop
//     BUILDS an ordered output (query streams, partner entries, miss inserts).
//   • monoprop/Threading.h (parallel_for_* / parallel_reduce_*): grain-scheduled work with the profiler
//     wired in, for loops that WRITE INTO PRE-SIZED disjoint slots or reduce commutatively — order does
//     not affect the result. Use for scatter/apply/inner-product style loops.
// (PareGraph.cpp::filter_layer_cosine_data open-codes this same chunk-and-concat pattern on a raw
// threading::run_static because it also AND-reduces a per-chunk `preserves` flag alongside the concat.)

// Shared chunk-count policy for both dimensions below: 0 for empty; 1 (serial) on a single worker or
// below the parallel floor; otherwise ~n/per_chunk chunks, capped at cap_per_worker × workers. The
// two callers differ ONLY in their three tuning constants, kept as named constexprs on the wrappers
// where the rationale that picks them lives.
inline auto chunk_count(size_t n, size_t min_parallel, size_t cap_per_worker, size_t per_chunk) -> size_t {
    if (n == 0) {
        return 0;
    }
    const size_t p = effective_parallelism();
    if (p <= 1 || n < min_parallel) {
        return 1;
    }
    return std::min<size_t>(p * cap_per_worker, std::max<size_t>(1, n / per_chunk));
}

// Index-space chunk count (resolve_self_queries etc.): ~256 elements/chunk, capped at 16× workers,
// serial below kMinParallelQueries or on one thread. The consumer probes the operator index with
// independent DRAM-latency-bound lookups, so fine chunks overlap more of them — but only once the
// index is large enough. Below the floor the query batch is cache-resident and parallelizing it is
// pure overhead (and steals threads from the serial scan of the next gate); the floor keeps it serial.
inline constexpr size_t kQueryChunkDivisor = 256;     // target elements per chunk
inline constexpr size_t kMinParallelQueries = 4096;   // run serial below this many queries
inline constexpr size_t kQueryChunkCapPerWorker = 16; // max chunks = cap × workers
inline auto partition_chunk_count(size_t n) -> size_t {
    return chunk_count(n, kMinParallelQueries, kQueryChunkCapPerWorker, kQueryChunkDivisor);
}

// Word-space chunk count for the inverted index XOR-column scan (fused_find_and_collect): the parallel
// dimension is the operator's word count = ceil(terms/64). Each word is ~|G| XOR/popcount ops over 64
// terms; target ~256 words/chunk, capped at 4× workers. Serial below kMinParallelWords — a floor set
// so small per-gate folds (where parallelism only adds dispatch overhead) stay serial, while larger
// folds, where parallelism pays, run chunked.
inline constexpr size_t kScanWordsPerChunk = 256;
inline constexpr size_t kMinParallelWords = 4000;
// Per-worker chunk cap for the find-scan word chunking (max chunks = cap × workers). 16 is the measured
// optimum on large Hubbard (c8/16T 208.2s→202.1s over a cap of 4): finer chunks cut the density-imbalance
// tail. The kScanWordsPerChunk (256 words/chunk) floor is UNAFFECTED — chunk_count independently caps the
// count at word_count/256, so this only refines chunks up to that floor. Small operators are untouched:
// the kMinParallelWords (4000) serial floor short-circuits chunk_count before the cap is ever read.
// Determinism is the chunk-order concat of disjoint word ranges, independent of the chunk count.
inline constexpr size_t kScanChunkCapPerWorker = 16;
inline auto partition_chunk_count_words(size_t word_count) -> size_t {
    return chunk_count(word_count, kMinParallelWords, kScanChunkCapPerWorker, kScanWordsPerChunk);
}

// Run body(chunk_idx, lo, hi) over `chunks` contiguous sub-ranges of [0, n) in parallel.
// One pool task per chunk, so each writes a distinct slot.
// (The adaptive gate-mode controller never reaches here with chunks > 1: under its serial override
// effective_parallelism() reports 1, so every chunk_count() policy already returned 1.)
template <typename Body>
inline auto for_each_chunk(size_t n, size_t chunks, Body &&body) -> void {
    if (n == 0 || chunks == 0) {
        return;
    }
    const profiling::Region prof_r = profiling::capture();
    if (chunks == 1) {
        profiling::TaskScope prof_ts(prof_r);
        body(size_t{0}, size_t{0}, n);
        return;
    }
    const size_t per = (n + chunks - 1) / chunks;
    threading::run_static(chunks, [&, prof_r](size_t c) {
        profiling::TaskScope prof_ts(prof_r);
        const size_t lo = c * per;
        if (lo >= n) {
            return;
        }
        body(c, lo, std::min(n, lo + per));
    });
}

// ── Order-preserving parallel gather core ──────────────────────────────────────
// Append `n_parts` per-chunk vectors (part_at(c) -> std::vector<T>&) onto `dst` in chunk order:
// chunk c lands at [base + prefix(c), base + prefix(c+1)). Deterministic and byte-identical
// regardless of thread count (unlike a per-THREAD merge). Each part is freed as consumed. Large totals
// scatter one task per chunk (sole writer per slice, no atomics); small/serial use one append pass.
template <typename Vec, typename PartAt>
inline auto append_parts_in_order(Vec &dst, size_t n_parts, PartAt &&part_at) -> void {
    if (n_parts == 0) {
        return;
    }
    std::vector<size_t> offsets(n_parts + 1, 0);
    for (size_t c = 0; c < n_parts; ++c) {
        offsets[c + 1] = offsets[c] + part_at(c).size();
    }
    const size_t total = offsets[n_parts];
    if (total == 0) {
        return;
    }
    // Single-chunk (serial-pass) fast path: steal the lone buffer outright when dst is empty.
    if (dst.empty() && n_parts == 1) {
        dst = std::move(part_at(0));
        Vec{}.swap(part_at(0));
        return;
    }
    // Small or single-threaded: one serial append pass (cheaper than spawning tasks).
    if (effective_parallelism() <= 1 || total < 4096) {
        dst.reserve(dst.size() + total);
        for (size_t c = 0; c < n_parts; ++c) {
            auto &part = part_at(c);
            dst.insert(dst.end(), part.begin(), part.end());
            Vec{}.swap(part);
        }
        return;
    }
    // Large: preallocate, then scatter each chunk into its disjoint slice in parallel.
    const size_t base = dst.size();
    dst.resize(base + total);
    const profiling::Region prof_r = profiling::capture();
    threading::run_static(n_parts, [&, prof_r](size_t c) {
        profiling::TaskScope prof_ts(prof_r);
        auto &part = part_at(c);
        std::copy(part.begin(), part.end(), dst.begin() + static_cast<std::ptrdiff_t>(base + offsets[c]));
        Vec{}.swap(part);
    });
}

// Append per-chunk vectors onto an existing destination in chunk order (frees inputs). Used by phases
// that accumulate across multiple passes (e.g. leader then follower) where replacing dst is not possible.
template <typename Vec>
inline auto append_gathered_chunks(Vec &dst, std::vector<Vec> &parts) -> void {
    append_parts_in_order(dst, parts.size(), [&](size_t c) -> Vec & { return parts[c]; });
}

// Append chunk-local per-rank vectors into per-rank destinations: for each rank r, the chunks
// chunk_by_rank[*][r] are appended onto dst_by_rank[r] in chunk order.
template <typename Vec>
inline auto append_chunked_rank_vectors(std::vector<Vec> &dst_by_rank, std::vector<std::vector<Vec>> &chunk_by_rank)
    -> void {
    const size_t chunks = chunk_by_rank.size();
    const size_t rank_count = dst_by_rank.size();
    if (chunks == 0 || rank_count == 0) {
        return;
    }
    for (size_t r = 0; r < rank_count; ++r) {
        append_parts_in_order(dst_by_rank[r], chunks, [&](size_t c) -> Vec & { return chunk_by_rank[c][r]; });
    }
}

} // namespace monoprop::detail
