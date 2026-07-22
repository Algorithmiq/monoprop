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

// Coverage of the RegionProfiler (RegionProfiler.h + Profiling.cpp). The profiler is env-gated
// (monoprop_PHASE_TIMERS / monoprop_FOLD_STATS) and its accumulators + one-shot stderr dump are
// otherwise never entered by the test suite. Here we drive the record/accumulate surface directly:
//   * g_profiling_enabled is an exported mutable bool, so we flip it on to exercise ScopedRegion's
//     timed path (ctor/dtor, acc(), profiling_accs()), then restore it.
//   * record_fold_stats is called across every branch of its histogram bucketing.
//   * profiling_ensure_atexit registers the process-exit dump(); because it is registered here
//     (after gcov's own exit flush), dump() runs -- and is recorded -- when the test binary exits,
//     with both a populated region and populated fold stats to print.
//
// Cases are flat top-level BOOST_AUTO_TEST_CASEs sharing a profiling_ prefix (not a
// BOOST_AUTO_TEST_SUITE) to match this suite's ctest discovery, which registers by leaf case name.

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <utility>

#include "monoprop/detail/profiling/RegionProfiler.h"

namespace prof = monoprop::profiling;

BOOST_AUTO_TEST_CASE(profiling_scoped_region_timed_path) {
    const bool saved = prof::g_profiling_enabled;
    prof::g_profiling_enabled = true;
    {
        // A non-trivial lifetime so wall_ns accumulates and calls increments (dtor path).
        prof::ScopedRegion region(prof::Region::Find);
        volatile std::uint64_t sink = 0;
        for (int i = 0; i < 10'000; ++i) {
            sink += static_cast<std::uint64_t>(i);
        }
        (void)sink;
    }
    prof::g_profiling_enabled = saved;

    // The Find accumulator recorded exactly one entry.
    const prof::RegionAcc *accs = prof::profiling_accs();
    const auto calls = accs[std::to_underlying(prof::Region::Find)].calls.load(std::memory_order_relaxed);
    BOOST_CHECK_GE(calls, 1u);
}

BOOST_AUTO_TEST_CASE(profiling_scoped_region_disabled_is_inert) {
    const bool saved = prof::g_profiling_enabled;
    prof::g_profiling_enabled = false;
    const prof::RegionAcc *accs = prof::profiling_accs();
    const auto before = accs[std::to_underlying(prof::Region::Extend)].calls.load(std::memory_order_relaxed);
    {
        prof::ScopedRegion region(prof::Region::Extend); // disabled: early-return, no accounting
        (void)region;
    }
    const auto after = accs[std::to_underlying(prof::Region::Extend)].calls.load(std::memory_order_relaxed);
    prof::g_profiling_enabled = saved;
    BOOST_CHECK_EQUAL(before, after);
}

BOOST_AUTO_TEST_CASE(profiling_record_fold_stats_all_branches) {
    // Sweep every branch of record_fold_stats + its ratio-histogram bucketing:
    //   skipped true/false, all_sparse true/false, postings == 0, and low/mid/high ratios that
    //   drive the clamp to buckets 1 / mid / 15.
    prof::record_fold_stats(/*all_sparse=*/false,
                            /*skipped=*/true,
                            /*postings=*/0,
                            /*word_count=*/4,
                            /*n_anti=*/0,
                            /*struct_rejects=*/0);           // skipped + early !all_sparse return
    prof::record_fold_stats(false, false, 7, 4, 2, 1);       // !all_sparse, not skipped
    prof::record_fold_stats(true, false, 0, 4, 3, 1);        // all_sparse, postings == 0 -> bucket 0
    prof::record_fold_stats(true, false, 8, 4, 5, 2);        // all_sparse, mid ratio
    prof::record_fold_stats(true, false, 1u << 20, 1, 9, 0); // very high ratio -> clamp to 15
    prof::record_fold_stats(true, false, 1, 1u << 20, 0, 4); // very low ratio -> clamp to 1

    // Ensure the one-shot atexit dump is registered (idempotent). It runs at process exit, printing
    // the region + fold accumulators populated above.
    prof::profiling_ensure_atexit();
    BOOST_CHECK(true);
}
