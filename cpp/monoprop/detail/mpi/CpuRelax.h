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

#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace monoprop::mpi::detail {

// One iteration of a polite busy-wait: a pause-class hint, off the syscall path, while a sibling stores.
inline auto cpu_relax() noexcept -> void {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

// How long a barrier spinner stays on-core before it starts yielding, in TIME rather than in cpu_relax()
// iterations. An iteration count does not carry across architectures: cpu_relax() is one PAUSE on x86 and
// one YIELD on aarch64, and those differ by more than an order of magnitude, so a count calibrated on one
// spends a wholly different budget on the other. On aarch64 the previous 2048-iteration budget came to
// roughly a microsecond -- the on-core spin was effectively skipped and every wait fell to sched_yield.
//
// 30 us is that old budget's equivalent on this Zen 2 part, measured rather than derived: a barrier stress
// at S=32 over 4 and 8 CPUs reproduces the old wall time at 30 us (0.130 s vs 0.153 s, 0.136 s vs 0.228 s)
// and drifts off it by 60 us. PAUSE here is therefore ~30 cycles, not the ~140 of Sapphire Rapids the old
// constant's comment was reasoning about.
//
// Which means one honest caveat: a single time budget cannot match the old spend on every x86 part,
// because the old spend itself varied with PAUSE latency -- that variance was the bug. On a part where
// PAUSE really is ~140 cycles the old constant bought ~100 us, so 30 us is a genuine shortening there, not
// a like-for-like port. It is calibrated where it was measured and where the benchmarks run; elsewhere
// monoprop_SPIN_BUDGET_US is the adjustment, and a part with a long PAUSE is the first place to reach for
// it.
//
// Retuning is a separate question, deliberately left to the cluster A/B: the same sweep has shorter
// budgets winning by 5-10x once partitions outnumber cores (10 us gives 0.028 s and 0.022 s), because a
// spinner holds the core its late peer needs to arrive on. Under the pinned one-partition-per-core layout
// the design targets, the trade runs the other way and staying on-core is the point --
// monoprop_SPIN_BUDGET_US exists so that can be swept on real workloads without a rebuild.
//
// Parking (sleep_for) past a second, longer budget was tried and rejected: at S=32 on 4 CPUs one
// sleeper's timer overshoot pushed its peers past their own budgets, and the cascade cost 2021 us/sync
// against 762 for plain yielding. A yielder is runnable the moment a core frees; a sleeper cannot answer
// before its timer expires.
inline constexpr int kDefaultSpinBudgetUs = 30;

// cpu_relax() calls between deadline checks. steady_clock::now() costs ~20 ns, which would dominate the
// pause it is meant to pace, so the check is amortized over a batch.
inline constexpr int kRelaxPerClockCheck = 64;

} // namespace monoprop::mpi::detail
