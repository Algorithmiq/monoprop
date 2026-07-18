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

/// One iteration of a polite busy-wait: a PAUSE-class hint that keeps the core out of the memory
/// speculation machinery (and off the syscall path) while a sibling finishes its store.
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

// How many cpu_relax() iterations a barrier spinner burns before it starts donating its timeslice
// via sched_yield. PAUSE is ~140 cycles on Sapphire Rapids, so 2048 iterations is ~0.1 ms of
// on-core waiting — well past the inter-shard arrival gaps of a balanced exchange, so pinned
// production shards never syscall; oversubscribed runs (tests, CI) still degrade gracefully to
// yield so a spinner can't starve the completer of a core.
inline constexpr int kSpinPauseIters = 2048;

} // namespace monoprop::mpi::detail
