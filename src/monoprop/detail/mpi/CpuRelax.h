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

// One iteration of a polite busy-wait: a PAUSE-class hint, off the syscall path, while a sibling
// finishes its store.
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

// cpu_relax() iterations a barrier spinner burns before donating its timeslice. PAUSE ~140 cycles on
// Sapphire Rapids ⇒ 2048 iters ≈ 0.1 ms, past a balanced exchange's arrival gaps; longer waits yield.
inline constexpr int kSpinPauseIters = 2048;

} // namespace monoprop::mpi::detail
