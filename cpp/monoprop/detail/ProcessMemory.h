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

#include <cstddef>

namespace monoprop::detail {

/// What the kernel and the allocator report, as opposed to what a byte ledger estimates.
///
/// Every field is PER PROCESS: report once, never summed over partitions; over RANKS it is a job total.
/// `alloc_retained_bytes` is what a ledger cannot reach -- freed chunks stay faulted and stay resident.
struct ProcessMemory {
    size_t rss_bytes{0uz};            ///< /proc/self/status VmRSS: pages faulted in right now.
    size_t peak_rss_bytes{0uz};       ///< VmHWM: the kernel's peak over the process's life.
    size_t alloc_in_use_bytes{0uz};   ///< malloc(3) chunks handed out and not yet freed.
    size_t alloc_retained_bytes{0uz}; ///< Freed chunks the allocator still holds.
    size_t alloc_system_bytes{0uz};   ///< What the allocator has taken from the kernel.
    size_t alloc_arenas{0uz};         ///< Arena count (MALLOC_ARENA_MAX bounds it).
};

/// Zero-filled where the platform cannot answer, and never throws: a diagnostic must not fail its caller.
auto process_memory() noexcept -> ProcessMemory;

} // namespace monoprop::detail
