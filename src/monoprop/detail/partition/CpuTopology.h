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

/**
 * @file
 * CPU-topology helpers for placing one partition per physical core, spread across L3/CCX domains and
 * restricted to the CPUs available to the process.
 */

#include <cstddef>
#include <vector>

#include "monoprop/monopropExport.h"

namespace monoprop::detail::partition {
/// A physical core represented by one allowed hardware thread and its L3/CCX domain.
struct PhysicalCore {
    /// OS index of an allowed SMT sibling representing this core.
    unsigned cpu = 0;
    /// Contiguous index of the L3/CCX domain containing this core.
    int l3_domain = 0;
};

/// A single-CPU partition placement represented by its OS CPU index.
struct CpuSet {
    /// OS index of the CPU to which the partition thread should bind.
    unsigned cpu = 0;
};

/**
 * Enumerates one allowed hardware thread per physical core, grouped by L3/CCX domain.
 *
 * @return The available physical cores and their topology domains, or an empty vector when topology
 * discovery fails.
 */
monoprop_EXPORT auto enumerate_physical_cores() -> std::vector<PhysicalCore>;

/**
 * Builds one placement per partition, assigning co-located MPI ranks disjoint physical cores.
 *
 * @param n Number of partition placements to build for this process.
 * @param group_index Zero-based index of this process among the co-located process group.
 * @param group_count Number of co-located processes sharing the available physical cores.
 * @return One placement per partition, or an empty vector when pinning is disabled, unsupported, or
 * would oversubscribe the available cores.
 */
monoprop_EXPORT auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1) -> std::vector<CpuSet>;

/**
 * Binds the current thread to a placement, ignoring failures because pinning only affects performance.
 *
 * @param set Single-CPU placement to apply to the current thread.
 */
monoprop_EXPORT auto pin_this_thread(const CpuSet &set) -> void;
} // namespace monoprop::detail::partition
