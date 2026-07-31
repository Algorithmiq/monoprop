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
#include <vector>

#include "monoprop/monopropExport.h"

// CPU-topology helpers for partition placement. Policy: one partition per physical core, spread across
// L3/CCX domains, using only the CPUs available to this process.

namespace monoprop::detail::partition {

struct PhysicalCore {
    unsigned cpu = 0; // representative hardware thread (an allowed SMT sibling of the core)
    int l3_domain = 0;
};

struct CpuSet {
    unsigned cpu = 0;
};

monoprop_EXPORT auto enumerate_physical_cores() -> std::vector<PhysicalCore>;

// Build `n` placements, one physical core each. Co-located MPI ranks receive disjoint cores.
// Empty means that pinning is disabled, unsupported, or would oversubscribe the available cores.
monoprop_EXPORT auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1) -> std::vector<CpuSet>;

// A binding failure is ignored because only performance depends on pinning.
monoprop_EXPORT auto pin_this_thread(const CpuSet &set) -> void;

} // namespace monoprop::detail::partition
