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

#include "monoprop/detail/EnvConfig.h"

#if defined(__linux__)
#include <sched.h>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// CPU-topology helpers for partition placement (the one platform-specific file). Policy: one partition per
// physical core, spread across L3/ccx domains. The Linux fast path parses /sys and pins each master,
// intersected with the process's allowed-CPU mask; elsewhere partitions run unpinned (still correct, no
// locality win).

namespace monoprop::detail::partition {

struct PhysicalCore {
    int cpu = 0; // representative hardware thread (an allowed SMT sibling of the core)
    int l3_domain = 0;
};

#if defined(__linux__)

using CpuSet = cpu_set_t;

namespace topo_detail {

// Parse a Linux cpulist ("0-3,16-19") into the set of CPU ids it names.
inline auto parse_cpulist(const std::string &text) -> std::vector<int> {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const auto dash = tok.find('-');
        if (dash == std::string::npos) {
            if (!tok.empty()) {
                out.push_back(std::stoi(tok));
            }
        }
        else {
            const int lo = std::stoi(tok.substr(0, dash));
            const int hi = std::stoi(tok.substr(dash + 1));
            for (int c = lo; c <= hi; ++c) {
                out.push_back(c);
            }
        }
    }
    return out;
}

inline auto read_line(const std::string &path) -> std::string {
    std::ifstream f(path);
    std::string line;
    if (f) {
        std::getline(f, line);
    }
    return line;
}

// The CPUs this process is allowed to run on (the cgroup / cpuset the launcher gave us). Empty ⇒ the
// query failed; callers then treat every CPU as allowed.
inline auto allowed_cpus() -> std::set<int> {
    std::set<int> allowed;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                allowed.insert(cpu);
            }
        }
    }
    return allowed;
}

} // namespace topo_detail

// Enumerate physical cores (one per smt sibling group) the process is allowed to use, tagged with their
// L3 domain. A core is included iff a sibling is in the allowed mask, with the smallest allowed sibling
// as representative, so a partial allocation never pins outside the mask. Empty if /sys cannot be read.
auto enumerate_physical_cores() -> std::vector<PhysicalCore>;

// Build `n` partition cpusets, one physical core each. `group_index`/`group_count` place one MPI rank's
// partitions among the ranks sharing this host, spread across L3 domains and disjoint from the other ranks'
// — two ranks must never share a core (one rank's busy-polling collectives would starve the other's
// barrier spins). Empty (⇒ unpinned) if the host lacks group_count*n cores.
auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1) -> std::vector<CpuSet>;

// A failing pthread call is ignored: only performance depends on it.
auto pin_this_thread(const CpuSet &set) -> void;

#else // portable fallback: no topology, no pinning

// A placeholder cpuset type so PartitionGroup's member/signatures are platform-independent.
struct CpuSet {};

// No /sys to parse. macOS reports its physical-core count so the partition-count policy stays accurate
// (threads still can't be pinned); other platforms return empty ⇒ hardware_concurrency()/2.
inline auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
#if defined(__APPLE__)
    int n = 0;
    size_t sz = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) {
        return std::vector<PhysicalCore>(static_cast<size_t>(n));
    }
#endif
    return {};
}

inline auto partition_cpusets(size_t /*n*/, size_t /*group_index*/ = 0, size_t /*group_count*/ = 1)
    -> std::vector<CpuSet> {
    return {};
}
inline auto pin_this_thread(const CpuSet & /*set*/) -> void {}

#endif // __linux__

} // namespace monoprop::detail::partition
