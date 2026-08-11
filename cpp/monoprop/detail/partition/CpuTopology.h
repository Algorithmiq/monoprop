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
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <cstdlib>
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
inline auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
    const std::set<int> allowed = topo_detail::allowed_cpus();
    const bool filter = !allowed.empty(); // no mask readable ⇒ accept every CPU
    const auto is_allowed = [&](int cpu) { return !filter || allowed.contains(cpu); };

    std::vector<PhysicalCore> cores;
    std::set<int> seen_cores;                 // sibling-group key (min sibling) already recorded
    std::vector<std::vector<int>> l3_members; // cpu-set per distinct L3 domain, in discovery order

    // Scan a bounded id range rather than stopping at the first gap: online CPU ids are not contiguous
    // (offlined or hot-plugged CPUs leave holes), and breaking on the first unreadable id truncates the
    // core list to whatever preceded the hole, silently under-partitioning and crowding the low CPUs.
    const int scan_limit = filter ? *allowed.rbegin() + 1 : CPU_SETSIZE;
    for (int cpu = 0; cpu < scan_limit; ++cpu) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const std::string sib = topo_detail::read_line(base + "/topology/thread_siblings_list");
        if (sib.empty()) {
            continue;
        }
        const auto siblings = topo_detail::parse_cpulist(sib);
        const int group_key = siblings.empty() ? cpu : *std::min_element(siblings.begin(), siblings.end());
        if (seen_cores.contains(group_key)) {
            continue;
        }
        seen_cores.insert(group_key);

        int rep = -1;
        if (siblings.empty()) {
            rep = is_allowed(cpu) ? cpu : -1;
        }
        else {
            for (int s : siblings) { // parse_cpulist yields ascending order
                if (is_allowed(s)) {
                    rep = s;
                    break;
                }
            }
        }
        if (rep < 0) {
            continue;
        }

        const auto l3 = topo_detail::parse_cpulist(topo_detail::read_line(base + "/cache/index3/shared_cpu_list"));
        int domain = -1;
        for (size_t d = 0; d < l3_members.size(); ++d) {
            if (std::find(l3_members[d].begin(), l3_members[d].end(), group_key) != l3_members[d].end()) {
                domain = static_cast<int>(d);
                break;
            }
        }
        if (domain < 0) {
            domain = static_cast<int>(l3_members.size());
            l3_members.push_back(l3.empty() ? std::vector<int>{group_key} : l3);
        }
        cores.push_back(PhysicalCore{rep, domain});
    }
    return cores;
}

// Build `n` partition cpusets, one physical core each. `group_index`/`group_count` place one MPI rank's
// partitions among the ranks sharing this host, spread across L3 domains and disjoint from the other ranks'
// — two ranks must never share a core (one rank's busy-polling collectives would starve the other's
// barrier spins). Empty (⇒ unpinned) if the host lacks group_count*n cores.
inline auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1) -> std::vector<CpuSet> {
    if (!config::get().partition_pinning) {
        return {};
    }
    const auto cores = enumerate_physical_cores();
    if (cores.empty() || group_count * n > cores.size()) {
        return {};
    }
    int max_domain = 0;
    for (const auto &c : cores) {
        max_domain = std::max(max_domain, c.l3_domain);
    }
    // Ordering: interleaved for a lone process, contiguous blocks for co-located ranks.
    std::vector<std::vector<int>> by_domain(static_cast<size_t>(max_domain) + 1);
    for (const auto &c : cores) {
        by_domain[static_cast<size_t>(c.l3_domain)].push_back(c.cpu);
    }
    // Interleave `buckets` depth-first: bucket0[0], bucket1[0], …, bucket0[1], bucket1[1], …
    const auto interleave = [](const std::vector<std::vector<int>> &buckets) {
        std::vector<int> out;
        for (size_t depth = 0;; ++depth) {
            bool any = false;
            for (const auto &bucket : buckets) {
                if (depth < bucket.size()) {
                    out.push_back(bucket[depth]);
                    any = true;
                }
            }
            if (!any) {
                return out;
            }
        }
    };

    std::vector<int> order;
    size_t offset = 0;
    if (group_count <= by_domain.size()) {
        // Domains dealt to this rank: group_index, +group_count, … (group_count == 1 ⇒ all of them).
        std::vector<std::vector<int>> mine;
        for (size_t d = group_index; d < by_domain.size(); d += group_count) {
            mine.push_back(by_domain[d]);
        }
        order = interleave(mine);
    }
    else {
        // More co-located ranks than L3 domains: flat domain-major order, one contiguous slice each.
        for (const auto &bucket : by_domain) {
            order.insert(order.end(), bucket.begin(), bucket.end());
        }
        offset = group_index * n;
    }
    if (offset + n > order.size()) {
        return {};
    }

    std::vector<CpuSet> sets(n);
    for (size_t i = 0; i < n; ++i) {
        CPU_ZERO(&sets[i]);
        CPU_SET(order[offset + i], &sets[i]);
    }
    return sets;
}

// The L3 domain each partition cpuset lands in, in partition_cpusets order -- what a two-level
// PartitionBarrier groups by. Derived from the sets, not from the placement logic, so the two cannot
// drift apart. Empty (⇒ flat barrier) if the sets are empty or one names no core the scan knows.
inline auto cpuset_l3_domains(const std::vector<CpuSet> &sets) -> std::vector<int> {
    if (sets.empty()) {
        return {};
    }
    const auto cores = enumerate_physical_cores();
    std::vector<int> domains;
    domains.reserve(sets.size());
    for (const CpuSet &set : sets) {
        int found = -1;
        for (const auto &core : cores) {
            if (CPU_ISSET(core.cpu, &set)) {
                found = core.l3_domain;
                break;
            }
        }
        if (found < 0) {
            return {};
        }
        domains.push_back(found);
    }
    return domains;
}

// A failing pthread call is ignored: only performance depends on it.
inline auto pin_this_thread(const CpuSet &set) -> void {
    pthread_setaffinity_np(pthread_self(), sizeof(CpuSet), &set);
}

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
inline auto cpuset_l3_domains(const std::vector<CpuSet> & /*sets*/) -> std::vector<int> {
    return {};
}
inline auto pin_this_thread(const CpuSet & /*set*/) -> void {}

#endif // __linux__

} // namespace monoprop::detail::partition
