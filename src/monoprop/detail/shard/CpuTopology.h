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

#include "monoprop/detail/EnvConfig.h" // config::get().shard_pinning

#if defined(__linux__)
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// CPU-topology helpers for shard placement. Phase-0 result: the best config is one single-threaded
// shard per PHYSICAL CORE, and small shard counts should spread across L3 (CCX) domains so each shard
// owns a distinct last-level cache.
//
// This is the ONE platform-specific file in the engine. The Linux fast path parses /sys to build that
// placement and pins each shard master to its core (intersected with the process's allowed-CPU mask,
// so a cgroup-restricted / partial Slurm allocation reports and pins only its own cores). Everything
// else runs the portable fallback: no topology, no pinning — the shards run unpinned, still correct,
// just without the locality win. macOS supplies an accurate physical-core COUNT (via sysctl) for the
// shard-count policy even though it cannot pin threads.

namespace monoprop::detail::shard {

/// One physical core: a representative hardware-thread id to pin to, and its L3-domain id.
struct PhysicalCore {
    int cpu = 0;       // representative hardware thread (an allowed SMT sibling of the core)
    int l3_domain = 0; // index of the shared-L3 group this core belongs to
};

#if defined(__linux__)

// On Linux a shard cpuset is a real affinity mask.
using CpuSet = cpu_set_t;

namespace topo_detail {

/// Parse a Linux cpulist ("0-3,16-19") into the set of CPU ids it names.
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

/// The CPUs this process/thread is allowed to run on (the cgroup / cpuset the launcher gave us).
/// Empty ⇒ the query failed; callers then treat every CPU as allowed.
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

/// Enumerate physical cores (one entry per SMT sibling group) whose CPUs the process is allowed to
/// use, each tagged with its L3 domain. A core is included iff at least one of its SMT siblings is in
/// the allowed mask, and its representative cpu is the smallest allowed sibling — so a partial
/// allocation yields exactly its own cores (never oversubscribing) and never pins outside the mask.
/// Empty if /sys cannot be read.
inline auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
    const std::set<int> allowed = topo_detail::allowed_cpus();
    const bool filter = !allowed.empty(); // no mask readable ⇒ accept every CPU
    const auto is_allowed = [&](int cpu) { return !filter || allowed.contains(cpu); };

    std::vector<PhysicalCore> cores;
    std::set<int> seen_cores;                 // sibling-group key (min sibling) already recorded
    std::vector<std::vector<int>> l3_members; // cpu-set per distinct L3 domain, in discovery order

    for (int cpu = 0;; ++cpu) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const std::string sib = topo_detail::read_line(base + "/topology/thread_siblings_list");
        if (sib.empty()) {
            break; // no more CPUs
        }
        const auto siblings = topo_detail::parse_cpulist(sib);
        const int group_key = siblings.empty() ? cpu : *std::min_element(siblings.begin(), siblings.end());
        if (seen_cores.contains(group_key)) {
            continue; // already recorded this physical core via another sibling
        }
        seen_cores.insert(group_key);

        // Pin target = smallest ALLOWED sibling; skip the core entirely if none of its siblings is ours.
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

/// Build `n` shard cpusets, one physical core each. `group_index`/`group_count` place the shards of
/// one MPI rank among `group_count` co-located ranks sharing this host (group_count == 1: the
/// single-process case). Placement:
///   - group_count == 1: cores ordered round-robin across L3 domains, so a small shard count spreads
///     over all caches (domain0 core0, domain1 core0, …, then core1s).
///   - group_count > 1: whole L3 domains are dealt to the co-located ranks round-robin and each rank
///     interleaves across its own domains (falling back to a flat domain-major slice when there are
///     more ranks than domains), so ranks get disjoint cores and maximally disjoint caches. Two ranks
///     must never share a core: MPI's busy-polling collectives on one rank would contend with the
///     other rank's barrier spins for the same timeslices, degrading lock-step progress
///     catastrophically.
/// If the host cannot supply group_count*n distinct physical cores, pinning is disabled (empty
/// vector ⇒ shards run unpinned; the OS spreads them — still correct, and better than doubling up).
inline auto shard_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1) -> std::vector<CpuSet> {
    // monoprop_SHARD_PINNING=0/false/n disables pinning (shards then run unpinned — still correct).
    if (!config::get().shard_pinning) {
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
    // Bucket cores by domain, then order: interleaved across domains for a lone process, contiguous
    // per domain when co-located ranks each take a block.
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
        // This rank's cores: interleaved across the domains dealt to it (domains group_index,
        // group_index + group_count, …). group_count == 1 degenerates to the all-domain interleave.
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

/// Pin the calling thread to `set`. No-op-safe: a failing pthread call is ignored (correctness does
/// not depend on pinning, only performance).
inline auto pin_this_thread(const CpuSet &set) -> void {
    pthread_setaffinity_np(pthread_self(), sizeof(CpuSet), &set);
}

#else // ─── portable fallback: no topology, no pinning ───────────────────────────

// A placeholder cpuset type so ShardGroup's member/signatures are platform-independent.
struct CpuSet {};

/// No /sys to parse. macOS reports its physical-core COUNT so the shard-count policy is accurate even
/// though threads cannot be pinned; every other platform returns empty (⇒ the shard-count policy falls
/// back to std::thread::hardware_concurrency()/2). All returned cores carry a placeholder cpu/domain —
/// they are only counted, never pinned.
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

inline auto shard_cpusets(size_t /*n*/, size_t /*group_index*/ = 0, size_t /*group_count*/ = 1)
    -> std::vector<CpuSet> {
    return {};
}
inline auto pin_this_thread(const CpuSet & /*set*/) -> void {}

#endif // __linux__

} // namespace monoprop::detail::shard
