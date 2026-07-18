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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// CPU-topology helpers for shard placement. Phase-0 result: the best config is one single-threaded
// shard per PHYSICAL CORE, and small shard counts should spread across L3 (CCX) domains so each shard
// owns a distinct last-level cache. This header parses /sys to build that placement and pins a shard
// master to its core. All functions degrade gracefully: on a non-Linux host or an unreadable /sys they
// return an empty placement / no-op pin, and the caller simply runs the shards unpinned (still correct,
// just without the locality win).

namespace monoprop::detail::shard {

// ─── portable /sys parsing (compiles everywhere; returns empty off Linux) ────────

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

} // namespace topo_detail

/// One physical core: a representative hardware-thread id to pin to, and its L3-domain id.
struct PhysicalCore {
    int cpu = 0;      // representative hardware thread (the core's first SMT sibling)
    int l3_domain = 0; // index of the shared-L3 group this core belongs to
};

/// Enumerate physical cores (one entry per SMT sibling group), each tagged with its L3 domain.
/// Empty if /sys cannot be read.
inline auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
    std::vector<PhysicalCore> cores;
    std::set<int> seen_cores;                 // representative cpu of each SMT group already taken
    std::vector<std::vector<int>> l3_members; // cpu-set per distinct L3 domain, in discovery order

    for (int cpu = 0;; ++cpu) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const std::string sib = topo_detail::read_line(base + "/topology/thread_siblings_list");
        if (sib.empty()) {
            break; // no more CPUs
        }
        const auto siblings = topo_detail::parse_cpulist(sib);
        const int rep = siblings.empty() ? cpu : *std::min_element(siblings.begin(), siblings.end());
        if (seen_cores.contains(rep)) {
            continue; // already recorded this physical core via another sibling
        }
        seen_cores.insert(rep);

        const auto l3 = topo_detail::parse_cpulist(topo_detail::read_line(base + "/cache/index3/shared_cpu_list"));
        int domain = -1;
        for (size_t d = 0; d < l3_members.size(); ++d) {
            if (std::find(l3_members[d].begin(), l3_members[d].end(), rep) != l3_members[d].end()) {
                domain = static_cast<int>(d);
                break;
            }
        }
        if (domain < 0) {
            domain = static_cast<int>(l3_members.size());
            l3_members.push_back(l3.empty() ? std::vector<int>{rep} : l3);
        }
        cores.push_back(PhysicalCore{rep, domain});
    }
    return cores;
}

// ─── Linux-only pinning (stubbed to no-ops elsewhere) ────────────────────────────

#if defined(__linux__)

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
inline auto shard_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1) -> std::vector<cpu_set_t> {
    // monoprop_SHARD_PINNING=0/false/n disables pinning (shards then run unpinned — still correct).
    if (const char *e = std::getenv("monoprop_SHARD_PINNING")) {
        const char c = e[0];
        if (c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N') {
            return {};
        }
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

    std::vector<cpu_set_t> sets(n);
    for (size_t i = 0; i < n; ++i) {
        CPU_ZERO(&sets[i]);
        CPU_SET(order[offset + i], &sets[i]);
    }
    return sets;
}

/// Pin the calling thread to `set`. No-op-safe: a failing pthread call is ignored (correctness does
/// not depend on pinning, only performance).
inline auto pin_this_thread(const cpu_set_t &set) -> void {
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
}

#else // non-Linux: no topology, no pinning

struct cpu_set_t_stub {};
using cpu_set_t = cpu_set_t_stub;
inline auto shard_cpusets(size_t /*n*/, size_t /*group_index*/ = 0, size_t /*group_count*/ = 1)
    -> std::vector<cpu_set_t> {
    return {};
}
inline auto pin_this_thread(const cpu_set_t & /*set*/) -> void {}

#endif // __linux__

} // namespace monoprop::detail::shard
