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
#include <optional>
#include <vector>

#include "monoprop/detail/EnvConfig.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// CPU-topology helpers for partition placement (the one platform-specific file). Policy: one partition per
// physical core, spread across locality domains. The Linux fast path parses /sys and pins each master,
// intersected with the process's allowed-CPU mask; elsewhere partitions run unpinned (still correct, no
// locality win).

namespace monoprop::detail::partition {

struct PhysicalCore {
    int cpu = 0; // representative hardware thread (an allowed SMT sibling of the core)
    int domain = 0;
};

// How the launcher divided this host's CPUs among the ranks sharing it. Mask size alone cannot tell the
// two apart -- four ranks holding 32 cores each and four ranks sharing one 32-core mask both report 32 --
// and they need opposite placement, so the caller that owns a node-local communicator measures this by
// exchanging masks rather than inferring it. See partition_cpusets and classify_node_mask.
enum class NodeMask {
    Shared,  ///< Every co-located rank sees the same mask (`mpirun --bind-to none`, or a lone process).
    PerRank, ///< Each co-located rank was given its own disjoint slice (`srun --cpu-bind=cores`).
};

#if defined(__linux__)

using CpuSet = cpu_set_t;

namespace topo_detail {

// Widest range a single cpulist token may expand to. Larger than any real machine, small enough that a
// corrupt token cannot turn into an allocation.
inline constexpr int kMaxCpuListSpan = 1 << 20;

// One non-negative integer, or nullopt when `text` is not exactly that. strtol rather than stoi because
// every caller here reads /sys, which a container's masked view or a truncated read can leave malformed:
// this path's contract is to degrade to "unpinned", never to throw out of a constructor.
inline auto parse_id(const std::string &text) -> std::optional<int> {
    if (text.empty()) {
        return std::nullopt;
    }
    const char *begin = text.c_str();
    char *end = nullptr;
    const long value = std::strtol(begin, &end, 10);
    // Deliberately not bounded by CPU_SETSIZE: this parses cache levels and NUMA node ids as well as CPU
    // ids, and a host with more than CPU_SETSIZE CPUs would otherwise have every cpulist it reports --
    // including /sys/.../cpu/online -- collapse to empty, silently disabling both pinning and the
    // two-level barrier. Ids too large to place are dropped where CPU_SET is called instead.
    if (end == begin || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

// Parse a Linux cpulist ("0-3,16-19") into the set of CPU ids it names. Malformed tokens are skipped
// rather than thrown on; see parse_id.
inline auto parse_cpulist(const std::string &text) -> std::vector<int> {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const auto dash = tok.find('-');
        if (dash == std::string::npos) {
            if (const auto one = parse_id(tok)) {
                out.push_back(*one);
            }
            continue;
        }
        const auto lo = parse_id(tok.substr(0, dash));
        const auto hi = parse_id(tok.substr(dash + 1));
        // A reversed range names nothing, and an absurdly wide one is garbage rather than a machine: with
        // parse_id no longer clamped to CPU_SETSIZE, expanding it unchecked would allocate on the strength
        // of a malformed /sys read.
        if (!lo || !hi || *hi < *lo || *hi - *lo > kMaxCpuListSpan) {
            continue;
        }
        for (int c = *lo; c <= *hi; ++c) {
            out.push_back(c);
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

// The CPUs sharing the deepest cache level that groups `cpu_base`'s core with another core, else the CPUs
// of its NUMA node, else empty. `siblings` is the core's SMT sibling list (empty ⇒ just `cpu`).
//
// Deliberately not keyed on index3/L3, and the NUMA fallback is not a nicety. Measured on Deucalion's
// A64FX nodes: /sys/devices/system/cpu/cpuN/cache does not exist at all -- no level is reported, not just
// no L3 -- so the cache walk finds nothing and the NUMA node is the only signal left. Keying on L3 made
// every core its own domain there, which turns the two-level barrier into a flat one carrying S extra
// cache lines while barrier_groups still reports S, reading as if the optimization had engaged. On x86 the
// deepest shared level *is* L3, so this keeps the CCX grouping the current placement was measured against.
inline auto shared_domain_cpus(const std::string &cpu_base, int cpu, const std::vector<int> &siblings)
    -> std::vector<int> {
    const std::vector<int> own = siblings.empty() ? std::vector<int>{cpu} : siblings;
    std::vector<int> deepest;
    int deepest_level = -1;
    // Cache indices are contiguous, so the first unreadable one ends the list.
    for (int idx = 0;; ++idx) {
        const std::string dir = cpu_base + "/cache/index" + std::to_string(idx);
        const auto level = parse_id(read_line(dir + "/level"));
        if (!level) {
            break;
        }
        const auto members = parse_cpulist(read_line(dir + "/shared_cpu_list"));
        // "Shared" has to mean shared with another *core*, not with our own SMT sibling: a per-core L1 or
        // L2 still lists every hardware thread of the core, so a size>=2 test would accept it, make each
        // core its own domain, and never reach the NUMA fallback. That would reintroduce the exact defect
        // this function exists to fix on any SMT part that exposes no shared cache.
        const bool groups_another_core = std::any_of(members.begin(), members.end(), [&](int m) {
            return std::find(own.begin(), own.end(), m) == own.end();
        });
        if (!groups_another_core) {
            continue;
        }
        if (*level > deepest_level) {
            deepest_level = *level;
            deepest = members;
        }
    }
    if (!deepest.empty()) {
        return deepest;
    }
    // No shared cache reported. The NUMA node is the next-coarsest thing a barrier can stay inside.
    for (int node : parse_cpulist(read_line("/sys/devices/system/node/possible"))) {
        const auto cpus = parse_cpulist(read_line("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist"));
        if (std::find(cpus.begin(), cpus.end(), cpu) != cpus.end()) {
            return cpus;
        }
    }
    return {};
}

} // namespace topo_detail

// Enumerate physical cores (one per smt sibling group) the process is allowed to use, tagged with the
// locality domain they share (deepest shared cache, else NUMA node; see shared_domain_cpus). A core is
// included iff a sibling is in the allowed mask, with the smallest allowed sibling as representative, so
// a partial allocation never pins outside the mask. Empty if /sys cannot be read.
inline auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
    const std::set<int> allowed = topo_detail::allowed_cpus();
    const bool filter = !allowed.empty(); // no mask readable ⇒ accept every CPU
    const auto is_allowed = [&](int cpu) { return !filter || allowed.contains(cpu); };

    std::vector<PhysicalCore> cores;
    std::set<int> seen_cores;                     // sibling-group key (min sibling) already recorded
    std::vector<std::vector<int>> domain_members; // cpu-set per distinct domain, in discovery order

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

        const auto shared = topo_detail::shared_domain_cpus(base, cpu, siblings);
        int domain = -1;
        for (size_t d = 0; d < domain_members.size(); ++d) {
            if (std::find(domain_members[d].begin(), domain_members[d].end(), group_key) != domain_members[d].end()) {
                domain = static_cast<int>(d);
                break;
            }
        }
        if (domain < 0) {
            domain = static_cast<int>(domain_members.size());
            domain_members.push_back(shared.empty() ? std::vector<int>{group_key} : shared);
        }
        cores.push_back(PhysicalCore{rep, domain});
    }
    return cores;
}

// This process's current affinity mask, for a caller that needs to compare it against its peers'. An
// unreadable mask comes back empty, which classify_node_mask reads as "cannot tell" ⇒ Shared.
inline auto this_thread_cpuset() -> CpuSet {
    CpuSet set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        CPU_ZERO(&set);
    }
    return set;
}

// Classify how the launcher divided the host, given every co-located rank's mask. The caller gathers them
// because it owns the node-local communicator and this header stays free of MPI.
//
// Pairwise disjoint and non-empty ⇒ PerRank. Anything else -- identical masks, partial overlap, or a mask
// that could not be read -- ⇒ Shared, which is the conservative answer; see partition_cpusets.
inline auto classify_node_mask(const std::vector<CpuSet> &masks) -> NodeMask {
    if (masks.size() < 2) {
        return NodeMask::Shared; // nobody to be disjoint from
    }
    for (size_t a = 0; a < masks.size(); ++a) {
        if (CPU_COUNT(&masks[a]) == 0) {
            return NodeMask::Shared;
        }
        for (size_t b = a + 1; b < masks.size(); ++b) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &masks[a]) && CPU_ISSET(c, &masks[b])) {
                    return NodeMask::Shared; // shares a CPU ⇒ not a per-rank split
                }
            }
        }
    }
    return NodeMask::PerRank;
}

// Build `n` partition cpusets, one physical core each. `group_index`/`group_count` place one MPI rank's
// partitions among the ranks sharing this host, spread across locality domains and disjoint from the other ranks'
// — two ranks must never share a core (one rank's busy-polling collectives would starve the other's
// barrier spins). Empty (⇒ unpinned) if the host lacks group_count*n cores.
inline auto partition_cpusets(size_t n,
                              size_t group_index = 0,
                              size_t group_count = 1,
                              NodeMask mask = NodeMask::Shared) -> std::vector<CpuSet> {
    if (!config::get().partition_pinning) {
        return {};
    }
    // enumerate_physical_cores() reports only cores inside this process's mask, so under a PerRank mask the
    // launcher has already handed each co-located rank a disjoint slice, and dividing by group_count a
    // second time splits an already-split machine: group_count * n exceeds the share, the guard below reads
    // that as "host too small", and every rank silently runs unpinned -- taking the two-level barrier's
    // domains with it, because cpuset_domains() derives them from these sets. Partition the whole slice and
    // leave the cross-rank split to whoever imposed it.
    //
    // The default is Shared because that is the pre-existing behaviour and the safe error direction:
    // collapsing a genuinely shared mask would point every co-located rank at the *same* cores, breaking
    // the disjointness invariant above, whereas dividing a PerRank mask only costs pinning.
    if (mask == NodeMask::PerRank) {
        group_index = 0;
        group_count = 1;
    }
    const auto cores = enumerate_physical_cores();
    if (cores.empty() || group_count * n > cores.size()) {
        return {};
    }
    int max_domain = 0;
    for (const auto &c : cores) {
        max_domain = std::max(max_domain, c.domain);
    }
    // Ordering: interleaved for a lone process, contiguous blocks for co-located ranks.
    std::vector<std::vector<int>> by_domain(static_cast<size_t>(max_domain) + 1);
    for (const auto &c : cores) {
        by_domain[static_cast<size_t>(c.domain)].push_back(c.cpu);
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
        // More co-located ranks than domains: flat domain-major order, one contiguous slice each.
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
        const int cpu = order[offset + i];
        // cpu_set_t addresses only CPU_SETSIZE CPUs, and CPU_SET past that is a silent no-op that would
        // leave an empty set pinning nothing. Refuse the whole placement instead: unpinned is a documented
        // outcome, whereas "pinned to no core" is not.
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            return {};
        }
        CPU_ZERO(&sets[i]);
        CPU_SET(cpu, &sets[i]);
    }
    return sets;
}

// The locality domain each partition cpuset lands in, in partition_cpusets order -- what a two-level
// PartitionBarrier groups by. Derived from the sets, not from the placement logic, so the two cannot
// drift apart. Empty (⇒ flat barrier) if the sets are empty or one names no core the scan knows.
inline auto cpuset_domains(const std::vector<CpuSet> &sets) -> std::vector<int> {
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
                found = core.domain;
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

// Correctness never depends on pinning, so a failure is not an error -- but it must not be invisible.
// Returns whether the affinity actually took: `barrier_groups = 0` has two legitimate causes (nothing
// was pinned, or each rank was confined to a single locality domain and so has nothing to fan in
// across), and without this they are indistinguishable from outside the process. CommProfile reports
// the count so a run states what happened instead of leaving it inferred from the timing.
[[nodiscard]] inline auto pin_this_thread(const CpuSet &set) -> bool {
    return pthread_setaffinity_np(pthread_self(), sizeof(CpuSet), &set) == 0;
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

inline auto partition_cpusets(size_t /*n*/,
                              size_t /*group_index*/ = 0,
                              size_t /*group_count*/ = 1,
                              NodeMask /*mask*/ = NodeMask::Shared) -> std::vector<CpuSet> {
    return {};
}
inline auto this_thread_cpuset() -> CpuSet {
    return {};
}
inline auto classify_node_mask(const std::vector<CpuSet> & /*masks*/) -> NodeMask {
    return NodeMask::Shared;
}
inline auto cpuset_domains(const std::vector<CpuSet> & /*sets*/) -> std::vector<int> {
    return {};
}
// Always false here: there is no pinning on this platform, so reporting "0 pinned" is accurate rather
// than a failure. See the Linux overload for why the result is surfaced at all.
[[nodiscard]] inline auto pin_this_thread(const CpuSet & /*set*/) -> bool {
    return false;
}

#endif // __linux__

} // namespace monoprop::detail::partition
