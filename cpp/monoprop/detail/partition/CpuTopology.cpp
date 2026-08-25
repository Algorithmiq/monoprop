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

#include "monoprop/detail/partition/CpuTopology.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <format>
#include <map>
#include <mutex>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include <hwloc.h>

namespace monoprop::detail::partition {

namespace {

/* ── Process-lifetime hwloc topology ──────────────────────────────────────── */

// hwloc_topology_t is safe for concurrent read-only access after hwloc_topology_load().
struct TopologyHolder {
    hwloc_topology_t topo = nullptr;

    TopologyHolder() noexcept {
        if (hwloc_topology_init(&topo) < 0) {
            topo = nullptr;
            return;
        }
        /* Keep all L3 cache objects so shared-L3 domains can always be identified, even on
         * topologies where the L3 appears private and would otherwise be suppressed by the
         * default HWLOC_TYPE_FILTER_KEEP_STRUCTURE filter. */
        hwloc_topology_set_type_filter(topo, HWLOC_OBJ_L3CACHE, HWLOC_TYPE_FILTER_KEEP_ALL);
        if (hwloc_topology_load(topo) < 0) {
            hwloc_topology_destroy(topo);
            topo = nullptr;
        }
    }

    ~TopologyHolder() {
        if (topo) {
            hwloc_topology_destroy(topo);
        }
    }

    TopologyHolder(const TopologyHolder &) = delete;
    auto operator=(const TopologyHolder &) -> TopologyHolder & = delete;
};

// Returns the loaded topology, or nullptr when initialization failed.
// The static local is initialized once; subsequent calls return the cached handle.
auto get_topology() -> hwloc_topology_t {
    static TopologyHolder holder;
    return holder.topo;
}

/* ── Effective allowed cpuset for the calling thread ──────────────────────── */

// Queries the current thread's affinity to respect any launcher-imposed restriction (cgroup, MPI
// process binding) narrower than the topology's own allowed cpuset. Falls back to the topology
// allowed cpuset when the cpubind query is unsupported on this platform. Caller must free the bitmap.
auto effective_allowed_cpuset(hwloc_topology_t topo) -> hwloc_cpuset_t {
    hwloc_cpuset_t set = hwloc_bitmap_alloc();
    if (!set) {
        return nullptr;
    }
    if (hwloc_get_cpubind(topo, set, HWLOC_CPUBIND_THREAD) == 0) {
        return set;
    }
    /* cpubind query not supported (e.g. macOS without OS X binding): fall back. */
    hwloc_bitmap_free(set);
    return hwloc_bitmap_dup(hwloc_topology_get_allowed_cpuset(topo));
}

} // anonymous namespace

/* ── topo_detail::placement_order ─────────────────────────────────────────── */

namespace topo_detail {

auto placement_order(const std::vector<PhysicalCore> &cores, size_t n, size_t group_index, size_t group_count)
    -> std::vector<int> {
    if (cores.empty() || group_count == 0 || group_count * n > cores.size()) {
        return {};
    }

    int max_domain = 0;
    for (const auto &c : cores) {
        max_domain = std::max(max_domain, c.l3_domain);
    }

    /* Bucket representative PU indices by L3 domain id. */
    std::vector<std::vector<int>> by_domain(static_cast<size_t>(max_domain) + 1);
    for (const auto &c : cores) {
        by_domain[static_cast<size_t>(c.l3_domain)].push_back(c.cpu);
    }

    /* Interleave buckets depth-first: b0[0], b1[0], …, b0[1], b1[1], … */
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
        /* Interleave arm: deal domains round-robin across ranks. */
        std::vector<std::vector<int>> mine;
        for (size_t d = group_index; d < by_domain.size(); d += group_count) {
            mine.push_back(by_domain[d]);
        }
        order = interleave(mine);
    }
    else {
        /* More co-located ranks than L3 domains: flat domain-major order, one contiguous slice each. */
        for (const auto &bucket : by_domain) {
            order.insert(order.end(), bucket.begin(), bucket.end());
        }
        offset = group_index * n;
    }

    if (offset + n > order.size()) {
        return {};
    }
    return {order.begin() + static_cast<std::ptrdiff_t>(offset),
            order.begin() + static_cast<std::ptrdiff_t>(offset + n)};
}

} // namespace topo_detail

/* ── enumerate_physical_cores ──────────────────────────────────────────────── */

auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
    auto *const topo = get_topology();
    if (!topo) {
        return {};
    }

    const hwloc_cpuset_t allowed = effective_allowed_cpuset(topo);
    if (!allowed) {
        return {};
    }

    const int core_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_CORE);
    if (core_depth == HWLOC_TYPE_DEPTH_UNKNOWN || core_depth == HWLOC_TYPE_DEPTH_MULTIPLE) {
        hwloc_bitmap_free(allowed);
        return {};
    }

    std::vector<PhysicalCore> cores;
    std::map<unsigned, int> l3_domain_map; // l3->logical_index → domain id
    int next_domain_id = 0;

    const unsigned num_cores = hwloc_get_nbobjs_by_depth(topo, core_depth);
    for (unsigned i = 0; i < num_cores; ++i) {
        auto *const core = hwloc_get_obj_by_depth(topo, core_depth, i);
        if (!core || !core->cpuset) {
            continue;
        }

        /* Skip cores that have no PU in the calling thread's effective allowed mask. */
        if (!hwloc_bitmap_intersects(core->cpuset, allowed)) {
            continue;
        }

        /* Lowest allowed PU on this core is the representative OS index. */
        hwloc_cpuset_t core_allowed = hwloc_bitmap_alloc();
        if (!core_allowed) {
            continue;
        }
        hwloc_bitmap_and(core_allowed, core->cpuset, allowed);
        const int rep = hwloc_bitmap_first(core_allowed);
        hwloc_bitmap_free(core_allowed);
        if (rep < 0) {
            continue;
        }

        /* Find the L3 cache ancestor and assign a stable domain id. Cores sharing an L3 object
         * (same logical_index) receive the same domain id. Cores without an L3 ancestor each
         * receive their own singleton domain so the placement algorithm can still spread across
         * whatever structure the topology does have. */
        int domain;
        auto *const l3 = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_L3CACHE, core);
        if (l3) {
            const auto [it, inserted] = l3_domain_map.emplace(l3->logical_index, next_domain_id);
            if (inserted) {
                ++next_domain_id;
            }
            domain = it->second;
        }
        else {
            domain = next_domain_id++;
        }

        cores.push_back(PhysicalCore{.cpu = rep, .l3_domain = domain});
    }

    hwloc_bitmap_free(allowed);
    return cores;
}

/* ── affinity_mask_words ───────────────────────────────────────────────────── */

auto affinity_mask_words(uint64_t *out, size_t nwords) -> bool {
    if (out == nullptr || nwords == 0) {
        return false;
    }
    std::fill_n(out, nwords, uint64_t{0});
    auto *const topo = get_topology();
    if (!topo) {
        return false;
    }
    const hwloc_cpuset_t allowed = effective_allowed_cpuset(topo);
    if (!allowed) {
        return false;
    }
    // Refused rather than truncated: a truncated mask could compare disjoint against a peer it overlaps.
    const int last = hwloc_bitmap_last(allowed);
    const bool representable = last >= 0 && static_cast<size_t>(last) < nwords * 64;
    if (representable) {
        for (int pu = hwloc_bitmap_first(allowed); pu >= 0; pu = hwloc_bitmap_next(allowed, pu)) {
            out[static_cast<size_t>(pu) / 64] |= uint64_t{1} << (static_cast<size_t>(pu) % 64);
        }
    }
    hwloc_bitmap_free(allowed);
    return representable;
}

/* ── masks_are_pairwise_disjoint ───────────────────────────────────────────── */

auto masks_are_pairwise_disjoint(const uint64_t *masks, size_t n, size_t words) -> bool {
    if (masks == nullptr || words == 0 || n < 2) {
        return false;
    }
    // An all-zero mask is disjoint from everything, so empty is rejected before the pairwise test.
    for (size_t r = 0; r < n; ++r) {
        bool any = false;
        for (size_t w = 0; w < words && !any; ++w) {
            any = masks[(r * words) + w] != 0;
        }
        if (!any) {
            return false;
        }
    }
    for (size_t a = 0; a < n; ++a) {
        for (size_t b = a + 1; b < n; ++b) {
            for (size_t w = 0; w < words; ++w) {
                if ((masks[(a * words) + w] & masks[(b * words) + w]) != 0) {
                    return false; // two peers share a CPU: not private
                }
            }
        }
    }
    return true;
}

/* ── summarize_masks ──────────────────────────────────────────────────────── */

// hwloc indexes a bitmap in unsigned long units, so a 64-bit word must be one of them.
static_assert(sizeof(unsigned long) == sizeof(uint64_t), "the mask word is not an hwloc bitmap unit");

auto summarize_masks(const uint64_t *masks, size_t n, size_t words, size_t self) -> std::optional<MaskSummary> {
    if (masks == nullptr || n == 0 || words == 0 || self >= n) {
        return std::nullopt;
    }
    auto *const row = hwloc_bitmap_alloc();
    auto *const all = hwloc_bitmap_alloc();
    if (row == nullptr || all == nullptr) {
        hwloc_bitmap_free(row);
        hwloc_bitmap_free(all);
        return std::nullopt;
    }
    MaskSummary out;
    bool ok = true;
    for (size_t r = 0; r < n; ++r) {
        hwloc_bitmap_zero(row);
        for (size_t w = 0; w < words; ++w) {
            hwloc_bitmap_set_ith_ulong(row, static_cast<unsigned>(w), masks[(r * words) + w]);
        }
        const int weight = hwloc_bitmap_weight(row);
        if (weight <= 0) { // an all-zero row is a mask that did not fit the exchange window
            ok = false;
            break;
        }
        if (r == self) {
            out.cpus = static_cast<size_t>(weight);
        }
        hwloc_bitmap_or(all, all, row);
    }
    if (ok) {
        out.node_cpus = static_cast<size_t>(hwloc_bitmap_weight(all));
        std::array<char, 512> text{};
        const int need = hwloc_bitmap_list_snprintf(text.data(), text.size(), all);
        out.cpu_list = text.data();
        // Truncation is stated, never silent: a cut list read as complete is a smaller machine. Cut
        // back to the last whole range first, so the marker never follows a half-written CPU id.
        if (std::cmp_greater_equal(need, text.size())) {
            const size_t last = out.cpu_list.rfind(',');
            out.cpu_list.resize(last == std::string::npos ? 0 : last + 1);
            out.cpu_list += "+";
        }
    }
    hwloc_bitmap_free(row);
    hwloc_bitmap_free(all);
    return ok ? std::optional{out} : std::nullopt;
}

auto format_place_line(int mpi_rank, int node_rank, int node_size, const char *verdict, const MaskSummary &summary)
    -> std::string {
    return std::format("COMMPLACE rank={} node_rank={} node_size={} masks={} cpus={} node_cpus={} cpu_list={}\n",
                       mpi_rank,
                       node_rank,
                       node_size,
                       verdict,
                       summary.cpus,
                       summary.node_cpus,
                       summary.cpu_list);
}

/* ── partition_cpusets ─────────────────────────────────────────────────────── */

auto partition_cpusets(size_t n, size_t group_index, size_t group_count, NodeMask mask) -> std::vector<CpuSet> {
    const auto cores = enumerate_physical_cores();

    // A private mask IS this rank's share: the launcher already separated co-located ranks.
    if (mask == NodeMask::PerRank) {
        group_index = 0;
        group_count = 1;
    }
    const auto order = topo_detail::placement_order(cores, n, group_index, group_count);

    if (order.empty()) {
        static std::once_flag warned;
        std::call_once(warned, [&] {
            std::print(stderr,
                       "monoprop: partition pinning requested but not possible "
                       "({} cores visible, {} groups x {} partitions); threads run unpinned.\n",
                       cores.size(),
                       group_count,
                       n);
            std::fflush(stderr);
        });
    }

    std::vector<CpuSet> sets(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        sets[i] = CpuSet{order[i]};
    }
    return sets;
}

/* ── pin_this_thread ───────────────────────────────────────────────────────── */

auto pin_this_thread(const CpuSet &set) -> void {
    if (set.pu < 0) {
        return;
    }
    auto *const topo = get_topology();
    if (!topo) {
        return;
    }
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    if (!cpuset) {
        return;
    }
    hwloc_bitmap_only(cpuset, static_cast<unsigned>(set.pu));
    /* Errors are intentionally ignored: pinning is performance-only, not a correctness requirement. */
    hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
    hwloc_bitmap_free(cpuset);
}

} // namespace monoprop::detail::partition
