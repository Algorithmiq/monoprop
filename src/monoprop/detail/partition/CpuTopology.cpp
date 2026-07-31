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

#include <hwloc.h>

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::detail::partition {
namespace {

class Topology {
public:
    Topology() {
        if (hwloc_topology_init(&topology_) < 0) {
            topology_ = nullptr;
            return;
        }
        if (hwloc_topology_load(topology_) < 0) {
            hwloc_topology_destroy(topology_);
            topology_ = nullptr;
            return;
        }
        allowed_ = hwloc_bitmap_alloc();
        if (allowed_ == nullptr) {
            hwloc_topology_destroy(topology_);
            topology_ = nullptr;
            return;
        }
        if (hwloc_get_cpubind(topology_, allowed_, HWLOC_CPUBIND_PROCESS) < 0) {
            hwloc_bitmap_copy(allowed_, hwloc_topology_get_allowed_cpuset(topology_));
        }
    }

    Topology(const Topology &) = delete;
    auto operator=(const Topology &) -> Topology & = delete;

    ~Topology() {
        hwloc_bitmap_free(allowed_);
        if (topology_ != nullptr) {
            hwloc_topology_destroy(topology_);
        }
    }

    auto get() const -> hwloc_topology_t { return topology_; }
    auto allowed() const -> hwloc_const_cpuset_t { return allowed_; }

private:
    hwloc_topology_t topology_ = nullptr;
    hwloc_cpuset_t allowed_ = nullptr;
};

auto topology_context() -> const Topology & {
    static const Topology instance;
    return instance;
}

auto topology() -> hwloc_topology_t {
    return topology_context().get();
}

auto binding_supported(hwloc_topology_t topo) -> bool {
    if (topo == nullptr) {
        return false;
    }
    const auto *support = hwloc_topology_get_support(topo);
    return support->cpubind != nullptr && support->cpubind->set_thisthread_cpubind != 0;
}

auto interleave(const std::vector<std::vector<unsigned>> &buckets) -> std::vector<unsigned> {
    std::vector<unsigned> out;
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
}

} // namespace

auto enumerate_physical_cores() -> std::vector<PhysicalCore> {
    auto *const topo = topology();
    if (topo == nullptr) {
        return {};
    }

    auto type = HWLOC_OBJ_CORE;
    if (hwloc_get_nbobjs_by_type(topo, type) <= 0) {
        type = HWLOC_OBJ_PU;
    }

    std::vector<PhysicalCore> cores;
    std::unordered_map<const hwloc_obj *, int> domains;
    const auto *const allowed = topology_context().allowed();
    auto *const candidate = hwloc_bitmap_alloc();
    if (allowed == nullptr || candidate == nullptr) {
        hwloc_bitmap_free(candidate);
        return {};
    }
    auto *object = hwloc_get_next_obj_by_type(topo, type, nullptr);
    while (object != nullptr) {
        hwloc_bitmap_and(candidate, object->cpuset, allowed);
        if (const auto cpu = hwloc_bitmap_first(candidate); cpu >= 0) {
            const auto *l3 = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_L3CACHE, object);
            auto domain = static_cast<int>(domains.size());
            if (l3 != nullptr) {
                const auto [it, inserted] = domains.try_emplace(l3, domain);
                domain = it->second;
            }
            else {
                domains.emplace(object, domain);
            }
            cores.push_back(PhysicalCore{.cpu = static_cast<unsigned>(cpu), .l3_domain = domain});
        }
        object = hwloc_get_next_obj_by_type(topo, type, object);
    }
    hwloc_bitmap_free(candidate);
    return cores;
}

auto partition_cpusets(size_t n, size_t group_index, size_t group_count) -> std::vector<CpuSet> {
    if (!config::get().partition_pinning || !binding_supported(topology()) || group_count == 0
        || group_index >= group_count) {
        return {};
    }
    const auto cores = enumerate_physical_cores();
    if (cores.empty() || n > cores.size() / group_count) {
        return {};
    }

    int max_domain = 0;
    for (const auto &core : cores) {
        max_domain = std::max(max_domain, core.l3_domain);
    }
    std::vector<std::vector<unsigned>> by_domain(static_cast<size_t>(max_domain) + 1);
    for (const auto &core : cores) {
        by_domain[static_cast<size_t>(core.l3_domain)].push_back(core.cpu);
    }

    std::vector<unsigned> order;
    size_t offset = 0;
    if (group_count <= by_domain.size()) {
        std::vector<std::vector<unsigned>> mine;
        for (size_t domain = group_index; domain < by_domain.size(); domain += group_count) {
            mine.push_back(by_domain[domain]);
        }
        order = interleave(mine);
    }
    else {
        for (const auto &bucket : by_domain) {
            order.insert(order.end(), bucket.begin(), bucket.end());
        }
        offset = group_index * n;
    }
    if (offset > order.size() || n > order.size() - offset) {
        return {};
    }

    std::vector<CpuSet> sets;
    sets.reserve(n);
    for (size_t index = 0; index < n; ++index) {
        sets.push_back(CpuSet{order[offset + index]});
    }
    return sets;
}

auto pin_this_thread(const CpuSet &set) -> void {
    auto *const topo = topology();
    if (!binding_supported(topo)) {
        return;
    }
    auto *const cpuset = hwloc_bitmap_alloc();
    if (cpuset == nullptr) {
        return;
    }
    hwloc_bitmap_only(cpuset, set.cpu);
    static_cast<void>(hwloc_set_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD));
    hwloc_bitmap_free(cpuset);
}

} // namespace monoprop::detail::partition
