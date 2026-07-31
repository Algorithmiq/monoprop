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

// Coverage of hwloc-backed core discovery, partition placement, and affinity pinning.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <set>

#include "monoprop/detail/partition/CpuTopology.h"

namespace partition = monoprop::detail::partition;

// The enumerate/place/pin surface exists on every platform; exercise it regardless of OS.
BOOST_AUTO_TEST_CASE(cpu_topology_enumerate_and_place) {
    const auto cores = partition::enumerate_physical_cores();
    std::set<unsigned> core_cpus;
    for (const auto &core : cores) {
        BOOST_CHECK(core_cpus.insert(core.cpu).second);
        BOOST_CHECK_GE(core.l3_domain, 0);
    }

    const auto one = partition::partition_cpusets(/*n=*/1);
    BOOST_CHECK(one.size() <= 1u);
    if (!one.empty()) {
        BOOST_CHECK(!cores.empty());
        BOOST_CHECK(core_cpus.contains(one.front().cpu));
        partition::pin_this_thread(one.front());
    }

    // Asking for more physical cores than exist disables pinning (empty), never oversubscribes.
    const auto too_many = partition::partition_cpusets(/*n=*/1'000'000);
    BOOST_CHECK(too_many.empty());
}

BOOST_AUTO_TEST_CASE(cpu_topology_place_co_located_ranks) {
    const auto cores = partition::enumerate_physical_cores();
    if (cores.size() < 2) {
        return; // need at least two cores to deal one to each of two co-located ranks
    }
    // Two co-located ranks, one partition each. Covers both placement arms -- interleave across the
    // dealt domains (group_count <= #L3), and the flat domain-major slice otherwise.
    const auto rank0 = partition::partition_cpusets(/*n=*/1, /*group_index=*/0, /*group_count=*/2);
    const auto rank1 = partition::partition_cpusets(/*n=*/1, /*group_index=*/1, /*group_count=*/2);
    BOOST_CHECK_EQUAL(rank0.size(), rank1.size());
    if (!rank0.empty()) {
        BOOST_CHECK_NE(rank0.front().cpu, rank1.front().cpu);
    }

    const auto past_end = partition::partition_cpusets(/*n=*/cores.size(), /*group_index=*/1, /*group_count=*/2);
    BOOST_CHECK(past_end.empty());
}
