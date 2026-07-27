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

// Coverage of CpuTopology.h (Linux /sys parsing + affinity pinning; count-only fallback elsewhere).
// Drives the cpulist parser across its token shapes and the enumerate/place/pin surface on this host.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "monoprop/detail/shard/CpuTopology.h"

namespace shard = monoprop::detail::shard;

// The enumerate/place/pin surface exists on every platform; exercise it regardless of OS.
BOOST_AUTO_TEST_CASE(cpu_topology_enumerate_and_place) {
    const auto cores = shard::enumerate_physical_cores();

    const auto one = shard::shard_cpusets(/*n=*/1);
    BOOST_CHECK(one.size() <= 1u);
    if (!one.empty()) {
        // A placement only comes back where the engine can pin (Linux /sys), which implies cores were
        // found. Pinning itself is best-effort and no-op-safe; drive it.
        BOOST_CHECK(!cores.empty());
        shard::pin_this_thread(one.front());
    }
#if defined(__linux__)
    // With a readable /sys and pinning enabled, a non-empty core list must yield a placement.
    if (!cores.empty()) {
        BOOST_CHECK_EQUAL(one.size(), 1u);
    }
#endif

    // Asking for more physical cores than exist disables pinning (empty), never oversubscribes.
    const auto too_many = shard::shard_cpusets(/*n=*/1'000'000);
    BOOST_CHECK(too_many.empty());
}

BOOST_AUTO_TEST_CASE(cpu_topology_place_co_located_ranks) {
    const auto cores = shard::enumerate_physical_cores();
    if (cores.size() < 2) {
        return; // need at least two cores to deal one to each of two co-located ranks
    }
    // Two co-located ranks, one shard each: each gets a disjoint core. Covers both placement arms --
    // interleave across the dealt domains (group_count <= #L3), and the flat domain-major slice otherwise.
    const auto rank0 = shard::shard_cpusets(/*n=*/1, /*group_index=*/0, /*group_count=*/2);
    const auto rank1 = shard::shard_cpusets(/*n=*/1, /*group_index=*/1, /*group_count=*/2);
    // Off Linux there is no pinning, so both come back empty (unpinned, still disjoint by the scheduler).
#if defined(__linux__)
    BOOST_CHECK_EQUAL(rank0.size(), 1u);
    BOOST_CHECK_EQUAL(rank1.size(), 1u);
#else
    BOOST_CHECK(rank0.empty());
    BOOST_CHECK(rank1.empty());
#endif

    const auto past_end = shard::shard_cpusets(/*n=*/cores.size(), /*group_index=*/1, /*group_count=*/2);
    BOOST_CHECK(past_end.empty());
}

#if defined(__linux__)

using shard::topo_detail::parse_cpulist;
using shard::topo_detail::read_line;

// Enumeration must span holes in the CPU id space (offline or hot-plugged CPUs leave unreadable ids).
// Oracle: the sibling groups re-derived straight from /sys over the whole allowed range.
BOOST_AUTO_TEST_CASE(cpu_topology_enumeration_spans_gaps_in_the_id_space) {
    const auto allowed = shard::topo_detail::allowed_cpus();
    if (allowed.empty()) {
        return; // affinity unreadable; enumeration accepts every CPU and there is nothing to compare
    }

    std::set<int> expected_groups; // one key (min sibling) per physical core with an allowed sibling
    for (int cpu = 0; cpu <= *allowed.rbegin(); ++cpu) {
        const std::string sib =
            read_line("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        if (sib.empty()) {
            continue;
        }
        const auto siblings = parse_cpulist(sib);
        if (siblings.empty()) {
            continue;
        }
        if (std::any_of(siblings.begin(), siblings.end(), [&](int s) { return allowed.contains(s); })) {
            expected_groups.insert(*std::min_element(siblings.begin(), siblings.end()));
        }
    }
    if (expected_groups.empty()) {
        return; // /sys unreadable on this host
    }

    const auto cores = shard::enumerate_physical_cores();
    BOOST_CHECK_EQUAL(cores.size(), expected_groups.size());
    for (const auto &core : cores) {
        BOOST_TEST(allowed.contains(core.cpu));
    }
}

BOOST_AUTO_TEST_CASE(cpu_topology_parse_cpulist_shapes) {
    BOOST_TEST(parse_cpulist("5") == (std::vector<int>{5}), boost::test_tools::per_element());
    BOOST_TEST(parse_cpulist("0-3") == (std::vector<int>{0, 1, 2, 3}), boost::test_tools::per_element());
    BOOST_TEST(parse_cpulist("0-3,16-17") == (std::vector<int>{0, 1, 2, 3, 16, 17}), boost::test_tools::per_element());
    BOOST_TEST(parse_cpulist("2,4,6") == (std::vector<int>{2, 4, 6}), boost::test_tools::per_element());

    // Empty input and empty tokens contribute nothing (the !tok.empty() guard).
    BOOST_CHECK(parse_cpulist("").empty());
    BOOST_TEST(parse_cpulist("1,,3") == (std::vector<int>{1, 3}), boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(cpu_topology_read_line_present_and_absent) {
    BOOST_CHECK(read_line("/nonexistent/monoprop/topology/does_not_exist").empty());

    // cpu0 always exists on Linux, and its thread_siblings_list is a non-empty cpulist.
    const std::string line = read_line("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    BOOST_CHECK(!line.empty());
    BOOST_CHECK(!parse_cpulist(line).empty());
}

BOOST_AUTO_TEST_CASE(cpu_topology_allowed_cpus_nonempty_on_ci) {
    // sched_getaffinity succeeds on Linux CI, so the process's allowed set is non-empty.
    const auto allowed = shard::topo_detail::allowed_cpus();
    BOOST_CHECK(!allowed.empty());
}

#endif // __linux__
