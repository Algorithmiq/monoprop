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

// Coverage of CpuTopology.h — the one platform-specific engine file (Linux /sys parsing + affinity
// pinning; portable count-only fallback elsewhere). The engine only calls this during shard setup,
// which the white-box suite runs single-partition, so the placement logic is otherwise unswept.
// Here we drive the pure cpulist parser across its token shapes and call the enumerate/placement/pin
// surface on the host running the tests (the coverage CI is Linux, so the /sys fast path is live).
//
// Cases are flat top-level BOOST_AUTO_TEST_CASEs sharing a cpu_topology_ prefix (not a
// BOOST_AUTO_TEST_SUITE) to match this suite's ctest discovery, which registers by leaf case name.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <vector>

#include "monoprop/detail/shard/CpuTopology.h"

namespace shard = monoprop::detail::shard;

// enumerate_physical_cores() + shard_cpusets() + pin_this_thread() exist on every platform (Linux
// parses /sys; macOS counts; other platforms return empty). Exercise them regardless of OS.
BOOST_AUTO_TEST_CASE(cpu_topology_enumerate_and_place) {
    const auto cores = shard::enumerate_physical_cores();

    // Placing one shard yields at most one cpuset -- never oversubscribing.
    const auto one = shard::shard_cpusets(/*n=*/1);
    BOOST_CHECK(one.size() <= 1u);
    if (!one.empty()) {
        // A real placement only comes back where the engine can actually pin (the Linux /sys path),
        // and it implies the host reported cores. Pinning is best-effort and no-op-safe; drive it.
        BOOST_CHECK(!cores.empty());
        shard::pin_this_thread(one.front());
    }
#if defined(__linux__)
    // On the Linux CI host with a readable /sys and pinning enabled, a non-empty core list must yield
    // a placement. Elsewhere (e.g. macOS counts cores but cannot pin) `one` stays empty by design.
    if (!cores.empty()) {
        BOOST_CHECK_EQUAL(one.size(), 1u);
    }
#endif

    // Asking for more physical cores than exist disables pinning (empty vector), never oversubscribes.
    const auto too_many = shard::shard_cpusets(/*n=*/1'000'000);
    BOOST_CHECK(too_many.empty());
}

BOOST_AUTO_TEST_CASE(cpu_topology_place_co_located_ranks) {
    const auto cores = shard::enumerate_physical_cores();
    if (cores.size() < 2) {
        return; // need at least two cores to deal one to each of two co-located ranks
    }
    // Two co-located ranks, one shard each: each gets a disjoint core. This drives both placement
    // arms -- interleave-across-dealt-domains when group_count <= #L3 domains, and the flat
    // domain-major slice when there are more ranks than domains (the common single-L3 CI runner).
    const auto rank0 = shard::shard_cpusets(/*n=*/1, /*group_index=*/0, /*group_count=*/2);
    const auto rank1 = shard::shard_cpusets(/*n=*/1, /*group_index=*/1, /*group_count=*/2);
    // Placement only materializes where the engine can pin (the Linux /sys path). macOS counts >=2
    // cores but cannot pin, so both come back empty -- correct (unpinned shards, still disjoint by the
    // OS scheduler).
#if defined(__linux__)
    BOOST_CHECK_EQUAL(rank0.size(), 1u);
    BOOST_CHECK_EQUAL(rank1.size(), 1u);
#else
    BOOST_CHECK(rank0.empty());
    BOOST_CHECK(rank1.empty());
#endif

    // A group_index past the available slices yields empty (offset + n > order.size()).
    const auto past_end = shard::shard_cpusets(/*n=*/cores.size(), /*group_index=*/1, /*group_count=*/2);
    BOOST_CHECK(past_end.empty());
}

#if defined(__linux__)

using shard::topo_detail::parse_cpulist;
using shard::topo_detail::read_line;

BOOST_AUTO_TEST_CASE(cpu_topology_parse_cpulist_shapes) {
    // Single id, explicit range, and a mixed comma list of both.
    BOOST_TEST(parse_cpulist("5") == (std::vector<int>{5}), boost::test_tools::per_element());
    BOOST_TEST(parse_cpulist("0-3") == (std::vector<int>{0, 1, 2, 3}), boost::test_tools::per_element());
    BOOST_TEST(parse_cpulist("0-3,16-17") == (std::vector<int>{0, 1, 2, 3, 16, 17}), boost::test_tools::per_element());
    BOOST_TEST(parse_cpulist("2,4,6") == (std::vector<int>{2, 4, 6}), boost::test_tools::per_element());

    // Empty input and empty tokens contribute nothing (the !tok.empty() guard).
    BOOST_CHECK(parse_cpulist("").empty());
    BOOST_TEST(parse_cpulist("1,,3") == (std::vector<int>{1, 3}), boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(cpu_topology_read_line_present_and_absent) {
    // A path that cannot be opened yields an empty string (the `if (f)` false arm).
    BOOST_CHECK(read_line("/nonexistent/monoprop/topology/does_not_exist").empty());

    // A real single-CPU sysfs-style file is read back as its first line. cpu0 always exists on a
    // Linux CI host; its thread_siblings_list is a non-empty cpulist.
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
