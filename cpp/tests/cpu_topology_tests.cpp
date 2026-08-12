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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "monoprop/detail/partition/CpuTopology.h"

namespace partition = monoprop::detail::partition;

// The enumerate/place/pin surface exists on every platform; exercise it regardless of OS.
BOOST_AUTO_TEST_CASE(cpu_topology_enumerate_and_place) {
    const auto cores = partition::enumerate_physical_cores();

    const auto one = partition::partition_cpusets(/*n=*/1);
    BOOST_CHECK(one.size() <= 1u);
    if (!one.empty()) {
        // A placement only comes back where the engine can pin (Linux /sys), which implies cores were
        // found. Pinning itself is best-effort and no-op-safe; drive it.
        BOOST_CHECK(!cores.empty());
        partition::pin_this_thread(one.front());
    }
#if defined(__linux__)
    // With a readable /sys and pinning enabled, a non-empty core list must yield a placement.
    if (!cores.empty()) {
        BOOST_CHECK_EQUAL(one.size(), 1u);
    }
#endif

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
    // Off Linux there is no pinning, so both come back empty (unpinned, still disjoint by the scheduler).
#if defined(__linux__)
    BOOST_CHECK_EQUAL(rank0.size(), 1u);
    BOOST_CHECK_EQUAL(rank1.size(), 1u);
#else
    BOOST_CHECK(rank0.empty());
    BOOST_CHECK(rank1.empty());
#endif

    const auto past_end = partition::partition_cpusets(/*n=*/cores.size(), /*group_index=*/1, /*group_count=*/2);
    BOOST_CHECK(past_end.empty());
}

#if defined(__linux__)

#include <sched.h>

using partition::topo_detail::parse_cpulist;
using partition::topo_detail::read_line;

// Enumeration must span holes in the CPU id space (offline or hot-plugged CPUs leave unreadable ids).
// Oracle: the sibling groups re-derived straight from /sys over the whole allowed range.
BOOST_AUTO_TEST_CASE(cpu_topology_enumeration_spans_gaps_in_the_id_space) {
    const auto allowed = partition::topo_detail::allowed_cpus();
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

    const auto cores = partition::enumerate_physical_cores();
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
    const auto allowed = partition::topo_detail::allowed_cpus();
    BOOST_CHECK(!allowed.empty());
}

// Restores the caller's affinity mask however the scope exits, so a failing check cannot leave the rest
// of the suite pinned to two cores.
class ScopedAffinity {
public:
    ScopedAffinity() { pinned_ = sched_getaffinity(0, sizeof(saved_), &saved_) == 0; }
    ScopedAffinity(const ScopedAffinity &) = delete;
    auto operator=(const ScopedAffinity &) -> ScopedAffinity & = delete;
    ~ScopedAffinity() {
        if (pinned_) {
            sched_setaffinity(0, sizeof(saved_), &saved_);
        }
    }

private:
    cpu_set_t saved_{};
    bool pinned_ = false;
};

// A rank whose mask is its own slice of the node must still get a placement. Slurm hands each co-located
// rank a disjoint mask and then tells the rank group_count = ranks-per-node; enumerate_physical_cores()
// already filtered by the mask, so re-dividing pushed group_count * n past the share and answered
// "unpinned", which also flattened the two-level barrier (measured on this cluster as barrier_groups=0 and
// 437 vs 15.5 us/sync at 8 ranks x 16 partitions).
BOOST_AUTO_TEST_CASE(cpu_topology_per_rank_mask_still_places) {
    const auto full = partition::enumerate_physical_cores();
    if (full.size() < 2) {
        return; // too small to simulate a slice of a larger host
    }

    const ScopedAffinity restore_on_exit;

    // Stand in for `srun --cpus-per-task=2`: keep two physical cores of a host that has more.
    cpu_set_t confined;
    CPU_ZERO(&confined);
    CPU_SET(full[0].cpu, &confined);
    CPU_SET(full[1].cpu, &confined);
    if (sched_setaffinity(0, sizeof(confined), &confined) != 0) {
        return; // not permitted here (seccomp, restrictive cgroup); nothing to assert
    }

    // n = the whole share, and a group_count as if seven sibling ranks shared the node.
    const auto sets =
        partition::partition_cpusets(/*n=*/2, /*group_index=*/3, /*group_count=*/8, partition::NodeMask::PerRank);
    BOOST_CHECK_EQUAL(sets.size(), 2u);
    for (const cpu_set_t &set : sets) {
        // Never pin outside the mask the launcher gave us.
        BOOST_CHECK(CPU_ISSET(full[0].cpu, &set) || CPU_ISSET(full[1].cpu, &set));
        BOOST_CHECK_EQUAL(CPU_COUNT(&set), 1);
    }
    // One domain entry per set, and the two sets must not be the same core.
    BOOST_CHECK_EQUAL(partition::cpuset_domains(sets).size(), 2u);
    BOOST_CHECK(!CPU_EQUAL(&sets[0], &sets[1]));
}

// The invariant the PerRank collapse above most endangers: under a genuinely Shared mask, two co-located
// ranks must still deal themselves DIFFERENT cores. Collapsing unconditionally on "the mask is narrower
// than the machine" pointed every rank at the same cores, because mask width cannot distinguish a per-rank
// slice from a shared one -- which is why the caller measures it instead (see classify_node_mask).
BOOST_AUTO_TEST_CASE(cpu_topology_shared_mask_keeps_co_located_ranks_disjoint) {
    const auto full = partition::enumerate_physical_cores();
    if (full.size() < 4) {
        return; // need two cores per rank for two ranks
    }

    const ScopedAffinity restore_on_exit;

    // A shared step mask narrower than the host: four cores that both ranks can see.
    cpu_set_t shared;
    CPU_ZERO(&shared);
    for (size_t i = 0; i < 4; ++i) {
        CPU_SET(full[i].cpu, &shared);
    }
    if (sched_setaffinity(0, sizeof(shared), &shared) != 0) {
        return;
    }

    const auto rank0 =
        partition::partition_cpusets(/*n=*/2, /*group_index=*/0, /*group_count=*/2, partition::NodeMask::Shared);
    const auto rank1 =
        partition::partition_cpusets(/*n=*/2, /*group_index=*/1, /*group_count=*/2, partition::NodeMask::Shared);
    BOOST_CHECK_EQUAL(rank0.size(), 2u);
    BOOST_CHECK_EQUAL(rank1.size(), 2u);
    for (const cpu_set_t &a : rank0) {
        for (const cpu_set_t &b : rank1) {
            // Two ranks sharing a core would have each one's busy-polling collectives starve the other's
            // barrier spins -- the failure this whole placement path exists to avoid.
            BOOST_CHECK(!CPU_EQUAL(&a, &b));
        }
    }
}

// Two ranks holding disjoint masks are a per-rank split; identical masks are a shared one. Nothing else
// distinguishes them, which is the whole reason this is measured rather than inferred.
BOOST_AUTO_TEST_CASE(cpu_topology_classify_node_mask_disjoint_vs_identical) {
    cpu_set_t a;
    cpu_set_t b;
    CPU_ZERO(&a);
    CPU_ZERO(&b);
    CPU_SET(0, &a);
    CPU_SET(1, &a);
    CPU_SET(2, &b);
    CPU_SET(3, &b);
    BOOST_CHECK(partition::classify_node_mask({a, b}) == partition::NodeMask::PerRank);
    BOOST_CHECK(partition::classify_node_mask({a, a}) == partition::NodeMask::Shared);

    // Partial overlap is pathological; Shared is the conservative answer because dividing never
    // double-books a core within a rank, whereas collapsing points every rank at the same cores.
    cpu_set_t c = a;
    CPU_SET(2, &c);
    BOOST_CHECK(partition::classify_node_mask({a, c}) == partition::NodeMask::Shared);

    // An unreadable mask arrives empty and must not be read as "disjoint from everything".
    cpu_set_t empty;
    CPU_ZERO(&empty);
    BOOST_CHECK(partition::classify_node_mask({a, empty}) == partition::NodeMask::Shared);
    // A lone rank has nobody to be disjoint from.
    BOOST_CHECK(partition::classify_node_mask({a}) == partition::NodeMask::Shared);
}

// Malformed and out-of-range /sys content. This path's contract is to degrade to "unpinned", never to
// throw out of a constructor, and never to answer with a cpulist it half-understood.
BOOST_AUTO_TEST_CASE(cpu_topology_parse_cpulist_rejects_malformed_ranges) {
    BOOST_CHECK(parse_cpulist("3-0").empty());  // reversed range names nothing
    BOOST_CHECK(parse_cpulist("0-").empty());   // missing endpoint drops the token
    BOOST_CHECK(parse_cpulist("-3").empty());   // ditto
    BOOST_CHECK(parse_cpulist("abc").empty());  // non-numeric
    BOOST_CHECK(parse_cpulist("1-2x").empty()); // trailing junk rejects the whole token
    BOOST_TEST(parse_cpulist("7,3-0,9") == (std::vector<int>{7, 9}), boost::test_tools::per_element());

    // Ids at and past CPU_SETSIZE must still PARSE: parse_id also reads cache levels and NUMA node ids,
    // and a host with more CPUs than cpu_set_t can address would otherwise have every cpulist it reports
    // -- including cpu/online -- collapse to empty, silently disabling pinning and the two-level barrier.
    // Placement drops them where CPU_SET is called instead.
    BOOST_TEST(parse_cpulist(std::to_string(CPU_SETSIZE)) == (std::vector<int>{CPU_SETSIZE}),
               boost::test_tools::per_element());
    BOOST_CHECK_EQUAL(parse_cpulist("0-2047").size(), 2048u);

    // An absurdly wide range is garbage rather than a machine, and must not become an allocation.
    BOOST_CHECK(parse_cpulist("0-999999999").empty());
}

// Fixture-driven cover for shared_domain_cpus, which is otherwise exercised only through this host's own
// /sys -- an x86 part that does have an L3, so neither the shared-L2 shape nor the SMT trap below is
// reachable from real hardware here. `cpu_base` is a parameter precisely so this is testable.
namespace {

// Write a synthetic /sys-shaped cache tree. `levels` is (level, shared_cpu_list) in index order.
auto write_cache_fixture(const std::filesystem::path &base, const std::vector<std::pair<int, std::string>> &levels)
    -> void {
    for (size_t i = 0; i < levels.size(); ++i) {
        const auto dir = base / "cache" / ("index" + std::to_string(i));
        std::filesystem::create_directories(dir);
        std::ofstream(dir / "level") << levels[i].first << "\n";
        std::ofstream(dir / "type") << "Unified\n";
        std::ofstream(dir / "shared_cpu_list") << levels[i].second << "\n";
    }
}

// Removes the fixture tree however the scope exits.
class ScopedTree {
public:
    explicit ScopedTree(std::string name)
        : path_(std::filesystem::temp_directory_path() / ("monoprop-topo-" + std::move(name))) {
        std::filesystem::remove_all(path_);
    }
    ScopedTree(const ScopedTree &) = delete;
    auto operator=(const ScopedTree &) -> ScopedTree & = delete;
    ~ScopedTree() { std::filesystem::remove_all(path_); }
    auto path() const -> const std::filesystem::path & { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace

BOOST_AUTO_TEST_CASE(cpu_topology_shared_domain_prefers_deepest_cross_core_level) {
    const ScopedTree tree("deepest");
    // L1 and L2 private to the core (they still list both SMT threads), L3 spanning four cores.
    write_cache_fixture(tree.path(), {{1, "0,64"}, {2, "0,64"}, {3, "0-3,64-67"}});
    const auto got = partition::topo_detail::shared_domain_cpus(tree.path().string(), 0, {0, 64});
    BOOST_TEST(got == (std::vector<int>{0, 1, 2, 3, 64, 65, 66, 67}), boost::test_tools::per_element());
}

// The SMT trap: a per-core L1/L2 lists every hardware thread of the core, so a "shared by >= 2 CPUs" test
// accepts it, makes each core its own domain, and never reaches the NUMA fallback. On an SMT part exposing
// no cross-core cache that silently reinstates the defect this function exists to fix.
BOOST_AUTO_TEST_CASE(cpu_topology_shared_domain_ignores_smt_siblings) {
    const ScopedTree tree("smt");
    write_cache_fixture(tree.path(), {{1, "0,64"}, {2, "0,64"}});
    const std::vector<int> siblings{0, 64};
    const auto got = partition::topo_detail::shared_domain_cpus(tree.path().string(), 0, siblings);
    // Must not have selected the sibling-only level. Whatever it returns comes from the NUMA fallback, so
    // it either found nothing or found a set wider than this one core.
    BOOST_CHECK(got != siblings);
}

// Levels are selected by reported cache level, not by index order, so a /sys that lists them out of order
// still groups by the deepest one.
BOOST_AUTO_TEST_CASE(cpu_topology_shared_domain_selects_by_level_not_index) {
    const ScopedTree tree("order");
    write_cache_fixture(tree.path(), {{3, "0-3"}, {1, "0"}, {2, "0-1"}});
    const auto got = partition::topo_detail::shared_domain_cpus(tree.path().string(), 0, {0});
    BOOST_TEST(got == (std::vector<int>{0, 1, 2, 3}), boost::test_tools::per_element());
}

// A64FX-shaped: /sys/devices/system/cpu/cpuN/cache does not exist at all on Deucalion's ARM nodes -- no
// level is reported, not just no L3 -- so the NUMA node is the only signal left. Verified on cna0001.
BOOST_AUTO_TEST_CASE(cpu_topology_shared_domain_falls_back_when_no_cache_tree) {
    const ScopedTree tree("nocache");
    std::filesystem::create_directories(tree.path()); // exists, but with no cache/ inside
    const auto got = partition::topo_detail::shared_domain_cpus(tree.path().string(), 0, {0});
    // Falls through to this host's real NUMA topology: either unreadable (empty) or a node containing cpu0.
    if (!got.empty()) {
        BOOST_CHECK(std::find(got.begin(), got.end(), 0) != got.end());
    }
}

#endif // __linux__
