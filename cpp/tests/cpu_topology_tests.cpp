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

// Coverage of CpuTopology (hwloc-based topology discovery + thread affinity pinning).
//
// Tests are split into two layers:
//   1. Live smoke tests — exercise enumerate_physical_cores / partition_cpusets / pin_this_thread
//      on the actual host topology; these validate end-to-end hwloc integration.
//   2. Policy unit tests — call topo_detail::placement_order with synthetic PhysicalCore vectors
//      so the L3-domain interleaving and MPI-rank slicing logic can be checked deterministically
//      without depending on live hardware or hwloc.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "monoprop/detail/partition/CpuTopology.h"

namespace partition = monoprop::detail::partition;
using partition::topo_detail::placement_order;

/* RAII helper: save and restore the calling thread's CPU affinity around pin_this_thread() calls
 * so that CTest is not left pinned to a single PU after the test completes. */
struct AffinityGuard {
#if defined(__linux__)
    cpu_set_t saved_{};
    AffinityGuard() { sched_getaffinity(0, sizeof(saved_), &saved_); }
    ~AffinityGuard() { sched_setaffinity(0, sizeof(saved_), &saved_); }
#endif
};

/* ── Live smoke tests ─────────────────────────────────────────────────────── */

BOOST_AUTO_TEST_CASE(cpu_topology_enumerate_and_place) {
    const auto cores = partition::enumerate_physical_cores();

    AffinityGuard guard; // save affinity before any potential pin
    const auto one = partition::partition_cpusets(/*n=*/1);
    BOOST_CHECK(one.size() <= 1u);
    if (!one.empty()) {
        // A placement only comes back when topology discovery succeeded and pinning is enabled.
        BOOST_CHECK(!cores.empty());
        // Not asserted: pinning is best-effort, and a restrictive cgroup or seccomp policy can refuse
        // it without that being a defect. The return value exists so the outcome is reportable (see
        // CommProfile's pinned count), not so a test can require success.
        static_cast<void>(partition::pin_this_thread(one.front()));
        // guard restores affinity on scope exit
    }
    // When topology discovery succeeds, a non-empty core list must produce a non-empty placement.
    if (!cores.empty()) {
        BOOST_CHECK_EQUAL(one.size(), 1u);
    }

    // Oversubscription must always return empty regardless of topology state.
    const auto too_many = partition::partition_cpusets(/*n=*/1'000'000);
    BOOST_CHECK(too_many.empty());
}

BOOST_AUTO_TEST_CASE(cpu_topology_place_co_located_ranks) {
    const auto cores = partition::enumerate_physical_cores();
    if (cores.size() < 2) {
        return; // need at least two cores for the disjoint-placement check
    }

    // Two co-located ranks each requesting one partition. This exercises both placement arms:
    //   - interleave (group_count ≤ #L3 domains)
    //   - domain-major slice (group_count > #L3 domains)
    const auto rank0 = partition::partition_cpusets(/*n=*/1, /*group_index=*/0, /*group_count=*/2);
    const auto rank1 = partition::partition_cpusets(/*n=*/1, /*group_index=*/1, /*group_count=*/2);
    if (rank0.empty() || rank1.empty()) {
        BOOST_CHECK(rank0.empty());
        BOOST_CHECK(rank1.empty());
        return;
    }
    BOOST_REQUIRE_EQUAL(rank0.size(), 1u);
    BOOST_REQUIRE_EQUAL(rank1.size(), 1u);
    // The two placements must be on distinct PUs; sharing would violate the MPI no-starvation
    // invariant (one rank's busy-polling collectives cannot starve the other's barrier spins).
    BOOST_CHECK(rank0.front().pu != rank1.front().pu);

    // Oversubscription: 2 ranks × cores.size() partitions > total physical cores.
    const auto past_end = partition::partition_cpusets(/*n=*/cores.size(), /*group_index=*/1, /*group_count=*/2);
    BOOST_CHECK(past_end.empty());
}

/* ── Policy unit tests (deterministic, no hwloc or live hardware) ─────────── */

BOOST_AUTO_TEST_CASE(cpu_topology_policy_interleave_across_l3) {
    // 4 cores across 2 L3 domains; single rank receives all.
    // by_domain[0] = {0, 4}, by_domain[1] = {2, 6}
    // depth-first interleave: 0, 2, 4, 6
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {2, 1}, {4, 0}, {6, 1}};
    const auto order = placement_order(cores, 4, 0, 1);
    BOOST_REQUIRE_EQUAL(order.size(), 4u);
    BOOST_CHECK_EQUAL(order[0], 0);
    BOOST_CHECK_EQUAL(order[1], 2);
    BOOST_CHECK_EQUAL(order[2], 4);
    BOOST_CHECK_EQUAL(order[3], 6);
}

BOOST_AUTO_TEST_CASE(cpu_topology_policy_disjoint_mpi_ranks) {
    // 4 cores across 2 L3 domains; 2 co-located ranks each get 1 partition.
    // rank0 is dealt domain 0, rank1 is dealt domain 1 ⇒ no shared PU.
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {2, 1}, {4, 0}, {6, 1}};
    const auto r0 = placement_order(cores, 1, 0, 2);
    const auto r1 = placement_order(cores, 1, 1, 2);
    BOOST_REQUIRE_EQUAL(r0.size(), 1u);
    BOOST_REQUIRE_EQUAL(r1.size(), 1u);
    BOOST_CHECK(r0.front() != r1.front());
}

BOOST_AUTO_TEST_CASE(cpu_topology_policy_domain_major_more_ranks_than_l3) {
    // 4 cores in 1 L3 domain; 2 ranks each get 2 partitions (flat domain-major arm).
    // order = [0, 2, 4, 6]; rank0 offset=0 → {0,2}, rank1 offset=2 → {4,6}.
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {2, 0}, {4, 0}, {6, 0}};
    const auto r0 = placement_order(cores, 2, 0, 2);
    const auto r1 = placement_order(cores, 2, 1, 2);
    BOOST_REQUIRE_EQUAL(r0.size(), 2u);
    BOOST_REQUIRE_EQUAL(r1.size(), 2u);

    const std::set<int> s0(r0.begin(), r0.end());
    const std::set<int> s1(r1.begin(), r1.end());
    for (const auto cpu : s1) {
        BOOST_CHECK(!s0.contains(cpu));
    }
}

BOOST_AUTO_TEST_CASE(cpu_topology_policy_insufficient_cores_returns_empty) {
    // 2 cores total; 2 ranks × 2 partitions = 4 > 2 ⇒ oversubscription.
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {2, 0}};
    BOOST_CHECK(placement_order(cores, 2, 0, 2).empty());
    BOOST_CHECK(placement_order(cores, 2, 1, 2).empty());

    // Single rank requesting more cores than exist.
    BOOST_CHECK(placement_order(cores, 3, 0, 1).empty());

    // Empty core list.
    BOOST_CHECK(placement_order({}, 1, 0, 1).empty());
}

BOOST_AUTO_TEST_CASE(cpu_topology_policy_singleton_l3_domains) {
    // 2 cores each in its own singleton domain (no shared L3).
    // by_domain[0] = {0}, by_domain[1] = {4}; interleaved: 0, 4.
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {4, 1}};
    const auto order = placement_order(cores, 2, 0, 1);
    BOOST_REQUIRE_EQUAL(order.size(), 2u);
    BOOST_CHECK_EQUAL(order[0], 0);
    BOOST_CHECK_EQUAL(order[1], 4);
}

BOOST_AUTO_TEST_CASE(cpu_topology_policy_uneven_domains) {
    // 3 cores: 2 in domain 0, 1 in domain 1; single rank, 3 partitions.
    // by_domain[0] = {0, 4}, by_domain[1] = {2}; interleaved: 0, 2, 4.
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {2, 1}, {4, 0}};
    const auto order = placement_order(cores, 3, 0, 1);
    BOOST_REQUIRE_EQUAL(order.size(), 3u);
    BOOST_CHECK_EQUAL(order[0], 0);
    BOOST_CHECK_EQUAL(order[1], 2);
    BOOST_CHECK_EQUAL(order[2], 4);
}

/* ── Node-mask classification and the per-rank collapse ───────────────────── */

// The mechanism the collapse in partition_cpusets exists for, stated without needing live hardware.
// Under a per-rank launcher split, enumerate_physical_cores() already reports only this rank's slice,
// so passing the node-wide group_count through asks placement_order for group_count x n cores out of
// a list that only ever held n -- and it correctly refuses. Refusing means unpinned, which also
// costs the two-level barrier its domains: measured at 437 us/sync against 15.5 us/sync placed.
BOOST_AUTO_TEST_CASE(cpu_topology_policy_per_rank_slice_starves_without_collapse) {
    // One rank's slice under `srun --cpu-bind=cores`: 2 cores of a 16-core host, one L3 domain.
    const std::vector<partition::PhysicalCore> slice = {{6, 0}, {7, 0}};

    // Told the node-wide truth (8 ranks x 2 partitions), the request cannot be met from a slice.
    BOOST_CHECK(placement_order(slice, 2, /*group_index=*/3, /*group_count=*/8).empty());

    // Collapsed to a single group -- what NodeMask::PerRank does -- the same slice places fully.
    const auto collapsed = placement_order(slice, 2, /*group_index=*/0, /*group_count=*/1);
    BOOST_REQUIRE_EQUAL(collapsed.size(), 2u);
    BOOST_CHECK_EQUAL(collapsed[0], 6);
    BOOST_CHECK_EQUAL(collapsed[1], 7);
}

// Two ranks holding disjoint masks are a per-rank split; identical masks are a shared one. Nothing else
// distinguishes them, which is the whole reason this is measured rather than inferred: mask *width*
// cannot tell "8 ranks holding 16 cores each" from "8 ranks sharing one 16-core mask", and the two need
// opposite placement.
BOOST_AUTO_TEST_CASE(cpu_topology_classify_node_mask_disjoint_vs_identical) {
    partition::CpuMask a;
    partition::CpuMask b;
    partition::cpumask_set(a, 0);
    partition::cpumask_set(a, 1);
    partition::cpumask_set(b, 2);
    partition::cpumask_set(b, 3);
    BOOST_CHECK(partition::classify_node_mask({a, b}) == partition::NodeMask::PerRank);
    BOOST_CHECK(partition::classify_node_mask({a, a}) == partition::NodeMask::Shared);

    // Partial overlap is pathological; Shared is the conservative answer because dividing never
    // double-books a core within a rank, whereas collapsing points every rank at the same cores.
    partition::CpuMask c = a;
    partition::cpumask_set(c, 2);
    BOOST_CHECK(partition::classify_node_mask({a, c}) == partition::NodeMask::Shared);

    // An unreadable mask arrives empty and must not be read as "disjoint from everything".
    const partition::CpuMask empty;
    BOOST_CHECK(partition::classify_node_mask({a, empty}) == partition::NodeMask::Shared);
    // A lone rank has nobody to be disjoint from.
    BOOST_CHECK(partition::classify_node_mask({a}) == partition::NodeMask::Shared);
}

// Disjointness must be tested per PU, not by popcount or by first/last index: two masks of equal size
// in different words are disjoint, and two overlapping in one word are not.
BOOST_AUTO_TEST_CASE(cpu_topology_classify_node_mask_spans_word_boundaries) {
    partition::CpuMask low;
    partition::CpuMask high;
    partition::cpumask_set(low, 5);
    partition::cpumask_set(high, 200); // a different 64-bit word
    BOOST_CHECK(partition::classify_node_mask({low, high}) == partition::NodeMask::PerRank);
    BOOST_CHECK_EQUAL(partition::cpumask_count(low), 1u);

    partition::CpuMask also_high;
    partition::cpumask_set(also_high, 200);
    BOOST_CHECK(partition::classify_node_mask({high, also_high}) == partition::NodeMask::Shared);

    // A PU index past the mask is dropped rather than wrapping onto an unrelated bit.
    partition::CpuMask overflow;
    partition::cpumask_set(overflow, partition::kCpuMaskBits + 7);
    BOOST_CHECK_EQUAL(partition::cpumask_count(overflow), 0u);
    BOOST_CHECK(!partition::cpumask_test(overflow, 7));
}

// This thread's mask must be non-empty on any host where hwloc found cores, and must agree with the
// cores enumerate_physical_cores() reports -- the two are read from the same hwloc cpuset, so a
// disagreement means the mask exchange would classify against a different machine than the placement.
BOOST_AUTO_TEST_CASE(cpu_topology_this_thread_cpumask_covers_enumerated_cores) {
    const auto cores = partition::enumerate_physical_cores();
    const auto mine = partition::this_thread_cpumask();
    if (cores.empty()) {
        return; // no topology (hwloc unavailable); nothing to agree with
    }
    BOOST_CHECK(partition::cpumask_count(mine) > 0u);
    for (const auto &core : cores) {
        BOOST_CHECK(partition::cpumask_test(mine, static_cast<size_t>(core.cpu)));
    }
}

/* ── Live placement under a restricted mask ───────────────────────────────── */

#if defined(__linux__)

// A Slurm-style per-rank confinement, end to end: narrow this thread's affinity to two cores, then ask
// for both of them while passing the node-wide group_count a launcher would report. Nothing asserted
// this before the collapse landed, which is why the bug surfaced only through monoprop_COMM_PROFILE.
BOOST_AUTO_TEST_CASE(cpu_topology_per_rank_mask_still_places) {
    const auto full = partition::enumerate_physical_cores();
    if (full.size() < 2) {
        return; // need two cores to confine to
    }

    const AffinityGuard guard;

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
    BOOST_REQUIRE_EQUAL(sets.size(), 2u);
    for (const auto &set : sets) {
        // Never pin outside the mask the launcher gave us.
        BOOST_CHECK(set.pu == full[0].cpu || set.pu == full[1].cpu);
    }
    // The two partitions must not land on the same core, and each must carry a domain for the barrier.
    BOOST_CHECK(sets[0].pu != sets[1].pu);
    BOOST_CHECK_EQUAL(partition::cpuset_domains(sets).size(), 2u);
}

// The invariant the PerRank collapse most endangers: under a genuinely Shared mask, two co-located
// ranks must still deal themselves DIFFERENT cores. Collapsing unconditionally on "the mask is narrower
// than the machine" pointed every rank at the same cores, because mask width cannot distinguish a
// per-rank slice from a shared one -- which is why the caller measures it instead.
BOOST_AUTO_TEST_CASE(cpu_topology_shared_mask_keeps_co_located_ranks_disjoint) {
    const auto full = partition::enumerate_physical_cores();
    if (full.size() < 4) {
        return; // need two cores per rank for two ranks
    }

    const AffinityGuard guard;

    // A shared mask narrower than the host: four cores that both ranks can see.
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
    BOOST_REQUIRE_EQUAL(rank0.size(), 2u);
    BOOST_REQUIRE_EQUAL(rank1.size(), 2u);
    for (const auto &a : rank0) {
        for (const auto &b : rank1) {
            // Two ranks sharing a core would have each one's busy-polling collectives starve the other's
            // barrier spins -- the failure this whole placement path exists to avoid.
            BOOST_CHECK(a.pu != b.pu);
        }
    }
}

#endif // __linux__
