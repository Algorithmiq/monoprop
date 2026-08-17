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
#include <cstdint>
#include <set>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "monoprop/detail/EnvConfig.h" // config::get().partition_pinning -- the one licensed empty placement
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
        partition::pin_this_thread(one.front());
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

    // 2 ranks × cores.size() partitions asks for more cores than this process can see. What that
    // MEANS depends on whether the co-located ranks' masks are disjoint, which no local test can
    // answer, so partition_cpusets takes it as an argument and both arms are pinned here:
    //
    //   shared mask  → a genuine oversubscription. Refuse: placing anyway would hand both ranks the
    //                  same cores, and each rank's busy-polling collectives would starve the other's
    //                  barrier spins — worse than not pinning at all.
    //   private mask → the batch system already split the node (Slurm per-rank cgroups) and this is
    //                  our own share. Refusing here is exactly what left every rank unpinned on the
    //                  benchmark nodes, voiding four A/B jobs. Fill our mask.
    //
    // Passing the flag explicitly is what makes this deterministic on any machine; the earlier
    // version branched on the ambient environment and so asserted whichever answer the host gave.
    const auto shared_mask = partition::partition_cpusets(/*n=*/cores.size(),
                                                          /*group_index=*/1,
                                                          /*group_count=*/2,
                                                          /*mask_is_private=*/false);
    BOOST_CHECK(shared_mask.empty());

    const auto private_mask = partition::partition_cpusets(/*n=*/cores.size(),
                                                           /*group_index=*/1,
                                                           /*group_count=*/2,
                                                           /*mask_is_private=*/true);
    BOOST_REQUIRE_EQUAL(private_mask.size(), cores.size());
    std::set<int> placed;
    for (const auto &set : private_mask) {
        placed.insert(set.pu);
    }
    // Distinct PUs, and every one of them a core this process was actually granted.
    BOOST_CHECK_EQUAL(placed.size(), cores.size());
    std::set<int> visible;
    for (const auto &core : cores) {
        visible.insert(core.cpu);
    }
    for (const int pu : placed) {
        BOOST_CHECK(visible.count(pu) == 1);
    }
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

/* ── The cgroup-placement classification ──────────────────────────────────────
 *
 * These are ported from the competing implementation of this fix on
 * perf/multinode-comm-scaling, which spelled the rule as a `NodeMask` enum over a `CpuMask` POD.
 * The mechanism is the same; only the surface differs, so the cases transfer with the enum
 * replaced by `masks_are_pairwise_disjoint` over a flat array and by `partition_cpusets`'
 * `mask_is_private` flag.
 *
 * They are worth porting because this branch adds NO hardware-free coverage of its own -- it
 * extends one live-topology case, which early-returns on a runner with fewer than two visible
 * cores, so the whole fix can be unexercised while the suite reports green. And the mechanism
 * has regressed once already, when topology discovery moved onto hwloc, because the guard lives
 * in the placement policy rather than in discovery.
 */

BOOST_AUTO_TEST_CASE(cpu_topology_policy_per_rank_slice_starves_without_collapse) {
    // One rank's slice under `srun --cpu-bind=cores`: 2 cores of a 16-core host, one L3 domain.
    const std::vector<partition::PhysicalCore> slice = {{6, 0}, {7, 0}};

    // Told the node-wide truth (8 ranks x 2 partitions), the request cannot be met from a slice.
    BOOST_CHECK(placement_order(slice, 2, /*group_index=*/3, /*group_count=*/8).empty());

    // Collapsed to a single group -- what mask_is_private does -- the same slice places fully.
    const auto collapsed = placement_order(slice, 2, /*group_index=*/0, /*group_count=*/1);
    BOOST_REQUIRE_EQUAL(collapsed.size(), 2u);
    BOOST_CHECK_EQUAL(collapsed[0], 6);
    BOOST_CHECK_EQUAL(collapsed[1], 7);
}

namespace {

/* Build the flat [n * words] array masks_are_pairwise_disjoint reads, from a list of PU-index
 * lists -- the shape MPI_Allgather leaves behind. */
auto packed_masks(const std::vector<std::vector<size_t>> &pus, size_t words) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(pus.size() * words, 0);
    for (size_t r = 0; r < pus.size(); ++r) {
        for (const size_t pu : pus[r]) {
            out[(r * words) + (pu / 64)] |= uint64_t{1} << (pu % 64);
        }
    }
    return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(cpu_topology_masks_disjoint_vs_identical) {
    constexpr size_t kWords = partition::kAffinityMaskWords;

    const auto disjoint = packed_masks({{0, 1}, {2, 3}}, kWords);
    BOOST_CHECK(partition::masks_are_pairwise_disjoint(disjoint.data(), 2, kWords));

    const auto identical = packed_masks({{0, 1}, {0, 1}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(identical.data(), 2, kWords));

    // Partial overlap is pathological; "not private" is the conservative answer, because dividing
    // never double-books a core within a rank whereas collapsing points every rank at the same ones.
    const auto partial = packed_masks({{0, 1}, {1, 2}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(partial.data(), 2, kWords));

    // An unreadable mask arrives empty and must not be read as "disjoint from everything".
    const auto with_empty = packed_masks({{0, 1}, {}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(with_empty.data(), 2, kWords));

    // A lone rank has nobody to be disjoint from, and the normal split already handles it.
    const auto lone = packed_masks({{0, 1}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(lone.data(), 1, kWords));

    // More than two peers: one overlapping pair anywhere is enough to make the node shared.
    const auto four_ok = packed_masks({{0}, {1}, {2}, {3}}, kWords);
    BOOST_CHECK(partition::masks_are_pairwise_disjoint(four_ok.data(), 4, kWords));
    const auto four_bad = packed_masks({{0}, {1}, {2}, {1}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(four_bad.data(), 4, kWords));
}

BOOST_AUTO_TEST_CASE(cpu_topology_masks_span_word_boundaries) {
    constexpr size_t kWords = partition::kAffinityMaskWords;

    // Bits in different 64-bit words: a per-word comparison that forgot to loop would call these
    // disjoint by luck on the first word and overlapping is what must be detected on the third.
    const auto low_high = packed_masks({{5}, {200}}, kWords);
    BOOST_CHECK(partition::masks_are_pairwise_disjoint(low_high.data(), 2, kWords));

    const auto both_high = packed_masks({{200}, {200}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(both_high.data(), 2, kWords));

    // Overlap in a later word than the first difference.
    const auto late_overlap = packed_masks({{1, 3000}, {2, 3000}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(late_overlap.data(), 2, kWords));

    // The truncation guard itself -- a PU index past the exchanged window making
    // affinity_mask_words report "cannot classify" -- needs a live hwloc mask and is not
    // reachable from here. It is covered only by the all_ok reduction in
    // PartitionGroup::classify_node_masks_, which needs a communicator.
}

BOOST_AUTO_TEST_CASE(cpu_topology_affinity_mask_covers_enumerated_cores) {
    // The only check that the mask EXCHANGED and the cores PLACED come from the same view of the
    // machine. affinity_mask_words otherwise has no test caller at all.
    const auto cores = partition::enumerate_physical_cores();
    if (cores.empty()) {
        return; // hwloc loaded no topology at all: there is no second view to agree with.
    }
    std::vector<uint64_t> mine(partition::kAffinityMaskWords, 0);
    if (!partition::affinity_mask_words(mine.data(), mine.size())) {
        /* NOT a free skip. enumerate_physical_cores just succeeded, so hwloc works here and the
         * only licensed reason to refuse is the truncation guard: some allowed PU sits past the
         * exchanged window. Assert that, or a mask function that had simply stopped working would
         * read as "nothing to test" -- which is the failure mode this whole file exists to close. */
        int highest = 0;
        for (const auto &core : cores) {
            highest = std::max(highest, core.cpu);
        }
        BOOST_CHECK_GE(static_cast<size_t>(highest), partition::kAffinityMaskWords * 64);
        return;
    }
    size_t set_bits = 0;
    for (const uint64_t w : mine) {
        set_bits += static_cast<size_t>(__builtin_popcountll(w));
    }
    BOOST_CHECK(set_bits > 0u);
    for (const auto &core : cores) {
        const auto pu = static_cast<size_t>(core.cpu);
        BOOST_CHECK((mine[pu / 64] >> (pu % 64)) & 1U);
    }
}

#if defined(__linux__)

namespace {

/* Confine this thread to the first `k` enumerated cores and assert the premise took hold.
 *
 * The premise is not free. effective_allowed_cpuset() re-reads hwloc_get_cpubind() on every call,
 * so a successful sched_setaffinity DOES narrow what enumerate_physical_cores() reports -- but if
 * it ever stopped doing so, the two cases below would keep asserting, against the UNCONFINED
 * machine, and keep passing. That is the same defect as an early return, only harder to see, so
 * the narrowing is checked rather than assumed.
 *
 * @returns false when the kernel refused the call (seccomp, a restrictive cgroup), which is the
 *          one condition here that no assertion can rescue. */
auto confine_to_first(const std::vector<partition::PhysicalCore> &full, size_t k) -> bool {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (size_t i = 0; i < k; ++i) {
        CPU_SET(full[i].cpu, &mask);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        return false;
    }
    BOOST_REQUIRE_EQUAL(partition::enumerate_physical_cores().size(), k);
    return true;
}

/* An empty placement is licensed by exactly one thing -- pinning turned off -- and by nothing
 * else. Spelled as a check so that "placed nothing", the bug this branch fixes, can never be
 * mistaken for a configuration. */
auto empty_placement_is_licensed() -> bool {
    if (monoprop::config::get().partition_pinning) {
        return false;
    }
    BOOST_TEST_MESSAGE("monoprop_PARTITION_PINNING is off; partition_cpusets places nothing");
    return true;
}

} // namespace

BOOST_AUTO_TEST_CASE(cpu_topology_per_rank_mask_still_places) {
    const auto full = partition::enumerate_physical_cores();
    if (full.empty()) {
        return; // hwloc loaded no topology: there is no mask to confine to.
    }
    /* ONE core is enough to exercise the collapse -- the request is group_count x n out of a list
     * holding n either way. Requiring two would let this pass vacuously on a single-core runner,
     * which is exactly the property that made the live case it was ported alongside worthless. */
    const size_t k = std::min<size_t>(full.size(), 2);

    const AffinityGuard guard;
    if (!confine_to_first(full, k)) {
        return; // the kernel refused the affinity call; nothing below is reachable
    }

    // n = the whole share, and a group_count as if seven sibling ranks shared the node. This is
    // the configuration `srun --cpu-bind=cores` produces, and without the collapse it places
    // NOTHING: the request is group_count x n cores out of a list holding n.
    const auto sets =
        partition::partition_cpusets(/*n=*/k, /*group_index=*/3, /*group_count=*/8, /*mask_is_private=*/true);
    if (sets.empty() && empty_placement_is_licensed()) {
        return;
    }
    BOOST_REQUIRE_EQUAL(sets.size(), k);
    for (const auto &set : sets) {
        // Never pin outside the mask the launcher gave us.
        bool inside = false;
        for (size_t i = 0; i < k; ++i) {
            inside = inside || set.pu == full[i].cpu;
        }
        BOOST_CHECK(inside);
    }
    if (k == 2) {
        // The two partitions must not land on the same core.
        BOOST_CHECK(sets[0].pu != sets[1].pu);
    }
}

BOOST_AUTO_TEST_CASE(cpu_topology_shared_mask_keeps_co_located_ranks_disjoint) {
    const auto full = partition::enumerate_physical_cores();
    if (full.size() < 2) {
        return; // two ranks cannot hold disjoint cores when there is only one
    }
    /* Two cores are enough: one per rank. The ported version asked for four, which a two-physical-
     * core CI runner does not have -- it would have returned here and reported green. */
    const size_t per_rank = full.size() >= 4 ? 2 : 1;
    const size_t shared = 2 * per_rank;

    const AffinityGuard guard;
    if (!confine_to_first(full, shared)) {
        return; // the kernel refused the affinity call; nothing below is reachable
    }

    const auto rank0 = partition::partition_cpusets(/*n=*/per_rank,
                                                    /*group_index=*/0,
                                                    /*group_count=*/2,
                                                    /*mask_is_private=*/false);
    const auto rank1 = partition::partition_cpusets(/*n=*/per_rank,
                                                    /*group_index=*/1,
                                                    /*group_count=*/2,
                                                    /*mask_is_private=*/false);
    if (rank0.empty() && rank1.empty() && empty_placement_is_licensed()) {
        return;
    }
    BOOST_REQUIRE_EQUAL(rank0.size(), per_rank);
    BOOST_REQUIRE_EQUAL(rank1.size(), per_rank);
    for (const auto &a : rank0) {
        for (const auto &b : rank1) {
            // Two ranks sharing a core would have each one's busy-polling collectives starve the
            // other's barrier spins -- the failure this whole placement path exists to avoid.
            BOOST_CHECK(a.pu != b.pu);
        }
    }
}

#endif // __linux__
