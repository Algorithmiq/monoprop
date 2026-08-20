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

namespace {

// An empty placement is licensed by pinning being off and by nothing else; "placed nothing" is the bug.
auto empty_placement_is_licensed() -> bool {
    if (monoprop::config::get().partition_pinning) {
        return false;
    }
    BOOST_TEST_MESSAGE("monoprop_PARTITION_PINNING is off; partition_cpusets places nothing");
    return true;
}

} // namespace

/* ── Live smoke tests ─────────────────────────────────────────────────────── */

BOOST_AUTO_TEST_CASE(cpu_topology_enumerate_and_place) {
    const auto cores = partition::enumerate_physical_cores();

    AffinityGuard guard; // save affinity before any potential pin
    const auto one = partition::partition_cpusets(/*n=*/1);
    BOOST_CHECK(one.size() <= 1u);
    if (!one.empty()) {
        // A placement only comes back when topology discovery succeeded and pinning is enabled.
        BOOST_CHECK(!cores.empty());
        // Reported, not asserted: [[nodiscard]], but a container or launcher cpuset may refuse a strict
        // bind and that is not a defect in the placement policy.
        const bool bound = partition::pin_this_thread(one.front());
        BOOST_TEST_MESSAGE("pin_this_thread: " << (bound ? "bound" : "refused"));
        // guard restores affinity on scope exit
    }
    // When topology discovery succeeds, a non-empty core list must produce a non-empty placement.
    if (!cores.empty() && !(one.empty() && empty_placement_is_licensed())) {
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

    // Both arms passed explicitly so neither depends on the host: refuse the shared one, fill the private.
    const auto shared_mask = partition::partition_cpusets(/*n=*/cores.size(),
                                                          /*group_index=*/1,
                                                          /*group_count=*/2,
                                                          partition::NodeMask::Shared);
    BOOST_CHECK(shared_mask.empty());

    const auto private_mask = partition::partition_cpusets(/*n=*/cores.size(),
                                                           /*group_index=*/1,
                                                           /*group_count=*/2,
                                                           partition::NodeMask::PerRank);
    if (private_mask.empty() && empty_placement_is_licensed()) {
        return;
    }
    BOOST_REQUIRE_EQUAL(private_mask.size(), cores.size());
    std::set<int> placed;
    for (const auto &set : private_mask) {
        placed.insert(set.pu);
    }
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

/* ── The cgroup-placement classification ──────────────────────────────────── */

BOOST_AUTO_TEST_CASE(cpu_topology_policy_per_rank_slice_starves_without_collapse) {
    // One rank's slice under `srun --cpu-bind=cores`: 2 cores of a 16-core host, one L3 domain.
    const std::vector<partition::PhysicalCore> slice = {{6, 0}, {7, 0}};

    BOOST_CHECK(placement_order(slice, 2, /*group_index=*/3, /*group_count=*/8).empty());

    // Collapsed to a single group -- what NodeMask::PerRank does -- the same slice places fully.
    const auto collapsed = placement_order(slice, 2, /*group_index=*/0, /*group_count=*/1);
    BOOST_REQUIRE_EQUAL(collapsed.size(), 2u);
    BOOST_CHECK_EQUAL(collapsed[0], 6);
    BOOST_CHECK_EQUAL(collapsed[1], 7);
}

BOOST_AUTO_TEST_CASE(cpu_topology_policy_private_mask_collapses_even_when_the_split_would_fit) {
    // 2 ranks x 2 partitions fits these 4 cores, so the collapse is not a fallback: it moves rank 1's cores.
    const std::vector<partition::PhysicalCore> cores = {{0, 0}, {2, 1}, {4, 0}, {6, 1}};

    const auto split = placement_order(cores, 2, /*group_index=*/1, /*group_count=*/2);
    BOOST_REQUIRE_EQUAL(split.size(), 2u);
    BOOST_CHECK_EQUAL(split[0], 2);
    BOOST_CHECK_EQUAL(split[1], 6);

    // What NodeMask::PerRank now passes: the head of this rank's own interleave over both domains.
    const auto collapsed = placement_order(cores, 2, /*group_index=*/0, /*group_count=*/1);
    BOOST_REQUIRE_EQUAL(collapsed.size(), 2u);
    BOOST_CHECK_EQUAL(collapsed[0], 0);
    BOOST_CHECK_EQUAL(collapsed[1], 2);
}

namespace {

// The flat [n * words] array MPI_Allgather leaves behind, built from per-rank PU-index lists.
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

    // Partial overlap: "not private" is conservative, since collapsing points every rank at the same cores.
    const auto partial = packed_masks({{0, 1}, {1, 2}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(partial.data(), 2, kWords));

    // An unreadable mask arrives empty and must not be read as "disjoint from everything".
    const auto with_empty = packed_masks({{0, 1}, {}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(with_empty.data(), 2, kWords));

    const auto lone = packed_masks({{0, 1}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(lone.data(), 1, kWords));

    const auto four_ok = packed_masks({{0}, {1}, {2}, {3}}, kWords);
    BOOST_CHECK(partition::masks_are_pairwise_disjoint(four_ok.data(), 4, kWords));
    const auto four_bad = packed_masks({{0}, {1}, {2}, {1}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(four_bad.data(), 4, kWords));
}

BOOST_AUTO_TEST_CASE(cpu_topology_masks_span_word_boundaries) {
    constexpr size_t kWords = partition::kAffinityMaskWords;

    // A per-word comparison that forgot to loop would answer from word 0 alone.
    const auto low_high = packed_masks({{5}, {200}}, kWords);
    BOOST_CHECK(partition::masks_are_pairwise_disjoint(low_high.data(), 2, kWords));

    const auto both_high = packed_masks({{200}, {200}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(both_high.data(), 2, kWords));

    const auto late_overlap = packed_masks({{1, 3000}, {2, 3000}}, kWords);
    BOOST_CHECK(!partition::masks_are_pairwise_disjoint(late_overlap.data(), 2, kWords));
}

BOOST_AUTO_TEST_CASE(cpu_topology_affinity_mask_covers_enumerated_cores) {
    // The only check that the mask EXCHANGED and the cores PLACED come from one view of the machine.
    const auto cores = partition::enumerate_physical_cores();
    if (cores.empty()) {
        return; // hwloc loaded no topology at all: there is no second view to agree with.
    }
    std::vector<uint64_t> mine(partition::kAffinityMaskWords, 0);
    if (!partition::affinity_mask_words(mine.data(), mine.size())) {
        // Not a skip: refusal keys on the HIGHEST allowed PU, and core.cpu is each core's LOWEST sibling.
        std::vector<uint64_t> wide(partition::kAffinityMaskWords * 64, 0);
        BOOST_REQUIRE(partition::affinity_mask_words(wide.data(), wide.size()));
        size_t highest = 0;
        for (size_t w = wide.size(); w-- > 0;) {
            if (wide[w] != 0) {
                highest = (w * 64) + static_cast<size_t>(63 - __builtin_clzll(wide[w]));
                break;
            }
        }
        BOOST_CHECK_GE(highest, partition::kAffinityMaskWords * 64);
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

// Confine to the first `k` cores, asserting the narrowing took hold; false only if the kernel refused.
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

} // namespace

BOOST_AUTO_TEST_CASE(cpu_topology_per_rank_mask_still_places) {
    const auto full = partition::enumerate_physical_cores();
    if (full.empty()) {
        return; // hwloc loaded no topology: there is no mask to confine to.
    }
    // ONE core exercises the collapse, so a single-core runner does not turn this into a free pass.
    const size_t k = std::min<size_t>(full.size(), 2);

    const AffinityGuard guard;
    if (!confine_to_first(full, k)) {
        return; // the kernel refused the affinity call; nothing below is reachable
    }

    // What `srun --cpu-bind=cores` produces: our whole share, told there are eight sibling ranks.
    const auto sets =
        partition::partition_cpusets(/*n=*/k, /*group_index=*/3, /*group_count=*/8, partition::NodeMask::PerRank);
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
        BOOST_CHECK(sets[0].pu != sets[1].pu);
    }
}

BOOST_AUTO_TEST_CASE(cpu_topology_shared_mask_keeps_co_located_ranks_disjoint) {
    const auto full = partition::enumerate_physical_cores();
    if (full.size() < 2) {
        return; // two ranks cannot hold disjoint cores when there is only one
    }
    // Two cores, one per rank: a two-physical-core CI runner cannot supply the four the port asked for.
    const size_t per_rank = full.size() >= 4 ? 2 : 1;
    const size_t shared = 2 * per_rank;

    const AffinityGuard guard;
    if (!confine_to_first(full, shared)) {
        return; // the kernel refused the affinity call; nothing below is reachable
    }

    const auto rank0 = partition::partition_cpusets(/*n=*/per_rank,
                                                    /*group_index=*/0,
                                                    /*group_count=*/2,
                                                    partition::NodeMask::Shared);
    const auto rank1 = partition::partition_cpusets(/*n=*/per_rank,
                                                    /*group_index=*/1,
                                                    /*group_count=*/2,
                                                    partition::NodeMask::Shared);
    if (rank0.empty() && rank1.empty() && empty_placement_is_licensed()) {
        return;
    }
    BOOST_REQUIRE_EQUAL(rank0.size(), per_rank);
    BOOST_REQUIRE_EQUAL(rank1.size(), per_rank);
    for (const auto &a : rank0) {
        for (const auto &b : rank1) {
            // Two ranks on one core starve each other: busy-polling collectives against barrier spins.
            BOOST_CHECK(a.pu != b.pu);
        }
    }
}

#endif // __linux__
