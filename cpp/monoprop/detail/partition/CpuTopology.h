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

/*!
 * @file CpuTopology.h
 * @brief CPU-topology helpers for partition placement.
 *
 * Uses hwloc for cross-platform topology discovery and thread binding.
 * Policy: one partition per physical core, spread across L3/CCX domains, each worker thread pinned
 * to its representative PU. Falls back to unpinned execution when hwloc cannot load the topology or
 * when binding is unsupported — pinning is a performance optimisation, not a correctness requirement.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::detail::partition {

/*!
 * @brief One physical CPU core the process may use, tagged with its L3 cache domain.
 *
 * Produced by enumerate_physical_cores(). The representative PU is the lowest OS index in the
 * calling thread's allowed affinity mask for the core, so a cgroup or launcher-imposed restriction
 * is always respected.
 */
struct PhysicalCore {
    int cpu = 0;       //!< OS index of the representative PU (lowest allowed SMT sibling of the core).
    int l3_domain = 0; //!< Sequential L3-cache domain id assigned by enumerate_physical_cores().
};

/*!
 * @brief Lightweight placement token: identifies the single PU a partition worker thread is pinned to.
 */
struct CpuSet {
    int pu = -1; //!< OS PU index. -1 ⇒ invalid / not placed.
};

namespace topo_detail {
/*!
 * @brief Apply the L3-domain interleaving placement policy to a synthetic core list.
 *
 * Factored out of partition_cpusets() so the scheduling logic can be exercised without depending
 * on hwloc or live hardware. Cores are bucketed by their l3_domain, then interleaved depth-first
 * across the domains dealt to this rank. When there are more co-located ranks than L3 domains the
 * algorithm falls back to flat domain-major order with one contiguous slice per rank.
 *
 * @param cores  Physical cores available to allocate from, with L3 domain tags.
 * @param n      Number of partitions (PUs) requested for this rank.
 * @param group_index  This rank's 0-based index among the co-located ranks on the host.
 * @param group_count  Total number of co-located ranks on the host.
 * @returns Ordered PU OS indices of length @p n for the slice assigned to this rank,
 *          or empty when the request cannot be filled (empty core list, oversubscription,
 *          or computed offset out of range).
 */
auto placement_order(const std::vector<PhysicalCore> &cores, size_t n, size_t group_index, size_t group_count)
    -> std::vector<int>;
} // namespace topo_detail

/*!
 * @brief Enumerate physical cores (one per SMT sibling group) the process may use.
 *
 * Queries the calling thread's CPU affinity via hwloc to respect any cgroup or launcher-imposed
 * restriction. For each core whose cpuset intersects that affinity mask, the lowest matching OS
 * index is recorded as the representative PU and the core is labelled with the sequential id of
 * its nearest L3 cache ancestor (or a unique singleton id when no L3 object is present).
 *
 * @returns Vector of PhysicalCore in hwloc logical-core order, or empty when hwloc cannot load
 *          the topology or when no core passes the affinity filter.
 *
 * @note This function deliberately ignores @c monoprop_PARTITION_PINNING so that the auto
 *       partition-count heuristic (one partition per physical core) works even when pinning is
 *       disabled by the user.
 */
auto enumerate_physical_cores() -> std::vector<PhysicalCore>;

/*!
 * @brief Number of 64-bit words used to exchange an affinity mask between co-located ranks.
 *
 * 64 words is 4096 CPUs. A mask needing more is reported as "cannot classify" by
 * affinity_mask_words(), never as private, so the fallback stays off and the old unplaced
 * behaviour returns rather than a wrong placement.
 *
 * Declared here, beside the two functions that must agree on it, rather than privately in the
 * caller: a width that lives next to only one of the two can drift silently.
 */
inline constexpr size_t kAffinityMaskWords = 64;

/*!
 * @brief This process's effective allowed cpuset, as a bit array of @p nwords 64-bit words.
 *
 * Exposed so callers can exchange masks between co-located ranks and test them for pairwise
 * disjointness -- the only sound basis for deciding whether each rank owns a private share.
 *
 * @returns false (leaving @p out zeroed) when hwloc cannot load the topology or the mask needs
 *          more than @p nwords words, which is treated as "cannot classify" and hence not private.
 */
auto affinity_mask_words(uint64_t *out, size_t nwords) -> bool;

/*!
 * @brief Whether @p n masks of @p words words each, laid end to end in @p masks, are pairwise
 *        disjoint AND none of them is empty.
 *
 * This is the whole classification rule, and it is pure bit arithmetic over an already-gathered
 * array: no communicator, no hwloc, no live hardware. It is a free function precisely so it can be
 * tested, because the mechanism it implements has regressed once already -- when topology discovery
 * moved onto hwloc -- and the guard lives in the placement policy rather than in discovery.
 *
 * Mask WIDTH cannot substitute for this. "8 ranks holding 16 cores each" and "8 ranks sharing one
 * 16-core mask" both leave a rank seeing 16 of 128 CPUs, and they need opposite placements: the
 * first should fill its own mask or nothing is pinned at all, the second must not or every rank
 * lands on identical cores and their busy-polling collectives starve each other.
 *
 * The empty-mask clause is not incidental. An all-zero mask is trivially disjoint from everything,
 * so a bare disjointness test would answer "private" for a rank that can see no CPU at all --
 * collapsing group_count in exactly the case where the caller knows least. Shared is the safe
 * error, so an empty mask is not private.
 *
 * @returns false when @p n < 2 (nobody to collide with, and the normal split already handles it),
 *          when any mask is empty, or when any two masks share a bit.
 */
[[nodiscard]] auto masks_are_pairwise_disjoint(const uint64_t *masks, size_t n, size_t words) -> bool;

/*!
 * @brief Build placement tokens for one MPI rank's partitions.
 *
 * Calls enumerate_physical_cores() and applies topo_detail::placement_order() to select @p n
 * distinct physical cores for this rank. Two co-located ranks must never share a core: one rank's
 * busy-polling collectives would starve the other's barrier spins.
 *
 * @param n            Number of partitions to place.
 * @param group_index  This rank's 0-based index among the co-located ranks on the host.
 * @param group_count  Total number of co-located ranks on the host.
 * @param mask_is_private  True only when the caller has established that the co-located ranks'
 *                     affinity masks are pairwise DISJOINT, so this rank's mask is its own share of
 *                     the node. See PartitionGroup::discover_node_peers_(), which allgathers the
 *                     masks over its node communicator to decide.
 * @returns Vector of @p n CpuSet tokens, or empty when @c monoprop_PARTITION_PINNING is disabled,
 *          hwloc is unavailable, or not even @p n distinct cores are visible to this process.
 *
 * @note With @p mask_is_private, @p group_index / @p group_count are ignored once the normal split
 *       has failed: the batch system has already partitioned the node (Slurm with per-rank cgroups),
 *       so subdividing our own share a second time asks for @p group_count × more cores than exist
 *       and places nothing at all. Disjointness is what makes ignoring them safe. Without it, a set
 *       of ranks SHARING one narrow mask would every one of them collapse onto the same cores --
 *       worse than not pinning, since each rank's busy-polling collectives would then starve the
 *       others' barrier spins.
 */
auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1, bool mask_is_private = false)
    -> std::vector<CpuSet>;

/*!
 * @brief Bind the calling thread to the PU identified by @p set.
 *
 * Allocates a temporary hwloc bitmap, sets the single bit for @c set.pu, and calls
 * @c hwloc_set_cpubind with @c HWLOC_CPUBIND_THREAD | @c HWLOC_CPUBIND_STRICT. The call is
 * best-effort: hwloc errors are silently ignored because only performance, not correctness,
 * depends on successful pinning.
 *
 * @param set  Placement token as returned by partition_cpusets(). A token with @c pu == -1
 *             is a no-op.
 */
auto pin_this_thread(const CpuSet &set) -> void;
} // namespace monoprop::detail::partition
