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

#include <climits>
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

//! Affinity-mask exchange width, in 64-bit words. A mask needing more is "cannot classify", never private.
inline constexpr size_t kAffinityMaskWords = 64;

static_assert(kAffinityMaskWords > 0 && kAffinityMaskWords <= static_cast<size_t>(INT_MAX),
              "the affinity-mask width is an MPI_Allgather element count, which is an int");

//! This process's allowed cpuset as @p nwords 64-bit words; false with @p out zeroed when it does not fit.
auto affinity_mask_words(uint64_t *out, size_t nwords) -> bool;

/*! @brief Whether the @p n masks of @p words words laid end to end in @p masks are pairwise disjoint.
 *  False for @p n < 2 and for any empty mask: all-zero is disjoint from everything, and shared is the safe error.
 */
[[nodiscard]] auto masks_are_pairwise_disjoint(const uint64_t *masks, size_t n, size_t words) -> bool;

//! Whether the launcher has already handed this rank a private slice of the node, or the node's CPUs are shared.
enum class NodeMask { Shared, PerRank };

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
 * @param mask         NodeMask::PerRank only when the co-located ranks' affinity masks have been measured
 *                     pairwise DISJOINT (PartitionGroup::classify_node_masks_), so this mask is our share.
 * @returns Vector of @p n CpuSet tokens, or empty when @c monoprop_PARTITION_PINNING is disabled,
 *          hwloc is unavailable, or fewer than @p group_count x @p n cores are visible (@p n under PerRank).
 *
 * @note Under NodeMask::PerRank the group split is skipped: our share is already this rank's alone.
 */
auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1, NodeMask mask = NodeMask::Shared)
    -> std::vector<CpuSet>;

/*!
 * @brief Bind the calling thread to the PU identified by @p set.
 *
 * Allocates a temporary hwloc bitmap, sets the single bit for @c set.pu, and calls
 * @c hwloc_set_cpubind with @c HWLOC_CPUBIND_THREAD | @c HWLOC_CPUBIND_STRICT. The call is
 * best-effort: a failure is not an error, because only performance and not correctness depends on
 * successful pinning.
 *
 * @c [[nodiscard]] because a failed bind must not be silent: a launcher that confines the process to
 * fewer PUs than it placed partitions on fails every strict bind, and the surviving count is what lets
 * a run's `pinned=` field tell an unpinned run apart from a pinned one.
 *
 * @param set  Placement token as returned by partition_cpusets(). A token with @c pu == -1
 *             is a no-op.
 * @returns @c true iff the calling thread's affinity was actually restricted to @c set.pu.
 */
[[nodiscard]] auto pin_this_thread(const CpuSet &set) -> bool;
} // namespace monoprop::detail::partition
