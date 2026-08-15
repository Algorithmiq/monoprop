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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/detail/EnvConfig.h"

namespace monoprop::detail::partition {

/*!
 * @brief How the launcher divided this host's CPUs among the ranks sharing it.
 *
 * Mask *size* alone cannot tell the two apart — four ranks holding 32 cores each and four ranks
 * sharing one 32-core mask both report 32 — and they need opposite placement. So the caller that
 * owns a node-local communicator measures this by exchanging masks rather than inferring it.
 * See partition_cpusets() and classify_node_mask().
 */
enum class NodeMask {
    Shared,  //!< Every co-located rank sees the same mask (`mpirun --bind-to none`, or a lone process).
    PerRank, //!< Each co-located rank was given its own disjoint slice (`srun --cpu-bind=cores`).
};

/*!
 * @brief Number of PU indices a CpuMask can represent.
 *
 * Larger than any current machine's PU count, and generous relative to glibc's @c CPU_SETSIZE of
 * 1024, because a PU index beyond the mask is silently invisible to classify_node_mask() — which
 * would misread a per-rank split as shared.
 */
inline constexpr size_t kCpuMaskBits = 4096;

/*!
 * @brief A fixed-size affinity mask, as a trivially copyable POD.
 *
 * Deliberately not an hwloc bitmap: the node-local peer exchange in PartitionGroup ships this
 * straight through @c MPI_Allgather as @c MPI_BYTE, so it must have the same size and layout in
 * every rank and own no heap. Keeping it hwloc-free also lets classify_node_mask() be unit-tested
 * without live hardware.
 */
struct CpuMask {
    std::array<uint64_t, kCpuMaskBits / 64> words{}; //!< Bit @c i of word @c i/64 ⇒ PU @c i is allowed.
};

/*!
 * @brief Set the bit for PU @p pu, ignoring an index beyond @c kCpuMaskBits.
 */
inline auto cpumask_set(CpuMask &mask, size_t pu) -> void {
    if (pu < kCpuMaskBits) {
        mask.words[pu / 64] |= (uint64_t{1} << (pu % 64));
    }
}

/*!
 * @brief Whether the bit for PU @p pu is set.
 */
[[nodiscard]] inline auto cpumask_test(const CpuMask &mask, size_t pu) -> bool {
    return pu < kCpuMaskBits && ((mask.words[pu / 64] >> (pu % 64)) & uint64_t{1}) != 0;
}

/*!
 * @brief Number of PUs the mask names.
 */
[[nodiscard]] inline auto cpumask_count(const CpuMask &mask) -> size_t {
    size_t total = 0;
    for (const uint64_t word : mask.words) {
        total += static_cast<size_t>(std::popcount(word));
    }
    return total;
}

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
 * @brief Build placement tokens for one MPI rank's partitions.
 *
 * Calls enumerate_physical_cores() and applies topo_detail::placement_order() to select @p n
 * distinct physical cores for this rank. Two co-located ranks must never share a core: one rank's
 * busy-polling collectives would starve the other's barrier spins.
 *
 * @param n            Number of partitions to place.
 * @param group_index  This rank's 0-based index among the co-located ranks on the host.
 * @param group_count  Total number of co-located ranks on the host.
 * @param mask         How the launcher divided the host, as measured by classify_node_mask(). Under
 *                     @c NodeMask::PerRank, @p group_index / @p group_count are ignored — see below.
 * @returns Vector of @p n CpuSet tokens, or empty when @c monoprop_PARTITION_PINNING is disabled,
 *          hwloc is unavailable, or the host cannot provide the requested distinct cores.
 *
 * @note enumerate_physical_cores() reports only cores inside this process's affinity mask, so under
 *       a @c PerRank mask the launcher has *already* handed each co-located rank a disjoint slice.
 *       Dividing by @p group_count a second time splits an already-split machine: @p group_count ×
 *       @p n exceeds the share, topo_detail::placement_order() reads that as "host too small", and
 *       every rank silently runs unpinned — taking the two-level barrier's domains with it, because
 *       cpuset_domains() derives them from these sets. Measured on Deucalion at 8 ranks × 16
 *       partitions: 437 µs/sync unpinned against 15.5 µs/sync placed. So a @c PerRank mask
 *       collapses to a single group and the whole slice is partitioned, leaving the cross-rank split
 *       to whoever imposed it. The default is @c Shared because that is both the pre-existing
 *       behaviour and the safe error direction: collapsing a genuinely shared mask would point every
 *       co-located rank at the *same* cores, breaking the disjointness invariant above, whereas
 *       dividing a @c PerRank mask only costs pinning.
 */
auto partition_cpusets(size_t n, size_t group_index = 0, size_t group_count = 1, NodeMask mask = NodeMask::Shared)
    -> std::vector<CpuSet>;

/*!
 * @brief This thread's current affinity mask, for a caller that compares it against its peers'.
 *
 * @returns The allowed-PU mask hwloc reports for the calling thread, or an all-zero mask when the
 *          topology or the cpubind query is unavailable — which classify_node_mask() reads as
 *          "cannot tell" ⇒ @c Shared.
 */
auto this_thread_cpumask() -> CpuMask;

/*!
 * @brief Classify how the launcher divided the host, given every co-located rank's mask.
 *
 * The caller gathers the masks because it owns the node-local communicator and this header stays
 * free of MPI. Pure and hwloc-free, so it is testable without live hardware.
 *
 * @param masks  One mask per rank sharing the host, in any order.
 * @returns @c PerRank when the masks are pairwise disjoint and all non-empty; otherwise @c Shared —
 *          identical masks, partial overlap, or a mask that could not be read all land there, which
 *          is the conservative answer (see partition_cpusets()).
 */
auto classify_node_mask(const std::vector<CpuMask> &masks) -> NodeMask;

/*!
 * @brief The locality domain each placement token lands in, in partition_cpusets() order.
 *
 * What a two-level PartitionBarrier groups by. Derived from the tokens rather than from the
 * placement logic, so the two cannot drift apart.
 *
 * @param sets  Placement tokens as returned by partition_cpusets().
 * @returns One L3-domain id per token, or empty (⇒ flat barrier) when @p sets is empty or one token
 *          names no PU that enumerate_physical_cores() knows.
 */
auto cpuset_domains(const std::vector<CpuSet> &sets) -> std::vector<int>;

/*!
 * @brief Bind the calling thread to the PU identified by @p set.
 *
 * Allocates a temporary hwloc bitmap, sets the single bit for @c set.pu, and calls
 * @c hwloc_set_cpubind with @c HWLOC_CPUBIND_THREAD | @c HWLOC_CPUBIND_STRICT.
 *
 * @param set  Placement token as returned by partition_cpusets(). A token with @c pu == -1
 *             is a no-op.
 * @returns Whether the affinity actually took. Correctness never depends on pinning, so a failure
 *          is not an error — but it must not be invisible: @c barrier_groups = 0 has two legitimate
 *          causes (nothing was pinned, or each rank was confined to a single locality domain and so
 *          has nothing to fan in across), and without this they are indistinguishable from outside
 *          the process. CommProfile reports the count so a run states what happened instead of
 *          leaving it inferred from the timing.
 */
[[nodiscard]] auto pin_this_thread(const CpuSet &set) -> bool;
} // namespace monoprop::detail::partition
