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

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace monoprop::mpi_detail {

static_assert(sizeof(size_t) == sizeof(uint64_t), "MPI serialization assumes 64-bit size_t");

/// Number of size_t words per MajoranaSet<NumModes>.
template <size_t NumModes>
inline constexpr size_t kWords = MajoranaSet<NumModes>::num_words();

template <size_t NumModes>
inline auto append_majorana_words(const MajoranaSet<NumModes> &maj, VecZ &buffer) -> void {
    const auto *src = maj.data();
    for (size_t i = 0; i < kWords<NumModes>; ++i)
        buffer.push_back(src[i]);
}

template <size_t NumModes>
inline auto write_majorana_words(const MajoranaSet<NumModes> &maj, VecZ &buffer, size_t start) -> void {
    std::memcpy(&buffer[start], maj.data(), kWords<NumModes> * sizeof(uint64_t));
}

template <size_t NumModes>
inline auto read_majorana_from_words(const VecZ &buffer, size_t start) -> MajoranaSet<NumModes> {
    MajoranaSet<NumModes> maj;
    std::memcpy(maj.data(), &buffer[start], kWords<NumModes> * sizeof(uint64_t));
    return maj;
}

template <size_t NumModes>
inline auto insert_or_find_operator_index(MPOperator<NumModes> &local_mp_op,
                                          const MajoranaSet<NumModes> &maj,
                                          size_t &current_size) -> size_t {
    const auto hash = MPHash<NumModes>{}(maj);
    auto it = local_mp_op.indexing.find_prehashed(maj, hash);
    if (it == local_mp_op.indexing.end_for_hash(hash)) {
        const auto idx = current_size++;
        local_mp_op.op.push_back(maj);
        local_mp_op.indexing.emplace(maj, idx);
        return idx;
    }

    return it->second;
}

} // namespace monoprop::mpi_detail

namespace monoprop {

/**
 * @brief MPI utility functions for distributed MPOperator operations
 */
template <size_t NumModes>
auto find_rank(const MajoranaSet<NumModes> &maj, const size_t n_ranks) -> size_t {
    if (n_ranks == 0) {
        return 0;
    }
    return MPHash<NumModes>{}(maj) % n_ranks;
}

/**
 * @brief Update an MP operator with newly created half-cycle terms.
 *
 * Inserts any missing operator terms, exchanges their resolved indices, and
 * appends the completed cycles and phases back into the caller-owned storage.
 * The caller still invokes split_and_exchange_cycles on the updated data.
 *
 * @tparam NumModes Number of fermionic modes
 * @param local_mp_op The local rank's MPOperator to update
 * @param half_op New Majorana terms organized by [target_rank]
 * @param half_cycles Half cycles organized by [target_rank]
 * @param half_phases Half phases organized by [target_rank]
 * @param cycles Full cycles organized by [target_rank]
 * @param phases Phases organized by [target_rank]
 * @param comm MPI communicator
 * @param all_op1_sizes Sizes of main operator for each rank (for offset calculations in operator updates)
 * @return Empty SplitCycleResult; cycle splitting remains a caller responsibility
 */
template <size_t NumModes>
auto update_mp(MPOperator<NumModes> &local_mp_op,
               const std::vector<MajoranaVector<NumModes>> &half_op,
               const CyclesType &half_cycles,
               const std::vector<VecI> &half_phases,
               CyclesType &cycles,
               std::vector<VecI> &phases,
               MPI_Comm comm = MPI_COMM_WORLD,
               const VecZ &all_op1_sizes = {}) -> SplitCycleResult {
    const int num_ranks = mpi::size(comm);
    const int my_rank = mpi::rank(comm);

    // For single-rank case, update the operator and complete half-cycles into cycles.
    // We do NOT build a SplitCycleResult here — the caller handles that
    // via split_and_exchange_cycles to ensure a single consistent code path.
    if (num_ranks == 1) {
        const size_t index_offset = all_op1_sizes.empty() ? 0 : all_op1_sizes[0];

        if (!half_op.empty() && !half_op[0].empty()) {
            const auto &local_half_op = half_op[0];
            const auto &local_half_cycles = half_cycles[0];
            const auto &local_half_phases = half_phases[0];

            size_t current_size = local_mp_op.op.size();
            for (size_t i = 0; i < local_half_op.size(); ++i) {
                const auto new_idx =
                    mpi_detail::insert_or_find_operator_index(local_mp_op, local_half_op[i], current_size);
                if (i < local_half_cycles.size()) {
                    cycles[0].push_back({local_half_cycles[i].first, new_idx + index_offset});
                    if (i < local_half_phases.size()) {
                        phases[0].push_back(local_half_phases[i]);
                    }
                }
            }
        }
        // Return empty result — caller calls split_and_exchange_cycles.
        return {};
    }
    // Multi-rank: two-phase protocol matching the proven d8a6f71 approach.
    // Phase 1: Exchange half_op, insert new terms, send back indices, update cycles.
    // Phase 2: Caller invokes split_and_exchange_cycles on the updated cycles/phases.
    //          update_mp only handles operator insertion; it does NOT build the SplitCycleResult.

    constexpr size_t W = mpi_detail::kWords<NumModes>;

    // Step 1: Serialize half_op for REMOTE target ranks only (local handled in Step 5).
    std::vector<VecZ> send_op_data(num_ranks);
    threading::parallel_for_indices(static_cast<size_t>(num_ranks), [&half_op, &send_op_data, my_rank](size_t tr) {
        if (static_cast<int>(tr) == my_rank)
            return;
        if (tr >= half_op.size())
            return;
        auto &buf = send_op_data[tr];
        buf.reserve(half_op[tr].size() * W);
        for (const auto &maj : half_op[tr])
            mpi_detail::append_majorana_words<NumModes>(maj, buf);
    });
    auto recv_op_data = mpi::alltoallv(send_op_data, comm);

    // Step 2: Insert received terms (with duplicate check).
    std::vector<VecZ> response_new_indices(num_ranks);
    for (int sr = 0; sr < num_ranks; ++sr) {
        const auto &incoming = recv_op_data[sr];
        if (incoming.empty())
            continue;
        size_t cur = local_mp_op.op.size();
        for (size_t off = 0; off < incoming.size(); off += W) {
            auto new_maj = mpi_detail::read_majorana_from_words<NumModes>(incoming, off);
            const auto idx = mpi_detail::insert_or_find_operator_index(local_mp_op, new_maj, cur);
            response_new_indices[sr].push_back(idx);
        }
    }

    // Step 3: Send new indices back.
    auto recv_new_indices = mpi::alltoallv(response_new_indices, comm);

    // Step 4: Complete half-cycles → update cycles/phases with per-rank offset.
    for (int tr = 0; tr < num_ranks; ++tr) {
        if (static_cast<size_t>(tr) >= half_op.size())
            continue;
        if (static_cast<size_t>(tr) >= half_cycles.size())
            continue;
        if (static_cast<size_t>(tr) >= half_phases.size())
            continue;
        const auto &thc = half_cycles[tr];
        const auto &thp = half_phases[tr];
        const auto &ni = recv_new_indices[tr];
        const size_t tgt_off = (static_cast<size_t>(tr) < all_op1_sizes.size()) ? all_op1_sizes[tr] : 0;
        for (size_t i = 0; i < thc.size() && i < ni.size(); ++i) {
            cycles[tr].push_back({thc[i].first, ni[i] + tgt_off});
            if (i < thp.size())
                phases[tr].push_back(thp[i]);
        }
    }

    // Step 5: Insert local half-cycles (target_rank == my_rank).
    if (static_cast<size_t>(my_rank) < half_op.size() && static_cast<size_t>(my_rank) < half_cycles.size()
        && static_cast<size_t>(my_rank) < half_phases.size()) {
        const auto &lhh = half_op[my_rank];
        const auto &lhc = half_cycles[my_rank];
        const auto &lhp = half_phases[my_rank];
        const size_t lo = (static_cast<size_t>(my_rank) < all_op1_sizes.size()) ? all_op1_sizes[my_rank] : 0;
        size_t cur = local_mp_op.op.size();
        for (size_t i = 0; i < lhh.size(); ++i) {
            const auto idx = mpi_detail::insert_or_find_operator_index(local_mp_op, lhh[i], cur);
            if (i < lhc.size()) {
                cycles[my_rank].push_back({lhc[i].first, idx + lo});
                if (i < lhp.size())
                    phases[my_rank].push_back(lhp[i]);
            }
        }
    }

    // Return empty result — caller must invoke split_and_exchange_cycles.
    mpi::barrier(comm);
    return {};
}
} // namespace monoprop
