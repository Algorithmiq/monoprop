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

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <vector>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/PairSlots.h"
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include <mpi.h>

#include "monoprop/detail/mpi/HybridComm.h"
#include "monoprop/detail/mpi/Pairwise.h"
#endif

/*
 * pair_exchange: the one-round symmetric gate exchange. Keeps #ifdef monoprop_ENABLE_MPI out of the
 * consumers; a non-MPI build has the in-rank transports and a size-1 raw communicator.
 *
 * Called by EVERY participant of `comm` -- every partition master of every rank -- once per gate, in
 * program order, with the same `rank_shift` on every rank (it is derived from a replicated generator
 * list). Peer rank = my_rank ^ rank_shift; rank_shift == 0 keeps the exchange inside the rank and
 * touches no MPI. Sub-stream (rank r, partition u) -> (rank r ^ shift, partition t) for every u, t;
 * `from[u]` is in ascending u, which the caller's index assignment depends on.
 *
 * Lifetime rule, the same on every transport so a caller need not know which it holds:
 *
 *   1. The memory behind the S `send` spans must stay valid and unmodified until this participant's NEXT
 *      pair_exchange on the same comm RETURNS. In-rank, peers read it in place AFTER this call returned,
 *      and the next gate's barrier is the first point that proves every peer is done -- the deferred
 *      second barrier of a two-barrier protocol, paid once per gate instead of twice. A caller therefore
 *      alternates two sets of send buffers gate by gate. The outer array of descriptors is copied before
 *      the barrier and may be reused on return. A participant's LAST gate on a comm has no next call to
 *      bound it: that set stays alive until every participant has consumed its views -- any later
 *      collective on the same comm, or joining the partitions, proves that.
 *   2. The `from` views are valid until this participant's next pair_exchange on the same comm is
 *      ENTERED. In-rank they alias peers' send buffers, which rule 1 frees once every partition has
 *      entered the next gate; across ranks they alias the rank master's receive buffer, which the next
 *      cross-rank gate overwrites after its first barrier -- grown, never shrunk, so no gate can shrink a
 *      view still being read.
 *
 * Cost per gate and rank: one barrier in-rank (Shm, or Hybrid at shift 0); two barriers and ONE message
 * each way across ranks. Partition 0 alone calls MPI (MPI_THREAD_SERIALIZED), sending every partition's
 * sub-streams in place through one derived datatype whose leading block is the S*S word counts, and
 * sizing its receive by probing the peer's message. The header makes both ends post exactly one message
 * even when a side has nothing to send, so there is no count round and no zero-length asymmetry to keep
 * symmetric. The `Kind::Mpi` receive state is per thread, not per comm: one participant per process is
 * what that kind means, and rule 2 then also covers a next call on a different raw communicator.
 *
 * Arguments are validated before any barrier and throw CollectiveArgumentError; they are replicated, so
 * every participant throws or none does and nobody is left waiting. A failure past a HybridComm's first
 * barrier aborts the job, as every verb there does: the peer rank is committed inside MPI.
 */

namespace monoprop::mpi {

namespace detail {

// A single-partition rank cannot address a sibling, so `send` has one sub-stream and `from` one view.
inline constexpr size_t kMpiKindStreams = 1;

/*!
 * The raw-communicator receive side. Grown and never shrunk (rule 2 in the header comment) and reached
 * through a function-local thread_local so it has one instance per thread across translation units.
 */
struct MpiPairState {
    std::vector<size_t> recv;                                    //!< the peer's sub-stream, HWM-sized
    std::array<std::span<const size_t>, kMpiKindStreams> from{}; //!< the one view handed back
};

inline auto mpi_pair_state() -> MpiPairState & {
    static thread_local MpiPairState state;
    return state;
}

inline auto check_pair_args(const Comm &comm, int rank_shift, SubStreams send, int partitions, int ranks) -> void {
    if (static_cast<int>(send.size()) != partitions) {
        throw CollectiveArgumentError(
            std::format("pair_exchange: {} sub-streams for a rank of {} partitions", send.size(), partitions));
    }
    // The XOR pairing is an involution only over a power-of-two world; there every shift below `ranks`
    // names a peer below `ranks`. Both conditions are rank-replicated, so no rank can throw alone.
    if (rank_shift < 0 || rank_shift >= ranks || (rank_shift != 0 && (ranks & (ranks - 1)) != 0)) {
        throw CollectiveArgumentError(
            std::format("pair_exchange: rank_shift {} does not name a peer in a {}-rank world (kind {})",
                        rank_shift,
                        ranks,
                        static_cast<int>(comm.kind)));
    }
}

#ifdef monoprop_ENABLE_MPI
// One participant per process: one message each way, the receive sized by probing. Zero words is still
// a message, so both ends always post one and a silent side cannot desynchronise the pair.
inline auto mpi_kind_exchange(MPI_Comm comm, int peer, std::span<const size_t> mine) -> std::span<const size_t> {
    MpiPairState &st = mpi_pair_state();
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Isend(mine.data(),
              checked_mpi_count(static_cast<long long>(mine.size()), "Pair sub-stream length"),
              MPI_UINT64_T,
              peer,
              kPairExchangeTag,
              comm,
              &req);
    MPI_Message msg = MPI_MESSAGE_NULL;
    MPI_Status status;
    MPI_Mprobe(peer, kPairExchangeTag, comm, &msg, &status);
    int total = 0;
    MPI_Get_count(&status, MPI_UINT64_T, &total);
    if (total == MPI_UNDEFINED) {
        throw CollectiveArgumentError("pair_exchange: the peer's sub-stream exceeds the MPI count range");
    }
    if (st.recv.size() < static_cast<size_t>(total)) {
        st.recv.resize(static_cast<size_t>(total));
    }
    MPI_Mrecv(st.recv.data(), total, MPI_UINT64_T, &msg, MPI_STATUS_IGNORE);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    return {st.recv.data(), static_cast<size_t>(total)};
}
#endif

} // namespace detail

/*!
 * The one-round gate exchange; the contract is the comment at the top of this header. Dispatches on the
 * comm kind: the in-rank transports gather in place after one barrier, HybridComm adds the cross-rank
 * message when `rank_shift` != 0, a raw communicator is the S == 1 case of the same protocol.
 */
inline auto pair_exchange(Comm comm, int rank_shift, SubStreams send) -> PairRecv {
    static_assert(sizeof(size_t) == sizeof(uint64_t), "pair_exchange sends words as MPI_UINT64_T");
    switch (comm.kind) {
        case Comm::Kind::Shm:
            detail::check_pair_args(comm, rank_shift, send, comm.shm->size(), /*ranks=*/1);
            return comm.shm->pair_exchange(comm.shm_rank, send);
        case Comm::Kind::Hybrid:
#ifdef monoprop_ENABLE_MPI
            detail::check_pair_args(comm, rank_shift, send, comm.hyb->partitions(), comm.hyb->ranks());
            return comm.hyb->pair_exchange(comm.shm_rank, rank_shift, send);
#else
            throw CollectiveArgumentError("pair_exchange: a Hybrid comm in a build without MPI");
#endif
        case Comm::Kind::Mpi:
            break;
    }
    detail::MpiPairState &st = detail::mpi_pair_state();
#ifdef monoprop_ENABLE_MPI
    int ranks = 1;
    int me = 0;
    MPI_Comm_size(comm.mpi, &ranks);
    MPI_Comm_rank(comm.mpi, &me);
    detail::check_pair_args(comm, rank_shift, send, static_cast<int>(detail::kMpiKindStreams), ranks);
    if (rank_shift != 0) {
        st.from[0] = detail::mpi_kind_exchange(comm.mpi, me ^ rank_shift, send[0]);
        return PairRecv{st.from};
    }
#else
    detail::check_pair_args(comm, rank_shift, send, static_cast<int>(detail::kMpiKindStreams), /*ranks=*/1);
#endif
    // The self peer: a view of the caller's own sub-stream, under the same lifetime rule as in-rank.
    st.from[0] = send[0];
    return PairRecv{st.from};
}

} // namespace monoprop::mpi
