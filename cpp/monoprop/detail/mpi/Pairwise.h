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

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/Comm.h"

namespace monoprop::mpi {

// One tag per (transport, verb), all six here so no two can collide unseen: one thread per rank calls
// MPI, so the tag is all that keeps a count round in flight from being matched by a payload receive.
//
// Why Engine.h's run_exchange may post BOTH its begin_alltoallv rounds under kFlatPayloadTag on one
// communicator: MPI does not overtake within a (src, dst, tag, comm); the query round's MPI_Waitall
// completes before round 2 posts; and both ends skip a zero-count leg on the same value -- what one
// sends a peer IS that peer's recv count, by transpose -- so the two posted sequences match element
// for element and a round-2 receive cannot match a round-1 send.
inline constexpr int kHybridCountTag = 0x6D70; // 'mp'
inline constexpr int kHybridPayloadTag = 0x6D71;
inline constexpr int kFlatPayloadTag = 0x6D72;
inline constexpr int kFlatCountTag = 0x6D73;
// Graph REPLAY payload (Exchange.h). Its own value, so a replay leg can match neither the build path's
// counts nor its payload even though both run over the same communicator.
inline constexpr int kFlatReplayTag = 0x6D74;
// pair_exchange (PairExchange.h): the S*S word counts and the payload travel in ONE message, so one tag.
// A gate g+1 message cannot be matched by gate g's receive: every participant calls the verb in program
// order with the rank-replicated shift, so a rank posts gate g+1's send only after gate g's receive has
// completed, and the peer's sends to this rank sit on one (src, dst, tag, comm) sequence that MPI does
// not reorder -- the probe that sizes gate g's receive therefore always meets gate g's message.
inline constexpr int kPairExchangeTag = 0x6D75;

// Per-peer element counts and offsets. Null `counts` is the fixed-block case: `block` each, at b*block.
struct PeerLayout {
    const int *counts = nullptr;
    const int *displs = nullptr;
    int block = 0;

    [[nodiscard]] auto count(int b) const -> int { return counts != nullptr ? counts[b] : block; }
    [[nodiscard]] auto displ(int b) const -> size_t {
        return static_cast<size_t>(displs != nullptr ? displs[b] : b * block);
    }
};

// A variable all-to-all as point-to-point over `plan`'s peers: one Irecv/Isend pair each, the self peer
// copied in place. Counts and displacements are in ELEMENTS of `dt`, whose extent must be `elem`.
// `reqs` is caller storage, grown then INDEXED: MPI holds these pointers until the wait, so a
// reallocating push_back would dangle them.
//
// POSTS ONLY, and returns how many of `reqs` are live. The caller waits, so `send`, `recv` and `reqs`
// must all outlive that wait -- which is what lets a caller hold the round open (PendingAlltoallv) the
// same way the dense branch holds an MPI_Ialltoallv.
//
// `active_legs` is an UPPER BOUND on the peers that will post, for a caller that already knows it (a
// dense plan over a mostly-empty layout sizes `reqs` at 2 * n_ranks otherwise); negative means "assume
// every peer posts". Too small an upper bound is caught by the assert below, not silently.
[[nodiscard]] inline auto sparse_pairwise(PeerPlan plan,
                                          int me,
                                          int n_ranks,
                                          MPI_Comm comm,
                                          int tag,
                                          MPI_Datatype dt,
                                          size_t elem,
                                          const std::byte *send,
                                          PeerLayout send_lay,
                                          std::byte *recv,
                                          PeerLayout recv_lay,
                                          std::vector<MPI_Request> &reqs,
                                          int active_legs = -1) -> int {
    const int f = plan.count(n_ranks);
    const auto cap = static_cast<size_t>(2 * (active_legs < 0 || active_legs > f ? f : active_legs));
    if (reqs.size() < cap) {
        reqs.resize(cap);
    }
    int n_req = 0;
    for (int k = 0; k < f; ++k) {
        const int b = plan.peer(me, k);
        const int sc = send_lay.count(b);
        const int rc = recv_lay.count(b);
        std::byte *rbuf = recv + recv_lay.displ(b) * elem;
        const std::byte *sbuf = send + send_lay.displ(b) * elem;
        if (b == me) {
            // The self slot is a copy, not a message: its two counts are each other's transpose.
            assert(sc == rc);
            if (rc != 0) {
                std::memcpy(rbuf, sbuf, static_cast<size_t>(rc) * elem);
            }
            continue;
        }
        assert(static_cast<size_t>(n_req) + 2 <= reqs.size() || (rc == 0 && sc == 0)); // active_legs too small
        if (rc != 0) {
            MPI_Irecv(rbuf, rc, dt, b, tag, comm, &reqs[static_cast<size_t>(n_req++)]);
        }
        if (sc != 0) {
            MPI_Isend(sbuf, sc, dt, b, tag, comm, &reqs[static_cast<size_t>(n_req++)]);
        }
    }
    return n_req;
}

} // namespace monoprop::mpi
