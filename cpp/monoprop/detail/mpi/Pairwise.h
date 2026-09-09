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

// One tag per (transport, verb), all five here so no two can collide unseen: one thread per rank calls
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

// Complete description of one point-to-point all-to-all post.
struct SparsePairwiseArgs {
    PeerPlan plan;
    int me;
    int num_ranks;
    MPI_Comm comm;
    int tag;
    MPI_Datatype datatype;
    size_t elem;
    const std::byte *send;
    PeerLayout send_layout;
    std::byte *recv;
    PeerLayout recv_layout;
};

// A variable all-to-all as point-to-point over `args.plan`'s peers: one Irecv/Isend pair each, the self
// peer copied in place. Counts and displacements are in elements of `args.datatype`, whose extent must
// be `args.elem`. `reqs` is caller storage, grown then indexed: MPI holds these pointers until the wait,
// so a reallocating push_back would dangle them.
//
// POSTS ONLY, and returns how many of `reqs` are live. The caller waits, so the buffers and `reqs` must
// all outlive that wait. `active_legs` is an upper bound on peers that post; negative means every peer.
[[nodiscard]] inline auto sparse_pairwise(const SparsePairwiseArgs &args,
                                          std::vector<MPI_Request> &reqs,
                                          int active_legs = -1) -> int {
    const int num_peers = args.plan.count(args.num_ranks);
    const int legs = active_legs < 0 || active_legs > num_peers ? num_peers : active_legs;
    if (const auto capacity = size_t{2} * static_cast<size_t>(legs); reqs.size() < capacity) {
        reqs.resize(capacity);
    }
    int num_requests = 0;
    for (int k = 0; k < num_peers; ++k) {
        const int peer = args.plan.peer(args.me, k);
        const int send_count = args.send_layout.count(peer);
        const int recv_count = args.recv_layout.count(peer);
        if (peer == args.me) {
            // The self slot is a copy, not a message: its two counts are each other's transpose.
            assert(send_count == recv_count);
            if (recv_count != 0) {
                std::memcpy(args.recv + (args.recv_layout.displ(peer) * args.elem),
                            args.send + (args.send_layout.displ(peer) * args.elem),
                            static_cast<size_t>(recv_count) * args.elem);
            }
            continue;
        }
        assert(static_cast<size_t>(num_requests) + 2 <= reqs.size()
               || (recv_count == 0 && send_count == 0)); // active_legs too small
        if (recv_count != 0) {
            MPI_Irecv(args.recv + (args.recv_layout.displ(peer) * args.elem),
                      recv_count,
                      args.datatype,
                      peer,
                      args.tag,
                      args.comm,
                      &reqs[static_cast<size_t>(num_requests++)]);
        }
        if (send_count != 0) {
            MPI_Isend(args.send + (args.send_layout.displ(peer) * args.elem),
                      send_count,
                      args.datatype,
                      peer,
                      args.tag,
                      args.comm,
                      &reqs[static_cast<size_t>(num_requests++)]);
        }
    }
    return num_requests;
}

} // namespace monoprop::mpi
