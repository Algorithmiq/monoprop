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
// completes before round 2 posts; and each round posts the SAME number of messages at both ends -- the
// probe arm one each way unconditionally, the known-layout arm because both ends skip a zero-count leg
// on the same value, what one sends a peer being that peer's recv count by transpose. So the two posted
// sequences match element for element and a round-2 receive cannot match a round-1 send.
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

// One matched-probe exchange with the single peer GF(2)-linear routing leaves. The recv size is READ OFF
// the peer's envelope instead of agreed beforehand, so the exposed one-int round trip alltoall_counts
// costs disappears -- the payload's own header carries what the count round used to carry.
//
// Only sound at fanout 1: a probe names the size of ONE message from ONE known peer, so with `R` sources
// the buffer could not be laid out without re-introducing the count round this replaces.
template <typename T>
struct ProbeExchangeArgs {
    int me = 0;
    int peer = 0; // me ^ shift; equal to `me` when the generator moves nothing between ranks
    MPI_Comm comm = MPI_COMM_NULL;
    int tag = 0;
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    const T *send = nullptr;
    int send_count = 0;
    std::vector<T> *recv = nullptr; // RESIZED to the probed count, so it may not alias `send`
};

struct ProbeExchangeResult {
    int recv_count = 0; // elements the peer sent, taken from the matched envelope
    int posted = 0;     // live entries at the front of `reqs`
};

// POSTS ONLY for the payload legs -- the probe itself blocks, exactly where the count round used to --
// and returns what the caller must still wait on. `args.recv` and `reqs` must outlive that wait.
//
// Two orderings are contracts, not preferences:
//
// - The send is posted BEFORE the probe. Probing first would have both ends waiting on a message neither
//   has posted, because the pairing is symmetric.
// - The send is UNCONDITIONAL, empty blocks included. A skipped send is a probe that never matches, i.e.
//   a hang; an empty message costs an envelope and keeps the two ends' message sequences in step under a
//   tag that later rounds reuse.
//
// MPI_Mprobe, not MPI_Iprobe + MPI_Recv: the matched form dequeues the message atomically, so the size
// used for the resize is the size of the message the MPI_Imrecv below receives, whatever else arrives
// under the same envelope in between.
template <typename T>
[[nodiscard]] inline auto probe_pairwise(const ProbeExchangeArgs<T> &args, std::vector<MPI_Request> &reqs)
    -> ProbeExchangeResult {
    assert(args.recv != nullptr);
    if (args.peer == args.me) {
        // The self slot is a copy, not a message -- as in sparse_pairwise, and for the extra reason that
        // a self-send would have to be matched by a probe on the same thread that posted it.
        args.recv->assign(args.send, args.send + args.send_count);
        return {.recv_count = args.send_count, .posted = 0};
    }
    if (reqs.size() < 2) {
        reqs.resize(2);
    }
    int posted = 0;
    MPI_Isend(args.send, args.send_count, args.datatype, args.peer, args.tag, args.comm, &reqs[0]);
    ++posted;

    MPI_Message message = MPI_MESSAGE_NULL;
    MPI_Status status{};
    MPI_Mprobe(args.peer, args.tag, args.comm, &message, &status);
    int recv_count = 0;
    MPI_Get_count(&status, args.datatype, &recv_count);
    // MPI_UNDEFINED means the byte count is not a whole number of elements, i.e. the two ends disagree on
    // the datatype -- a misroute this path can still catch, unlike a wrong-but-agreed offset.
    assert(recv_count != MPI_UNDEFINED && "matched a message whose size is not a whole number of elements");
    args.recv->resize(static_cast<size_t>(recv_count));
    // The message is already matched, so the receive is not optional: dropping the handle on a zero count
    // would strand it. data() may be null at count 0, which MPI accepts.
    MPI_Imrecv(args.recv->data(), recv_count, args.datatype, &message, &reqs[1]);
    ++posted;
    return {.recv_count = recv_count, .posted = posted};
}

} // namespace monoprop::mpi
