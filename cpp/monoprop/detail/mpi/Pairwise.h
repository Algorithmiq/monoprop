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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Routing.h"

namespace monoprop::mpi {

// One tag per (transport, verb), kept together so none collide: the tag is all that stops a count round
// in flight matching a payload receive. Engine.h's two rounds may share kFlatPayloadTag: MPI does not
// overtake within (src, dst, tag, comm), round 1 completes before round 2 posts, and both ends skip a
// zero-count leg on the same (transposed) value.
inline constexpr int kHybridCountTag = 0x6D70; // 'mp'
inline constexpr int kHybridPayloadTag = 0x6D71;
inline constexpr int kFlatPayloadTag = 0x6D72;
inline constexpr int kFlatCountTag = 0x6D73;
// Graph replay payload (Exchange.h), distinct from the build path's rounds on the same communicator.
inline constexpr int kFlatReplayTag = 0x6D74;

// Per-peer element counts and offsets; null `counts` means `block` elements each, at b * block.
struct PeerLayout {
    const int *counts = nullptr;
    const int *displs = nullptr;
    int block = 0;

    [[nodiscard]] auto count(int b) const -> int { return counts != nullptr ? counts[b] : block; }
    [[nodiscard]] auto displ(int b) const -> size_t {
        return displs != nullptr ? static_cast<size_t>(displs[b]) : static_cast<size_t>(b) * static_cast<size_t>(block);
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

// A variable all-to-all as point-to-point over `args.plan`'s peers: an Irecv/Isend per non-empty leg,
// the self leg copied. Counts are in elements of `args.datatype`, whose extent is `args.elem`.
// POSTS ONLY and returns how many of `reqs` are live; buffers and `reqs` must outlive the caller's wait.
// `reqs` is sized by a pre-pass, never grown mid-loop: MPI holds pointers into it once a request posts.
[[nodiscard]] inline auto sparse_pairwise(const SparsePairwiseArgs &args, std::vector<MPI_Request> &reqs) -> int {
    require_routable(args.plan, args.num_ranks);
    const int num_peers = args.plan.count(args.num_ranks);
    size_t needed = 0;
    for (int k = 0; k < num_peers; ++k) {
        const int peer = args.plan.peer(args.me, k);
        if (peer != args.me) {
            needed += static_cast<size_t>(args.recv_layout.count(peer) != 0)
                      + static_cast<size_t>(args.send_layout.count(peer) != 0);
        }
    }
    if (reqs.size() < needed) {
        reqs.resize(needed);
    }
    int num_requests = 0;
    for (int k = 0; k < num_peers; ++k) {
        const int peer = args.plan.peer(args.me, k);
        const int send_count = args.send_layout.count(peer);
        const int recv_count = args.recv_layout.count(peer);
        if (peer == args.me) {
            assert(send_count == recv_count);
            if (recv_count != 0) {
                std::memcpy(args.recv + (args.recv_layout.displ(peer) * args.elem),
                            args.send + (args.send_layout.displ(peer) * args.elem),
                            static_cast<size_t>(recv_count) * args.elem);
            }
            continue;
        }
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

// Allreduce {v, ~v} under MPI_MAX, giving max and min at once: max == min == mine is exact agreement.
// Ranks that disagree would hang, so every rank throws together instead.
inline auto agree_routes_pairwise(MPI_Comm comm, int ranks, const routing::Config &mine) -> bool {
    const std::array<uint64_t, 6> probe{mine.linear,
                                        ~mine.linear,
                                        mine.partitions,
                                        ~mine.partitions,
                                        mine.seed,
                                        ~mine.seed};
    std::array<uint64_t, 6> agreed{};
    MPI_Allreduce(probe.data(), agreed.data(), 6, MPI_UINT64_T, MPI_MAX, comm);
    if (agreed != probe) {
        throw routing::RoutingDisagreement(
            std::format("routing configuration differs across the {} ranks (this one: {}). "
                        "monoprop_ROUTING / monoprop_ROUTE_SEED must reach every rank identically -- "
                        "a disagreement deadlocks the exchange rather than corrupting it.",
                        ranks,
                        mine.describe()));
    }
    return routing::routes_pairwise(mine, static_cast<size_t>(ranks));
}

} // namespace monoprop::mpi
