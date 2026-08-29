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
#include <span>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/MPICompat.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/Pairwise.h"
#endif

// Keeps #ifdef monoprop_ENABLE_MPI out of the consumers; non-MPI builds get self-copy stubs.

namespace monoprop::mpi {

// Precondition, not a diagnostic: MPI_Alltoallv reads one count and one displacement per rank
// whatever the span holds, so a layout built for a differently sized communicator reads out of bounds.
auto check_exchange_layout_width(std::span<const int> send_counts, const Comm &comm) -> void;

// Legs carrying a payload in either direction, i.e. an upper bound on what the pairwise path posts.
[[nodiscard]] inline auto active_leg_count(const int *send_counts, const int *recv_counts, int num_ranks) -> int {
    int legs = 0;
    for (int i = 0; i < num_ranks; ++i) {
        legs += static_cast<int>(send_counts[i] != 0 || recv_counts[i] != 0);
    }
    return legs;
}

// Idempotent completion handle for a posted payload transfer; move-only, so a request is waited on
// exactly once. wait() is a no-op on the blocking path and in non-MPI builds. Owns its requests: the
// destructor completes anything still in flight, because a dropped in-flight transfer -- what an
// exception between post and wait does -- keeps writing into a thread_local buffer the next exchange
// reallocates. The pairwise arm's request vector lives here for the same reason: MPI reads it until the
// wait, and a vector move keeps its heap block, so the handle can travel.
class [[nodiscard("call wait() on the Ticket to complete the posted transfer")]] Ticket {
public:
    Ticket() = default;
    Ticket(const Ticket &) = delete;
    auto operator=(const Ticket &) -> Ticket & = delete;
    Ticket(Ticket &&other) noexcept { *this = std::move(other); }
    auto operator=(Ticket &&other) noexcept -> Ticket & {
#ifdef monoprop_ENABLE_MPI
        if (this != &other) {
            wait(); // never drop a request this handle already owns
            request_ = other.request_;
            other.request_ = MPI_REQUEST_NULL;
            requests_ = std::move(other.requests_);
            posted_ = std::exchange(other.posted_, 0);
        }
#endif
        (void)other;
        return *this;
    }
    ~Ticket() { wait(); }

    auto wait() -> void {
#ifdef monoprop_ENABLE_MPI
        if (request_ != MPI_REQUEST_NULL) {
            MPI_Wait(&request_, MPI_STATUS_IGNORE);
            request_ = MPI_REQUEST_NULL;
        }
        if (posted_ != 0) {
            MPI_Waitall(posted_, requests_.data(), MPI_STATUSES_IGNORE);
            posted_ = 0;
        }
#endif
    }

    // Requests wait() still has to drain: 1 for the collective, two per pairwise leg, 0 for nothing
    // posted. The only handle on which transport a post took.
    [[nodiscard]] auto in_flight() const -> int {
#ifdef monoprop_ENABLE_MPI
        return static_cast<int>(request_ != MPI_REQUEST_NULL) + posted_;
#else
        return 0;
#endif
    }

#ifdef monoprop_ENABLE_MPI
    explicit Ticket(MPI_Request request) : request_(request) {}
    Ticket(std::vector<MPI_Request> requests, int posted) : requests_(std::move(requests)), posted_(posted) {}

private:
    MPI_Request request_ = MPI_REQUEST_NULL; // the dense collective
    std::vector<MPI_Request> requests_;      // the pairwise arm's pairs; `posted_` of them are live
    int posted_ = 0;
#endif
};

// Never skipped on zero total: the collective arm needs all ranks or it deadlocks. Non-blocking in an
// MPI build -- MPI_Ialltoallv, or Isend/Irecv over the legs that carry a payload (the Ticket completes
// either); non-MPI build does a per-rank self-copy (recv layout == send layout).
//
// `wire_bits` picks the transport and MUST be RANK-UNIFORM: a rank choosing MPI_Ialltoallv waits forever
// on ranks that chose point-to-point, and no predicate over a rank's OWN row can promise that (rows vary,
// so any threshold on one straddles). It is the resolved linear-routing bit count when that routing gives
// fanout 1 -- one destination rank per generator, which is what empties the other legs -- and 0, today's
// collective, for every other geometry. The caller owns that derivation; see Evolution.cpp.
template <typename T>
inline auto post_flat_alltoallv(const FlatAlltoallvArgs<T> &args, int num_ranks, Comm comm, int wire_bits = 0)
    -> Ticket {
    // The in-process transports address the buffers as raw bytes; MPI_Ialltoallv below still takes the
    // typed pointers plus a datatype. Offsets stay in elements on both paths.
    if (comm.kind == Comm::Kind::Shm) {
        // Synchronous: the transfer completes here, so the Ticket's wait() is a no-op. ShmComm needs no
        // send counts: its peers pull using the publisher's displacements.
        const auto bytes = args.bytes();
        comm.shm->alltoallv(comm.shm_rank,
                            bytes.send,
                            bytes.send_displs,
                            bytes.recv,
                            bytes.recv_counts,
                            bytes.recv_displs,
                            bytes.elem);
        return Ticket{};
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        // The wire is narrowed inside the verb, not here: only partition 0 reaches MPI, and it cannot
        // name the rank's peer from its own row alone (its row may be the empty one). See HybridComm.
        comm.hyb->alltoallv(comm.shm_rank, args.bytes(), datatype<T>::get(), PeerPlan{}, wire_bits);
        return Ticket{};
    }
    if (wire_bits > 0) {
        // Which legs to drop needs no plan and no count round: the count matrix is symmetric, so what
        // this rank sends a peer IS that peer's recv count and both ends drop the same legs on the same
        // value. The plan stays DENSE -- it walks all N and posts only the non-zero legs, which is
        // exactly that -- so a mis-derived shift cannot drop a block here; `wire_bits` only chooses the
        // transport. `legs` sizes the request vector, which a dense plan would otherwise take to 2N.
        const int legs = active_leg_count(args.send_counts, args.recv_counts, num_ranks);
        std::vector<MPI_Request> requests;
        const int posted = sparse_pairwise(PeerPlan{},
                                           rank(comm),
                                           num_ranks,
                                           comm.mpi,
                                           kFlatReplayTag,
                                           datatype<T>::get(),
                                           sizeof(T),
                                           reinterpret_cast<const std::byte *>(args.send),
                                           PeerLayout{.counts = args.send_counts, .displs = args.send_displs},
                                           reinterpret_cast<std::byte *>(args.recv),
                                           PeerLayout{.counts = args.recv_counts, .displs = args.recv_displs},
                                           requests,
                                           legs);
        return Ticket(std::move(requests), posted);
    }
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Ialltoallv(args.send,
                   args.send_counts,
                   args.send_displs,
                   datatype<T>::get(),
                   args.recv,
                   args.recv_counts,
                   args.recv_displs,
                   datatype<T>::get(),
                   comm.mpi,
                   &request);
    return Ticket(request);
#else
    for (int i = 0; i < num_ranks; ++i) {
        const int c = args.recv_counts[i];
        for (int j = 0; j < c; ++j) {
            args.recv[args.recv_displs[i] + j] = args.send[args.send_displs[i] + j];
        }
    }
    return Ticket{};
#endif
}

} // namespace monoprop::mpi
