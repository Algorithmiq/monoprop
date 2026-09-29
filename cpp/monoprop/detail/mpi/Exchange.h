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

// Move-only, idempotent completion handle for a posted payload transfer. The destructor waits: a
// dropped in-flight transfer keeps writing into a thread_local buffer the next exchange reallocates.
// Moving is safe because a vector move keeps the request block MPI points into.
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
        if (posted_ != 0) {
            MPI_Waitall(posted_, requests_.data(), MPI_STATUSES_IGNORE);
            posted_ = 0;
        }
#endif
    }

    // Requests wait() still has to drain: 1 for the collective, two per pairwise leg, 0 for none.
    [[nodiscard]] auto in_flight() const -> int {
#ifdef monoprop_ENABLE_MPI
        return posted_;
#else
        return 0;
#endif
    }

#ifdef monoprop_ENABLE_MPI
    Ticket(std::vector<MPI_Request> requests, int posted) : requests_(std::move(requests)), posted_(posted) {}

private:
    std::vector<MPI_Request> requests_; // the collective's one request, or the pairwise legs
    int posted_ = 0;                    // live entries of requests_
#endif
};

// Never skipped on zero total: every rank must take part. Non-blocking under MPI (MPI_Ialltoallv, or
// Isend/Irecv over the non-empty legs); the non-MPI build self-copies.
//
// `pairwise` picks the transport and MUST be rank-uniform, or a rank in MPI_Ialltoallv waits forever on
// one that went point-to-point. Take it from mpi::routes_pairwise(comm), never from a rank's own layout.
template <typename T>
inline auto post_flat_alltoallv(const FlatAlltoallvArgs<T> &args, int num_ranks, Comm comm, bool pairwise = false)
    -> Ticket {
    // In-process transports take raw bytes, MPI takes typed pointers; offsets are in elements on both.
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
        // Narrowed inside the verb: only there is the whole rank's traffic visible.
        comm.hyb->alltoallv(comm.shm_rank, args.bytes(), datatype<T>::get(), PeerPlan{}, pairwise);
        return Ticket{};
    }
    if (pairwise) {
        // Dense plan: walk every rank and post only non-empty legs, so nothing can be dropped. The layout
        // is symmetric, so both ends skip the same legs.
        std::vector<MPI_Request> requests;
        const SparsePairwiseArgs post{
            .plan = {},
            .me = rank(comm),
            .num_ranks = num_ranks,
            .comm = comm.mpi,
            .tag = kFlatReplayTag,
            .datatype = datatype<T>::get(),
            .elem = sizeof(T),
            .send = reinterpret_cast<const std::byte *>(args.send),
            .send_layout = {.counts = args.send_counts, .displs = args.send_displs},
            .recv = reinterpret_cast<std::byte *>(args.recv),
            .recv_layout = {.counts = args.recv_counts, .displs = args.recv_displs},
        };
        const int posted = sparse_pairwise(post, requests);
        return {std::move(requests), posted};
    }
    std::vector<MPI_Request> request(1, MPI_REQUEST_NULL);
    MPI_Ialltoallv(args.send,
                   args.send_counts,
                   args.send_displs,
                   datatype<T>::get(),
                   args.recv,
                   args.recv_counts,
                   args.recv_displs,
                   datatype<T>::get(),
                   comm.mpi,
                   request.data());
    return {std::move(request), 1};
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
