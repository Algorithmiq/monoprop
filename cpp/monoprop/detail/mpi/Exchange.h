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

#include <span>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/MPICompat.h"

// Keeps #ifdef monoprop_ENABLE_MPI out of the consumers; non-MPI builds get self-copy stubs.

namespace monoprop::mpi {

// Precondition, not a diagnostic: MPI_Alltoallv reads one count and one displacement per rank
// whatever the span holds, so a layout built for a differently sized communicator reads out of bounds.
auto check_exchange_layout_width(std::span<const int> send_counts, const Comm &comm) -> void;

// Idempotent completion handle for a posted payload transfer; move-only, so a request is waited on
// exactly once. wait() is a no-op on the blocking path and in non-MPI builds. Owns its request: the
// destructor completes anything still in flight, because a dropped in-flight MPI_Ialltoallv -- what an
// exception between post and wait does -- keeps writing into a thread_local buffer the next exchange
// reallocates.
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
#endif
    }

#ifdef monoprop_ENABLE_MPI
    explicit Ticket(MPI_Request request) : request_(request) {}

private:
    MPI_Request request_ = MPI_REQUEST_NULL;
#endif
};

// Never skipped on zero total: all ranks must participate or the collective deadlocks. Non-blocking
// (MPI_Ialltoallv) in an MPI build (the Ticket completes it); non-MPI build does a per-rank self-copy
// (recv layout == send layout).
template <typename T>
inline auto post_flat_alltoallv(const FlatAlltoallvArgs<T> &args, int num_ranks, Comm comm) -> Ticket {
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
        comm.hyb->alltoallv(comm.shm_rank, args.bytes(), datatype<T>::get());
        return Ticket{};
    }
    (void)num_ranks;
    auto request = MPI_REQUEST_NULL;
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
