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
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/RecvLayout.h"

// Variable-size all-to-all over caller-owned flat buffers, so consumers (replay/pare exchange) hold
// no #ifdef monoprop_ENABLE_MPI; non-MPI builds get self-copy stubs.

namespace monoprop::mpi {

// Resolve the recv side of a send-count vector, reusing `cache` when comm size is unchanged: a
// replayed graph's send pattern is fixed, so a hit removes one blocking count round-trip per layer
// per evaluation. The resolved layout is stored in `cache` and returned by reference.
inline auto resolve_recv(std::span<const int> send_counts, const Comm &comm, RecvLayoutCache &cache)
    -> const RecvLayout & {
    const auto n = static_cast<int>(send_counts.size());
    const int comm_size = mpi::size(comm);
    // alltoall_counts moves comm_size ints each way regardless of `n`, so a send vector that is not
    // exactly one entry per rank reads and writes out of bounds — and layouts outlive propagator copies
    // and pare rebuilds, so a graph replayed on a differently-sized comm would arrive with the old width.
    if (n != comm_size) {
        throw CollectiveArgumentError(
            std::format("Exchange layout has {} send counts but the communicator has {} ranks — a graph built for one "
                        "communicator cannot be replayed on another of a different size.",
                        n,
                        comm_size));
    }
    if (cache.comm_size == comm_size && static_cast<int>(cache.layout.counts.size()) == n) {
        return cache.layout;
    }

    RecvLayout &out = cache.layout;
    out.counts.resize(static_cast<size_t>(n));
    alltoall_counts(send_counts.data(), out.counts.data(), n, comm);
    out.displs.resize(static_cast<size_t>(n));
    // Accumulate wide, narrow through the checked helper (see MPICompat.h).
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        out.displs[static_cast<size_t>(i)] = checked_mpi_count(total);
        total += out.counts[static_cast<size_t>(i)];
    }
    out.total = checked_mpi_count(total);
    cache.comm_size = comm_size;
    return out;
}

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

// Post a variable-size all-to-all over caller-owned flat buffers. never skipped on zero total: all
// ranks must participate or the collective deadlocks. Non-blocking (MPI_Ialltoallv) in an MPI build
// (the Ticket completes it); non-MPI build does a per-rank self-copy (recv layout == send layout).
template <class T>
inline auto post_flat_alltoallv(const T *send,
                                const int *send_counts,
                                const int *send_displs,
                                T *recv,
                                const int *recv_counts,
                                const int *recv_displs,
                                int num_ranks,
                                Comm comm) -> Ticket {
    if (comm.kind == Comm::Kind::Shm) {
        // Synchronous under the hood: the transfer completes here and the Ticket's wait() is a no-op.
        comm.shm->alltoallv(comm.shm_rank, send, send_displs, recv, recv_counts, recv_displs, sizeof(T));
        return Ticket{};
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        // Synchronous (the MPI_Alltoallv runs inside); the Ticket's wait() is a no-op.
        comm.hyb->alltoallv(comm.shm_rank,
                            send,
                            send_counts,
                            send_displs,
                            recv,
                            recv_counts,
                            recv_displs,
                            sizeof(T),
                            datatype<T>::get());
        return Ticket{};
    }
    (void)num_ranks;
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Ialltoallv(send,
                   send_counts,
                   send_displs,
                   datatype<T>::get(),
                   recv,
                   recv_counts,
                   recv_displs,
                   datatype<T>::get(),
                   comm.mpi,
                   &request);
    return Ticket(request);
#else
    (void)send_counts;
    for (int i = 0; i < num_ranks; ++i) {
        const int c = recv_counts[i];
        for (int j = 0; j < c; ++j) {
            recv[recv_displs[i] + j] = send[send_displs[i] + j];
        }
    }
    return Ticket{};
#endif
}

} // namespace monoprop::mpi
