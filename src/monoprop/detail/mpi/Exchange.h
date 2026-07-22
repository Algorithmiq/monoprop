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
#include "monoprop/detail/mpi/RecvLayout.h"

// Variable-size all-to-all facade over caller-owned FLAT buffers, so consumers (replay/pare exchange)
// hold no #ifdef monoprop_ENABLE_MPI. States the "every rank must participate — never skip on zero
// counts" deadlock discipline; non-MPI builds get self-copy stubs.

namespace monoprop::mpi {

/// Resolve the recv side of a send-count vector, reusing `cache` when comm size is unchanged: a
/// replayed graph's send pattern is fixed, so a hit removes one blocking count round-trip per layer
/// per evaluation. The resolved layout is stored in `cache` and returned by reference.
inline auto resolve_recv(std::span<const int> send_counts, const Comm &comm, RecvLayoutCache &cache)
    -> const RecvLayout & {
    const auto n = static_cast<int>(send_counts.size());
    const int comm_size = mpi::size(comm);
    if (cache.comm_size == comm_size && static_cast<int>(cache.layout.counts.size()) == n) {
        return cache.layout;
    }

    RecvLayout &out = cache.layout;
    out.counts.resize(static_cast<size_t>(n));
    alltoall_counts(send_counts.data(), out.counts.data(), n, comm);
    out.displs.resize(static_cast<size_t>(n));
    int total = 0;
    for (int i = 0; i < n; ++i) {
        out.displs[static_cast<size_t>(i)] = total;
        total += out.counts[static_cast<size_t>(i)];
    }
    out.total = total;
    cache.comm_size = comm_size;
    return out;
}

/// Idempotent completion handle for a posted payload transfer. wait() finishes a non-blocking
/// transfer; it is a no-op for the blocking path and for non-MPI builds. Move-only so a request is
/// waited on exactly once.
class [[nodiscard("call wait() on the Ticket to complete the posted transfer")]] Ticket {
public:
    Ticket() = default;
    Ticket(const Ticket &) = delete;
    auto operator=(const Ticket &) -> Ticket & = delete;
    Ticket(Ticket &&other) noexcept { *this = std::move(other); }
    auto operator=(Ticket &&other) noexcept -> Ticket & {
#ifdef monoprop_ENABLE_MPI
        request_ = other.request_;
        other.request_ = MPI_REQUEST_NULL;
#endif
        (void)other;
        return *this;
    }
    ~Ticket() = default;

    auto wait() -> void {
#ifdef monoprop_ENABLE_MPI
        if (request_ != MPI_REQUEST_NULL) {
            MPI_Wait(&request_, MPI_STATUS_IGNORE);
            request_ = MPI_REQUEST_NULL;
        }
#endif
    }

#ifdef monoprop_ENABLE_MPI
    // Constructed by post_flat_alltoallv; not intended for direct use.
    explicit Ticket(MPI_Request request) : request_(request) {}

private:
    MPI_Request request_ = MPI_REQUEST_NULL;
#endif
};

/// Post a variable-size all-to-all over caller-owned FLAT buffers. NEVER skipped on zero total: all
/// ranks must participate or the collective deadlocks. Non-blocking (MPI_Ialltoallv) in an MPI build
/// (the Ticket completes it); non-MPI build does a per-rank self-copy (recv layout == send layout).
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
