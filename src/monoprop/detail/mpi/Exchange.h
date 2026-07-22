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

// Variable-size all-to-all facade over caller-owned FLAT buffers. This is the ONE place (besides the
// vector-of-vectors begin_alltoallv in MPICompat.h) that names MPI_Alltoall / MPI_[I]alltoallv /
// MPI_Wait / MPI_Request, and the ONE place that states the "every rank must participate — never skip
// on zero counts" deadlock discipline. Non-MPI builds get self-copy stubs so callers compile and run
// unchanged. Consumers (replay exchange, pare exchange) hold no #ifdef monoprop_ENABLE_MPI.

namespace monoprop::mpi {

// ─── count exchange ──────────────────────────────────────────────────────────

/// Resolve the recv side of a send-count vector, reusing `cache` when the communicator size is
/// unchanged (the send pattern of a replayed graph is fixed ⇒ recv layout is identical every call, so
/// a hit removes one blocking count round-trip per layer per evaluation). Resolution order:
///   1. cache hit (cache.comm_size == size and same rank count) → no MPI;
///   2. otherwise → one MPI_Alltoall via alltoall_counts.
/// The resolved layout is stored in `cache` and returned by reference.
inline auto resolve_recv(std::span<const int> send_counts, Comm comm, RecvLayoutCache &cache) -> const RecvLayout & {
    const int n = static_cast<int>(send_counts.size());
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

// ─── payload exchange ────────────────────────────────────────────────────────

/// Idempotent completion handle for a posted payload transfer. wait() finishes a non-blocking
/// transfer; it is a no-op for the blocking path and for non-MPI builds. Move-only so a request is
/// waited on exactly once.
class [[nodiscard]] Ticket {
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

/// Post a variable-size all-to-all over caller-owned FLAT send/recv buffers with the given per-rank
/// counts/displacements. NEVER skipped on zero total: all ranks must participate in the collective
/// (skipping on one while another has data deadlocks). Non-blocking (MPI_Ialltoallv) in an MPI build;
/// the returned Ticket completes the transfer. Non-MPI build: per-rank copy of send into recv (recv
/// layout must equal send layout, which holds at communicator size 1).
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
