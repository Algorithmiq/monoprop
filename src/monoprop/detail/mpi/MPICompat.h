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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <print>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Comm.h owns the MPI_Comm typedef (real <mpi.h> or the int fallback) and the runtime-tagged
// mpi::Comm handle; ShmComm.h is the in-process transport a Kind::Shm handle dispatches to.
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/HybridComm.h" // Kind::Hybrid transport (R MPI ranks x S shards)
#endif

// These includes are here on purpose and should not be moved to the top
#include "monoprop/detail/print_compat.h"
#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::mpi {

// ─── lifecycle (MPI only; ShmComm needs no global init) ──────────────────────────

#ifdef monoprop_ENABLE_MPI
/**
 * @brief Initialize MPI environment. Should be called once at program start. Safe to call repeatedly.
 */
inline auto init(int *argc = nullptr, char ***argv = nullptr) -> void {
    monoprop::threading::init_from_env();
    auto initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        // SERIALIZED (not FUNNELED): under the MPI hybrid, each rank's shard-0 master thread — not the
        // main thread — makes the MPI calls, always one-at-a-time (bracketed by the HybridComm
        // barriers). mpi4py already requests MULTIPLE >= SERIALIZED, so Python is unaffected.
        auto required = MPI_THREAD_SERIALIZED;
        auto provided = 0;
        MPI_Init_thread(argc, argv, required, &provided);
        if (provided < required) {
            auto comm = MPI_COMM_WORLD;
            std::print("Sorry, the MPI library does not provide MPI_THREAD_SERIALIZED support, which is required "
                       "by the shard/MPI hybrid transport.\n");
            MPI_Abort(comm, 1);
        }
    }
}

/**
 * @brief Finalize MPI environment. Should be called once at program end. Safe to call repeatedly.
 */
inline auto finalize() -> void {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

// Template for MPI datatypes (only referenced in the Kind::Mpi transport arms below).
namespace detail {
template <class>
inline constexpr bool unsupported_mpi_datatype_v = false;
} // namespace detail

template <class T>
struct datatype {
    static inline auto get() -> MPI_Datatype {
        if constexpr (std::is_same_v<T, int>) {
            return MPI_INT;
        }
        else if constexpr (std::is_same_v<T, double>) {
            return MPI_DOUBLE;
        }
        else if constexpr (std::is_same_v<T, uint32_t>) {
            return MPI_UINT32_T;
        }
        else if constexpr (std::is_same_v<T, uint64_t>) {
            return MPI_UINT64_T;
        }
        else if constexpr (std::is_same_v<T, size_t>) {
            if constexpr (std::is_same_v<size_t, unsigned int>) {
                return MPI_UNSIGNED;
            }
            else if constexpr (std::is_same_v<size_t, unsigned long>) {
                return MPI_UNSIGNED_LONG;
            }
            else if constexpr (std::is_same_v<size_t, unsigned long long>) {
                return MPI_UNSIGNED_LONG_LONG;
            }
            else {
                static_assert(detail::unsupported_mpi_datatype_v<T>, "Unsupported size_t representation for MPI");
            }
        }
        else {
            static_assert(detail::unsupported_mpi_datatype_v<T>, "Unsupported MPI datatype");
        }
    }
};
#else
inline auto init(int * /*argc*/ = nullptr, char *** /*argv*/ = nullptr) -> void {
    monoprop::threading::init_from_env();
}
inline auto finalize() -> void {}
#endif // monoprop_ENABLE_MPI

// ─── rank / size ─────────────────────────────────────────────────────────────

/// Rank of the caller in `comm`.
inline auto rank(Comm comm) -> int {
    if (comm.kind == Comm::Kind::Shm) {
        return comm.shm_rank;
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        return comm.hyb->global_rank(comm.shm_rank);
    }
    int r = 0;
    if (MPI_Comm_rank(comm.mpi, &r) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Comm_rank failed");
    }
    return r;
#else
    return 0;
#endif
}

/// Total number of participants in `comm`.
inline auto size(Comm comm) -> int {
    if (comm.kind == Comm::Kind::Shm) {
        return comm.shm->size();
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        return comm.hyb->size();
    }
    int s = 0;
    if (MPI_Comm_size(comm.mpi, &s) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Comm_size failed");
    }
    return s;
#else
    return 1;
#endif
}

// ─── allreduce ───────────────────────────────────────────────────────────────

/**
 * @brief Allreduce sum for a single value.
 */
template <class T>
inline auto allreduce_sum(T local_val, Comm comm) -> T {
    if (comm.kind == Comm::Kind::Shm) {
        return comm.shm->allreduce_sum<T>(comm.shm_rank, local_val);
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        return comm.hyb->allreduce_sum<T>(comm.shm_rank, local_val);
    }
    T global_val{};
    MPI_Allreduce(&local_val, &global_val, 1, datatype<T>::get(), MPI_SUM, comm.mpi);
    return global_val;
#else
    return local_val;
#endif
}

/**
 * @brief Allreduce sum for a vector of doubles (in-place).
 */
inline auto allreduce_sum_inplace(VecD &values, Comm comm) -> void {
    if (comm.kind == Comm::Kind::Shm) {
        comm.shm->allreduce_sum_inplace(comm.shm_rank, values.data(), values.size());
        return;
    }
#ifdef monoprop_ENABLE_MPI
    if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->allreduce_sum_inplace(comm.shm_rank, values.data(), values.size());
        return;
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_SUM, comm.mpi);
#else
    (void)values; // single participant: identity
#endif
}

// ─── variable all-to-all (vector-of-vectors) ─────────────────────────────────

/**
 * @brief In-flight variable-size all-to-all. Owns its send/recv buffers + layout so MULTIPLE
 * exchanges can be in flight at once. The per-rank COUNT exchange has already completed when a handle
 * is returned from begin_alltoallv, so recv_counts is valid immediately; wait_into completes the
 * PAYLOAD transfer (a no-op for the Shm and single-process paths, which transfer synchronously in
 * begin_alltoallv) and unpacks by source.
 */
template <class T>
struct PendingAlltoallv {
    int num_ranks = 0;
    std::vector<int> send_counts, send_displs, recv_counts, recv_displs;
    std::vector<T> send_buffer, recv_buffer;
#ifdef monoprop_ENABLE_MPI
    MPI_Request request = MPI_REQUEST_NULL; // set only on the Kind::Mpi async path
#endif

    // Per-source recv counts, known after begin_alltoallv (the transpose of the peers' send counts).
    auto received_counts() const -> const std::vector<int> & { return recv_counts; }

    auto wait_into(std::vector<std::vector<T>> &recv_data) -> void {
#ifdef monoprop_ENABLE_MPI
        if (request != MPI_REQUEST_NULL) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            request = MPI_REQUEST_NULL;
        }
#endif
        recv_data.resize(static_cast<size_t>(num_ranks));
        for (int i = 0; i < num_ranks; ++i) {
            const auto lo = recv_buffer.begin() + recv_displs[static_cast<size_t>(i)];
            recv_data[static_cast<size_t>(i)].assign(lo, lo + recv_counts[static_cast<size_t>(i)]);
        }
    }
};

/**
 * @brief Post a variable-size all-to-all. The per-rank COUNT exchange runs eagerly (recv_counts is
 * known on return). On the Kind::Mpi path the PAYLOAD is posted NON-BLOCKING (MPI_Ialltoallv) so the
 * caller can compute during the transfer, and PendingAlltoallv::wait_into completes it; on the
 * Kind::Shm path (and the single-process build) the transfer runs synchronously here and wait_into
 * just unpacks.
 *
 * @param send_data          Vectors indexed by target rank.
 * @param skip_self          Do not send the self slot (caller handles self inline): self send/recv=0.
 * @param known_recv_counts  Skip the count exchange — the recv counts are already known (e.g. response
 *                           counts are the transpose of the query counts). Self slot is zeroed here
 *                           too when skip_self is set.
 */
template <class T>
inline auto begin_alltoallv(const std::vector<std::vector<T>> &send_data,
                            Comm comm,
                            bool skip_self = false,
                            const std::vector<int> *known_recv_counts = nullptr) -> PendingAlltoallv<T> {
    const int num_ranks = size(comm);
    if (static_cast<int>(send_data.size()) != num_ranks) {
        throw std::runtime_error(std::format("begin_alltoallv: send_data size ({}) must equal number of ranks ({})",
                                             send_data.size(),
                                             num_ranks));
    }
    PendingAlltoallv<T> h;
    h.num_ranks = num_ranks;
    h.send_counts.resize(static_cast<size_t>(num_ranks));
    h.send_displs.resize(static_cast<size_t>(num_ranks));
    h.recv_displs.resize(static_cast<size_t>(num_ranks));

    const int self = skip_self ? rank(comm) : -1;
    size_t total_send = 0;
    for (int i = 0; i < num_ranks; ++i) {
        const int c = (i == self) ? 0 : static_cast<int>(send_data[static_cast<size_t>(i)].size());
        h.send_counts[static_cast<size_t>(i)] = c;
        total_send += static_cast<size_t>(c);
    }
    h.send_displs[0] = 0;
    for (int i = 1; i < num_ranks; ++i) {
        h.send_displs[static_cast<size_t>(i)] =
            h.send_displs[static_cast<size_t>(i - 1)] + h.send_counts[static_cast<size_t>(i - 1)];
    }
    h.send_buffer.resize(total_send);
    for (int i = 0; i < num_ranks; ++i) {
        const int c = h.send_counts[static_cast<size_t>(i)];
        if (c == 0) {
            continue;
        }
        std::copy(send_data[static_cast<size_t>(i)].begin(),
                  send_data[static_cast<size_t>(i)].begin() + c,
                  h.send_buffer.begin() + h.send_displs[static_cast<size_t>(i)]);
    }

    // Resolve recv counts (known transpose, or one count exchange).
    h.recv_counts.resize(static_cast<size_t>(num_ranks));
    if (known_recv_counts != nullptr) {
        std::copy(known_recv_counts->begin(),
                  known_recv_counts->begin()
                      + std::min<size_t>(known_recv_counts->size(), static_cast<size_t>(num_ranks)),
                  h.recv_counts.begin());
        if (self >= 0) {
            h.recv_counts[static_cast<size_t>(self)] = 0;
        }
    }
    else if (comm.kind == Comm::Kind::Shm) {
        comm.shm->alltoall_counts(comm.shm_rank, h.send_counts.data(), h.recv_counts.data());
    }
#ifdef monoprop_ENABLE_MPI
    else if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoall_counts(comm.shm_rank, h.send_counts.data(), h.recv_counts.data());
    }
#endif
    else {
#ifdef monoprop_ENABLE_MPI
        MPI_Alltoall(h.send_counts.data(), 1, MPI_INT, h.recv_counts.data(), 1, MPI_INT, comm.mpi);
#else
        h.recv_counts = h.send_counts; // single participant: recv counts == send counts
#endif
    }

    h.recv_displs[0] = 0;
    for (int i = 1; i < num_ranks; ++i) {
        h.recv_displs[static_cast<size_t>(i)] =
            h.recv_displs[static_cast<size_t>(i - 1)] + h.recv_counts[static_cast<size_t>(i - 1)];
    }
    const int total_recv =
        h.recv_displs[static_cast<size_t>(num_ranks - 1)] + h.recv_counts[static_cast<size_t>(num_ranks - 1)];
    h.recv_buffer.resize(static_cast<size_t>(total_recv));

    // Transport.
    if (comm.kind == Comm::Kind::Shm) {
        comm.shm->alltoallv(comm.shm_rank,
                            h.send_buffer.data(),
                            h.send_displs.data(),
                            h.recv_buffer.data(),
                            h.recv_counts.data(),
                            h.recv_displs.data(),
                            sizeof(T));
    }
#ifdef monoprop_ENABLE_MPI
    else if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoallv(comm.shm_rank,
                            h.send_buffer.data(),
                            h.send_counts.data(),
                            h.send_displs.data(),
                            h.recv_buffer.data(),
                            h.recv_counts.data(),
                            h.recv_displs.data(),
                            sizeof(T),
                            datatype<T>::get());
    }
#endif
    else {
#ifdef monoprop_ENABLE_MPI
        MPI_Ialltoallv(h.send_buffer.data(),
                       h.send_counts.data(),
                       h.send_displs.data(),
                       datatype<T>::get(),
                       h.recv_buffer.data(),
                       h.recv_counts.data(),
                       h.recv_displs.data(),
                       datatype<T>::get(),
                       comm.mpi,
                       &h.request);
#else
        h.recv_buffer = h.send_buffer; // single participant: self round-trip (layouts identical)
#endif
    }
    return h;
}

} // namespace monoprop::mpi
