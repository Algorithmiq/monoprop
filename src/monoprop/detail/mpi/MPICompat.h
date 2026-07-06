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

#if defined(monoprop_ENABLE_MPI)
#include <mpi.h>
#else
// Fallback MPI types for non-MPI builds.
using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr MPI_Comm MPI_COMM_SELF = 0;
#endif

// These includes are here on purpose and should not be moved to the top
#include "monoprop/detail/print_compat.h"
#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::mpi {

#ifdef monoprop_ENABLE_MPI

/**
 * @brief Initialize MPI environment
 *
 * Should be called once at program start. Safe to call multiple times.
 */
inline auto init(int* argc = nullptr, char*** argv = nullptr) -> void {
    monoprop::threading::init_from_env();
    auto initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        auto required = MPI_THREAD_FUNNELED;
        auto provided = 0;
        MPI_Init_thread(argc, argv, required, &provided);

        if (provided < required) {
            auto comm = MPI_COMM_WORLD;
            std::print("Sorry, the MPI library does not provide MPI_THREAD_FUNNELED support, which is required by\n");
            MPI_Abort(comm, 1);
        }
    }
}

/**
 * @brief Finalize MPI environment
 *
 * Should be called once at program end. Safe to call multiple times.
 */
inline auto finalize() -> void {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

/**
 * @brief Get the rank of the calling process in the given communicator
 */
inline auto rank(MPI_Comm comm) -> int {
    int r = 0;
    if (MPI_Comm_rank(comm, &r) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Comm_rank failed");
    }
    return r;
}

/**
 * @brief Get the total number of processes in the given communicator
 */
inline auto size(MPI_Comm comm) -> int {
    int s = 0;
    if (MPI_Comm_size(comm, &s) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Comm_size failed");
    }
    return s;
}

// Template for MPI datatypes
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

/**
 * @brief Allreduce sum for a single value
 */

template <class T>
inline auto allreduce_sum(T local_val, MPI_Comm comm) -> T {
    T global_val{};
    MPI_Allreduce(&local_val, &global_val, 1, datatype<T>::get(), MPI_SUM, comm);
    return global_val;
}

/**
 * @brief Allreduce sum for a vector of doubles (in-place)
 */
inline auto allreduce_sum_inplace(VecD& values, MPI_Comm comm) -> void {
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_SUM, comm);
}

/**
 * @brief In-flight variable-size all-to-all. Owns its send/recv buffers + layout so MULTIPLE
 * exchanges can be in flight at once (the former thread_local staging could not). The per-rank
 * COUNT exchange has already completed when a handle is returned from begin_alltoallv, so
 * recv_counts is valid immediately; wait_into completes the PAYLOAD transfer and unpacks by source.
 */
template <class T>
struct PendingAlltoallv {
    int num_ranks = 0;
    std::vector<int> send_counts, send_displs, recv_counts, recv_displs;
    std::vector<T> send_buffer, recv_buffer;
#ifdef monoprop_ENABLE_MPI
    MPI_Request request = MPI_REQUEST_NULL;
#endif

    // Per-source recv counts, known after begin_alltoallv (the transpose of the peers' send counts).
    auto received_counts() const -> const std::vector<int>& { return recv_counts; }

    auto wait_into(std::vector<std::vector<T>>& recv_data) -> void {
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
 * known on return); the PAYLOAD is posted NON-BLOCKING (MPI_Ialltoallv) so the caller can compute
 * during the transfer, and PendingAlltoallv::wait_into completes it.
 *
 * @param send_data          Vectors indexed by target rank.
 * @param skip_self          Do not send the self slot (caller handles self inline): self send/recv=0.
 * @param known_recv_counts  Skip the count Alltoall — the recv counts are already known (E4: response
 *                           counts are the transpose of the query counts). Self slot is zeroed here
 *                           too when skip_self is set.
 */
template <class T>
inline auto begin_alltoallv(const std::vector<std::vector<T>>& send_data,
                            MPI_Comm comm,
                            bool skip_self = false,
                            const std::vector<int>* known_recv_counts = nullptr) -> PendingAlltoallv<T> {
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

    if (known_recv_counts != nullptr) {
        h.recv_counts = *known_recv_counts;
        h.recv_counts.resize(static_cast<size_t>(num_ranks));
        if (self >= 0) {
            h.recv_counts[static_cast<size_t>(self)] = 0;
        }
    }
    else {
        h.recv_counts.resize(static_cast<size_t>(num_ranks));
        MPI_Alltoall(h.send_counts.data(), 1, MPI_INT, h.recv_counts.data(), 1, MPI_INT, comm);
    }

    h.recv_displs[0] = 0;
    for (int i = 1; i < num_ranks; ++i) {
        h.recv_displs[static_cast<size_t>(i)] =
            h.recv_displs[static_cast<size_t>(i - 1)] + h.recv_counts[static_cast<size_t>(i - 1)];
    }
    const int total_recv = h.recv_displs[static_cast<size_t>(num_ranks - 1)]
                           + h.recv_counts[static_cast<size_t>(num_ranks - 1)];
    h.recv_buffer.resize(static_cast<size_t>(total_recv));

    MPI_Ialltoallv(h.send_buffer.data(),
                   h.send_counts.data(),
                   h.send_displs.data(),
                   datatype<T>::get(),
                   h.recv_buffer.data(),
                   h.recv_counts.data(),
                   h.recv_displs.data(),
                   datatype<T>::get(),
                   comm,
                   &h.request);
    return h;
}

#else // monoprop_ENABLE_MPI is disabled
// Stub implementations for non-MPI builds (single process)

inline auto init(int* /*argc*/ = nullptr, char*** /*argv*/ = nullptr) -> void {
    monoprop::threading::init_from_env();
}
inline auto rank(MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> int {
    return 0;
}
inline auto size(MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> int {
    return 1;
}
inline auto finalize() -> void {}

template <class T>
inline auto allreduce_sum(T local_val, MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> T {
    return local_val;
}

inline auto allreduce_sum_inplace(VecD& values, MPI_Comm comm = MPI_COMM_WORLD) -> void {
    values = allreduce_sum(values, comm);
}

// Single-process stubs of the nonblocking primitive: the self slot round-trips to itself, so the
// handle just holds the self data and wait_into returns it. Present so headers that call
// begin_alltoallv (guarded by rank_count > 1) still COMPILE in a non-MPI build.
template <class T>
struct PendingAlltoallv {
    std::vector<std::vector<T>> self_data;
    std::vector<int> recv_counts;
    auto received_counts() const -> const std::vector<int>& { return recv_counts; }
    auto wait_into(std::vector<std::vector<T>>& recv_data) -> void { recv_data = std::move(self_data); }
};

template <class T>
inline auto begin_alltoallv(const std::vector<std::vector<T>>& send_data,
                            MPI_Comm /*comm*/ = MPI_COMM_WORLD,
                            bool /*skip_self*/ = false,
                            const std::vector<int>* /*known_recv_counts*/ = nullptr) -> PendingAlltoallv<T> {
    PendingAlltoallv<T> h;
    h.self_data = send_data.empty() ? std::vector<std::vector<T>>{{}} : std::vector<std::vector<T>>{send_data[0]};
    h.recv_counts = {send_data.empty() ? 0 : static_cast<int>(send_data[0].size())};
    return h;
}

#endif // monoprop_ENABLE_MPI
} // namespace monoprop::mpi
