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
#include <format>
#include <print>
#include <stdexcept>
#include <type_traits>
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

/**
 * @brief Barrier synchronization
 */
inline auto barrier(MPI_Comm comm) -> void {
    MPI_Barrier(comm);
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
 * @brief Allgather for vectors with variable sizes per rank
 *
 * Each rank provides a local vector, and receives all vectors concatenated.
 * The order is by rank: rank 0's data first, then rank 1's, etc.
 *
 * @param local_data Local vector to contribute
 * @param comm MPI communicator
 * @return Vector containing all data from all ranks
 */
template <class T>
inline auto allgatherv(const std::vector<T>& local_data, MPI_Comm comm) -> std::vector<T> {
    const int num_ranks = size(comm);
    const int local_count = static_cast<int>(local_data.size());

    // Gather counts from all ranks
    std::vector<int> counts(num_ranks);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    // Compute displacements
    std::vector<int> displs(num_ranks);
    displs[0] = 0;
    for (int i = 1; i < num_ranks; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    // Compute total size
    const int total_size = displs[num_ranks - 1] + counts[num_ranks - 1];

    // Gather all data
    std::vector<T> all_data(total_size);
    MPI_Allgatherv(local_data.data(),
                   local_count,
                   datatype<T>::get(),
                   all_data.data(),
                   counts.data(),
                   displs.data(),
                   datatype<T>::get(),
                   comm);

    return all_data;
}

/**
 * @brief Allgather for a single value (returns values from all ranks)
 */
template <class T>
inline auto allgather(T local_val, MPI_Comm comm) -> std::vector<T> {
    const int num_ranks = size(comm);
    std::vector<T> all_vals(num_ranks);
    MPI_Allgather(&local_val, 1, datatype<T>::get(), all_vals.data(), 1, datatype<T>::get(), comm);
    return all_vals;
}

/**
 * @brief All-to-all variable-size exchange into caller-provided receive buffers
 *
 * @param send_data Vectors indexed by target rank
 * @param recv_data Output vectors indexed by source rank
 * @param comm MPI communicator
 */
template <class T>
inline auto alltoallv_into(const std::vector<std::vector<T>>& send_data,
                           std::vector<std::vector<T>>& recv_data,
                           MPI_Comm comm) -> void {
    const int num_ranks = size(comm);

    if (static_cast<int>(send_data.size()) != num_ranks) {
        throw std::runtime_error(std::format("alltoallv_into: send_data size ({}) must equal number of ranks ({})",
                                             send_data.size(),
                                             num_ranks));
    }

    static thread_local std::vector<int> send_counts;
    static thread_local std::vector<int> send_displs;
    static thread_local std::vector<int> recv_counts;
    static thread_local std::vector<int> recv_displs;
    static thread_local std::vector<T> send_buffer;
    static thread_local std::vector<T> recv_buffer;

    send_counts.resize(static_cast<size_t>(num_ranks));
    send_displs.resize(static_cast<size_t>(num_ranks));
    recv_counts.resize(static_cast<size_t>(num_ranks));
    recv_displs.resize(static_cast<size_t>(num_ranks));

    size_t total_send = 0;
    for (int i = 0; i < num_ranks; ++i) {
        send_counts[static_cast<size_t>(i)] = static_cast<int>(send_data[static_cast<size_t>(i)].size());
        total_send += send_data[static_cast<size_t>(i)].size();
    }

    send_displs[0] = 0;
    for (int i = 1; i < num_ranks; ++i) {
        send_displs[static_cast<size_t>(i)] =
            send_displs[static_cast<size_t>(i - 1)] + send_counts[static_cast<size_t>(i - 1)];
    }

    send_buffer.resize(total_send);
    for (int i = 0; i < num_ranks; ++i) {
        std::copy(send_data[static_cast<size_t>(i)].begin(),
                  send_data[static_cast<size_t>(i)].end(),
                  send_buffer.begin() + send_displs[static_cast<size_t>(i)]);
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    recv_displs[0] = 0;
    for (int i = 1; i < num_ranks; ++i) {
        recv_displs[static_cast<size_t>(i)] =
            recv_displs[static_cast<size_t>(i - 1)] + recv_counts[static_cast<size_t>(i - 1)];
    }

    const int total_recv =
        recv_displs[static_cast<size_t>(num_ranks - 1)] + recv_counts[static_cast<size_t>(num_ranks - 1)];
    recv_buffer.resize(static_cast<size_t>(total_recv));

    MPI_Alltoallv(send_buffer.data(),
                  send_counts.data(),
                  send_displs.data(),
                  datatype<T>::get(),
                  recv_buffer.data(),
                  recv_counts.data(),
                  recv_displs.data(),
                  datatype<T>::get(),
                  comm);

    recv_data.resize(static_cast<size_t>(num_ranks));
    for (int i = 0; i < num_ranks; ++i) {
        auto& target = recv_data[static_cast<size_t>(i)];
        target.assign(recv_buffer.begin() + recv_displs[static_cast<size_t>(i)],
                      recv_buffer.begin() + recv_displs[static_cast<size_t>(i)] + recv_counts[static_cast<size_t>(i)]);
    }
}

template <class T>
inline auto alltoallv(const std::vector<std::vector<T>>& send_data, MPI_Comm comm) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> recv_data;
    alltoallv_into(send_data, recv_data, comm);
    return recv_data;
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
inline auto barrier(MPI_Comm comm = MPI_COMM_WORLD) -> void {
    static_cast<void>(size(comm)); // no-op for a single process; discard size() to silence unused-param
}

inline auto finalize() -> void {
    barrier();
}

template <class T>
inline auto allreduce_sum(T local_val, MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> T {
    return local_val;
}

inline auto allreduce_sum_inplace(VecD& values, MPI_Comm comm = MPI_COMM_WORLD) -> void {
    values = allreduce_sum(values, comm);
}

template <typename T>
inline auto alltoallv(const std::vector<std::vector<T>>& send_data, MPI_Comm /*comm*/ = MPI_COMM_WORLD)
    -> std::vector<std::vector<T>> {
    // Single rank: just return the data sent to self (index 0)
    if (send_data.empty()) {
        return {{}};
    }
    return {send_data[0]};
}

template <class T>
inline auto alltoallv_into(const std::vector<std::vector<T>>& send_data,
                           std::vector<std::vector<T>>& recv_data,
                           MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> void {
    if (send_data.empty()) {
        recv_data = {{}};
        return;
    }
    recv_data = {send_data[0]};
}

template <class T>
inline auto allgather(T local_val, MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> std::vector<T> {
    return {local_val};
}

template <class T>
inline auto allgatherv(const std::vector<T>& local_data, MPI_Comm /*comm*/ = MPI_COMM_WORLD) -> std::vector<T> {
    return local_data;
}

#endif // monoprop_ENABLE_MPI
} // namespace monoprop::mpi
