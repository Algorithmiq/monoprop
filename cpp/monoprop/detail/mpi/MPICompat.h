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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <print>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Comm.h owns the MPI_Comm typedef (real or non-MPI fallback) and the runtime-tagged mpi::Comm handle.
#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/ShmComm.h"
#ifdef monoprop_ENABLE_MPI
#include "monoprop/detail/mpi/HybridComm.h"
#include "monoprop/detail/mpi/Pairwise.h"
#endif

// These includes are here on purpose and should not be moved to the top
#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop::mpi {

#ifdef monoprop_ENABLE_MPI
monoprop_EXPORT auto init(int *argc = nullptr, char ***argv = nullptr) -> void;
monoprop_EXPORT auto finalize() -> void;

namespace detail {
template <class>
inline constexpr bool unsupported_mpi_datatype_v = false;
} // namespace detail

template <typename T>
struct datatype {
    static auto get() -> MPI_Datatype {
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
inline auto init(int * /*argc*/ = nullptr, char *** /*argv*/ = nullptr) -> void {}
inline auto finalize() -> void {}
#endif // monoprop_ENABLE_MPI

monoprop_EXPORT auto rank(const Comm &comm) -> int;
monoprop_EXPORT auto size(const Comm &comm) -> int;

// How the flat world of size() is actually built: ranks * partitions. Routing needs the split, because
// an inter-rank message costs a network hop while an inter-partition one is a shared-memory copy --
// size() alone cannot tell them apart. ranks * partitions == size() for every Kind.
struct Geometry {
    int ranks = 1;
    int partitions = 1;
};
auto geometry(const Comm &comm) -> Geometry;

template <typename T>
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

monoprop_EXPORT auto allreduce_sum_inplace(VecD &values, Comm comm) -> void;

// `n` is the comm size. `plan` narrows the exchange to the destination ranks it can reach (see PeerPlan);
// the default is dense, i.e. today's collective.
monoprop_EXPORT auto alltoall_counts(const int *send_counts, int *recv_counts, int n, Comm comm, PeerPlan plan = {})
    -> void;

// In-flight variable-size all-to-all owning its buffers + layout, so several can be in flight.
// recv_counts is valid on return from begin_alltoallv; wait_into completes the payload transfer (a
// no-op on the synchronous Shm / single-process paths) and unpacks by source.
template <typename T>
struct PendingAlltoallv {
    int num_ranks = 0;
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<T> send_buffer;
    std::vector<T> recv_buffer;
#ifdef monoprop_ENABLE_MPI
    MPI_Request request = MPI_REQUEST_NULL; // set only on the Kind::Mpi async path
#endif

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

// The count exchange runs eagerly (recv_counts known on return); the Kind::Mpi payload is non-blocking
// (wait_into completes it), Shm / single-process transfer here.
// skip_self: do not send the self slot (the caller handles self inline) — self send/recv = 0.
// known_recv_counts: recv counts already known (e.g. the transpose of the query counts), so skip the
// count exchange. The self slot is also zeroed when skip_self is set.
template <typename T>
inline auto begin_alltoallv(const std::vector<std::vector<T>> &send_data,
                            Comm comm,
                            bool skip_self = false,
                            const std::vector<int> *known_recv_counts = nullptr,
                            PeerPlan plan = {}) -> PendingAlltoallv<T> {
    const int num_ranks = size(comm);
    if (static_cast<int>(send_data.size()) != num_ranks) {
        throw CollectiveArgumentError(
            std::format("begin_alltoallv: send_data size ({}) must equal number of ranks ({})",
                        send_data.size(),
                        num_ranks));
    }
    PendingAlltoallv<T> h;
    h.num_ranks = num_ranks;
    h.send_counts.resize(static_cast<size_t>(num_ranks));
    h.send_displs.resize(static_cast<size_t>(num_ranks));
    h.recv_displs.resize(static_cast<size_t>(num_ranks));

    const int self = skip_self ? rank(comm) : -1;
    // Wide accumulator + checked narrowing: a wrapped count would size send_buffer short and then feed
    // MPI a negative count/displacement.
    long long total_send = 0;
    for (int i = 0; i < num_ranks; ++i) {
        const size_t n = (i == self) ? 0 : send_data[static_cast<size_t>(i)].size();
        const int c = checked_mpi_count(n, "Send count");
        h.send_counts[static_cast<size_t>(i)] = c;
        total_send += c;
    }
    long long running_send = 0;
    for (int i = 0; i < num_ranks; ++i) {
        h.send_displs[static_cast<size_t>(i)] = checked_mpi_count(running_send, "Send displacement");
        running_send += h.send_counts[static_cast<size_t>(i)];
    }
    h.send_buffer.resize(static_cast<size_t>(checked_mpi_count(total_send, "Total send count")));
    for (int i = 0; i < num_ranks; ++i) {
        const int c = h.send_counts[static_cast<size_t>(i)];
        if (c == 0) {
            continue;
        }
        std::copy(send_data[static_cast<size_t>(i)].begin(),
                  send_data[static_cast<size_t>(i)].begin() + c,
                  h.send_buffer.begin() + h.send_displs[static_cast<size_t>(i)]);
    }

    h.recv_counts.resize(static_cast<size_t>(num_ranks));

    const AlltoallvResolveArgs<T> resolve_args{.send = h.send_buffer.data(),
                                               .send_counts = h.send_counts.data(),
                                               .send_displs = h.send_displs.data(),
                                               .recv = h.recv_buffer,
                                               .recv_counts = h.recv_counts.data(),
                                               .recv_displs = h.recv_displs.data()};
    // Fused fast path (query round, recv layout unknown): resolve recv counts AND move payload in one
    // in-process verb, folding away the count exchange's barriers (Shm 4→2, Hybrid 6→4). It fills
    // recv_counts/recv_displs and resizes recv_buffer; known-layout and pure-MPI paths fall through.
    if (known_recv_counts == nullptr && comm.kind == Comm::Kind::Shm) {
        comm.shm->alltoallv_resolve<T>(comm.shm_rank, resolve_args);
        return h;
    }
#ifdef monoprop_ENABLE_MPI
    if (known_recv_counts == nullptr && comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoallv_resolve<T>(comm.shm_rank, resolve_args, datatype<T>::get(), plan);
        return h;
    }
#endif

    if (known_recv_counts != nullptr) {
        std::copy(
            known_recv_counts->begin(),
            known_recv_counts->begin() + std::min<size_t>(known_recv_counts->size(), static_cast<size_t>(num_ranks)),
            h.recv_counts.begin());
        if (self >= 0) {
            h.recv_counts[static_cast<size_t>(self)] = 0;
        }
        // Mask the caller's array through the plan, as alltoall_counts already does for the counts it
        // exchanges: no receive is ever posted for a non-peer, so a non-zero count there sizes
        // recv_buffer for bytes nothing writes and wait_into hands the caller uninitialised memory.
        if (!plan.dense()) {
            const auto geom = geometry(comm);
            const int me = rank(comm) / geom.partitions;
            for (int g = 0; g < num_ranks; ++g) {
                if (!plan.contains(me, g / geom.partitions)) {
                    h.recv_counts[static_cast<size_t>(g)] = 0;
                }
            }
        }
    }
    else {
        alltoall_counts(h.send_counts.data(), h.recv_counts.data(), num_ranks, comm, plan);
    }

    // Wide accumulator + checked narrowing: see checked_mpi_count.
    long long running = 0;
    for (int i = 0; i < num_ranks; ++i) {
        h.recv_displs[static_cast<size_t>(i)] = checked_mpi_count(running, "Recv displacement");
        running += h.recv_counts[static_cast<size_t>(i)];
    }
    h.recv_buffer.resize(static_cast<size_t>(checked_mpi_count(running, "Total recv count")));

    // Taken after the resize above: recv_buffer may have reallocated.
    const auto flat = FlatAlltoallvArgs<T>{.send = h.send_buffer.data(),
                                           .send_counts = h.send_counts.data(),
                                           .send_displs = h.send_displs.data(),
                                           .recv = h.recv_buffer.data(),
                                           .recv_counts = h.recv_counts.data(),
                                           .recv_displs = h.recv_displs.data()}
                          .bytes();
    if (comm.kind == Comm::Kind::Shm) {
        // ShmComm needs no send counts: its peers pull using the publisher's displacements.
        comm.shm->alltoallv(comm.shm_rank,
                            flat.send,
                            flat.send_displs,
                            flat.recv,
                            flat.recv_counts,
                            flat.recv_displs,
                            flat.elem);
    }
#ifdef monoprop_ENABLE_MPI
    else if (comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoallv(comm.shm_rank, flat, datatype<T>::get(), plan);
    }
#endif
    else {
#ifdef monoprop_ENABLE_MPI
        if (plan.dense()) {
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
        }
        else {
            // S == 1 world: the same pairing as the Hybrid path, one message per reachable peer. Blocking
            // here rather than through the Ticket, because the request set is per-peer, not one handle.
            std::vector<MPI_Request> reqs;
            sparse_pairwise(plan,
                            rank(comm),
                            num_ranks,
                            comm.mpi,
                            kFlatPayloadTag,
                            datatype<T>::get(),
                            sizeof(T),
                            reinterpret_cast<const std::byte *>(h.send_buffer.data()),
                            PeerLayout{.counts = h.send_counts.data(), .displs = h.send_displs.data()},
                            reinterpret_cast<std::byte *>(h.recv_buffer.data()),
                            PeerLayout{.counts = h.recv_counts.data(), .displs = h.recv_displs.data()},
                            reqs);
        }
#else
        h.recv_buffer = h.send_buffer; // single participant: self round-trip (layouts identical)
#endif
    }
    return h;
}

} // namespace monoprop::mpi
