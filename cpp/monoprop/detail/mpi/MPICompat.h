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

// size() split into ranks * partitions: routing needs to tell a network hop from a shared-memory copy.
struct Geometry {
    int ranks = 1;
    int partitions = 1;
};
monoprop_EXPORT auto geometry(const Comm &comm) -> Geometry;

// Whether this communicator's exchanges go point-to-point: agreed collectively once and cached on it,
// because ranks that disagree hang. Throws RoutingDisagreement on every rank together on a mismatch.
monoprop_EXPORT auto routes_pairwise(const Comm &comm) -> bool;

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

// `n` is the comm size; `plan` narrows the exchange to its peer ranks (dense by default).
monoprop_EXPORT auto alltoall_counts(const int *send_counts, int *recv_counts, int n, Comm comm, PeerPlan plan = {})
    -> void;

// An in-flight variable all-to-all owning its buffers and layout. recv_counts is valid once
// begin_alltoallv returns; wait_into completes the payload and unpacks it by source.
// Move-only and self-draining: MPI reads the buffers until `requests` retire, and a vector move keeps
// the heap blocks those requests point into.
template <typename T>
struct PendingAlltoallv {
    int num_ranks = 0;
    SlotWindow window; // the slots this round touches; counts/displs are zero outside it
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<T> send_buffer;
    std::vector<T> recv_buffer;
#ifdef monoprop_ENABLE_MPI
    std::vector<MPI_Request> requests;
    int posted = 0; // how many of `requests` are live
#endif

    PendingAlltoallv() = default;
    PendingAlltoallv(const PendingAlltoallv &) = delete;
    auto operator=(const PendingAlltoallv &) -> PendingAlltoallv & = delete;

    PendingAlltoallv(PendingAlltoallv &&other) noexcept { adopt_(other); }

    auto operator=(PendingAlltoallv &&other) noexcept -> PendingAlltoallv & {
        if (this != &other) {
            drain();
            adopt_(other);
        }
        return *this;
    }

    ~PendingAlltoallv() { drain(); }

    auto drain() noexcept -> void {
#ifdef monoprop_ENABLE_MPI
        if (posted != 0) {
            MPI_Waitall(posted, requests.data(), MPI_STATUSES_IGNORE);
            posted = 0;
        }
#endif
    }

    auto wait_into(WindowVec<std::vector<T>> &recv_data) -> void {
        drain();
        recv_data.reset(window);
        for (const auto wi : window.indices()) {
            const size_t i = window.slot(wi);
            const auto lo = recv_buffer.begin() + recv_displs[i];
            recv_data[wi].assign(lo, lo + recv_counts[i]);
        }
    }

private:
    auto adopt_(PendingAlltoallv &other) noexcept -> void {
        num_ranks = other.num_ranks;
        window = other.window;
        send_counts = std::move(other.send_counts);
        send_displs = std::move(other.send_displs);
        recv_counts = std::move(other.recv_counts);
        recv_displs = std::move(other.recv_displs);
        send_buffer = std::move(other.send_buffer);
        recv_buffer = std::move(other.recv_buffer);
#ifdef monoprop_ENABLE_MPI
        requests = std::move(other.requests);
        posted = std::exchange(other.posted, 0);
#endif
    }
};

// Debug-only: blocks the caller left outside the plan's window would be dropped, not refused.
template <typename T>
inline auto assert_outside_window_is_empty_([[maybe_unused]] const WindowVec<std::vector<T>> &send_data,
                                            [[maybe_unused]] SlotWindow window) -> void {
#ifndef NDEBUG
    const SlotWindow supplied = send_data.window();
    for (size_t i = supplied.base; i < supplied.stop(); ++i) {
        assert((window.contains(i) || send_data.at_slot(i).empty())
               && "a block outside the plan's peer window would be dropped in silence");
    }
#endif
}

template <typename T>
inline auto prepare_recv_layout_(PendingAlltoallv<T> &pending,
                                 const std::vector<int> *known_recv_counts,
                                 int self,
                                 Comm comm,
                                 PeerPlan plan) -> void {
    const auto window = pending.window;
    if (known_recv_counts != nullptr) {
        // Refused, not truncated: a short array would leave a posted send unmatched.
        if (known_recv_counts->size() < window.stop()) {
            throw CollectiveArgumentError(
                std::format("begin_alltoallv: known_recv_counts has {} entries, short of the plan's [{}, {})",
                            known_recv_counts->size(),
                            window.base,
                            window.stop()));
        }
        for (const auto wi : window.indices()) {
            const size_t i = window.slot(wi);
            pending.recv_counts[i] = (*known_recv_counts)[i];
        }
        if (self >= 0) {
            pending.recv_counts[static_cast<size_t>(self)] = 0;
        }
    }
    else {
        alltoall_counts(pending.send_counts.data(), pending.recv_counts.data(), pending.num_ranks, comm, plan);
    }

    long long running = 0;
    for (const auto wi : window.indices()) {
        const size_t i = window.slot(wi);
        pending.recv_displs[i] = checked_mpi_count(running, "Recv displacement");
        running += pending.recv_counts[i];
    }
    pending.recv_buffer.resize(static_cast<size_t>(checked_mpi_count(running, "Total recv count")));
}

// The count exchange runs eagerly (recv_counts known on return); the Kind::Mpi payload is non-blocking
// (wait_into completes it), Shm / single-process transfer here.
// skip_self: do not send the self slot (the caller handles self inline) — self send/recv = 0.
// known_recv_counts: recv counts already known (e.g. the transpose of the query counts), so skip the
// count exchange. The self slot is also zeroed when skip_self is set.
template <typename T>
[[nodiscard]] inline auto begin_alltoallv(const WindowVec<std::vector<T>> &send_data,
                                          Comm comm,
                                          bool skip_self = false,
                                          const std::vector<int> *known_recv_counts = nullptr,
                                          PeerPlan plan = {}) -> PendingAlltoallv<T> {
    const int num_ranks = size(comm);
    const int me = rank(comm);
    const auto geom = geometry(comm);
    require_routable(plan, geom.ranks);
    PendingAlltoallv<T> h;
    h.num_ranks = num_ranks;
    // The plan's window is the mask. send_data need only cover it; what lies outside must be empty.
    h.window =
        plan.window(static_cast<size_t>(me), static_cast<size_t>(geom.ranks), static_cast<size_t>(geom.partitions));
    const SlotWindow supplied = send_data.window();
    if (h.window.stop() > static_cast<size_t>(num_ranks) || supplied.base > h.window.base
        || supplied.stop() < h.window.stop()) {
        throw CollectiveArgumentError(
            std::format("begin_alltoallv: send_data covers slots [{}, {}), which does not cover the plan's "
                        "[{}, {}) in a {}-slot world",
                        supplied.base,
                        supplied.stop(),
                        h.window.base,
                        h.window.stop(),
                        num_ranks));
    }
    assert_outside_window_is_empty_(send_data, h.window);
    h.send_counts.assign(static_cast<size_t>(num_ranks), 0);
    h.send_displs.assign(static_cast<size_t>(num_ranks), 0);
    h.recv_displs.assign(static_cast<size_t>(num_ranks), 0);

    const int self = skip_self ? me : -1;
    // Wide accumulator + checked narrowing: a wrapped count would feed MPI a negative count.
    long long running_send = 0;
    const auto window = h.window;
    for (const auto wi : window.indices()) {
        const size_t i = window.slot(wi);
        const size_t n = self >= 0 && std::cmp_equal(i, self) ? size_t{0} : send_data.at_slot(i).size();
        const int c = checked_mpi_count(n, "Send count");
        h.send_counts[i] = c;
        h.send_displs[i] = checked_mpi_count(running_send, "Send displacement");
        running_send += c;
    }
    h.send_buffer.resize(static_cast<size_t>(checked_mpi_count(running_send, "Total send count")));
    for (const auto wi : window.indices()) {
        const size_t i = window.slot(wi);
        const int c = h.send_counts[i];
        if (c == 0) {
            continue;
        }
        const auto &block = send_data.at_slot(i);
        std::copy(block.begin(), block.begin() + c, h.send_buffer.begin() + h.send_displs[i]);
    }

    h.recv_counts.assign(static_cast<size_t>(num_ranks), 0);

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

    prepare_recv_layout_(h, known_recv_counts, self, comm, plan);

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
            h.requests.resize(1);
            h.posted = 1;
            MPI_Ialltoallv(h.send_buffer.data(),
                           h.send_counts.data(),
                           h.send_displs.data(),
                           datatype<T>::get(),
                           h.recv_buffer.data(),
                           h.recv_counts.data(),
                           h.recv_displs.data(),
                           datatype<T>::get(),
                           comm.mpi,
                           h.requests.data());
        }
        else {
            // S == 1 world: point-to-point with the peer, left in flight like MPI_Ialltoallv.
            const SparsePairwiseArgs pairwise{
                .plan = plan,
                .me = me,
                .num_ranks = num_ranks,
                .comm = comm.mpi,
                .tag = kFlatPayloadTag,
                .datatype = datatype<T>::get(),
                .elem = sizeof(T),
                .send = reinterpret_cast<const std::byte *>(h.send_buffer.data()),
                .send_layout = {.counts = h.send_counts.data(), .displs = h.send_displs.data()},
                .recv = reinterpret_cast<std::byte *>(h.recv_buffer.data()),
                .recv_layout = {.counts = h.recv_counts.data(), .displs = h.recv_displs.data()},
            };
            h.posted = sparse_pairwise(pairwise, h.requests);
        }
#else
        h.recv_buffer = h.send_buffer; // single participant: self round-trip (layouts identical)
#endif
    }
    return h;
}

} // namespace monoprop::mpi
