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
monoprop_EXPORT auto geometry(const Comm &comm) -> Geometry;

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

// The per-slot block arrays begin_alltoallv and wait_into accept: a plain [P] vector-of-vectors, or a
// WindowVec over the slots a PeerPlan can reach. These four overload pairs are the whole difference --
// the verbs below are one code path walking one SlotWindow.
template <typename Blocks>
using SlotBlockValue = typename Blocks::value_type::value_type;

template <typename T>
inline auto slot_window_of(const std::vector<std::vector<T>> &v) -> SlotWindow {
    return SlotWindow{.base = 0, .count = v.size()};
}
template <typename T>
inline auto slot_window_of(const WindowVec<std::vector<T>> &v) -> SlotWindow {
    return v.window();
}

template <typename T>
inline auto slot_block(const std::vector<std::vector<T>> &v, size_t slot) -> const std::vector<T> & {
    return v[slot];
}
template <typename T>
inline auto slot_block(std::vector<std::vector<T>> &v, size_t slot) -> std::vector<T> & {
    return v[slot];
}
template <typename T>
inline auto slot_block(const WindowVec<std::vector<T>> &v, size_t slot) -> const std::vector<T> & {
    return v.at_slot(slot);
}
template <typename T>
inline auto slot_block(WindowVec<std::vector<T>> &v, size_t slot) -> std::vector<T> & {
    return v.at_slot(slot);
}

// A plain destination keeps the full-world shape (a non-peer's block is empty, not absent); a WindowVec
// takes the round's window.
template <typename T>
inline auto reset_slots(std::vector<std::vector<T>> &v, SlotWindow /*w*/, size_t world) -> void {
    v.assign(world, std::vector<T>{});
}
template <typename T>
inline auto reset_slots(WindowVec<std::vector<T>> &v, SlotWindow w, size_t /*world*/) -> void {
    v.reset(w);
}

// In-flight variable-size all-to-all owning its buffers + layout, so several can be in flight.
// recv_counts is valid on return from begin_alltoallv; wait_into completes the payload transfer (a
// no-op on the synchronous Shm / single-process paths) and unpacks by source.
template <typename T>
struct PendingAlltoallv {
    int num_ranks = 0;
    // The slots this round touches; counts/displs are zero outside it. Set by begin_alltoallv.
    SlotWindow window;
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    std::vector<T> send_buffer;
    std::vector<T> recv_buffer;
#ifdef monoprop_ENABLE_MPI
    MPI_Request request = MPI_REQUEST_NULL; // set only on the Kind::Mpi dense async path
    std::vector<MPI_Request> requests;      // the Kind::Mpi sparse path's pairs; `posted` of them live
    int posted = 0;                         // MPI reads send_buffer/recv_buffer until these complete,
                                            // and both move with the handle, so the pointers hold
#endif

    template <typename Dest>
    auto wait_into(Dest &recv_data) -> void {
#ifdef monoprop_ENABLE_MPI
        if (request != MPI_REQUEST_NULL) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            request = MPI_REQUEST_NULL;
        }
        if (posted != 0) {
            MPI_Waitall(posted, requests.data(), MPI_STATUSES_IGNORE);
            posted = 0;
        }
#endif
        reset_slots(recv_data, window, static_cast<size_t>(num_ranks));
        for (size_t k = 0; k < window.count; ++k) {
            const size_t i = window.slot(WindowIndex{k});
            const auto lo = recv_buffer.begin() + recv_displs[i];
            slot_block(recv_data, i).assign(lo, lo + recv_counts[i]);
        }
    }
};

// Debug-only: a caller may supply more slots than the plan reaches, and anything it left outside the
// window is DROPPED rather than refused -- the silent failure mode a wrong-but-agreed shift produces.
template <typename Blocks>
inline auto assert_outside_window_is_empty_([[maybe_unused]] const Blocks &send_data,
                                            [[maybe_unused]] SlotWindow supplied,
                                            [[maybe_unused]] SlotWindow window) -> void {
#ifndef NDEBUG
    for (size_t i = supplied.base; i < supplied.stop(); ++i) {
        assert((window.contains(i) || slot_block(send_data, i).empty())
               && "a block outside the plan's peer window would be dropped in silence");
    }
#endif
}

// The count exchange runs eagerly (recv_counts known on return); the Kind::Mpi payload is non-blocking
// (wait_into completes it), Shm / single-process transfer here.
// skip_self: do not send the self slot (the caller handles self inline) — self send/recv = 0.
// known_recv_counts: recv counts already known (e.g. the transpose of the query counts), so skip the
// count exchange. The self slot is also zeroed when skip_self is set.
template <typename Blocks, typename T = SlotBlockValue<Blocks>>
inline auto begin_alltoallv(const Blocks &send_data,
                            Comm comm,
                            bool skip_self = false,
                            const std::vector<int> *known_recv_counts = nullptr,
                            PeerPlan plan = {}) -> PendingAlltoallv<T> {
    const int num_ranks = size(comm);
    const int me = rank(comm);
    const auto geom = geometry(comm);
    PendingAlltoallv<T> h;
    h.num_ranks = num_ranks;
    // The plan IS the mask, dense included -- it is the count == P value of the same window. A caller may
    // hand a whole [P] array under a sparse plan (the tests do), so the supplied array only has to COVER
    // the window; assert_outside_window_is_empty_ catches what it leaves outside, which is the silent
    // drop a wrong-but-agreed shift produces.
    h.window =
        plan.window(static_cast<size_t>(me), static_cast<size_t>(geom.ranks), static_cast<size_t>(geom.partitions));
    const SlotWindow supplied = slot_window_of(send_data);
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
    assert_outside_window_is_empty_(send_data, supplied, h.window);
    h.send_counts.assign(static_cast<size_t>(num_ranks), 0);
    h.send_displs.assign(static_cast<size_t>(num_ranks), 0);
    h.recv_displs.assign(static_cast<size_t>(num_ranks), 0);

    const int self = skip_self ? me : -1;
    // Counts and their prefix in ONE sweep over the window; the rest stay zero from the assign above.
    // Wide accumulator + checked narrowing: a wrapped count would size send_buffer short and then feed
    // MPI a negative count/displacement.
    long long running_send = 0;
    for (size_t k = 0; k < h.window.count; ++k) {
        const size_t i = h.window.slot(WindowIndex{k});
        const size_t n = (static_cast<int>(i) == self) ? 0 : slot_block(send_data, i).size();
        const int c = checked_mpi_count(n, "Send count");
        h.send_counts[i] = c;
        h.send_displs[i] = checked_mpi_count(running_send, "Send displacement");
        running_send += c;
    }
    h.send_buffer.resize(static_cast<size_t>(checked_mpi_count(running_send, "Total send count")));
    for (size_t k = 0; k < h.window.count; ++k) {
        const size_t i = h.window.slot(WindowIndex{k});
        const int c = h.send_counts[i];
        if (c == 0) {
            continue;
        }
        const auto &block = slot_block(send_data, i);
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

    if (known_recv_counts != nullptr) {
        // The caller's array is FLAT [P]; the window is the mask, as alltoall_counts already masks the
        // counts it exchanges. No receive is ever posted for a non-peer, so a non-zero count there sizes
        // recv_buffer for bytes nothing writes and wait_into hands the caller uninitialised memory.
        const size_t avail = known_recv_counts->size();
        for (size_t k = 0; k < h.window.count; ++k) {
            const size_t i = h.window.slot(WindowIndex{k});
            if (i < avail) {
                h.recv_counts[i] = (*known_recv_counts)[i];
            }
        }
        if (self >= 0) {
            h.recv_counts[static_cast<size_t>(self)] = 0;
        }
    }
    else {
        alltoall_counts(h.send_counts.data(), h.recv_counts.data(), num_ranks, comm, plan);
    }

    // Wide accumulator + checked narrowing: see checked_mpi_count.
    long long running = 0;
    for (size_t k = 0; k < h.window.count; ++k) {
        const size_t i = h.window.slot(WindowIndex{k});
        h.recv_displs[i] = checked_mpi_count(running, "Recv displacement");
        running += h.recv_counts[i];
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
            // S == 1 world: the same pairing as the Hybrid path, one message per reachable peer, left
            // in flight in the handle exactly as MPI_Ialltoallv is. The buffers MPI holds live in `h`
            // and travel with it: a vector move keeps its heap block, so returning `h` moves nothing
            // MPI is reading.
            h.posted = sparse_pairwise(plan,
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
                                       h.requests);
        }
#else
        h.recv_buffer = h.send_buffer; // single participant: self round-trip (layouts identical)
#endif
    }
    return h;
}

} // namespace monoprop::mpi
