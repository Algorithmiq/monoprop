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
#endif

// These includes are here on purpose and should not be moved to the top
#include "monoprop/TypeAliases.h"

namespace monoprop::mpi {

#ifdef monoprop_ENABLE_MPI
auto init(int *argc = nullptr, char ***argv = nullptr) -> void;
auto finalize() -> void;

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
inline auto init(int * /*argc*/ = nullptr, char *** /*argv*/ = nullptr) -> void {}
inline auto finalize() -> void {}
#endif // monoprop_ENABLE_MPI

auto rank(const Comm &comm) -> int;
auto size(const Comm &comm) -> int;

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

auto allreduce_sum_inplace(VecD &values, Comm comm) -> void;

// `n` is the comm size.
auto alltoall_counts(const int *send_counts, int *recv_counts, int n, Comm comm) -> void;

namespace detail {

// Every buffer one begin_alltoallv call needs, kept together so a call leases them as a unit rather than
// heap-allocating each one. Nothing here carries state between calls: every logical size is set exactly
// by the call that borrows the bundle, only capacity survives.
template <class T>
struct AlltoallvScratch {
    std::vector<int> send_counts;         // [P] elements sent to each destination
    std::vector<int> send_displs;         // [P] element offsets into `send`
    std::vector<int> recv_counts;         // [P] elements received from each source
    std::vector<int> recv_displs;         // [P] element offsets into `recv`
    std::vector<T> send;                  // flat pack of send_data; peers read it in place (see Comm.h)
    std::vector<T> recv;                  // flat receive, laid out by recv_displs
    std::vector<std::vector<T>> unpacked; // wait_into()'s per-source destination
};

// Free list of scratch bundles for element type T, owned by the calling thread.
//
// Ownership scope is the partition master thread, which is exactly the per-partition scope: each
// partition runs its whole gate sequence on one pinned master (see PartitionGroup), so capacity carries
// gate to gate and every buffer stays on the NUMA node whose thread first grew it. Same reasoning as
// Scan.h's `nz` and InvertedIndex.h's fold blocks. Templated on T so the query leg (size_t monomial
// words) can never be handed a bundle typed for the response leg (TermIndex or double) or the reverse:
// the transports address payloads as bytes, but these vectors are the typed owners and
// alltoallv_resolve resizes one of them through a std::vector<T> &.
//
// A free list rather than one cached bundle because PendingAlltoallv promises that several exchanges may
// be in flight at once: concurrent handles each hold a distinct bundle, so no live send buffer is ever
// handed to a second call. Capacity is a high-water mark held until the thread exits — the same trade
// HybridComm's stage_send_ / stage_recv_ already make.
template <class T>
inline auto scratch_free_list() -> std::vector<AlltoallvScratch<T>> & {
    static thread_local std::vector<AlltoallvScratch<T>> free_list;
    return free_list;
}

// Move-only lease of one scratch bundle, returned to the thread's free list on destruction.
template <class T>
class ScratchLease {
public:
    ScratchLease() {
        auto &free_list = scratch_free_list<T>();
        if (!free_list.empty()) {
            buf_ = std::move(free_list.back()); // moving a vector keeps its capacity
            free_list.pop_back();
        }
    }
    ScratchLease(const ScratchLease &) = delete;
    auto operator=(const ScratchLease &) -> ScratchLease & = delete;
    ScratchLease(ScratchLease &&other) noexcept
        : buf_(std::move(other.buf_)),
          held_(std::exchange(other.held_, false)) {}
    auto operator=(ScratchLease &&other) noexcept -> ScratchLease & {
        if (this != &other) {
            give_back_();
            buf_ = std::move(other.buf_);
            held_ = std::exchange(other.held_, false);
        }
        return *this;
    }
    ~ScratchLease() { give_back_(); }

    auto get() -> AlltoallvScratch<T> & { return buf_; }

private:
    auto give_back_() noexcept -> void {
        if (!std::exchange(held_, false)) {
            return;
        }
        // Deliberately no clear(): capacity is the point, and every buffer's logical size is set exactly
        // by the next borrower before anything reads it (see begin_alltoallv / wait_into).
        try {
            scratch_free_list<T>().push_back(std::move(buf_));
        }
        catch (...) {
            // Dropping a bundle only costs its capacity, and this runs in a destructor.
        }
    }

    AlltoallvScratch<T> buf_;
    bool held_ = true;
};

} // namespace detail

// In-flight variable-size all-to-all owning its buffers + layout, so several can be in flight.
// recv_counts is valid on return from begin_alltoallv; wait_into completes the payload transfer (a
// no-op on the synchronous Shm / single-process paths) and unpacks by source.
//
// The buffers are leased from a per-thread free list rather than allocated per call, and the handle's
// lifetime is what makes that safe against the send-buffer contract in Comm.h ("must stay alive and
// unmodified until the verb's second barrier"):
//   - Kind::Shm / Kind::Hybrid: begin_alltoallv returns only after the verb's last barrier, past which
//     no peer reads our send buffer, so the bundle is free the moment the handle dies.
//   - Kind::Mpi: MPI_Ialltoallv still reads the send buffer until the wait, so the destructor completes
//     the request before the bundle goes back to the pool. Recycling it under a live request is the
//     silent-wrong-answer failure this class of change invites, so it is closed here rather than
//     documented as a caller obligation (same reasoning as Exchange.h's Ticket).
// Move-only for the same reason: a copied handle would wait on one request twice.
template <class T>
class PendingAlltoallv {
public:
    PendingAlltoallv() = default;
    PendingAlltoallv(const PendingAlltoallv &) = delete;
    auto operator=(const PendingAlltoallv &) -> PendingAlltoallv & = delete;
    PendingAlltoallv(PendingAlltoallv &&other) noexcept { *this = std::move(other); }
    auto operator=(PendingAlltoallv &&other) noexcept -> PendingAlltoallv & {
        if (this != &other) {
            complete_(); // never drop a request this handle already owns
            num_ranks = std::exchange(other.num_ranks, 0);
            lease_ = std::move(other.lease_);
#ifdef monoprop_ENABLE_MPI
            request = std::exchange(other.request, MPI_REQUEST_NULL);
#endif
        }
        return *this;
    }
    ~PendingAlltoallv() { complete_(); }

    int num_ranks = 0;
#ifdef monoprop_ENABLE_MPI
    MPI_Request request = MPI_REQUEST_NULL; // set only on the Kind::Mpi async path
#endif

    // The leased buffers. begin_alltoallv fills them; callers read `unpacked` through wait_into.
    auto buffers() -> detail::AlltoallvScratch<T> & { return lease_.get(); }

    auto wait_into(std::vector<std::vector<T>> &recv_data) -> void {
        complete_();
        auto &b = buffers();
        recv_data.resize(static_cast<size_t>(num_ranks));
        for (int i = 0; i < num_ranks; ++i) {
            // assign, not insert: the logical size is set exactly from recv_counts, so a reused
            // recv_data keeps only capacity and can carry no element of a longer earlier round.
            const auto lo = b.recv.begin() + b.recv_displs[static_cast<size_t>(i)];
            recv_data[static_cast<size_t>(i)].assign(lo, lo + b.recv_counts[static_cast<size_t>(i)]);
        }
    }

    // Unpack into this handle's own scratch. The result is only valid while the handle lives, which is
    // what lets the per-source blocks keep their capacity from one call to the next.
    auto wait_into() -> std::vector<std::vector<T>> & {
        wait_into(buffers().unpacked);
        return buffers().unpacked;
    }

    // The last wait_into() result. Precondition: wait_into() (the no-argument form) has been called.
    auto received() -> std::vector<std::vector<T>> & { return buffers().unpacked; }

private:
    auto complete_() noexcept -> void {
#ifdef monoprop_ENABLE_MPI
        if (request != MPI_REQUEST_NULL) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            request = MPI_REQUEST_NULL;
        }
#endif
    }

    detail::ScratchLease<T> lease_;
};

// The count exchange runs eagerly (recv_counts known on return); the Kind::Mpi payload is non-blocking
// (wait_into completes it), Shm / single-process transfer here.
// skip_self: do not send the self slot (the caller handles self inline) — self send/recv = 0.
// known_recv_counts: recv counts already known (e.g. the transpose of the query counts), so skip the
// count exchange. The self slot is also zeroed when skip_self is set.
template <class T>
inline auto begin_alltoallv(const std::vector<std::vector<T>> &send_data,
                            Comm comm,
                            bool skip_self = false,
                            const std::vector<int> *known_recv_counts = nullptr) -> PendingAlltoallv<T> {
    const int num_ranks = size(comm);
    if (static_cast<int>(send_data.size()) != num_ranks) {
        throw CollectiveArgumentError(
            std::format("begin_alltoallv: send_data size ({}) must equal number of ranks ({})",
                        send_data.size(),
                        num_ranks));
    }
    PendingAlltoallv<T> h;
    h.num_ranks = num_ranks;
    auto &b = h.buffers();
    const auto p = static_cast<size_t>(num_ranks);
    // resize, not assign: the three loops below write every one of the P entries, so a leased buffer's
    // stale values are all overwritten before anything reads them.
    b.send_counts.resize(p);
    b.send_displs.resize(p);
    b.recv_displs.resize(p);

    const int self = skip_self ? rank(comm) : -1;
    // Wide accumulator + checked narrowing: a wrapped count would size the send buffer short and then
    // feed MPI a negative count/displacement.
    long long total_send = 0;
    for (int i = 0; i < num_ranks; ++i) {
        const size_t n = (i == self) ? 0 : send_data[static_cast<size_t>(i)].size();
        const int c = checked_mpi_count(n, "Send count");
        b.send_counts[static_cast<size_t>(i)] = c;
        total_send += c;
    }
    long long running_send = 0;
    for (int i = 0; i < num_ranks; ++i) {
        b.send_displs[static_cast<size_t>(i)] = checked_mpi_count(running_send, "Send displacement");
        running_send += b.send_counts[static_cast<size_t>(i)];
    }
    // The displacements tile [0, total_send) exactly, so the pack loop below rewrites every live element
    // whatever the previous borrower left behind; resize alone makes the logical size exact.
    b.send.resize(static_cast<size_t>(checked_mpi_count(total_send, "Total send count")));
    for (int i = 0; i < num_ranks; ++i) {
        const int c = b.send_counts[static_cast<size_t>(i)];
        if (c == 0) {
            continue;
        }
        std::copy(send_data[static_cast<size_t>(i)].begin(),
                  send_data[static_cast<size_t>(i)].begin() + c,
                  b.send.begin() + b.send_displs[static_cast<size_t>(i)]);
    }

    // assign, not resize: the known_recv_counts branch below copies only a prefix when the caller passes
    // a short vector, and a leased buffer would otherwise keep an earlier round's counts in the tail.
    b.recv_counts.assign(p, 0);

    const AlltoallvResolveArgs<T> resolve_args{.send = b.send.data(),
                                               .send_counts = b.send_counts.data(),
                                               .send_displs = b.send_displs.data(),
                                               .recv = b.recv,
                                               .recv_counts = b.recv_counts.data(),
                                               .recv_displs = b.recv_displs.data()};
    // Fused fast path (query round, recv layout unknown): resolve recv counts AND move payload in one
    // in-process verb, folding away the count exchange's barriers (Shm 4→2, Hybrid 6→4). It fills
    // recv_counts/recv_displs and resizes recv_buffer; known-layout and pure-MPI paths fall through.
    if (known_recv_counts == nullptr && comm.kind == Comm::Kind::Shm) {
        comm.shm->alltoallv_resolve<T>(comm.shm_rank, resolve_args);
        return h;
    }
#ifdef monoprop_ENABLE_MPI
    if (known_recv_counts == nullptr && comm.kind == Comm::Kind::Hybrid) {
        comm.hyb->alltoallv_resolve<T>(comm.shm_rank, resolve_args, datatype<T>::get());
        return h;
    }
#endif

    if (known_recv_counts != nullptr) {
        std::copy(
            known_recv_counts->begin(),
            known_recv_counts->begin() + std::min<size_t>(known_recv_counts->size(), static_cast<size_t>(num_ranks)),
            b.recv_counts.begin());
        if (self >= 0) {
            b.recv_counts[static_cast<size_t>(self)] = 0;
        }
    }
    else {
        alltoall_counts(b.send_counts.data(), b.recv_counts.data(), num_ranks, comm);
    }

    // Wide accumulator + checked narrowing: see checked_mpi_count.
    long long running = 0;
    for (int i = 0; i < num_ranks; ++i) {
        b.recv_displs[static_cast<size_t>(i)] = checked_mpi_count(running, "Recv displacement");
        running += b.recv_counts[static_cast<size_t>(i)];
    }
    b.recv.resize(static_cast<size_t>(checked_mpi_count(running, "Total recv count")));

    // Taken after the resize above: the recv buffer may have reallocated.
    const auto flat = FlatAlltoallvArgs<T>{.send = b.send.data(),
                                           .send_counts = b.send_counts.data(),
                                           .send_displs = b.send_displs.data(),
                                           .recv = b.recv.data(),
                                           .recv_counts = b.recv_counts.data(),
                                           .recv_displs = b.recv_displs.data()}
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
        comm.hyb->alltoallv(comm.shm_rank, flat, datatype<T>::get());
    }
#endif
    else {
#ifdef monoprop_ENABLE_MPI
        MPI_Ialltoallv(b.send.data(),
                       b.send_counts.data(),
                       b.send_displs.data(),
                       datatype<T>::get(),
                       b.recv.data(),
                       b.recv_counts.data(),
                       b.recv_displs.data(),
                       datatype<T>::get(),
                       comm.mpi,
                       &h.request);
#else
        b.recv = b.send; // single participant: self round-trip (layouts identical)
#endif
    }
    return h;
}

} // namespace monoprop::mpi
