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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/PartitionBarrier.h"

// In-process shared-memory SPMD transport: S partition-master threads each call the same collective
// sequence in program order, every collective two-phase (publish slot → barrier → read peers →
// barrier). Two guarantees the engine relies on: alltoallv delivers each source's block contiguously
// in ascending source-rank order (Resolve.h's positional pairing needs it), and allreduce sums in
// ascending rank order, so results are bit-identical and deterministic per S.

namespace monoprop::mpi {

class ShmComm {
public:
    explicit ShmComm(int n) : n_(n), slots_(static_cast<size_t>(n)), barrier_(n) {}

    ShmComm(const ShmComm &) = delete;
    auto operator=(const ShmComm &) -> ShmComm & = delete;

    auto size() const -> int { return n_; }

    // recv_counts[s] = what rank s sends to me (the transpose of the send-count matrix).
    auto alltoall_counts(int rank, const int *send_counts, int *recv_counts) -> void {
        slots_[static_cast<size_t>(rank)].counts = send_counts;
        sync();
        for (int s = 0; s < n_; ++s) {
            recv_counts[s] = slots_[static_cast<size_t>(s)].counts[rank];
        }
        sync();
    }

    // Variable all-to-all over caller-owned flat buffers, addressed as raw bytes (counts/displs stay in
    // elements, `elem` = element bytes — the pointers carry no element type to scale by). Fills recv per
    // source s ascending at recv_displs[s]; recv_counts must already hold the transpose — same contract
    // as MPI_Alltoallv. `send` must stay alive and unmodified until the second barrier: peers read it.
    auto alltoallv(int rank,
                   const std::byte *send,
                   const int *send_displs,
                   std::byte *recv,
                   const int *recv_counts,
                   const int *recv_displs,
                   size_t elem) -> void {
        Slot &me = slots_[static_cast<size_t>(rank)];
        me.ptr = send;
        me.displs = send_displs;
        sync();
        auto *dst = recv;
        for (int s = 0; s < n_; ++s) {
            const Slot &src = slots_[static_cast<size_t>(s)];
            const auto count = static_cast<size_t>(recv_counts[s]);
            // Bail before offsetting src: a source that sends us nothing need not have published a
            // buffer at all, and its displs[rank] may then point past the end of one.
            if (count == 0) {
                continue;
            }
            std::memcpy(dst + static_cast<size_t>(recv_displs[s]) * elem,
                        src.ptr + static_cast<size_t>(src.displs[rank]) * elem,
                        count * elem);
        }
        sync();
    }

    // Fused count-resolve + payload all-to-all in one round (2 syncs vs 4). Same contiguous
    // ascending-source ordering as alltoallv, which the query/response positional pairing depends on.
    // See AlltoallvResolveArgs for the lifetime and element-offset contract.
    template <class T>
    auto alltoallv_resolve(int rank, const AlltoallvResolveArgs<T> &args) -> void {
        Slot &me = slots_[static_cast<size_t>(rank)];
        // Typed here but byte-addressed in the slot: peers scatter by (displ, count) in elements and
        // never reconstruct T, so the slot stays type-erased for the untyped alltoallv above.
        me.ptr = reinterpret_cast<const std::byte *>(args.send);
        me.displs = args.send_displs;
        me.counts = args.send_counts;
        sync(); // B1: send buffers published
        long long total = 0;
        for (int s = 0; s < n_; ++s) {
            const int c = slots_[static_cast<size_t>(s)].counts[rank]; // what s sends to me
            args.recv_counts[s] = c;
            args.recv_displs[s] = checked_mpi_count(total, "Recv displacement");
            total += c;
        }
        args.recv.resize(static_cast<size_t>(checked_mpi_count(total, "Total recv count")));
        auto *dst = reinterpret_cast<std::byte *>(args.recv.data()); // after the resize: it may reallocate
        for (int s = 0; s < n_; ++s) {
            const Slot &src = slots_[static_cast<size_t>(s)];
            const auto count = static_cast<size_t>(args.recv_counts[s]);
            // See alltoallv: an unpublished source must not be offset by its displacement.
            if (count == 0) {
                continue;
            }
            std::memcpy(dst + static_cast<size_t>(args.recv_displs[s]) * sizeof(T),
                        src.ptr + static_cast<size_t>(src.displs[rank]) * sizeof(T),
                        count * sizeof(T));
        }
        sync(); // B2: peers finished reading our send buffer before the caller may reuse it
    }

    template <class T>
    auto allreduce_sum(int rank, T local_val) -> T {
        Slot &me = slots_[static_cast<size_t>(rank)];
        if constexpr (std::is_floating_point_v<T>) {
            me.f64 = static_cast<double>(local_val);
        }
        else {
            me.u64 = static_cast<uint64_t>(local_val);
        }
        sync();
        T acc{};
        for (int s = 0; s < n_; ++s) {
            const Slot &src = slots_[static_cast<size_t>(s)];
            if constexpr (std::is_floating_point_v<T>) {
                acc += static_cast<T>(src.f64);
            }
            else {
                acc += static_cast<T>(src.u64);
            }
        }
        sync();
        return acc;
    }

    // Safe in place: each element is read then overwritten by its single slice owner, and slices are
    // cache-line-rounded.
    auto allreduce_sum_inplace(int rank, double *values, size_t len) -> void {
        slots_[static_cast<size_t>(rank)].vec = values;
        sync();
        constexpr size_t kLine = 64 / sizeof(double);
        const size_t lines = (len + kLine - 1) / kLine;
        const size_t per = (lines + static_cast<size_t>(n_) - 1) / static_cast<size_t>(n_);
        const size_t lo = std::min(len, static_cast<size_t>(rank) * per * kLine);
        const size_t hi = std::min(len, lo + per * kLine);
        for (size_t k = lo; k < hi; ++k) {
            double acc = 0.0;
            for (int s = 0; s < n_; ++s) { // ascending rank order
                acc += slots_[static_cast<size_t>(s)].vec[k];
            }
            for (int s = 0; s < n_; ++s) { // publish the same bits into every rank's buffer
                slots_[static_cast<size_t>(s)].vec[k] = acc;
            }
        }
        sync(); // peers write into our buffer (and read from it) until here
    }

    // See PartitionBarrier::poison / ::reset for when each is legal to call.
    auto poison() -> void { barrier_.poison(); }

    auto reset() -> void { barrier_.reset(); }

private:
    // One cache-line-isolated publish slot per rank. A rank writes only its own slot, and reads peers'
    // slots only between the two barriers of a collective.
    struct alignas(64) Slot {
        const std::byte *ptr = nullptr; // byte view of the publisher's send buffer; null until published
        const int *counts = nullptr;
        const int *displs = nullptr;
        double *vec = nullptr; // mutable: allreduce_sum_inplace writes results back through peers' slots
        double f64 = 0.0;
        uint64_t u64 = 0;
    };

    auto sync() -> void { barrier_.sync(); }

    int n_;
    std::vector<Slot> slots_;
    PartitionBarrier barrier_;
};

} // namespace monoprop::mpi
