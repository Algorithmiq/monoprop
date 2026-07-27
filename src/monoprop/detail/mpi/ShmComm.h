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

#include "monoprop/detail/mpi/ShardBarrier.h"

// In-process shared-memory SPMD transport: S shard-master threads each call the same collective
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

    // Transpose per-rank send counts into per-rank recv counts: recv[s] = amount rank s sends to me.
    auto alltoall_counts(int rank, const int *send_counts, int *recv_counts) -> void {
        slots_[static_cast<size_t>(rank)].counts = send_counts;
        sync();
        for (int s = 0; s < n_; ++s) {
            recv_counts[s] = slots_[static_cast<size_t>(s)].counts[rank];
        }
        sync();
    }

    // Variable all-to-all over caller-owned FLAT buffers (counts/displs in ELEMENTS, `elem` = element
    // bytes). Fills recv per source s ascending at recv_displs[s]; recv_counts must already hold the
    // transpose — same contract as MPI_Alltoallv.
    auto alltoallv(int rank,
                   const void *send,
                   const int *send_displs,
                   void *recv,
                   const int *recv_counts,
                   const int *recv_displs,
                   size_t elem) -> void {
        Slot &me = slots_[static_cast<size_t>(rank)];
        me.ptr = send;
        me.displs = send_displs;
        sync();
        auto *dst = static_cast<char *>(recv);
        for (int s = 0; s < n_; ++s) {
            const Slot &src = slots_[static_cast<size_t>(s)];
            const auto *sp = static_cast<const char *>(src.ptr);
            const auto count = static_cast<size_t>(recv_counts[s]);
            if (count == 0) {
                continue;
            }
            std::memcpy(dst + static_cast<size_t>(recv_displs[s]) * elem,
                        sp + static_cast<size_t>(src.displs[rank]) * elem,
                        count * elem);
        }
        sync();
    }

    // Fused count-resolve + payload all-to-all in ONE round (2 syncs vs 4). recv_counts / recv_displs
    // and `recv` (resized) are OUTPUTS. Same contiguous ascending-source ordering as alltoallv, which
    // the query/response positional pairing depends on.
    template <class T>
    auto alltoallv_resolve(int rank,
                           const T *send,
                           const int *send_counts,
                           const int *send_displs,
                           std::vector<T> &recv,
                           int *recv_counts,
                           int *recv_displs) -> void {
        Slot &me = slots_[static_cast<size_t>(rank)];
        me.ptr = send;
        me.displs = send_displs;
        me.counts = send_counts;
        sync(); // B1: send buffers published
        size_t total = 0;
        for (int s = 0; s < n_; ++s) {
            const int c = slots_[static_cast<size_t>(s)].counts[rank]; // what s sends to me
            recv_counts[s] = c;
            recv_displs[s] = static_cast<int>(total);
            total += static_cast<size_t>(c);
        }
        recv.resize(total);
        auto *dst = reinterpret_cast<char *>(recv.data());
        for (int s = 0; s < n_; ++s) {
            const Slot &src = slots_[static_cast<size_t>(s)];
            const auto count = static_cast<size_t>(recv_counts[s]);
            if (count == 0) {
                continue;
            }
            std::memcpy(dst + static_cast<size_t>(recv_displs[s]) * sizeof(T),
                        reinterpret_cast<const char *>(src.ptr) + static_cast<size_t>(src.displs[rank]) * sizeof(T),
                        count * sizeof(T));
        }
        sync(); // B2: peers finished reading our send buffer before the caller may reuse it
    }

    // Summed in ascending rank order.
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

    // In-place element-wise allreduce-sum, ascending rank order (bit-identical on every rank). Safe in
    // place: each element is read then overwritten by its single slice owner, and slices are cache-line-rounded.
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

    // Called by the shard dispatcher when a participant unwinds. See ShardBarrier::poison.
    auto poison() -> void { barrier_.poison(); }

    // Clear the poison flag and arrival counter between collective rounds. See ShardBarrier::reset.
    auto reset() -> void { barrier_.reset(); }

private:
    // One cache-line-isolated publish slot per rank (no false sharing). A rank writes only its own slot,
    // and reads peers' slots only between the two barriers of a collective.
    struct alignas(64) Slot {
        const void *ptr = nullptr;
        const int *counts = nullptr;
        const int *displs = nullptr;
        double *vec = nullptr; // mutable: allreduce_sum_inplace writes results back through peers' slots
        double f64 = 0.0;
        uint64_t u64 = 0;
    };

    auto sync() -> void { barrier_.sync(); }

    int n_;
    std::vector<Slot> slots_;
    ShardBarrier barrier_;
};

} // namespace monoprop::mpi
