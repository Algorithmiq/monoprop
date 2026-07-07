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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

// In-process shared-memory SPMD transport. S participant threads (shard masters) each hold a
// mpi::Comm{Kind::Shm, this, rank} and call the SAME sequence of collectives in program order — the
// exact SPMD discipline an MPI rank set follows. Every collective is a two-phase barrier:
// publish-my-slot → barrier → read-peers'-slots → barrier. This is the sole shared-memory analogue
// of MPI_Alltoall / MPI_Alltoallv / MPI_Allreduce; the vector-of-vectors packing and the flat-buffer
// facade stay in MPICompat.h / Exchange.h, which call these primitives, so this class is a pure
// transport with no knowledge of the engine's payload types.
//
// Two properties the engine relies on are guaranteed here:
//   • alltoallv delivers each source's block CONTIGUOUSLY in ASCENDING source-rank order — the exact
//     ordering MPI_Alltoallv provides, on which the positional query/response pairing in Resolve.h
//     depends (bit-exact term-index assignment across shard counts).
//   • allreduce sums in ASCENDING rank order on every rank, so the result is bit-identical on all
//     ranks and deterministic for a given S (stronger than MPI, whose reduction order is unspecified;
//     the codebase already accepts rank-count-dependent FP results).

namespace monoprop::mpi {

/// Thrown by a collective when a peer shard unwound (set the poison flag) instead of arriving —
/// turns a would-be permanent barrier hang into a propagating exception on every participant.
class ShmCommPoisoned : public std::runtime_error {
public:
    ShmCommPoisoned() : std::runtime_error("ShmComm poisoned: a peer shard threw during a collective") {}
};

class ShmComm {
public:
    explicit ShmComm(int n) : n_(n), slots_(static_cast<size_t>(n)) {}

    ShmComm(const ShmComm &) = delete;
    auto operator=(const ShmComm &) -> ShmComm & = delete;

    auto size() const -> int { return n_; }

    /// Transpose per-rank send counts into per-rank recv counts: recv[s] = amount rank s sends to me.
    auto alltoall_counts(int rank, const int *send_counts, int *recv_counts) -> void {
        slots_[static_cast<size_t>(rank)].counts = send_counts;
        sync();
        for (int s = 0; s < n_; ++s) {
            recv_counts[s] = slots_[static_cast<size_t>(s)].counts[rank];
        }
        sync();
    }

    /// Variable all-to-all over caller-owned FLAT buffers (counts/displs in ELEMENTS, `elem` = element
    /// size in bytes). Rank r's recv buffer is filled, for each source s ascending, with s's block
    /// destined for r placed at recv_displs[s]. recv_counts[s] must equal what s sends to r (the
    /// caller establishes this via alltoall_counts or a known transpose — same contract as MPI).
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
        char *dst = static_cast<char *>(recv);
        for (int s = 0; s < n_; ++s) {
            const Slot &src = slots_[static_cast<size_t>(s)];
            const char *sp = static_cast<const char *>(src.ptr);
            const size_t count = static_cast<size_t>(recv_counts[s]);
            if (count == 0) {
                continue;
            }
            std::memcpy(dst + static_cast<size_t>(recv_displs[s]) * elem,
                        sp + static_cast<size_t>(src.displs[rank]) * elem,
                        count * elem);
        }
        sync();
    }

    /// Allreduce-sum of a scalar (double or an unsigned integer type), summed in ascending rank order.
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

    /// In-place element-wise allreduce-sum of a double vector (all ranks pass the same length), summed
    /// in ascending rank order so every rank ends with the bit-identical result.
    auto allreduce_sum_inplace(int rank, double *values, size_t len) -> void {
        slots_[static_cast<size_t>(rank)].vec = values;
        sync();
        std::vector<double> tmp(len, 0.0);
        for (int s = 0; s < n_; ++s) {
            const double *vs = slots_[static_cast<size_t>(s)].vec;
            for (size_t k = 0; k < len; ++k) {
                tmp[k] += vs[k];
            }
        }
        sync();
        std::memcpy(values, tmp.data(), len * sizeof(double));
    }

    /// Signal that this participant is unwinding (e.g. an engine exception): release peers spinning in
    /// a barrier so they throw ShmCommPoisoned rather than hang forever. Idempotent.
    auto poison() -> void { poisoned_.store(true, std::memory_order_release); }

    /// Clear the poison flag and the barrier's arrival counter. MUST be called only when every
    /// participant is quiescent (between collective rounds, e.g. by the shard dispatcher before a new
    /// job), so a round aborted by poison leaves no dirty state for the next round. The generation is
    /// left monotonic (each participant re-reads it at its next barrier).
    auto reset() -> void {
        poisoned_.store(false, std::memory_order_relaxed);
        arrived_.store(0, std::memory_order_relaxed);
    }

private:
    // One cache-line-isolated publish slot per rank (no false sharing between publishers). A rank only
    // ever writes its own slot and only reads peers' slots between the two barriers of a collective.
    struct alignas(64) Slot {
        const void *ptr = nullptr;
        const int *counts = nullptr;
        const int *displs = nullptr;
        const double *vec = nullptr;
        double f64 = 0.0;
        uint64_t u64 = 0;
    };

    // Sense-reversing generation barrier with a poison escape. The completer (last arriver) resets the
    // counter then bumps the generation, releasing spinners; a poisoned peer that never arrives is
    // covered because spinners also break on the poison flag and throw.
    auto sync() -> void {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            arrived_.store(0, std::memory_order_relaxed);
            gen_.store(g + 1, std::memory_order_release);
        }
        else {
            while (gen_.load(std::memory_order_acquire) == g) {
                if (poisoned_.load(std::memory_order_acquire)) {
                    throw ShmCommPoisoned();
                }
                std::this_thread::yield();
            }
        }
        if (poisoned_.load(std::memory_order_acquire)) {
            throw ShmCommPoisoned();
        }
    }

    int n_;
    std::vector<Slot> slots_;
    std::atomic<int> arrived_{0};
    std::atomic<unsigned> gen_{0};
    std::atomic<bool> poisoned_{false};
};

} // namespace monoprop::mpi
