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
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/ShmComm.h" // reuse the ShmCommPoisoned exception type

// Hybrid transport: compose R MPI ranks x S in-process shards into ONE flat SPMD world of P = R*S
// partitions, so the unchanged engine — which only ever asks its comm for size()/rank() and issues
// alltoall/alltoallv/allreduce — sees a single P-partition world and needs zero changes. Global
// partition id is RANK-MAJOR: g = mpi_rank*S + local_shard. Rank-major keeps every rank's S shards
// CONTIGUOUS in ascending-global-source order, so cross-rank aggregation reproduces the
// per-source-contiguous ascending-global-order contract that Resolve.h's positional pairing relies on
// with one memcpy per (source rank, receiving shard) — no interleaving.
//
// Only the local shard-0 master ever calls MPI (bracketed by the intra-rank barriers below, so no two
// threads are ever in MPI at once on a rank). That needs MPI_THREAD_SERIALIZED (see mpi::init); the
// ctor asserts the provided level. Every verb is: publish-to-slots -> barrier -> [shard 0 does the one
// MPI collective] -> barrier -> read-results. Determinism: local partial sums run in ascending shard
// order and MPI_Allreduce/Alltoallv are order-preserving, so results are bit-identical across ranks
// and repeatable for a fixed (R, S) — the same standard the pure-MPI path already meets.

namespace monoprop::mpi {

class HybridComm {
public:
    // parent = the R-rank MPI communicator; n_local_shards = S (same on every rank — the facade ctor
    // allreduces S for a min==max consistency check before constructing this).
    HybridComm(MPI_Comm parent, int n_local_shards)
        : parent_(parent), s_(n_local_shards), slots_(static_cast<size_t>(n_local_shards)) {
        MPI_Comm_size(parent_, &r_);
        MPI_Comm_rank(parent_, &mpi_rank_);
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        if (provided < MPI_THREAD_SERIALIZED) {
            throw std::runtime_error("HybridComm requires MPI_THREAD_SERIALIZED (shard-0 masters call "
                                     "MPI while peers are parked); provided level is lower. Ensure "
                                     "mpi::init / mpi4py requests SERIALIZED or MULTIPLE.");
        }
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    auto size() const -> int { return r_ * s_; }                                      // P = R*S
    auto global_rank(int local_shard) const -> int { return mpi_rank_ * s_ + local_shard; } // rank-major

    // recv_counts[g] = amount global partition g sends to this (local_shard) partition, in the flat
    // P-world. 3 barriers + one MPI_Alltoall of S*S ints per rank pair.
    auto alltoall_counts(int local_shard, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        const size_t u = static_cast<size_t>(local_shard);
        slots_[u].counts = send_counts;
        sync();
        if (local_shard == 0) {
            // Pack per-dest-rank blocks of S*S ints, dest-shard-major (t) then source-shard-minor (u):
            //   scratch[b][t][u] = (shard u on this rank) -> (shard t on rank b).
            counts_send_.assign(static_cast<size_t>(r_) * static_cast<size_t>(s_) * static_cast<size_t>(s_), 0);
            counts_recv_.assign(counts_send_.size(), 0);
            for (int b = 0; b < r_; ++b) {
                for (int t = 0; t < s_; ++t) {
                    for (int su = 0; su < s_; ++su) {
                        const size_t idx = ((static_cast<size_t>(b) * static_cast<size_t>(s_)) + static_cast<size_t>(t))
                                               * static_cast<size_t>(s_)
                                           + static_cast<size_t>(su);
                        counts_send_[idx] = slots_[static_cast<size_t>(su)].counts[b * s_ + t];
                    }
                }
            }
            MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
        }
        sync();
        // Shard t extracts its row: recv from (rank a, shard su) is contiguous per source rank a.
        const int t = local_shard;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const size_t idx = (static_cast<size_t>(a) * static_cast<size_t>(s_) * static_cast<size_t>(s_))
                                   + (static_cast<size_t>(t) * static_cast<size_t>(s_)) + static_cast<size_t>(su);
                recv_counts[a * s_ + su] = counts_recv_[idx];
            }
        }
        sync();
    }

    // Flat variable all-to-all over caller-owned buffers (counts/displs in ELEMENTS; `elem` = element
    // size in bytes; `dt` = the matching MPI datatype). recv_counts must already hold the transpose
    // (from alltoall_counts or a known transpose) — the same contract as MPI_Alltoallv / ShmComm.
    auto alltoallv(int local_shard,
                   const void *send,
                   const int *send_counts /*[P]*/,
                   const int *send_displs /*[P]*/,
                   void *recv,
                   const int *recv_counts /*[P]*/,
                   const int *recv_displs /*[P]*/,
                   size_t elem,
                   MPI_Datatype dt) -> void {
        const size_t u = static_cast<size_t>(local_shard);
        Slot &me = slots_[u];
        me.ptr = send;
        me.send_counts = send_counts;
        me.send_displs = send_displs;
        me.recv_counts = recv_counts;
        sync(); // B1: every slot published

        // B2: shard 0 sizes the shared staging buffers from the published count matrix. This MUST
        // complete (with its reallocation) before any shard packs into stage_send_ — hence its own
        // barrier, separate from packing.
        if (local_shard == 0) {
            size_staging_(elem);
        }
        sync(); // B2: stage_send_/stage_recv_ allocated at final size, mpi counts/displs ready

        // B3: every shard packs its own cross-rank blocks into the now-stable stage_send_ (disjoint
        // writes — each shard owns distinct source-shard sub-blocks, so no coordination needed).
        pack_send_(local_shard, elem);
        sync(); // B3: stage_send_ fully packed

        // B4: shard 0 runs the single MPI_Alltoallv while peers park at the barrier.
        if (local_shard == 0) {
            MPI_Alltoallv(stage_send_.data(),
                          mpi_send_counts_.data(),
                          mpi_send_displs_.data(),
                          dt,
                          stage_recv_.data(),
                          mpi_recv_counts_.data(),
                          mpi_recv_displs_.data(),
                          dt,
                          parent_);
        }
        sync(); // B4: stage_recv_ filled

        // Scatter: for every global source g (ascending), copy its contiguous run out of stage_recv_
        // into the caller's recv buffer at recv_displs[g]. All legs — including this rank's own shards
        // (MPI does the self-rank block as a local copy) — go through the staged buffer uniformly, so
        // there is exactly one offset scheme. Within source rank a's block, the data for this shard t
        // begins after all lower dest-shards' data (dest-shard-major packing on the sender), and the
        // sources su run ascending within.
        char *dst = static_cast<char *>(recv);
        const int t = local_shard;
        for (int a = 0; a < r_; ++a) {
            size_t within = 0;
            for (int tp = 0; tp < t; ++tp) {
                for (int su = 0; su < s_; ++su) {
                    within += static_cast<size_t>(recv_counts_of_shard_(tp, a, su));
                }
            }
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data()
                                    + (static_cast<size_t>(mpi_recv_displs_[static_cast<size_t>(a)]) + within) * elem,
                                static_cast<size_t>(cnt) * elem);
                }
                within += static_cast<size_t>(cnt);
            }
        }
        sync();
    }

    template <class T>
    auto allreduce_sum(int local_shard, T local_val) -> T {
        Slot &me = slots_[static_cast<size_t>(local_shard)];
        if constexpr (std::is_floating_point_v<T>) {
            me.f64 = static_cast<double>(local_val);
        }
        else {
            me.u64 = static_cast<uint64_t>(local_val);
        }
        sync();
        if (local_shard == 0) {
            if constexpr (std::is_floating_point_v<T>) {
                double local = 0.0;
                for (int s = 0; s < s_; ++s) {
                    local += slots_[static_cast<size_t>(s)].f64;
                }
                MPI_Allreduce(&local, &red_f64_, 1, MPI_DOUBLE, MPI_SUM, parent_);
            }
            else {
                uint64_t local = 0;
                for (int s = 0; s < s_; ++s) {
                    local += slots_[static_cast<size_t>(s)].u64;
                }
                MPI_Allreduce(&local, &red_u64_, 1, MPI_UINT64_T, MPI_SUM, parent_);
            }
        }
        sync();
        T out{};
        if constexpr (std::is_floating_point_v<T>) {
            out = static_cast<T>(red_f64_);
        }
        else {
            out = static_cast<T>(red_u64_);
        }
        sync();
        return out;
    }

    auto allreduce_sum_inplace(int local_shard, double *values, size_t len) -> void {
        slots_[static_cast<size_t>(local_shard)].vec = values;
        sync();
        if (local_shard == 0) {
            red_vec_.assign(len, 0.0);
            for (int s = 0; s < s_; ++s) {
                const double *vs = slots_[static_cast<size_t>(s)].vec;
                for (size_t k = 0; k < len; ++k) {
                    red_vec_[k] += vs[k];
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, red_vec_.data(), static_cast<int>(len), MPI_DOUBLE, MPI_SUM, parent_);
        }
        sync();
        std::memcpy(values, red_vec_.data(), len * sizeof(double));
        sync();
    }

    auto poison() -> void { poisoned_.store(true, std::memory_order_release); }
    auto reset() -> void {
        poisoned_.store(false, std::memory_order_relaxed);
        arrived_.store(0, std::memory_order_relaxed);
    }

private:
    struct alignas(64) Slot {
        const void *ptr = nullptr;
        const int *counts = nullptr;
        const int *send_counts = nullptr;
        const int *send_displs = nullptr;
        const int *recv_counts = nullptr;
        const double *vec = nullptr;
        double f64 = 0.0;
        uint64_t u64 = 0;
    };

    // recv_counts[g] published by shard tp on this rank (its view of what global partition g sends it).
    auto recv_counts_of_shard_(int tp, int a, int su) const -> int {
        return slots_[static_cast<size_t>(tp)].recv_counts[a * s_ + su];
    }

    // Shard 0: aggregate the S*P published send/recv count matrices into per-MPI-rank counts/displs and
    // size the staging buffers. (overflow-guarded: aggregated counts sum S^2 shard-pair blocks and can
    // exceed INT_MAX sooner than any single block.) Packing is a separate, barriered phase (all shards).
    auto size_staging_(size_t elem) -> void {
        mpi_send_counts_.assign(static_cast<size_t>(r_), 0);
        mpi_recv_counts_.assign(static_cast<size_t>(r_), 0);
        mpi_send_displs_.assign(static_cast<size_t>(r_), 0);
        mpi_recv_displs_.assign(static_cast<size_t>(r_), 0);
        for (int b = 0; b < r_; ++b) {
            long long send_sum = 0;
            long long recv_sum = 0;
            for (int t = 0; t < s_; ++t) {
                for (int su = 0; su < s_; ++su) {
                    send_sum += slots_[static_cast<size_t>(su)].send_counts[b * s_ + t];
                    recv_sum += slots_[static_cast<size_t>(t)].recv_counts[b * s_ + su];
                }
            }
            mpi_send_counts_[static_cast<size_t>(b)] = checked_int_(send_sum);
            mpi_recv_counts_[static_cast<size_t>(b)] = checked_int_(recv_sum);
        }
        for (int b = 1; b < r_; ++b) {
            mpi_send_displs_[static_cast<size_t>(b)] =
                mpi_send_displs_[static_cast<size_t>(b - 1)] + mpi_send_counts_[static_cast<size_t>(b - 1)];
            mpi_recv_displs_[static_cast<size_t>(b)] =
                mpi_recv_displs_[static_cast<size_t>(b - 1)] + mpi_recv_counts_[static_cast<size_t>(b - 1)];
        }
        const size_t total_send =
            static_cast<size_t>(mpi_send_displs_[static_cast<size_t>(r_ - 1)] + mpi_send_counts_[static_cast<size_t>(r_ - 1)]);
        const size_t total_recv =
            static_cast<size_t>(mpi_recv_displs_[static_cast<size_t>(r_ - 1)] + mpi_recv_counts_[static_cast<size_t>(r_ - 1)]);
        stage_send_.assign(total_send * elem, std::byte{0});
        stage_recv_.assign(total_recv * elem, std::byte{0});
    }

    // Pack local shard `u`'s cross-rank send blocks into stage_send_. Destination offset for the block
    // (u -> rank b, shard t) is: mpi_send_displs_[b] + (sum over t'<t, all u' of cnt(u'-> (b,t')))
    //                                                 + (sum over u'<u of cnt(u' -> (b,t))).
    // Every shard computes this independently from the published matrix — disjoint writes, no coordination.
    auto pack_send_(int local_shard, size_t elem) -> void {
        const int u = local_shard;
        const char *src = static_cast<const char *>(slots_[static_cast<size_t>(u)].ptr);
        const int *my_send_counts = slots_[static_cast<size_t>(u)].send_counts;
        const int *my_send_displs = slots_[static_cast<size_t>(u)].send_displs;
        for (int b = 0; b < r_; ++b) {
            size_t off = static_cast<size_t>(mpi_send_displs_[static_cast<size_t>(b)]);
            for (int t = 0; t < s_; ++t) {
                // offset contribution from lower dest-shards t' (all source shards u')
                // is already accumulated in `off` as we iterate t ascending.
                size_t within = 0;
                for (int up = 0; up < u; ++up) {
                    within += static_cast<size_t>(slots_[static_cast<size_t>(up)].send_counts[b * s_ + t]);
                }
                const int cnt = my_send_counts[b * s_ + t];
                if (cnt != 0) {
                    std::memcpy(stage_send_.data() + (off + within) * elem,
                                src + static_cast<size_t>(my_send_displs[b * s_ + t]) * elem,
                                static_cast<size_t>(cnt) * elem);
                }
                // advance `off` past this whole dest-shard-t block (all source shards)
                for (int up = 0; up < s_; ++up) {
                    off += static_cast<size_t>(slots_[static_cast<size_t>(up)].send_counts[b * s_ + t]);
                }
            }
        }
    }

    static auto checked_int_(long long v) -> int {
        if (v < 0 || v > static_cast<long long>(2147483647)) {
            throw std::runtime_error("HybridComm: aggregated per-rank count overflows int (message too large)");
        }
        return static_cast<int>(v);
    }

    // Sense-reversing S-participant barrier with poison escape (same design as ShmComm::sync).
    auto sync() -> void {
        const unsigned g = gen_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == s_) {
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

    MPI_Comm parent_;
    int s_;
    int r_ = 1;
    int mpi_rank_ = 0;
    std::vector<Slot> slots_;

    // Shard-0-managed shared state (written by shard 0, read by all between barriers).
    std::vector<int> counts_send_, counts_recv_;                           // S*S per rank, the counts alltoall
    std::vector<int> mpi_send_counts_, mpi_send_displs_, mpi_recv_counts_, mpi_recv_displs_; // [R]
    std::vector<std::byte> stage_send_, stage_recv_;                       // aggregated MPI payload staging
    double red_f64_ = 0.0;
    uint64_t red_u64_ = 0;
    std::vector<double> red_vec_;

    std::atomic<int> arrived_{0};
    std::atomic<unsigned> gen_{0};
    std::atomic<bool> poisoned_{false};
};

} // namespace monoprop::mpi
