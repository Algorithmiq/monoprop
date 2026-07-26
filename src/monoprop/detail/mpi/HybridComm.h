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
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/ShardBarrier.h"

// Composes R MPI ranks x S in-process shards into one flat P=R*S SPMD world so the engine needs zero
// changes. Global id is RANK-MAJOR (g = mpi_rank*S + shard), keeping each rank's shards contiguous in
// ascending-global order — the contract Resolve.h's positional pairing relies on. Only shard-0 masters
// call MPI (bracketed by intra-rank barriers ⇒ requires MPI_THREAD_SERIALIZED); ascending-order local
// sums + order-preserving MPI ⇒ bit-identical, repeatable results for fixed (R, S).

namespace monoprop::mpi {

class HybridComm {
public:
    // n_local_shards = S, identical on every rank (the facade ctor checks that before constructing).
    HybridComm(MPI_Comm parent, int n_local_shards)
        : parent_(parent),
          s_(n_local_shards),
          slots_(static_cast<size_t>(n_local_shards)),
          barrier_(n_local_shards) {
        MPI_Comm_size(parent_, &r_);
        MPI_Comm_rank(parent_, &mpi_rank_);
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        if (provided < MPI_THREAD_SERIALIZED) {
            throw std::runtime_error("HybridComm requires MPI_THREAD_SERIALIZED (shard-0 masters call "
                                     "MPI while peers are parked); provided level is lower. Ensure "
                                     "mpi::init / mpi4py requests SERIALIZED or MULTIPLE.");
        }
        // Size all (R,S)-fixed scratch once here so per-call paths never allocate (the staging buffers
        // and red_vec_ grow to a high-water mark on demand instead).
        const size_t rss = static_cast<size_t>(r_) * static_cast<size_t>(s_) * static_cast<size_t>(s_);
        counts_send_.resize(rss);
        counts_recv_.resize(rss);
        mpi_send_counts_.resize(static_cast<size_t>(r_));
        mpi_recv_counts_.resize(static_cast<size_t>(r_));
        mpi_send_displs_.resize(static_cast<size_t>(r_));
        mpi_recv_displs_.resize(static_cast<size_t>(r_));
        pack_off_.resize(rss);
        scatter_off_.resize(rss);
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    auto size() const -> int { return r_ * s_; }
    auto global_rank(int local_shard) const -> int { return mpi_rank_ * s_ + local_shard; }

    // recv_counts[g] = amount global partition g sends to this partition. 2 barriers + one S*S-int MPI_Alltoall.
    auto alltoall_counts(int local_shard, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        const size_t u = static_cast<size_t>(local_shard);
        slots_[u].counts = send_counts;
        sync();
        if (local_shard == 0) {
            // Pack the S*S count matrix per dest rank, dest-shard-major (t) then source-shard-minor (su);
            // the loop writes every element and MPI_Alltoall fills counts_recv_ fully, so neither is pre-zeroed.
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
        // No trailing barrier: past the last sync only counts_recv_ is read, and shard 0 cannot rewrite
        // it until a future call's second barrier — unreachable until every extractor here has arrived.
        // send_counts was consumed before the last sync, so peers may free/reuse it on return.
    }

    // Flat variable all-to-all over caller-owned buffers (counts/displs in ELEMENTS, `elem` = element
    // bytes, `dt` = MPI datatype). recv_counts must already hold the transpose — same contract as MPI_Alltoallv.
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
        sync(); // B1

        // B2: shard 0 sizes/reallocates staging; MUST finish before any shard packs into stage_send_,
        // hence a barrier separate from packing.
        if (local_shard == 0) {
            size_staging_(elem);
        }
        sync(); // B2

        // B3: each shard packs its own cross-rank blocks into stage_send_ (disjoint writes, no coordination).
        pack_send_(local_shard, elem);
        sync(); // B3

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
        sync(); // B4

        // Scatter each global source's contiguous run out of stage_recv_ to recv_displs[g]. All legs
        // (incl. self-rank) go through staging uniformly; block starts come from scatter_off_, so no
        // peer slot is read past B4.
        char *dst = static_cast<char *>(recv);
        const int t = local_shard;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data() + scatter_off_[block_idx_(a, t, su)] * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
        // No trailing barrier: past B4 a shard reads only stage_recv_/scatter_off_/its own buffers,
        // all rewritten only inside a future call's shard-0 size_staging_ (after that call's B1).
    }

    // Fused count-resolve + payload alltoallv: folds the standalone count exchange into this verb's
    // B1→B2 window (4 syncs instead of 6). recv_counts / recv_displs and `recv` (resized) are OUTPUTS.
    // Bit-identical to alltoall_counts + alltoallv.
    template <class T>
    auto alltoallv_resolve(int local_shard,
                           const T *send,
                           const int *send_counts /*[P]*/,
                           const int *send_displs /*[P]*/,
                           std::vector<T> &recv,
                           int *recv_counts /*[P]*/,
                           int *recv_displs /*[P]*/,
                           size_t elem,
                           MPI_Datatype dt) -> void {
        const size_t u = static_cast<size_t>(local_shard);
        Slot &me = slots_[u];
        me.ptr = send;
        me.send_counts = send_counts;
        me.send_displs = send_displs;
        // recv_counts is an OUTPUT here — deliberately NOT published; the count Alltoall resolves it.
        sync(); // B1

        if (local_shard == 0) {
            // Pack the S*S count matrix (dest-shard-major t, source-shard-minor su) and MPI_Alltoall it.
            for (int b = 0; b < r_; ++b) {
                for (int t = 0; t < s_; ++t) {
                    for (int su = 0; su < s_; ++su) {
                        counts_send_[block_idx_(b, t, su)] = slots_[static_cast<size_t>(su)].send_counts[b * s_ + t];
                    }
                }
            }
            MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
            // Size staging from counts_recv_ (transpose layout: recv of shard t from (rank, su) at rank*S*S + t*S +
            // su).
            size_staging_impl_(elem, [this](int t, int rank, int su) -> int {
                return counts_recv_[(static_cast<size_t>(rank) * static_cast<size_t>(s_) * static_cast<size_t>(s_))
                                    + (static_cast<size_t>(t) * static_cast<size_t>(s_)) + static_cast<size_t>(su)];
            });
        }
        sync(); // B2

        const int t = local_shard;
        size_t total = 0;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int c =
                    counts_recv_[(static_cast<size_t>(a) * static_cast<size_t>(s_) * static_cast<size_t>(s_))
                                 + (static_cast<size_t>(t) * static_cast<size_t>(s_)) + static_cast<size_t>(su)];
                recv_counts[g] = c;
                recv_displs[g] = static_cast<int>(total);
                total += static_cast<size_t>(c);
            }
        }
        recv.resize(total);

        pack_send_(local_shard, elem);
        sync(); // B3

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
        sync(); // B4

        char *dst = reinterpret_cast<char *>(recv.data());
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data() + scatter_off_[block_idx_(a, t, su)] * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
        // No trailing barrier: same discipline as alltoallv (past B4 only stage_recv_/scatter_off_/own buffers read).
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
        // No trailing barrier: red_f64_/red_u64_ are rewritten only inside a future verb's barriered window.
        return out;
    }

    // In-place element-wise allreduce-sum across the flat P-world, slice-partitioned across shards in
    // ascending order (bit-identical to a sequential sum). red_vec_ sizing gets its own barrier phase so
    // no shard writes into a buffer that may still reallocate.
    auto allreduce_sum_inplace(int local_shard, double *values, size_t len) -> void {
        slots_[static_cast<size_t>(local_shard)].vec = values;
        sync(); // all inputs published
        if (local_shard == 0) {
            grow_(red_vec_, len);
        }
        sync(); // red_vec_ sized
        constexpr size_t kLine = 64 / sizeof(double);
        const size_t lines = (len + kLine - 1) / kLine;
        const size_t per = (lines + static_cast<size_t>(s_) - 1) / static_cast<size_t>(s_);
        const size_t lo = std::min(len, static_cast<size_t>(local_shard) * per * kLine);
        const size_t hi = std::min(len, lo + per * kLine);
        for (size_t k = lo; k < hi; ++k) {
            double acc = 0.0;
            for (int s = 0; s < s_; ++s) {
                acc += slots_[static_cast<size_t>(s)].vec[k];
            }
            red_vec_[k] = acc; // disjoint line-rounded slices: no two shards store to one line
        }
        sync(); // local reduction complete
        if (local_shard == 0) {
            MPI_Allreduce(MPI_IN_PLACE, red_vec_.data(), static_cast<int>(len), MPI_DOUBLE, MPI_SUM, parent_);
        }
        sync(); // global result in red_vec_
        std::memcpy(values, red_vec_.data(), len * sizeof(double));
        // No trailing barrier: red_vec_ is rewritten only inside a future verb's barriered phases.
    }

    auto poison() -> void { barrier_.poison(); }
    auto reset() -> void { barrier_.reset(); }

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

    // Flat index of the (rank, dest shard, source shard) block in the R*S*S offset/count tables.
    auto block_idx_(int b, int t, int u) const -> size_t {
        return (static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t)) * static_cast<size_t>(s_)
               + static_cast<size_t>(u);
    }

    // Grow-only (high-water-mark) sizing: never shrink, never zero what will be overwritten anyway.
    template <class V>
    static auto grow_(V &v, size_t need) -> void {
        if (v.size() < need) {
            v.resize(need);
        }
    }

    // Shard 0: aggregate the published count matrices into per-rank counts/displs, size staging, and
    // precompute the pack/scatter offset tables. Overflow-guarded throughout: both the S^2-block
    // per-rank sums and their prefix sums across the R ranks can pass INT_MAX.
    // Default recv-count source is shard t's published recv_counts; alltoallv_resolve passes an accessor
    // reading the just-computed counts_recv_ instead, so no shard need publish recv_counts.
    auto size_staging_(size_t elem) -> void {
        size_staging_impl_(elem, [this](int t, int rank, int su) -> int {
            return slots_[static_cast<size_t>(t)].recv_counts[rank * s_ + su];
        });
    }

    // recv_count(t, rank, su) yields the count shard t on this rank receives from (rank, source shard su).
    template <class RecvCountFn>
    auto size_staging_impl_(size_t elem, RecvCountFn recv_count) -> void {
        for (int b = 0; b < r_; ++b) {
            long long send_sum = 0;
            long long recv_sum = 0;
            for (int t = 0; t < s_; ++t) {
                for (int su = 0; su < s_; ++su) {
                    send_sum += slots_[static_cast<size_t>(su)].send_counts[b * s_ + t];
                    recv_sum += recv_count(t, b, su);
                }
            }
            mpi_send_counts_[static_cast<size_t>(b)] = checked_int_(send_sum);
            mpi_recv_counts_[static_cast<size_t>(b)] = checked_int_(recv_sum);
        }
        // Prefix sums accumulate WIDE and narrow through checked_int_: each per-rank count above fits in
        // an int, but their running total need not, and a plain int accumulator would be UB on overflow
        // before being cast to size_t to size the staging buffers (a wrapped positive value would
        // silently under-size them and MPI_Alltoallv would run past the end).
        long long send_running = 0;
        long long recv_running = 0;
        for (int b = 0; b < r_; ++b) {
            mpi_send_displs_[static_cast<size_t>(b)] = checked_int_(send_running);
            mpi_recv_displs_[static_cast<size_t>(b)] = checked_int_(recv_running);
            send_running += mpi_send_counts_[static_cast<size_t>(b)];
            recv_running += mpi_recv_counts_[static_cast<size_t>(b)];
        }
        const size_t total_send = static_cast<size_t>(checked_int_(send_running));
        const size_t total_recv = static_cast<size_t>(checked_int_(recv_running));
        // Grow-only, no zero-fill: pack_send_'s blocks tile [0, total_send) exactly and MPI_Alltoallv
        // fills every live byte of stage_recv_, so stale bytes past a prior high-water mark are never read.
        grow_(stage_send_, total_send * elem);
        grow_(stage_recv_, total_recv * elem);
        // Precompute each (rank b, dest shard t, source shard u) block start (ELEMENTS), dest-major/
        // source-minor to match the staged layout, so packing/scatter do O(1) lookups instead of
        // re-summing peer count matrices (O(R*S^3), and it kept peer-slot reads alive past B4).
        for (int b = 0; b < r_; ++b) {
            size_t cur = static_cast<size_t>(mpi_send_displs_[static_cast<size_t>(b)]);
            for (int t = 0; t < s_; ++t) {
                for (int u = 0; u < s_; ++u) {
                    pack_off_[block_idx_(b, t, u)] = cur;
                    cur += static_cast<size_t>(slots_[static_cast<size_t>(u)].send_counts[b * s_ + t]);
                }
            }
        }
        for (int a = 0; a < r_; ++a) {
            size_t cur = static_cast<size_t>(mpi_recv_displs_[static_cast<size_t>(a)]);
            for (int t = 0; t < s_; ++t) {
                for (int su = 0; su < s_; ++su) {
                    scatter_off_[block_idx_(a, t, su)] = cur;
                    cur += static_cast<size_t>(recv_count(t, a, su));
                }
            }
        }
    }

    // Pack local shard `u`'s cross-rank send blocks into stage_send_ at the block starts shard 0
    // precomputed in pack_off_ — disjoint writes, no coordination, no peer-slot reads.
    auto pack_send_(int local_shard, size_t elem) -> void {
        const int u = local_shard;
        const char *src = static_cast<const char *>(slots_[static_cast<size_t>(u)].ptr);
        const int *my_send_counts = slots_[static_cast<size_t>(u)].send_counts;
        const int *my_send_displs = slots_[static_cast<size_t>(u)].send_displs;
        for (int b = 0; b < r_; ++b) {
            for (int t = 0; t < s_; ++t) {
                const int cnt = my_send_counts[b * s_ + t];
                if (cnt != 0) {
                    std::memcpy(stage_send_.data() + pack_off_[block_idx_(b, t, u)] * elem,
                                src + static_cast<size_t>(my_send_displs[b * s_ + t]) * elem,
                                static_cast<size_t>(cnt) * elem);
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

    // Intra-rank barrier between the s_ shards (shard 0 brackets its MPI call between two). See ShardBarrier.
    auto sync() -> void { barrier_.sync(); }

    MPI_Comm parent_;
    int s_;
    int r_ = 1;
    int mpi_rank_ = 0;
    std::vector<Slot> slots_;

    // Shard-0-managed shared state (written by shard 0, read by all between barriers).
    std::vector<int> counts_send_, counts_recv_; // S*S per rank, the counts alltoall
    std::vector<int> mpi_send_counts_, mpi_send_displs_, mpi_recv_counts_, mpi_recv_displs_; // [R]
    std::vector<size_t> pack_off_, scatter_off_;     // [R*S*S] block starts (elements) in the staging buffers
    std::vector<std::byte> stage_send_, stage_recv_; // aggregated MPI payload staging, HWM-sized
    double red_f64_ = 0.0;
    uint64_t red_u64_ = 0;
    std::vector<double> red_vec_;

    ShardBarrier barrier_;
};

} // namespace monoprop::mpi
