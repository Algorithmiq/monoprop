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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <print>
#include <stdexcept>
#include <thread>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/CommProfile.h"
#include "monoprop/detail/mpi/PartitionBarrier.h"

// Composes R MPI ranks x S in-process partitions into one flat P=R*S SPMD world. Global id is rank-major
// (g = mpi_rank*S + partition), keeping each rank's partitions contiguous in ascending-global order — the
// contract Resolve.h's positional pairing relies on. Only partition-0 masters call MPI (bracketed by
// intra-rank barriers ⇒ requires MPI_THREAD_SERIALIZED); ascending-order local sums + order-preserving
// MPI ⇒ bit-identical, repeatable results for fixed (R, S).

namespace monoprop::mpi {

class MpiThreadLevelUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class HybridComm {
public:
    // n_local_partitions = S, identical on every rank (the facade ctor checks that before constructing).
    // partition_l3_domains (optional, one entry per partition) makes the barrier two-level; see
    // PartitionBarrier.
    HybridComm(MPI_Comm parent, int n_local_partitions, const std::vector<int> &partition_l3_domains = {})
        : parent_(parent),
          s_(n_local_partitions),
          slots_(static_cast<size_t>(n_local_partitions)),
          barrier_(n_local_partitions, partition_l3_domains) {
        MPI_Comm_size(parent_, &r_);
        MPI_Comm_rank(parent_, &mpi_rank_);
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        if (provided < MPI_THREAD_SERIALIZED) {
            throw MpiThreadLevelUnsupported("HybridComm requires MPI_THREAD_SERIALIZED (partition-0 masters call "
                                     "MPI while peers are parked); provided level is lower. Ensure "
                                     "mpi::init / mpi4py requests SERIALIZED or MULTIPLE.");
        }
        // Size all (R,S)-fixed scratch once so per-call paths never allocate; staging grows on demand.
        const size_t rss = static_cast<size_t>(r_) * static_cast<size_t>(s_) * static_cast<size_t>(s_);
        counts_send_.resize(rss);
        counts_recv_.resize(rss);
        mpi_send_counts_.resize(static_cast<size_t>(r_));
        mpi_recv_counts_.resize(static_cast<size_t>(r_));
        mpi_send_displs_.resize(static_cast<size_t>(r_));
        mpi_recv_displs_.resize(static_cast<size_t>(r_));
        pack_off_.resize(rss);
        scatter_off_.resize(rss);
        const size_t rs = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        col_send_.resize(rs);
        col_recv_.resize(rs);
        tpre_send_.resize(rs);
        tpre_recv_.resize(rs);
        rev_send_counts_.resize(static_cast<size_t>(r_));
        rev_send_displs_.resize(static_cast<size_t>(r_));
        rev_recv_counts_.resize(static_cast<size_t>(r_));
        rev_recv_displs_.resize(static_cast<size_t>(r_));
        mask_words_ = pad_to_line_(static_cast<size_t>((r_ + 63) / 64));
        send_mask_.assign(static_cast<size_t>(s_) * mask_words_, 0);
        run_stride_ = pad_to_line_(static_cast<size_t>(r_));
        run_scratch_.assign(static_cast<size_t>(s_) * run_stride_, 0);
        if (config::get().comm_profile) {
            prof_ = std::make_unique<CommProfile>(s_, mpi_rank_);
            prof_->barrier_groups = barrier_.group_count();
        }
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    // Profiling is opt-in, so the common case destroys nothing and prints nothing.
    ~HybridComm() {
        if (prof_ != nullptr) {
            prof_->dump();
        }
    }

    auto size() const -> int { return r_ * s_; }
    auto global_rank(int local_partition) const -> int { return mpi_rank_ * s_ + local_partition; }

    // Once partition 0 -- this rank's only participant on parent_ -- is inside a collective, the peer ranks
    // are committed: their partition-0 threads enter theirs and block inside MPI with no timeout. So a
    // rank-local failure here must MPI_Abort with the underlying error rather than throw, which would
    // hang the job. Single-rank partitioned runs use ShmComm, not HybridComm, and keep their exceptions.
    template <class Body>
    auto guard_partition0_(int local_partition, const char *verb, Body &&body) -> decltype(body()) {
        if (local_partition != 0) {
            return body();
        }
        try {
            return body();
        }
        catch (const std::exception &e) {
            abort_rank_(verb, e.what());
        }
        catch (...) {
            abort_rank_(verb, "unknown error");
        }
    }

    auto alltoall_counts(int local_partition, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        guard_partition0_(local_partition, "alltoall_counts", [&] {
            alltoall_counts_impl_(local_partition, send_counts, recv_counts);
        });
    }

    // See AlltoallvArgs for the send-buffer lifetime and the element-vs-byte convention; `dt` is the MPI
    // datatype whose extent is args.elem, and it stays a separate argument because the bundle is shared
    // with the non-MPI-capable transport. The bundle is unpacked here rather than threaded through the
    // implementation, which keeps the per-argument signature the barrier reasoning is written against.
    auto alltoallv(int local_partition, const AlltoallvArgs &args, MPI_Datatype dt) -> void {
        guard_partition0_(local_partition, "alltoallv", [&] {
            alltoallv_impl_(local_partition,
                            args.send,
                            args.send_counts,
                            args.send_displs,
                            args.recv,
                            args.recv_counts,
                            args.recv_displs,
                            args.elem,
                            dt);
        });
    }

    // See AlltoallvResolveArgs: the recv side is an output and args.recv is resized here. Element bytes
    // are sizeof(T) by construction, so unlike alltoallv they are derived rather than carried.
    template <class T>
    auto alltoallv_resolve(int local_partition, const AlltoallvResolveArgs<T> &args, MPI_Datatype dt) -> void {
        // `args` by reference, not by value: the impl resizes args.recv and then writes through it.
        guard_partition0_(local_partition, "alltoallv_resolve", [&] {
            alltoallv_resolve_impl_<T>(local_partition,
                                       args.send,
                                       args.send_counts,
                                       args.send_displs,
                                       args.recv,
                                       args.recv_counts,
                                       args.recv_displs,
                                       sizeof(T),
                                       dt);
        });
    }

    // Payload-only reverse of the immediately preceding alltoallv_resolve: same legs, opposite
    // direction, one element back per element received. See alltoallv_reverse_impl_ for the contract.
    auto alltoallv_reverse(int local_partition,
                           const void *send,
                           const int *send_counts /*[P]*/,
                           const int *send_displs /*[P]*/,
                           void *recv,
                           const int *recv_counts /*[P]*/,
                           const int *recv_displs /*[P]*/,
                           size_t elem,
                           MPI_Datatype dt,
                           int forward_stride) -> void {
        guard_partition0_(local_partition, "alltoallv_reverse", [&] {
            alltoallv_reverse_impl_(local_partition,
                                    send,
                                    send_counts,
                                    send_displs,
                                    recv,
                                    recv_counts,
                                    recv_displs,
                                    elem,
                                    dt,
                                    forward_stride);
        });
    }

    template <class T>
    auto allreduce_sum(int local_partition, T local_val) -> T {
        return guard_partition0_(local_partition, "allreduce_sum", [&] {
            return allreduce_sum_impl_<T>(local_partition, local_val);
        });
    }

    auto allreduce_sum_inplace(int local_partition, double *values, size_t len) -> void {
        guard_partition0_(local_partition, "allreduce_sum_inplace", [&] {
            allreduce_sum_inplace_impl_(local_partition, values, len);
        });
    }

    // recv_counts[g] = amount global partition g sends to this partition. 2 barriers + one S*S-int
    // MPI_Alltoall; the count matrix is filled row-per-partition before the first barrier.
    auto alltoall_counts_impl_(int local_partition, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        count_verb(local_partition);
        if (local_partition == 0) {
            ++tables_gen_;
        }
        // Each partition writes its OWN source row of the count matrix before the barrier that publishes
        // it; the rows are disjoint, so this costs no extra barrier. Every element is written here and
        // MPI_Alltoall fills counts_recv_ fully, so neither buffer is pre-zeroed.
        {
            ScopedNs timer{par_table_ns(local_partition)};
            fill_count_row_(local_partition, send_counts);
        }
        sync(local_partition);
        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
            MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
        }
        sync(local_partition);
        // Partition t extracts its column: what (rank a, partition su) sends to it.
        const int t = local_partition;
        ScopedNs timer{par_table_ns(local_partition)};
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                recv_counts[a * s_ + su] = counts_recv_[cnt_idx_(a, su, t)];
            }
        }
        // No trailing barrier: past the last sync only counts_recv_ is read, and partition 0 cannot rewrite
        // it until a future call's second barrier — unreachable until every extractor here has arrived.
        // send_counts was consumed before the last sync, so peers may free/reuse it on return.
    }

    // Flat variable all-to-all over caller-owned buffers (counts/displs in elements, `elem` = element bytes).
    // recv_counts must already hold the transpose — same contract as MPI_Alltoallv.
    auto alltoallv_impl_(int local_partition,
                         const void *send,
                         const int *send_counts /*[P]*/,
                         const int *send_displs /*[P]*/,
                         void *recv,
                         const int *recv_counts /*[P]*/,
                         const int *recv_displs /*[P]*/,
                         size_t elem,
                         MPI_Datatype dt) -> void {
        const size_t u = static_cast<size_t>(local_partition);
        count_verb(local_partition);
        if (local_partition == 0) {
            ++tables_gen_;
        }
        Slot &me = slots_[u];
        me.ptr = send;
        me.send_counts = send_counts;
        me.send_displs = send_displs;
        me.recv_counts = recv_counts;
        // No count matrix on this path, so the veto row phase A reads is filled on its own.
        {
            ScopedNs timer{par_table_ns(local_partition)};
            fill_send_mask_(local_partition, send_counts);
        }
        sync(local_partition); // B1

        // Sizing is S-way parallel and carries its own two barriers (see size_staging_parallel_); on
        // return every base is visible and staging is grown, so packing into stage_send_ is safe.
        size_staging_parallel_(local_partition, elem);

        // Each partition packs its own cross-rank blocks into stage_send_ (disjoint writes).
        {
            ScopedNs timer{move_ns(local_partition)};
            pack_send_(local_partition, elem);
        }
        sync(local_partition); // B3

        // Partition 0 runs the single MPI_Alltoallv while peers park at the barrier.
        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
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
        sync(local_partition); // B4

        // Scatter each global source's contiguous run from stage_recv_ to recv_displs[g] (all legs, incl.
        // self-rank, go through staging). Block starts come from the offset tables, so no peer slot is
        // read after the last barrier.
        ScopedNs timer{move_ns(local_partition)};
        char *dst = static_cast<char *>(recv);
        const int t = local_partition;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data() + recv_block_off_(a, t, su) * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
        // No trailing barrier (see alltoall_counts_impl_): past B4 only stage_recv_/offset tables/own buffers.
    }

    // Fused count-resolve + payload alltoallv: folds the standalone count exchange into this verb's
    // B1→B2 window (4 syncs instead of 6). recv_counts / recv_displs and `recv` (resized) are outputs.
    // Bit-identical to alltoall_counts + alltoallv.
    template <class T>
    auto alltoallv_resolve_impl_(int local_partition,
                                 const T *send,
                                 const int *send_counts /*[P]*/,
                                 const int *send_displs /*[P]*/,
                                 std::vector<T> &recv,
                                 int *recv_counts /*[P]*/,
                                 int *recv_displs /*[P]*/,
                                 size_t elem,
                                 MPI_Datatype dt) -> void {
        const size_t u = static_cast<size_t>(local_partition);
        count_verb(local_partition);
        if (local_partition == 0) {
            ++tables_gen_;
        }
        Slot &me = slots_[u];
        me.ptr = send;
        me.send_counts = send_counts;
        me.send_displs = send_displs;
        // recv_counts is an output here — deliberately not published; the count Alltoall resolves it. The
        // count row comes from the local argument, so it rides the same barrier as the slot publish.
        {
            ScopedNs timer{par_table_ns(local_partition)};
            fill_count_row_(local_partition, send_counts);
        }
        sync(local_partition); // B1: slots and the whole send-count matrix published

        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
            MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
        }
        sync(local_partition); // B2: counts_recv_ visible to every partition

        // Size staging from counts_recv_: what partition t gets from (rank, su) is at cnt_idx_(rank, su, t).
        // Carries its own two barriers and its own timers — do not wrap it in one.
        size_staging_parallel_(local_partition, elem, [this](int t, int rank, int su) {
            return counts_recv_[cnt_idx_(rank, su, t)];
        });

        const int t = local_partition;
        {
            ScopedNs timer{par_table_ns(local_partition)};
            long long total = 0;
            for (int a = 0; a < r_; ++a) {
                for (int su = 0; su < s_; ++su) {
                    const int g = a * s_ + su;
                    const int c = counts_recv_[cnt_idx_(a, su, t)];
                    recv_counts[g] = c;
                    recv_displs[g] = checked_mpi_count(total);
                    total += c;
                }
            }
            recv.resize(static_cast<size_t>(checked_mpi_count(total)));
        }
        {
            ScopedNs timer{move_ns(local_partition)};
            pack_send_(local_partition, elem);
        }
        sync(local_partition); // B3

        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
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
        sync(local_partition); // B4

        ScopedNs scatter_timer{move_ns(local_partition)};
        char *dst = reinterpret_cast<char *>(recv.data());
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data() + recv_block_off_(a, t, su) * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
        if (local_partition == 0) {
            reverse_ready_gen_ = tables_gen_; // this round's tables may be reversed exactly once
        }
        // No trailing barrier: same discipline as alltoallv_impl_.
    }

    // Reverses the immediately preceding alltoallv_resolve: every leg carries one element back per RECORD
    // it delivered, so this round's geometry is that round's with the send/recv roles swapped and every
    // count and offset divided by `forward_stride`, the forward leg's elements per record (a query is >= 2
    // words, its answer one value). Every forward count is a multiple of the stride, so the division is
    // exact and no count exchange or sizing pass is needed -- that is the 3*R*S^2 table entries and one
    // barrier saved. Reusing the offsets undivided would still route correctly but would stage and
    // transmit `forward_stride` times the bytes.
    //
    // The reuse is legal because both ends see the same nesting: rank X ordered its query block to Y as
    // (dest partition of Y major, source partition of X minor), and Y's recv view of it has that nesting
    // and those counts, so Y answering in place produces the layout X expects back.
    //
    // CONTRACT: must be the next table-touching verb after an alltoallv_resolve on this comm, with
    // send_counts/recv_counts the transpose of that round's. Checked on partition 0 via tables_gen_.
    auto alltoallv_reverse_impl_(int local_partition,
                                 const void *send,
                                 const int *send_counts /*[P]*/,
                                 const int *send_displs /*[P]*/,
                                 void *recv,
                                 const int *recv_counts /*[P]*/,
                                 const int *recv_displs /*[P]*/,
                                 size_t elem,
                                 MPI_Datatype dt,
                                 int forward_stride) -> void {
        count_verb(local_partition);
        if (local_partition == 0 && tables_gen_ != reverse_ready_gen_) {
            throw std::runtime_error("HybridComm::alltoallv_reverse must directly follow the "
                                     "alltoallv_resolve whose layout it reverses; another collective "
                                     "has overwritten the offset tables since.");
        }
        sync(local_partition); // B1: the contract check is global before anyone reuses a table

        if (local_partition == 0) {
            ScopedNs timer{p0_table_ns(local_partition)};
            // Per-rank counts/displs are the forward round's, roles swapped and scaled down by the
            // stride. Kept in their own scratch so the forward tables stay intact for the scatter below.
            long long send_total = 0;
            long long recv_total = 0;
            for (int b = 0; b < r_; ++b) {
                const size_t i = static_cast<size_t>(b);
                rev_send_counts_[i] = mpi_recv_counts_[i] / forward_stride;
                rev_recv_counts_[i] = mpi_send_counts_[i] / forward_stride;
                rev_send_displs_[i] = checked_mpi_count(send_total);
                rev_recv_displs_[i] = checked_mpi_count(recv_total);
                send_total += rev_send_counts_[i];
                recv_total += rev_recv_counts_[i];
            }
            grow_(stage_send_, static_cast<size_t>(send_total) * elem);
            grow_(stage_recv_, static_cast<size_t>(recv_total) * elem);
            reverse_ready_gen_ = ~0ULL; // one reverse per resolve
        }
        sync(local_partition); // B2: staging sized

        // Pack into the query round's RECV geometry: partition t answers (rank a, partition su) at the
        // very offset that query block occupied.
        {
            ScopedNs timer{move_ns(local_partition)};
            const int t = local_partition;
            const char *src = static_cast<const char *>(send);
            for (int a = 0; a < r_; ++a) {
                for (int su = 0; su < s_; ++su) {
                    const int g = a * s_ + su;
                    const int cnt = send_counts[g];
                    if (cnt != 0) {
                        const size_t off = recv_block_off_(a, t, su) / static_cast<size_t>(forward_stride);
                        std::memcpy(stage_send_.data() + off * elem,
                                    src + static_cast<size_t>(send_displs[g]) * elem,
                                    static_cast<size_t>(cnt) * elem);
                    }
                }
            }
        }
        sync(local_partition); // B3

        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
            MPI_Alltoallv(stage_send_.data(),
                          rev_send_counts_.data(), // send side = the query round's recv side / stride
                          rev_send_displs_.data(),
                          dt,
                          stage_recv_.data(),
                          rev_recv_counts_.data(), // recv side = the query round's send side / stride
                          rev_recv_displs_.data(),
                          dt,
                          parent_);
        }
        sync(local_partition); // B4

        // Scatter from the query round's SEND geometry: partition u collects the answers to the queries
        // it sent, block (rank b, dest partition t) sitting where it packed that query.
        ScopedNs timer{move_ns(local_partition)};
        const int u = local_partition;
        char *dst = static_cast<char *>(recv);
        for (int b = 0; b < r_; ++b) {
            const size_t base = static_cast<size_t>(b) * static_cast<size_t>(s_);
            for (int t = 0; t < s_; ++t) {
                const int g = b * s_ + t;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    const size_t off = (tpre_send_[base + static_cast<size_t>(t)] + pack_off_[block_idx_(b, t, u)])
                                       / static_cast<size_t>(forward_stride);
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data() + off * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
    }

    template <class T>
    auto allreduce_sum_impl_(int local_partition, T local_val) -> T {
        count_verb(local_partition);
        Slot &me = slots_[static_cast<size_t>(local_partition)];
        if constexpr (std::is_floating_point_v<T>) {
            me.f64 = static_cast<double>(local_val);
        }
        else {
            me.u64 = static_cast<uint64_t>(local_val);
        }
        sync(local_partition);
        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
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
        sync(local_partition);
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

    // In-place element-wise allreduce-sum across the flat P-world, slice-partitioned across partitions in
    // ascending order (bit-identical to a sequential sum).
    auto allreduce_sum_inplace_impl_(int local_partition, double *values, size_t len) -> void {
        count_verb(local_partition);
        slots_[static_cast<size_t>(local_partition)].vec = values;
        sync(local_partition); // all inputs published
        if (local_partition == 0) {
            grow_(red_vec_, len);
        }
        sync(local_partition); // red_vec_ sized
        constexpr size_t kLine = 64 / sizeof(double);
        const size_t lines = (len + kLine - 1) / kLine;
        const size_t per = (lines + static_cast<size_t>(s_) - 1) / static_cast<size_t>(s_);
        const size_t lo = std::min(len, static_cast<size_t>(local_partition) * per * kLine);
        const size_t hi = std::min(len, lo + per * kLine);
        {
            ScopedNs timer{move_ns(local_partition)};
            for (size_t k = lo; k < hi; ++k) {
                double acc = 0.0;
                for (int s = 0; s < s_; ++s) {
                    acc += slots_[static_cast<size_t>(s)].vec[k];
                }
                red_vec_[k] = acc; // disjoint line-rounded slices: no two partitions store to one line
            }
        }
        sync(local_partition); // local reduction complete
        if (local_partition == 0) {
            ScopedNs timer{mpi_ns(local_partition)};
            MPI_Allreduce(MPI_IN_PLACE, red_vec_.data(), static_cast<int>(len), MPI_DOUBLE, MPI_SUM, parent_);
        }
        sync(local_partition); // global result in red_vec_
        std::memcpy(values, red_vec_.data(), len * sizeof(double));
        // No trailing barrier: red_vec_ is rewritten only inside a future verb's barriered phases.
    }

    auto poison() -> void { barrier_.poison(); }
    auto reset() -> void { barrier_.reset(); }

private:
    struct alignas(64) Slot {
        const void *ptr = nullptr;
        const int *send_counts = nullptr;
        const int *send_displs = nullptr;
        const int *recv_counts = nullptr;
        const double *vec = nullptr;
        double f64 = 0.0;
        uint64_t u64 = 0;
    };

    // Flat index of the (rank, dest partition, source partition) block in the R*S*S offset tables.
    auto block_idx_(int b, int t, int u) const -> size_t {
        return (static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t)) * static_cast<size_t>(s_)
               + static_cast<size_t>(u);
    }

    // Flat index into the exchanged count matrix, SOURCE-partition-major within each rank's block, so
    // partition u owns the contiguous run [cnt_idx_(b,u,0), cnt_idx_(b,u,s_)) and fills it itself: the
    // O(R*S^2) transpose partition 0 used to run alone becomes R*S contiguous writes per partition with
    // no false sharing. Dest-partition-major would put s_ partitions on every cache line instead.
    // MPI_Alltoall is indifferent: `b` stays outermost, and both sides index through this one helper.
    auto cnt_idx_(int b, int u, int t) const -> size_t {
        return (static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(u)) * static_cast<size_t>(s_)
               + static_cast<size_t>(t);
    }

    template <class V>
    static auto grow_(V &v, size_t need) -> void {
        if (v.size() < need) {
            v.resize(need);
        }
    }

    // Round a row of 8-byte entries up to whole cache lines, so one partition's row never shares a line
    // with a peer's.
    static auto pad_to_line_(size_t entries) -> size_t {
        constexpr size_t kPerLine = 64 / 8;
        return ((entries + kPerLine - 1) / kPerLine) * kPerLine;
    }

    // Partition u's veto row: bit b set iff u sends at least one element to rank b. `mask_words_` is the
    // padded row stride; only the first ceil(R/64) words carry bits.
    auto mask_row_(int u) -> uint64_t * { return send_mask_.data() + static_cast<size_t>(u) * mask_words_; }
    static auto mask_test_(const uint64_t *row, int b) -> bool {
        return (row[static_cast<size_t>(b) >> 6] & (1ULL << (static_cast<unsigned>(b) & 63U))) != 0;
    }
    auto mask_empty_(const uint64_t *row) const -> bool {
        const size_t live = static_cast<size_t>((r_ + 63) / 64);
        for (size_t w = 0; w < live; ++w) {
            if (row[w] != 0) {
                return false;
            }
        }
        return true;
    }

    // Phase A's per-rank running offsets, one padded row per partition (r_ can be in the hundreds, so
    // this is a member and not a stack array).
    auto run_row_(int t) -> size_t * { return run_scratch_.data() + static_cast<size_t>(t) * run_stride_; }

    // Partition u summarizes its own send counts into its veto row, from its own argument and before the
    // publishing barrier, so it costs no barrier and reads no peer.
    auto fill_send_mask_(int local_partition, const int *send_counts /*[P]*/) -> void {
        uint64_t *row = mask_row_(local_partition);
        std::fill(row, row + mask_words_, uint64_t{0});
        for (int b = 0; b < r_; ++b) {
            const int *counts = send_counts + b * s_;
            for (int t = 0; t < s_; ++t) {
                if (counts[t] != 0) {
                    row[static_cast<size_t>(b) >> 6] |= 1ULL << (static_cast<unsigned>(b) & 63U);
                    break;
                }
            }
        }
    }

    // Partition u writes its own source row of the exchanged count matrix. Disjoint per u and
    // contiguous in t, so it needs no barrier of its own and no coherence ping-pong.
    auto fill_count_row_(int local_partition, const int *send_counts /*[P]*/) -> void {
        const int u = local_partition;
        for (int b = 0; b < r_; ++b) {
            const int *row = send_counts + b * s_;
            int *dst = counts_send_.data() + cnt_idx_(b, u, 0);
            for (int t = 0; t < s_; ++t) {
                dst[t] = row[t];
            }
        }
        fill_send_mask_(local_partition, send_counts);
    }

    // Non-resolve alltoallv path: recv counts come from partition t's own published recv_counts, so
    // in phase A below every partition reads only its own slot.
    auto size_staging_parallel_(int local_partition, size_t elem) -> void {
        size_staging_parallel_(local_partition, elem, [this](int t, int rank, int su) {
            return slots_[static_cast<size_t>(t)].recv_counts[rank * s_ + su];
        });
    }

    // Replaces the O(R*S^2) serial sizing pass with a parallel prefix. recv_count(t, rank, su) is the
    // count partition t on this rank receives from (rank, source partition su).
    //
    // The offset of block (b, t, u) in the staged message decomposes as
    //     mpi_send_displs_[b]  +  sum_{t' < t} col_send_[b][t']  +  sum_{u' < u} c[u'][b][t]
    //     \------------- tpre_send_[b][t], one O(R*S) scan -----/     \--- pack_off_[b][t][u] ---/
    // The right-hand term is an independent scan per (b,t) pair, so partition t owns every pair with its
    // own t: R*S work each, S-way parallel, covering the same R*S^2 entries. Only the middle term needs
    // global knowledge, and at O(R*S) it is small enough to leave on partition 0.
    //
    // The two barriers this costs (phase A complete before partition 0 reduces it, bases visible before
    // anyone packs) buy S-way parallelism over what measured 84% of wall time at S=112.
    template <class RecvCountFn>
    auto size_staging_parallel_(int local_partition, size_t elem, RecvCountFn recv_count) -> void {
        const int t = local_partition;
        {
            ScopedNs timer{par_table_ns(local_partition)};
            // Send half, SOURCE-partition-outer so each peer's veto row is loaded once for all R ranks
            // instead of probing its count array R times on R separate cache lines -- the dominant cost
            // of this phase once the serial fill was gone. Skipped blocks leave a stale pack_off_ entry,
            // which is safe: an offset is read only where the matching count is nonzero.
            size_t *run = run_row_(t);
            std::fill(run, run + r_, size_t{0});
            for (int u = 0; u < s_; ++u) {
                const uint64_t *mask = mask_row_(u);
                if (mask_empty_(mask)) {
                    continue;
                }
                const int *counts = slots_[static_cast<size_t>(u)].send_counts;
                for (int b = 0; b < r_; ++b) {
                    if (!mask_test_(mask, b)) {
                        continue; // u sends nothing to rank b, so it contributes 0 to every (b,t) scan
                    }
                    pack_off_[block_idx_(b, t, u)] = run[b];
                    run[b] += static_cast<size_t>(counts[b * s_ + t]);
                }
            }
            for (int b = 0; b < r_; ++b) {
                col_send_[static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t)] = run[b];
                size_t got = 0;
                for (int su = 0; su < s_; ++su) {
                    scatter_off_[block_idx_(b, t, su)] = got;
                    got += static_cast<size_t>(recv_count(t, b, su));
                }
                col_recv_[static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t)] = got;
            }
        }
        sync(local_partition); // phase A complete on every partition

        if (local_partition == 0) {
            ScopedNs timer{p0_table_ns(local_partition)};
            long long send_running = 0;
            long long recv_running = 0;
            for (int b = 0; b < r_; ++b) {
                long long send_sum = 0;
                long long recv_sum = 0;
                const size_t base = static_cast<size_t>(b) * static_cast<size_t>(s_);
                for (int k = 0; k < s_; ++k) {
                    send_sum += static_cast<long long>(col_send_[base + static_cast<size_t>(k)]);
                    recv_sum += static_cast<long long>(col_recv_[base + static_cast<size_t>(k)]);
                }
                mpi_send_counts_[static_cast<size_t>(b)] = checked_mpi_count(send_sum);
                mpi_recv_counts_[static_cast<size_t>(b)] = checked_mpi_count(recv_sum);
                mpi_send_displs_[static_cast<size_t>(b)] = checked_mpi_count(send_running);
                mpi_recv_displs_[static_cast<size_t>(b)] = checked_mpi_count(recv_running);
                // tpre_*_ folds the per-rank base in, so pack/scatter add exactly two numbers.
                size_t cur_s = static_cast<size_t>(send_running);
                size_t cur_r = static_cast<size_t>(recv_running);
                for (int k = 0; k < s_; ++k) {
                    tpre_send_[base + static_cast<size_t>(k)] = cur_s;
                    cur_s += col_send_[base + static_cast<size_t>(k)];
                    tpre_recv_[base + static_cast<size_t>(k)] = cur_r;
                    cur_r += col_recv_[base + static_cast<size_t>(k)];
                }
                send_running += mpi_send_counts_[static_cast<size_t>(b)];
                recv_running += mpi_recv_counts_[static_cast<size_t>(b)];
            }
            // Grow-only, no zero-fill: pack_send_'s blocks tile [0, total_send) exactly and MPI_Alltoallv
            // fills every live byte of stage_recv_, so stale bytes past a prior high-water mark are never read.
            grow_(stage_send_, static_cast<size_t>(checked_mpi_count(send_running)) * elem);
            grow_(stage_recv_, static_cast<size_t>(checked_mpi_count(recv_running)) * elem);
        }
        sync(local_partition); // bases and staging visible to every packer
    }

    auto pack_send_(int local_partition, size_t elem) -> void {
        const int u = local_partition;
        const char *src = static_cast<const char *>(slots_[static_cast<size_t>(u)].ptr);
        const int *my_send_counts = slots_[static_cast<size_t>(u)].send_counts;
        const int *my_send_displs = slots_[static_cast<size_t>(u)].send_displs;
        for (int b = 0; b < r_; ++b) {
            const size_t base = static_cast<size_t>(b) * static_cast<size_t>(s_);
            for (int t = 0; t < s_; ++t) {
                const int cnt = my_send_counts[b * s_ + t];
                if (cnt != 0) {
                    // Absolute start = per-(b,t) base (partition 0) + within-(b,t) scan (partition t).
                    const size_t off = tpre_send_[base + static_cast<size_t>(t)] + pack_off_[block_idx_(b, t, u)];
                    std::memcpy(stage_send_.data() + off * elem,
                                src + static_cast<size_t>(my_send_displs[b * s_ + t]) * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
    }

    // Absolute start of the block partition t receives from (rank a, source partition su).
    auto recv_block_off_(int a, int t, int su) const -> size_t {
        return tpre_recv_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)]
               + scatter_off_[block_idx_(a, t, su)];
    }

    [[noreturn]] auto abort_rank_(const char *verb, const char *what) -> void {
        std::print(stderr,
                   "monoprop: rank {} cannot complete the collective '{}' ({}). Its peer ranks are "
                   "blocked inside MPI with no way to be released, so the job is aborted rather than hung.\n",
                   mpi_rank_,
                   verb,
                   what);
        std::fflush(stderr);
        MPI_Abort(parent_, 1);
        std::abort(); // MPI_Abort is not marked [[noreturn]]; unreachable in practice
    }

    // Barrier wait is attributed to the partition that waited, which is the whole point: when the
    // master monopolises a serial phase, its own wait stays near zero while every peer's grows.
    auto sync(int local_partition) -> void {
        if (prof_ == nullptr) {
            barrier_.sync(local_partition);
            return;
        }
        CommProfile::Slot &sl = prof_->slot(local_partition);
        ++sl.n_barriers;
        ScopedNs t{&sl.barrier_ns};
        barrier_.sync(local_partition);
    }

    // nullptr target ⇒ ScopedNs is inert, so instrumented regions need no #ifdef or duplicate code.
    auto p0_table_ns(int u) -> uint64_t * { return prof_ == nullptr ? nullptr : &prof_->slot(u).table_p0_ns; }
    auto par_table_ns(int u) -> uint64_t * { return prof_ == nullptr ? nullptr : &prof_->slot(u).table_par_ns; }
    auto move_ns(int u) -> uint64_t * { return prof_ == nullptr ? nullptr : &prof_->slot(u).table_move_ns; }
    auto mpi_ns(int u) -> uint64_t * { return prof_ == nullptr ? nullptr : &prof_->slot(u).mpi_ns; }
    auto count_verb(int u) -> void {
        if (prof_ != nullptr) {
            ++prof_->slot(u).n_verbs;
        }
    }

    std::unique_ptr<CommProfile> prof_;

    MPI_Comm parent_;
    int s_;
    int r_ = 1;
    int mpi_rank_ = 0;
    std::vector<Slot> slots_;

    // Partition-0-managed shared state (written by partition 0, read by all between barriers).
    // S*S per rank, the counts alltoall.
    std::vector<int> counts_send_;
    std::vector<int> counts_recv_;
    // Per-rank [R] counts/displs for the aggregated payload alltoallv.
    std::vector<int> mpi_send_counts_;
    std::vector<int> mpi_send_displs_;
    std::vector<int> mpi_recv_counts_;
    std::vector<int> mpi_recv_displs_;
    // [R*S*S] within-(rank, dest partition) exclusive scans over the SOURCE partition, filled by the
    // owning dest partition; absolute starts are these plus tpre_*_ (see size_staging_parallel_).
    std::vector<size_t> pack_off_;
    std::vector<size_t> scatter_off_;
    // [R*S] per-(rank, dest partition) column totals and their prefix, the only globally-reduced part.
    std::vector<size_t> col_send_;
    std::vector<size_t> col_recv_;
    std::vector<size_t> tpre_send_;
    std::vector<size_t> tpre_recv_;
    // [S * mask_words_] one cache-line-padded veto row per source partition, and [S * run_stride_]
    // per-partition scratch for phase A's per-rank running offsets. Both written only by their owner.
    std::vector<uint64_t> send_mask_;
    std::vector<size_t> run_scratch_;
    size_t mask_words_ = 8;
    size_t run_stride_ = 8;
    // Aggregated MPI payload staging, HWM-sized.
    std::vector<std::byte> stage_send_;
    std::vector<std::byte> stage_recv_;
    // [R] reverse-round per-rank counts/displs (the forward round's, swapped and divided by the stride).
    // Only partition 0 touches these and tables_gen_/reverse_ready_gen_, inside a barriered window.
    std::vector<int> rev_send_counts_;
    std::vector<int> rev_send_displs_;
    std::vector<int> rev_recv_counts_;
    std::vector<int> rev_recv_displs_;
    uint64_t tables_gen_ = 0;
    uint64_t reverse_ready_gen_ = ~0ULL;
    double red_f64_ = 0.0;
    uint64_t red_u64_ = 0;
    std::vector<double> red_vec_;

    PartitionBarrier barrier_;
};

} // namespace monoprop::mpi
