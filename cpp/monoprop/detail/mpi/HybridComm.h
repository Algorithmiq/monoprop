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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <print>
#include <stdexcept>
#include <thread>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
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
    HybridComm(MPI_Comm parent, int n_local_partitions)
        : parent_(parent),
          s_(n_local_partitions),
          slots_(static_cast<size_t>(n_local_partitions)),
          barrier_(n_local_partitions) {
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
        // That contract is live: every array touched inside a barriered window, including the Phase P0
        // publish rows and the two staging scratch vectors, is sized here and only here.
        const size_t rss = static_cast<size_t>(r_) * static_cast<size_t>(s_) * static_cast<size_t>(s_);
        const size_t p = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        counts_send_.resize(rss);
        counts_recv_.resize(rss);
        mpi_send_counts_.resize(static_cast<size_t>(r_));
        mpi_recv_counts_.resize(static_cast<size_t>(r_));
        mpi_send_displs_.resize(static_cast<size_t>(r_));
        mpi_recv_displs_.resize(static_cast<size_t>(r_));
        pack_off_.resize(static_cast<size_t>(s_) * p); // same S*R*S elements as before, indexed (u, g)
        base_send_.resize(p);
        base_recv_.resize(p);
        col_sum_.resize(p);
        recv_col_.resize(p);
        // Row u of counts_matrix_ is written ONLY by partition u, concurrently with every other row, so
        // the row stride is rounded up to a whole number of 64-byte lines: without that, two partitions
        // share a line and the Phase P0 publish becomes a false-sharing storm across S cores. At P=256
        // a row is exactly 1 KiB and the padding costs nothing; at other P it does.
        counts_stride_ = round_up_(p, kIntsPerLine);
        rows_stride_ = round_up_(static_cast<size_t>(r_), kLongsPerLine);
        // Over-allocated by one line each so the base pointer can be advanced to a line boundary:
        // std::vector's allocator guarantees alignof(T), not 64, and a padded stride only keeps rows
        // apart once row 0 starts on a line of its own.
        counts_matrix_store_.assign(static_cast<size_t>(s_) * counts_stride_ + kIntsPerLine, 0);
        counts_matrix_ = align_to_line_(counts_matrix_store_.data());
        rows_store_.assign(static_cast<size_t>(s_) * rows_stride_ + kLongsPerLine, 0LL);
        rows_ = align_to_line_(rows_store_.data());
        // The realignment is what could be wrong, not the padding: `stride % kLineBytes == 0` is a
        // property of round_up_. Both hold by arithmetic too -- Release sets -DNDEBUG.
        assert(reinterpret_cast<uintptr_t>(counts_matrix_) % kLineBytes == 0);
        assert(reinterpret_cast<uintptr_t>(rows_) % kLineBytes == 0);
        assert(counts_matrix_ + static_cast<size_t>(s_) * counts_stride_
               <= counts_matrix_store_.data() + counts_matrix_store_.size());
        assert(rows_ + static_cast<size_t>(s_) * rows_stride_ <= rows_store_.data() + rows_store_.size());
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    auto size() const -> int { return r_ * s_; }
    auto global_rank(int local_partition) const -> int { return mpi_rank_ * s_ + local_partition; }

    auto alltoall_counts(int local_partition, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        guard_partition0_(local_partition, "alltoall_counts", [this, local_partition, send_counts, recv_counts] {
            alltoall_counts_impl_(local_partition, send_counts, recv_counts);
        });
    }

    // See AlltoallvArgs for the send-buffer lifetime and the element-vs-byte convention; `dt` is the MPI
    // datatype whose extent is args.elem, and it stays a separate argument because the bundle is shared
    // with the non-MPI-capable transport.
    auto alltoallv(int local_partition, const AlltoallvArgs &args, MPI_Datatype dt) -> void {
        guard_partition0_(local_partition, "alltoallv", [this, local_partition, &args, dt] {
            alltoallv_impl_(local_partition, args, dt);
        });
    }

    // See AlltoallvResolveArgs: the recv side is an output, and args.recv is resized here.
    template <typename T>
    auto alltoallv_resolve(int local_partition, const AlltoallvResolveArgs<T> &args, MPI_Datatype dt) -> void {
        // `args` by reference, not by value: the impl resizes args.recv and then writes through it.
        guard_partition0_(local_partition, "alltoallv_resolve", [this, local_partition, &args, dt] {
            alltoallv_resolve_impl_<T>(local_partition, args, dt);
        });
    }

    template <typename T>
    auto allreduce_sum(int local_partition, T local_val) -> T {
        return guard_partition0_(local_partition, "allreduce_sum", [this, local_partition, local_val] {
            return allreduce_sum_impl_<T>(local_partition, local_val);
        });
    }

    auto allreduce_sum_inplace(int local_partition, double *values, size_t len) -> void {
        guard_partition0_(local_partition, "allreduce_sum_inplace", [this, local_partition, values, len] {
            allreduce_sum_inplace_impl_(local_partition, values, len);
        });
    }

    auto poison() -> void { barrier_.poison(); }
    auto reset() -> void { barrier_.reset(); }

private:
    // No count pointer is published here any more: counts travel through counts_matrix_ / rows_, which
    // each partition fills from its own arguments in Phase P0. What is left is read only by the owning
    // partition itself (pack_send_) or by partition 0 inside a barriered window (the reductions).
    struct alignas(64) Slot {
        const std::byte *ptr = nullptr; // byte view of this partition's send buffer; null until published
        const int *send_displs = nullptr;
        const double *vec = nullptr;
        double f64 = 0.0;
        uint64_t u64 = 0;
    };

    // Once partition 0 -- this rank's only participant on parent_ -- is inside a collective, the peer ranks
    // are committed: their partition-0 threads enter theirs and block inside MPI with no timeout. So a
    // rank-local failure here must MPI_Abort with the underlying error rather than throw, which would
    // hang the job. Single-rank partitioned runs use ShmComm, not HybridComm, and keep their exceptions.
    // `body` is invoked synchronously and never stored, so the callers' lambdas may capture by reference.
    template <typename Body>
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

    // recv_counts[g] = amount global partition g sends to this partition. 2 barriers + one S*S-int MPI_Alltoall.
    auto alltoall_counts_impl_(int local_partition, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        // Phase P0, before the first barrier: this partition copies its own count row into the shared
        // matrix, reading only its own argument. See publish_counts_row_ for why that needs no barrier.
        publish_counts_row_(local_partition, send_counts);
        sync();
        if (local_partition == 0) {
            pack_count_matrix_();
            MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
        }
        sync();
        // Partition t extracts its row: recv from (rank a, partition su) is contiguous per source rank a.
        const int t = local_partition;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                recv_counts[a * s_ + su] = counts_recv_[counts_idx_(a, t, su)];
            }
        }
        // No trailing barrier: past the last sync only counts_recv_ is read, and partition 0 cannot rewrite
        // it until a future call's second barrier — unreachable until every extractor here has arrived.
        // send_counts was consumed before the FIRST sync now (Phase P0 copies it), so peers may free or
        // reuse it on return, as before but sooner.
    }

    // Flat variable all-to-all over caller-owned buffers; see AlltoallvArgs for the conventions.
    auto alltoallv_impl_(int local_partition, const AlltoallvArgs &args, MPI_Datatype dt) -> void {
        const size_t u = static_cast<size_t>(local_partition);
        Slot &me = slots_[u];
        me.ptr = args.send;
        me.send_displs = args.send_displs;
        // Phase P0, BEFORE B1: each partition publishes its own count row and its per-rank recv totals,
        // reading only its own arguments. No peer state is touched, so this needs no barrier of its own
        // and the verb still costs exactly 4.
        //
        // Do not overstate it: partition 0 still sweeps every partition's row in B1→B2, so the volume
        // it pulls across cores is unchanged at S rows x P ints. What changed is the access pattern
        // (three strided passes over S separate arrays become sequential sweeps of one matrix) and the
        // recv row TOTALS, which are the only part genuinely moved onto the owners, at O(R*S).
        // See publish_counts_row_ for the lifetime rule.
        publish_counts_row_(local_partition, args.send_counts);
        publish_recv_rows_(local_partition, args.recv_counts);
        sync(); // B1

        // B2: partition 0 sizes/reallocates staging; must finish before any partition packs into stage_send_.
        if (local_partition == 0) {
            size_staging_send_(args.elem);
            fill_recv_col_from_rows_();
            size_staging_recv_(args.elem);
        }
        sync(); // B2

        // B3: each partition packs its own cross-rank blocks into stage_send_ (disjoint writes).
        pack_send_(local_partition, args.elem);
        sync(); // B3

        // B4: partition 0 runs the single MPI_Alltoallv while peers park at the barrier.
        if (local_partition == 0) {
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

        // Scatter each global source's contiguous run from stage_recv_ to recv_displs[g] (all legs, incl.
        // self-rank, go through staging). There is no R*S*S scatter table any more: the old one held a
        // prefix over su at fixed (a, t), computed on partition 0 out of partition t's OWN published
        // recv row — work that never needed to be on partition 0 at all. This loop already walks (a, su)
        // in exactly the order those offsets accumulate, so `cur` re-derives them for free from the
        // per-(rank, partition) base and this partition's own counts. No peer slot is read past B4.
        std::byte *dst = args.recv;
        const int t = local_partition;
        for (int a = 0; a < r_; ++a) {
            size_t cur = base_recv_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)];
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = args.recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(args.recv_displs[g]) * args.elem,
                                stage_recv_.data() + cur * args.elem,
                                static_cast<size_t>(cnt) * args.elem);
                }
                cur += static_cast<size_t>(cnt);
            }
        }
        // No trailing barrier (see alltoall_counts_impl_): past B4 only stage_recv_/base_recv_/own buffers.
        // base_recv_ is rewritten only in a later verb's B1→B2 window, which no partition can reach until
        // every partition — including one still scattering here — has arrived at that verb's B1.
    }

    // Fused count-resolve + payload alltoallv: folds the standalone count exchange into this verb's
    // B1→B2 window (4 syncs instead of 6). recv_counts / recv_displs and `recv` (resized) are outputs.
    // Bit-identical to alltoall_counts + alltoallv.
    template <typename T>
    auto alltoallv_resolve_impl_(int local_partition, const AlltoallvResolveArgs<T> &args, MPI_Datatype dt) -> void {
        // Typed verb: element bytes are sizeof(T) by construction, so they are derived rather than passed.
        constexpr size_t elem = sizeof(T);
        const size_t u = static_cast<size_t>(local_partition);
        Slot &me = slots_[u];
        // Typed here but byte-addressed in the slot: pack_send_ copies by (displ, count) in elements and
        // never reconstructs T, so the slot stays type-erased for the untyped alltoallv_impl_ above.
        me.ptr = reinterpret_cast<const std::byte *>(args.send);
        me.send_displs = args.send_displs;
        // Phase P0, BEFORE B1: the count row only. This verb cannot publish recv rows, because its recv
        // counts do not exist until the count MPI_Alltoall has run on partition 0 in the B1→B2 window;
        // that side is served by fill_recv_col_from_counts_recv_ instead.
        publish_counts_row_(local_partition, args.send_counts);
        sync(); // B1

        if (local_partition == 0) {
            pack_count_matrix_();
            MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
            size_staging_send_(elem);
            // Size the recv side from counts_recv_, which partition 0 now holds in full: recv of
            // partition t from (rank a, su) sits at a*S*S + t*S + su, so the S entries this needs per
            // (a, t) are one contiguous run.
            fill_recv_col_from_counts_recv_();
            size_staging_recv_(elem);
        }
        sync(); // B2

        const int t = local_partition;
        long long total = 0;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int c = counts_recv_[counts_idx_(a, t, su)];
                args.recv_counts[g] = c;
                args.recv_displs[g] = checked_mpi_count(total, "Recv displacement");
                total += c;
            }
        }
        args.recv.resize(static_cast<size_t>(checked_mpi_count(total, "Total recv count")));

        pack_send_(local_partition, elem);
        sync(); // B3

        if (local_partition == 0) {
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

        std::byte *dst = reinterpret_cast<std::byte *>(args.recv.data()); // after the resize: it may reallocate
        // Same running-cursor scatter as alltoallv_impl_; see the comment there for why the R*S*S
        // scatter table is gone.
        for (int a = 0; a < r_; ++a) {
            size_t cur = base_recv_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)];
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = args.recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(args.recv_displs[g]) * elem,
                                stage_recv_.data() + cur * elem,
                                static_cast<size_t>(cnt) * elem);
                }
                cur += static_cast<size_t>(cnt);
            }
        }
        // No trailing barrier: same discipline as alltoallv_impl_.
    }

    template <typename T>
    auto allreduce_sum_impl_(int local_partition, T local_val) -> T {
        Slot &me = slots_[static_cast<size_t>(local_partition)];
        if constexpr (std::is_floating_point_v<T>) {
            me.f64 = static_cast<double>(local_val);
        }
        else {
            me.u64 = static_cast<uint64_t>(local_val);
        }
        sync();
        if (local_partition == 0) {
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

    // In-place element-wise allreduce-sum across the flat P-world, slice-partitioned across partitions in
    // ascending order (bit-identical to a sequential sum).
    auto allreduce_sum_inplace_impl_(int local_partition, double *values, size_t len) -> void {
        slots_[static_cast<size_t>(local_partition)].vec = values;
        sync(); // all inputs published
        if (local_partition == 0) {
            grow_(red_vec_, len);
        }
        sync(); // red_vec_ sized
        constexpr size_t kLine = 64 / sizeof(double);
        const size_t lines = (len + kLine - 1) / kLine;
        const size_t per = (lines + static_cast<size_t>(s_) - 1) / static_cast<size_t>(s_);
        const size_t lo = std::min(len, static_cast<size_t>(local_partition) * per * kLine);
        const size_t hi = std::min(len, lo + per * kLine);
        for (size_t k = lo; k < hi; ++k) {
            double acc = 0.0;
            for (int s = 0; s < s_; ++s) {
                acc += slots_[static_cast<size_t>(s)].vec[k];
            }
            red_vec_[k] = acc; // disjoint line-rounded slices: no two partitions store to one line
        }
        sync(); // local reduction complete
        if (local_partition == 0) {
            MPI_Allreduce(MPI_IN_PLACE, red_vec_.data(), static_cast<int>(len), MPI_DOUBLE, MPI_SUM, parent_);
        }
        sync(); // global result in red_vec_
        std::memcpy(values, red_vec_.data(), len * sizeof(double));
        // No trailing barrier: red_vec_ is rewritten only inside a future verb's barriered phases.
    }

    static constexpr size_t kLineBytes = 64;
    static constexpr size_t kIntsPerLine = kLineBytes / sizeof(int);
    static constexpr size_t kLongsPerLine = kLineBytes / sizeof(long long);

    static auto round_up_(size_t n, size_t m) -> size_t { return ((n + m - 1) / m) * m; }

    // Advance a vector's data pointer to the next 64-byte boundary. The gap is a whole number of T,
    // because the allocator hands back storage at least alignof(T)-aligned and 64 is a multiple of
    // sizeof(T) for both element types used here; the caller over-allocates one line to pay for it.
    template <typename T>
    static auto align_to_line_(T *p) -> T * {
        static_assert(kLineBytes % sizeof(T) == 0);
        const auto addr = reinterpret_cast<uintptr_t>(p);
        const size_t pad = (kLineBytes - static_cast<size_t>(addr % kLineBytes)) % kLineBytes;
        return p + pad / sizeof(T);
    }

    // Flat index of the (rank, dest partition, source partition) entry in the R*S*S count matrices the
    // counts MPI_Alltoall exchanges. This is that message's WIRE layout and nothing else. The payload
    // offset table no longer has this shape, which is why there are two indexers below rather than one
    // `block_idx_` meaning different things at different call sites.
    auto counts_idx_(int b, int t, int su) const -> size_t {
        return (static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t)) * static_cast<size_t>(s_)
               + static_cast<size_t>(su);
    }

    // Flat index into the [S x P] payload offset table: row u holds source partition u's own staging
    // start for each global destination g, so pack_send_ sweeps it contiguously instead of striding by S.
    auto pack_idx_(int u, int g) const -> size_t {
        return static_cast<size_t>(u) * static_cast<size_t>(r_) * static_cast<size_t>(s_) + static_cast<size_t>(g);
    }

    // Partition u's own line-aligned rows of the two Phase P0 tables.
    auto counts_row_(int u) -> int * { return counts_matrix_ + static_cast<size_t>(u) * counts_stride_; }
    auto row_recv_(int u) -> long long * { return rows_ + static_cast<size_t>(u) * rows_stride_; }

    /* Phase P0 — what a partition publishes BEFORE the verb's first barrier.
     *
     * Each partition writes only its OWN row of counts_matrix_ / rows_, from its own arguments. No
     * partition reads another's state here, so the phase needs no barrier and the verbs stay at 4. It
     * exists to move the R*S*S strided cross-partition read off partition 0: what used to be one thread
     * dereferencing a different thread's array on every iteration is now S owner-local walks running in
     * parallel, each touching only memory it just wrote.
     *
     * LIFETIME — this invariant must survive future edits or the next change silently breaks it. A fast
     * peer may enter verb k+1's Phase P0 while a slow peer is still in verb k's post-B4 tail, so the
     * write of row u for verb k+1 races the tail of verb k unless something orders them. It is ordered:
     * the only CROSS-partition reader of counts_matrix_ / rows_ is partition 0, and it reads them
     * strictly inside the B1→B2 window. Partition 0 cannot leave B2 of verb k until every partition has
     * arrived at B2, so its verb-k reads are complete before ANY partition passes B2 of verb k; and a
     * peer cannot reach verb k+1's Phase P0 without first passing the LAST barrier of verb k (B4 for the
     * payload verbs, B2 for the count exchange), hence B2 of verb k. The rewrite therefore strictly
     * follows the read. This is the same argument alltoall_counts_impl_ makes for counts_recv_ at the end
     * of its own window, pushed one verb further out.
     *
     * Partition u also reads its OWN counts row, in pack_send_ (B2→B3): one thread reading what it wrote,
     * and it cannot rewrite that row before verb k+1's Phase P0, which is past B4.
     *
     * Row u is written only by partition u, and the row strides are padded to whole cache lines
     * (see the constructor), so concurrent publishes neither race nor false-share.
     */
    auto publish_counts_row_(int local_partition, const int *send_counts) -> void {
        std::memcpy(counts_row_(local_partition),
                    send_counts,
                    static_cast<size_t>(r_) * static_cast<size_t>(s_) * sizeof(int));
    }

    // The recv side, for the verb whose recv counts are a caller input. There is no send counterpart:
    // partition 0 derives the per-rank send totals from col_sum_, which it must build anyway.
    // long long, not int: it sums up to S int counts, and only the per-rank total is int-checked.
    auto publish_recv_rows_(int local_partition, const int *recv_counts) -> void {
        long long *rr = row_recv_(local_partition);
        for (int a = 0; a < r_; ++a) {
            long long sum = 0;
            for (int su = 0; su < s_; ++su) {
                sum += recv_counts[a * s_ + su];
            }
            rr[a] = sum;
        }
    }

    // Transpose the published count rows into counts_send_, dest-partition-major (t) then
    // source-partition-minor (su), ready for the one S*S-int MPI_Alltoall. Partition 0 only, and only
    // inside a barriered window.
    //
    // Source partition OUTER, so the read side streams one partition's contiguous P-int row at a time
    // out of counts_matrix_ and the strided side is the write into counts_send_ — this rank's own
    // freshly-touched buffer. The old form had it the other way round: a dependent load into a
    // different thread's array on each of the R*S*S iterations.
    //
    // Every element of counts_send_ is written here and MPI_Alltoall fills counts_recv_ fully, so
    // neither buffer is pre-zeroed.
    auto pack_count_matrix_() -> void {
        for (int su = 0; su < s_; ++su) {
            const int *row = counts_row_(su);
            for (int b = 0; b < r_; ++b) {
                for (int t = 0; t < s_; ++t) {
                    counts_send_[counts_idx_(b, t, su)] = row[b * s_ + t];
                }
            }
        }
    }

    template <typename V>
    static auto grow_(V &v, size_t need) -> void {
        if (v.size() < need) {
            v.resize(need);
        }
    }

    /* Partition 0's send-side staging sizing, between B1 and B2. Both passes are CONTIGUOUS: they
     * sweep counts_matrix_ row by row instead of striding across S separate argument arrays. This is
     * still another thread's memory — the rows were written by their owners in Phase P0 — so the
     * cross-core volume is the same; it is the pattern and the pass count that improved.
     *
     * BIT-IDENTITY. The staging BLOCK ORDER is unchanged — within the message to rank b, destination
     * partition t's region comes first in ascending t, and inside it the source partitions u ascend.
     * Only the way a block start is computed changes. The old code carried one running cursor:
     *
     *     cur = mpi_send_displs_[b]; for t: for u: pack_off_(b,t,u) = cur; cur += c[u][b*S+t];
     *
     * and the new code splits that same recurrence into a per-destination base and a per-source offset:
     *
     *     W[g]                 = sum_u c[u][g]                             (pass A, u outer)
     *     mpi_send_counts_[b]  = sum_t W[b*S+t]
     *     base_send_[g]        = mpi_send_displs_[b] + sum_{t'<t} W[b*S+t']  for g = b*S + t
     *     pack_off_(u,g)       = base_send_[g] + sum_{u'<u} c[u'][g]        (pass B, u outer)
     *
     * These are elementwise EQUAL, not merely equivalent. W[b*S+t] is exactly the total the old inner
     * u-loop accumulated before it moved on to t+1, so base_send_[b*S+t] is the value `cur` held when
     * the old loop entered that t; and sum_{u'<u} c[u'][g] is what it had added by the time it reached
     * u. Every changed quantity is an integer prefix sum reassociated over an exact operation, and no
     * floating-point accumulation order anywhere in the transport is touched, so the bytes on the wire
     * — and every reduction fed from them — stay bit-identical at fixed (R, S).
     */
    auto size_staging_send_(size_t elem) -> void {
        const size_t p = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        // Pass A: the column sums W, u OUTER and g INNER, so both the count row and the accumulator are
        // swept in address order. It feeds the checked counts below and cannot throw, so the order in
        // which those throw is unchanged.
        std::fill(col_sum_.begin(), col_sum_.end(), 0LL);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row_(u);
            for (size_t g = 0; g < p; ++g) {
                col_sum_[g] += row[g];
            }
        }
        // The per-rank total is that same W summed over t: one contiguous S-run of this rank's own array.
        for (int b = 0; b < r_; ++b) {
            const long long *col = col_sum_.data() + static_cast<size_t>(b) * static_cast<size_t>(s_);
            long long send_sum = 0;
            for (int t = 0; t < s_; ++t) {
                send_sum += col[t];
            }
            mpi_send_counts_[static_cast<size_t>(b)] = checked_mpi_count(send_sum, "Per-rank send count");
        }
        long long send_running = 0;
        for (int b = 0; b < r_; ++b) {
            mpi_send_displs_[static_cast<size_t>(b)] = checked_mpi_count(send_running, "Send displacement");
            send_running += mpi_send_counts_[static_cast<size_t>(b)];
        }
        const size_t total_send = static_cast<size_t>(checked_mpi_count(send_running, "Total send count"));
        for (int b = 0; b < r_; ++b) {
            size_t cur = static_cast<size_t>(mpi_send_displs_[static_cast<size_t>(b)]);
            for (int t = 0; t < s_; ++t) {
                const size_t g = static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t);
                base_send_[g] = cur;
                cur += static_cast<size_t>(col_sum_[g]);
            }
        }
        // Pass B: the exclusive prefix over source partitions, u OUTER again. col_sum_ carries it: its
        // last reader is the base_send_ loop above, and pass B ends up rebuilding the same column sums.
        std::fill(col_sum_.begin(), col_sum_.end(), 0LL);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row_(u);
            size_t *off = pack_off_.data() + pack_idx_(u, 0);
            for (size_t g = 0; g < p; ++g) {
                off[g] = base_send_[g] + static_cast<size_t>(col_sum_[g]);
                col_sum_[g] += row[g];
            }
        }
        // Grow-only, no zero-fill: pack_send_'s blocks tile [0, total_send) exactly, so stale bytes past
        // a prior high-water mark are never read.
        grow_(stage_send_, total_send * elem);
    }

    // recv_col_[a*S + t] = the total partition t on this rank receives from rank a. For the payload verb
    // that is exactly the row every partition published in Phase P0; the R*S reads here are strided
    // across partitions, but R*S is 256 at layout A where the removed term was R*S*S = 32768.
    auto fill_recv_col_from_rows_() -> void {
        for (int a = 0; a < r_; ++a) {
            for (int t = 0; t < s_; ++t) {
                recv_col_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)] = row_recv_(t)[a];
            }
        }
    }

    // The same vector for the fused resolve verb, where partition 0 holds every recv count itself:
    // counts_recv_(a, t, .) is a contiguous run of S ints, so this is a block sum rather than the old
    // strided per-(t, su) closure call.
    auto fill_recv_col_from_counts_recv_() -> void {
        for (int a = 0; a < r_; ++a) {
            for (int t = 0; t < s_; ++t) {
                const int *blk = counts_recv_.data() + counts_idx_(a, t, 0);
                long long sum = 0;
                for (int su = 0; su < s_; ++su) {
                    sum += blk[su];
                }
                recv_col_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)] = sum;
            }
        }
    }

    /* Partition 0's recv-side staging sizing, from recv_col_. There is no R*S*S scatter table any more.
     * The old scatter_off_(a, t, su) was a prefix over su at fixed (a, t) — a prefix over partition t's
     * OWN published recv row — so it never needed to be computed on partition 0 at all. Only the
     * per-(rank, partition) base survives here; the post-B4 loop walks (a, su) in exactly the order
     * those offsets accumulate and re-derives the rest locally.
     */
    auto size_staging_recv_(size_t elem) -> void {
        for (int a = 0; a < r_; ++a) {
            long long recv_sum = 0;
            for (int t = 0; t < s_; ++t) {
                recv_sum += recv_col_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)];
            }
            mpi_recv_counts_[static_cast<size_t>(a)] = checked_mpi_count(recv_sum, "Per-rank recv count");
        }
        long long recv_running = 0;
        for (int a = 0; a < r_; ++a) {
            mpi_recv_displs_[static_cast<size_t>(a)] = checked_mpi_count(recv_running, "Recv displacement");
            recv_running += mpi_recv_counts_[static_cast<size_t>(a)];
        }
        const size_t total_recv = static_cast<size_t>(checked_mpi_count(recv_running, "Total recv count"));
        for (int a = 0; a < r_; ++a) {
            size_t cur = static_cast<size_t>(mpi_recv_displs_[static_cast<size_t>(a)]);
            for (int t = 0; t < s_; ++t) {
                const size_t g = static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t);
                base_recv_[g] = cur;
                cur += static_cast<size_t>(recv_col_[g]);
            }
        }
        // MPI_Alltoallv fills every live byte of stage_recv_, so this grow is likewise fill-free.
        grow_(stage_recv_, total_recv * elem);
    }

    auto pack_send_(int local_partition, size_t elem) -> void {
        const int u = local_partition;
        // Own slot only — no peer's published send buffer is read here, which is what lets every
        // partition pack concurrently in the B2→B3 window.
        const std::byte *src = slots_[static_cast<size_t>(u)].ptr;
        // Lengths from this partition's own published row -- the same snapshot pack_off_ was derived from.
        const int *my_send_counts = counts_row_(u);
        const int *my_send_displs = slots_[static_cast<size_t>(u)].send_displs;
        // One flat sweep over g = b*S + t: the same (b, t) visiting order the two-level loop had, but the
        // offset row is now contiguous in g where pack_off_[block_idx_(b, t, u)] strode by S.
        const size_t *off = pack_off_.data() + pack_idx_(u, 0);
        const int p = r_ * s_;
        for (int g = 0; g < p; ++g) {
            const int cnt = my_send_counts[g];
            if (cnt != 0) {
                std::memcpy(stage_send_.data() + off[g] * elem,
                            src + static_cast<size_t>(my_send_displs[g]) * elem,
                            static_cast<size_t>(cnt) * elem);
            }
        }
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

    auto sync() -> void { barrier_.sync(); }

    MPI_Comm parent_;
    int s_;
    int r_ = 1;
    int mpi_rank_ = 0;
    std::vector<Slot> slots_;

    // Partition-0-managed shared state (written by partition 0, read by all between barriers). The two
    // Phase P0 tables at the end of this block are the exception: each partition owns and writes its own
    // row of those, and partition 0 only reads them.
    // S*S per rank, the counts alltoall.
    std::vector<int> counts_send_;
    std::vector<int> counts_recv_;
    // Per-rank [R] counts/displs for the aggregated payload alltoallv.
    std::vector<int> mpi_send_counts_;
    std::vector<int> mpi_send_displs_;
    std::vector<int> mpi_recv_counts_;
    std::vector<int> mpi_recv_displs_;
    // [S x P] payload block starts (elements) in stage_send_: row u is source partition u's own starts,
    // one per global destination g. Same element count as the old [R*S*S] table, transposed so that the
    // row a partition reads is contiguous. There is no scatter-side table: see size_staging_recv_.
    std::vector<size_t> pack_off_;
    // [P] staging bases. base_send_[b*S + t] starts destination partition (b, t)'s region of the message
    // to rank b; base_recv_[a*S + t] starts local partition t's region of the message from rank a.
    std::vector<size_t> base_send_;
    std::vector<size_t> base_recv_;
    // Partition-0 scratch for the two contiguous staging passes and the recv-side totals. Members, not
    // locals: the constructor's no-per-call-allocation contract covers these too. long long throughout,
    // because each entry sums up to S int counts and only the per-rank totals are int-checked.
    std::vector<long long> col_sum_;
    std::vector<long long> recv_col_;
    // [S x counts_stride_] send-count matrix: row u is written ONLY by partition u, in Phase P0, and read
    // by partition 0 in the B1→B2 window and by partition u itself in pack_send_. The stride is padded and
    // the base realigned so that no two partitions' rows share a 64-byte line; the ctor asserts the
    // padding. See publish_counts_row_ for the lifetime argument that makes the pre-barrier write safe.
    std::vector<int> counts_matrix_store_;
    int *counts_matrix_ = nullptr;
    size_t counts_stride_ = 0;
    // [S x rows_stride_] per-partition per-rank recv totals, row u = row_recv_(u)[0..R), under the same
    // ownership, padding and lifetime rules as counts_matrix_.
    std::vector<long long> rows_store_;
    long long *rows_ = nullptr;
    size_t rows_stride_ = 0;
    // Aggregated MPI payload staging, HWM-sized.
    std::vector<std::byte> stage_send_;
    std::vector<std::byte> stage_recv_;
    double red_f64_ = 0.0;
    uint64_t red_u64_ = 0;
    std::vector<double> red_vec_;

    PartitionBarrier barrier_;
};

} // namespace monoprop::mpi
