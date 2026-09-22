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
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <print>
#include <stdexcept>
#include <thread>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Pairwise.h"
#include "monoprop/detail/mpi/PartitionBarrier.h"
#include "monoprop/detail/mpi/Routing.h"

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

class HybridComm { // NOSONAR(cpp:S1448): transport phases share one barrier-owned state machine.
public:
    // n_local_partitions = S, identical on every rank (the facade ctor checks that before constructing).
    HybridComm(MPI_Comm parent, int n_local_partitions)
        : parent_(parent),
          s_(n_local_partitions),
          slots_(static_cast<size_t>(n_local_partitions)),
          row_peer_(static_cast<size_t>(n_local_partitions), kNoPeer),
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
        // After the local preconditions: this is a collective, so a rank that fails one of those must not
        // already have committed its peers to it.
        pairwise_ = agree_routing_(routing::Config::from_env(static_cast<size_t>(s_)));
        // Size all (R,S)-fixed scratch once so per-call paths never allocate; staging grows on demand.
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
        // Rounded up to whole 64-byte lines: each row has its own writer, so a shared line false-shares.
        counts_stride_ = round_up_(p, kIntsPerLine);
        rows_stride_ = round_up_(static_cast<size_t>(r_), kLongsPerLine);
        // One spare line each: the allocator guarantees alignof(T), not 64, so row 0 must be realigned.
        counts_matrix_store_.assign((static_cast<size_t>(s_) * counts_stride_) + kIntsPerLine, 0);
        counts_matrix_ = align_to_line_(counts_matrix_store_.data());
        rows_store_.assign((static_cast<size_t>(s_) * rows_stride_) + kLongsPerLine, 0LL);
        rows_ = align_to_line_(rows_store_.data());
        assert(reinterpret_cast<uintptr_t>(counts_matrix_) % kLineBytes == 0);
        assert(reinterpret_cast<uintptr_t>(rows_) % kLineBytes == 0);
        assert(counts_matrix_ + static_cast<size_t>(s_) * counts_stride_
               <= counts_matrix_store_.data() + counts_matrix_store_.size());
        assert(rows_ + static_cast<size_t>(s_) * rows_stride_ <= rows_store_.data() + rows_store_.size());
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    auto size() const -> int { return r_ * s_; }
    auto ranks() const -> int { return r_; }
    auto partitions() const -> int { return s_; }
    auto global_rank(int local_partition) const -> int { return (mpi_rank_ * s_) + local_partition; }
    // Agreed across parent_ at construction, so every rank branches the same way. See agree_routing_.
    auto routes_pairwise() const -> bool { return pairwise_; }

    auto alltoall_counts(int local_partition,
                         const int *send_counts /*[P]*/,
                         int *recv_counts /*[P]*/,
                         PeerPlan plan = {}) -> void {
        require_routable(plan, r_);
        guard_partition0_(local_partition, "alltoall_counts", [this, local_partition, send_counts, recv_counts, plan] {
            alltoall_counts_impl_(local_partition, send_counts, recv_counts, plan);
        });
    }

    // See AlltoallvArgs for the send-buffer lifetime and the element-vs-byte convention; `dt` is the MPI
    // datatype whose extent is args.elem, and it stays a separate argument because the bundle is shared
    // with the non-MPI-capable transport.
    //
    // `derive_wire_plan` asks partition 0 to read the peer set off the whole rank's traffic instead of
    // taking it from the caller. A caller cannot supply it: only partition 0 reaches MPI, and its own row
    // may be the empty one while a sibling partition holds the rank's only traffic. It must be
    // rank-uniform (routes_pairwise()), and legal only for a SYMMETRIC layout -- recv counts are the send
    // counts -- which is what lets the peer set be read off the published recv rows.
    auto alltoallv(int local_partition,
                   const AlltoallvArgs &args,
                   MPI_Datatype dt,
                   PeerPlan plan = {},
                   bool derive_wire_plan = false) -> void {
        require_routable(plan, r_);
        guard_partition0_(local_partition, "alltoallv", [this, local_partition, &args, dt, plan, derive_wire_plan] {
            alltoallv_impl_(local_partition, args, dt, plan, derive_wire_plan);
        });
    }

    // See AlltoallvResolveArgs: the recv side is an output, and args.recv is resized here.
    template <typename T>
    auto alltoallv_resolve(int local_partition,
                           const AlltoallvResolveArgs<T> &args,
                           MPI_Datatype dt,
                           PeerPlan plan = {}) -> void {
        require_routable(plan, r_);
        // `args` by reference, not by value: the impl resizes args.recv and then writes through it.
        guard_partition0_(local_partition, "alltoallv_resolve", [this, local_partition, &args, dt, plan] {
            alltoallv_resolve_impl_<T>(local_partition, args, dt, plan);
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
    // Counts travel through counts_matrix_ / rows_, not through here.
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

    // recv_counts[g] = amount global partition g sends to this partition. 2 barriers + one count round.
    auto alltoall_counts_impl_(int local_partition,
                               const int *send_counts /*[P]*/,
                               int *recv_counts /*[P]*/,
                               PeerPlan plan) -> void {
        publish_counts_row_(local_partition, send_counts, plan.sparse);
        sync();
        if (local_partition == 0) {
            require_plan_covers_traffic_(plan);
            fill_peers_(plan);
            pack_count_matrix_();
            exchange_count_blocks_(plan);
        }
        sync();
        // Partition t extracts its row: recv from (rank a, partition su) is contiguous per source rank a.
        // Under a plan only the f peer ranks were exchanged, so the rest of the row is zero by definition
        // (a non-peer cannot own the partner of any term this rank owns).
        const int t = local_partition;
        std::fill(recv_counts, recv_counts + (static_cast<size_t>(r_) * static_cast<size_t>(s_)), 0);
        for (const int a : peers_) {
            for (int su = 0; su < s_; ++su) {
                recv_counts[(a * s_) + su] = counts_recv_[counts_idx_(a, t, su)];
            }
        }
        // No trailing barrier: past the last sync only counts_recv_ is read, and partition 0 cannot rewrite
        // it until a future call's second barrier — unreachable until every extractor here has arrived.
        // send_counts is consumed before the first sync, so peers may free or reuse it on return.
    }

    // Flat variable all-to-all over caller-owned buffers; see AlltoallvArgs for the conventions.
    auto alltoallv_impl_(int local_partition,
                         const AlltoallvArgs &args,
                         MPI_Datatype dt,
                         PeerPlan plan,
                         bool derive_wire_plan = false) -> void {
        const size_t u = static_cast<size_t>(local_partition);
        Slot &me = slots_[u];
        me.ptr = args.send;
        me.send_displs = args.send_displs;
        publish_counts_row_(local_partition, args.send_counts, plan.sparse || derive_wire_plan);
        publish_recv_rows_(local_partition, args.recv_counts, plan);
        sync(); // B1

        // B2: partition 0 sizes/reallocates staging; must finish before any partition packs into stage_send_.
        if (local_partition == 0) {
            // Written here, read again in the B3->B4 window: partition 0 is this member's only toucher.
            if (derive_wire_plan) {
                wire_plan_ = derived_wire_plan_(); // cross-checks both sides itself
            }
            else {
                wire_plan_ = plan;
                require_plan_covers_traffic_(wire_plan_);
            }
            fill_peers_(wire_plan_);
            size_staging_send_(args.elem);
            fill_recv_col_([this](int a, int t) { return row_recv_(t)[a]; });
            size_staging_recv_(args.elem);
        }
        sync(); // B2

        // B3: each partition packs its own cross-rank blocks into stage_send_ (disjoint writes).
        pack_send_(local_partition, args.elem);
        sync(); // B3

        // B4: partition 0 moves the payload while peers park at the barrier.
        if (local_partition == 0) {
            exchange_payload_(dt, args.elem, wire_plan_);
        }
        sync(); // B4

        scatter_recv_(local_partition, args.recv, args.recv_counts, args.recv_displs, args.elem);
        // No trailing barrier: base_recv_ is rewritten only in a later verb's B1→B2 window.
    }

    // Fused count-resolve + payload alltoallv: folds the standalone count exchange into this verb's
    // barriered windows (4 syncs instead of 6). recv_counts / recv_displs and `recv` (resized) are
    // outputs. Bit-identical to alltoall_counts + alltoallv.
    //
    // The count round is POSTED in B1→B2 and only waited on in B3→B4, so the whole B2→B3 packing runs
    // underneath it. Split, not fused, because nothing before fill_recv_col_ reads counts_recv_: the
    // send side sizes from the locally published rows, and pack_send_ from that sizing. The dense arm
    // is MPI_Alltoall, blocking, and completes inside post_count_blocks_ regardless.
    template <typename T>
    auto alltoallv_resolve_impl_(int local_partition,
                                 const AlltoallvResolveArgs<T> &args,
                                 MPI_Datatype dt,
                                 PeerPlan plan) -> void {
        // Typed verb: element bytes are sizeof(T) by construction, so they are derived rather than passed.
        constexpr size_t elem = sizeof(T);
        const size_t u = static_cast<size_t>(local_partition);
        Slot &me = slots_[u];
        // Typed here but byte-addressed in the slot: pack_send_ copies by (displ, count) in elements and
        // never reconstructs T, so the slot stays type-erased for the untyped alltoallv_impl_ above.
        me.ptr = reinterpret_cast<const std::byte *>(args.send);
        me.send_displs = args.send_displs;
        // Count row only: the recv counts do not exist until the count round is drained in B3→B4.
        publish_counts_row_(local_partition, args.send_counts, plan.sparse);
        sync(); // B1

        if (local_partition == 0) {
            require_plan_covers_traffic_(plan);
            fill_peers_(plan);
            pack_count_matrix_();
            post_count_blocks_(plan);
            size_staging_send_(elem);
        }
        sync(); // B2

        // B3: each partition packs its own cross-rank blocks into stage_send_, the count round in flight.
        pack_send_(local_partition, elem);
        sync(); // B3

        // B4: the counts land here -- fill_recv_col_ is their first reader -- then the payload moves.
        if (local_partition == 0) {
            wait_count_blocks_();
            fill_recv_col_([this](int a, int t) { return block_sum_(a, t); });
            size_staging_recv_(elem);
            exchange_payload_(dt, elem, plan);
        }
        sync(); // B4

        // Past B4 now, not before B3: counts_recv_ does not exist until the wait above. Partition 0
        // cannot rewrite it before a later verb's B1→B2 window, unreachable until every reader here
        // has arrived at that verb's B1.
        const int t = local_partition;
        long long total = 0;
        const size_t p = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        std::fill(args.recv_counts, args.recv_counts + p, 0);
        std::fill(args.recv_displs, args.recv_displs + p, 0);
        for (const int a : peers_) {
            for (int su = 0; su < s_; ++su) {
                const int g = (a * s_) + su;
                const int c = counts_recv_[counts_idx_(a, t, su)];
                args.recv_counts[g] = c;
                args.recv_displs[g] = checked_mpi_count(total, "Recv displacement");
                total += c;
            }
        }
        args.recv.resize(static_cast<size_t>(checked_mpi_count(total, "Total recv count")));

        scatter_recv_(local_partition,
                      reinterpret_cast<std::byte *>(args.recv.data()), // after the resize: it may reallocate
                      args.recv_counts,
                      args.recv_displs,
                      elem);
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
        const size_t hi = std::min(len, lo + (per * kLine));
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

    template <typename T>
    static auto align_to_line_(T *p) -> T * {
        static_assert(kLineBytes % sizeof(T) == 0);
        const auto addr = reinterpret_cast<uintptr_t>(p);
        const size_t pad = (kLineBytes - static_cast<size_t>(addr % kLineBytes)) % kLineBytes;
        const auto offset = static_cast<std::ptrdiff_t>(pad / sizeof(T));
        return p + offset;
    }

    // (rank, dest partition, source partition) in the R*S*S count matrices: the count message's wire layout.
    auto counts_idx_(int b, int t, int su) const -> size_t {
        return (((static_cast<size_t>(b) * static_cast<size_t>(s_)) + static_cast<size_t>(t)) * static_cast<size_t>(s_))
               + static_cast<size_t>(su);
    }

    // The plan's peer ranks, materialised once per verb: the sizing sweeps index them S times each.
    // Written by partition 0 in the B1→B2 window, like base_recv_, so every reader past B2 sees it.
    auto fill_peers_(PeerPlan plan) -> void {
        const int f = plan.count(r_);
        peers_.resize(static_cast<size_t>(f));
        for (int k = 0; k < f; ++k) {
            peers_[static_cast<size_t>(k)] = plan.peer(mpi_rank_, k);
        }
    }

    // Zero only the peer ranks' slots of a [P] table. Every sweep that writes one of these tables and
    // every sweep that reads it walks the same peer set, so the rest of the row is never looked at --
    // and these run SERIALLY on partition 0 while S-1 partitions park, so the width is the cost.
    auto zero_peer_slots_(std::vector<long long> &table) const -> void {
        for (const int b : peers_) {
            std::fill_n(table.begin() + (static_cast<std::ptrdiff_t>(b) * s_), s_, 0LL);
        }
    }

    // What rank a's block of the count message holds for partition t, summed over source partitions.
    auto block_sum_(int a, int t) const -> long long {
        const int *blk = counts_recv_.data() + counts_idx_(a, t, 0);
        long long sum = 0;
        for (int su = 0; su < s_; ++su) {
            sum += blk[su];
        }
        return sum;
    }

    // [S x P] payload offset table: row u is source partition u's staging starts, one per destination g.
    auto pack_idx_(int u, int g) const -> size_t {
        return (static_cast<size_t>(u) * static_cast<size_t>(r_) * static_cast<size_t>(s_)) + static_cast<size_t>(g);
    }

    auto counts_row_(int u) -> int * { return counts_matrix_ + (static_cast<size_t>(u) * counts_stride_); }
    auto row_recv_(int u) -> long long * { return rows_ + (static_cast<size_t>(u) * rows_stride_); }

    // Phase P0: partition u writes only its own row, so the pre-barrier write needs no barrier. The one
    // cross-partition reader is partition 0 inside B1→B2, and no peer reaches verb k+1's P0 without
    // passing verb k's B2, so the rewrite always follows the read.
    auto publish_counts_row_(int local_partition, const int *send_counts, bool track_peer) -> void {
        const size_t u = static_cast<size_t>(local_partition);
        std::memcpy(counts_row_(local_partition),
                    send_counts,
                    static_cast<size_t>(r_) * static_cast<size_t>(s_) * sizeof(int));
        row_peer_[u] = kNoPeer;
        if (!track_peer) {
            return; // a dense round reaches every rank, so there is nothing a peer could exclude
        }
        // The destination rank this row's live blocks touch, so partition 0 can check a plan against the
        // traffic without an O(R*S^2) sweep of its own: S rows of O(R*S), each on its own partition.
        int peer = kNoPeer;
        for (int a = 0; a < r_ && peer != kManyPeers; ++a) {
            const int *block = send_counts + (static_cast<size_t>(a) * static_cast<size_t>(s_));
            if (std::all_of(block, block + s_, [](int n) { return n == 0; })) {
                continue;
            }
            peer = peer == kNoPeer ? a : kManyPeers;
        }
        row_peer_[u] = peer;
    }

    // The S published rows folded: kNoPeer, the one destination rank, or kManyPeers. Partition 0 only.
    auto folded_send_peer_() const -> int {
        int peer = kNoPeer;
        for (int u = 0; u < s_; ++u) {
            const int a = row_peer_[static_cast<size_t>(u)];
            if (a == kNoPeer || a == peer) {
                continue;
            }
            if (a == kManyPeers || peer != kNoPeer) {
                return kManyPeers;
            }
            peer = a;
        }
        return peer;
    }

    // A plan every rank agrees on but that excludes a live destination does not hang -- it silently DROPS
    // those blocks, because the staging sweeps only ever visit peers. Raised in partition 0's B1->B2
    // window, before any count or payload is posted, so guard_partition0_ can abort with nobody waiting.
    auto require_plan_covers_traffic_(PeerPlan plan) const -> void {
        if (plan.dense()) {
            return;
        }
        const int peer = folded_send_peer_();
        if (peer == kNoPeer || (peer != kManyPeers && plan.contains(mpi_rank_, peer))) {
            return;
        }
        throw std::runtime_error(
            std::format("peer plan shift {} excludes destination ranks this rank is sending to (rank {} of {}); "
                        "those blocks would be dropped rather than refused",
                        plan.shift,
                        mpi_rank_,
                        r_));
    }

    // long long: it sums S int counts. Masked through the plan, symmetric with its one reader
    // (fill_recv_col_): a non-peer's row is zero by definition, and a count left there would size
    // staging for a block no receive is posted for.
    auto publish_recv_rows_(int local_partition, const int *recv_counts, PeerPlan plan) -> void {
        long long *rr = row_recv_(local_partition);
        // Every entry is written, not just the peers': derived_wire_plan_ reads the whole row.
        std::fill_n(rr, r_, 0LL);
        const int f = plan.count(r_);
        for (int k = 0; k < f; ++k) {
            const int a = plan.peer(mpi_rank_, k);
            long long sum = 0;
            for (int su = 0; su < s_; ++su) {
                sum += recv_counts[(a * s_) + su];
            }
            rr[a] = sum;
        }
    }

    // Transpose the published count rows into counts_send_, dest-major then source-minor, for the one
    // S*S-int MPI_Alltoall. Partition 0 only, inside a barriered window. Source partition outer, so the
    // peer-owned side streams. Every element is written here, so no pre-zeroing.
    auto pack_count_matrix_() -> void {
        for (int su = 0; su < s_; ++su) {
            const int *row = counts_row_(su);
            for (const int b : peers_) {
                for (int t = 0; t < s_; ++t) {
                    counts_send_[counts_idx_(b, t, su)] = row[(b * s_) + t];
                }
            }
        }
    }

    // The rank-level peer set, read off the rows every partition published before B1 -- the first point
    // with a view wider than one partition's row. Under fanout-1 routing a layer's traffic is all on ONE
    // rank, so this resolves to a shift; with nothing occupied it resolves to the self peer, whose legs
    // are then all zero. That keeps the collective-vs-pairwise branch a function of the rank-uniform
    // routes_pairwise() alone and never of a rank's data, which straddles and hangs.
    //
    // Both sides are read, because the caller's symmetry claim (recv counts ARE the send counts) is what
    // licenses reading the peer off the recv rows at all. Anything wider than one rank, or a send side
    // disagreeing with the recv side, is that claim broken: throw, never fall back to a locally dense
    // plan, or this rank enters MPI_Alltoallv while its degree-one peers are already in Isend/Irecv.
    auto derived_wire_plan_() -> PeerPlan {
        int recv_peer = kNoPeer;
        for (int u = 0; u < s_ && recv_peer != kManyPeers; ++u) {
            const long long *rr = row_recv_(u);
            for (int a = 0; a < r_; ++a) {
                if (rr[a] == 0 || a == recv_peer) {
                    continue;
                }
                if (recv_peer != kNoPeer) {
                    recv_peer = kManyPeers;
                    break;
                }
                recv_peer = a;
            }
        }
        const int send_peer = folded_send_peer_();
        const bool spans_one = recv_peer != kManyPeers && send_peer != kManyPeers;
        const bool sides_agree = recv_peer == kNoPeer || send_peer == kNoPeer || recv_peer == send_peer;
        if (!spans_one || !sides_agree) {
            throw std::runtime_error(std::format(
                "routing claims fanout 1, but rank {} of {} has a layer spanning several peer ranks "
                "(send peer {}, recv peer {}); the layout is not the symmetric one the derived wire plan needs",
                mpi_rank_,
                r_,
                send_peer,
                recv_peer));
        }
        const int peer = recv_peer != kNoPeer ? recv_peer : (send_peer != kNoPeer ? send_peer : mpi_rank_);
        return PeerPlan{.sparse = true, .shift = mpi_rank_ ^ peer};
    }

    // The count blocks: one S*S-int MPI_Alltoall when dense, else a pair per peer (with the full linear
    // bits a zero rank shift keeps the whole round on-rank). Partition 0 only, in a barriered window.
    //
    // Sparse arm POSTS ONLY, so counts_recv_ and counts_send_ must stay put until wait_count_blocks_;
    // the dense MPI_Alltoall is blocking and has completed on return, which is why plan.dense() leaves
    // nothing live. The requests go in count_reqs_, never reqs_: see the member declaration.
    auto post_count_blocks_(PeerPlan plan) -> void {
        assert(count_posted_ == 0); // an un-drained round would be waited on twice
        const int block = s_ * s_;
        if (plan.dense()) {
            MPI_Alltoall(counts_send_.data(), block, MPI_INT, counts_recv_.data(), block, MPI_INT, parent_);
            return;
        }
        const PeerLayout blocks{.block = block};
        const SparsePairwiseArgs pairwise{
            .plan = plan,
            .me = mpi_rank_,
            .num_ranks = r_,
            .comm = parent_,
            .tag = kHybridCountTag,
            .datatype = MPI_INT,
            .elem = sizeof(int),
            .send = reinterpret_cast<const std::byte *>(counts_send_.data()),
            .send_layout = blocks,
            .recv = reinterpret_cast<std::byte *>(counts_recv_.data()),
            .recv_layout = blocks,
        };
        count_posted_ = sparse_pairwise(pairwise, count_reqs_);
    }

    // Drains post_count_blocks_. A no-op on the dense arm, and on a plan whose only peer is this rank
    // itself -- a count block is a fixed S*S ints, so no other peer can be skipped for a zero count.
    auto wait_count_blocks_() -> void {
        if (count_posted_ != 0) {
            MPI_Waitall(count_posted_, count_reqs_.data(), MPI_STATUSES_IGNORE);
            count_posted_ = 0;
        }
    }

    // Post and drain in one step, for callers with no work to overlap.
    auto exchange_count_blocks_(PeerPlan plan) -> void {
        post_count_blocks_(plan);
        wait_count_blocks_();
    }

    // The staged payload: one MPI_Alltoallv when dense, else a pair per peer over the same per-rank
    // counts and displacements (a non-peer's count is zero, so nothing is dropped). `elem` is dt's
    // extent: needed to reach a block, and known exactly to both callers.
    auto exchange_payload_(MPI_Datatype dt, size_t elem, PeerPlan plan) -> void {
        if (plan.dense()) {
            MPI_Alltoallv(stage_send_.data(),
                          mpi_send_counts_.data(),
                          mpi_send_displs_.data(),
                          dt,
                          stage_recv_.data(),
                          mpi_recv_counts_.data(),
                          mpi_recv_displs_.data(),
                          dt,
                          parent_);
            return;
        }
        const SparsePairwiseArgs pairwise{
            .plan = plan,
            .me = mpi_rank_,
            .num_ranks = r_,
            .comm = parent_,
            .tag = kHybridPayloadTag,
            .datatype = dt,
            .elem = elem,
            .send = stage_send_.data(),
            .send_layout = {.counts = mpi_send_counts_.data(), .displs = mpi_send_displs_.data()},
            .recv = stage_recv_.data(),
            .recv_layout = {.counts = mpi_recv_counts_.data(), .displs = mpi_recv_displs_.data()},
        };
        const int posted = sparse_pairwise(pairwise, reqs_);
        MPI_Waitall(posted, reqs_.data(), MPI_STATUSES_IGNORE);
    }

    template <typename V>
    static auto grow_(V &v, size_t need) -> void {
        if (v.size() < need) {
            v.resize(need);
        }
    }

    // Partition 0's send-side staging sizing, between B1 and B2, in two sweeps of counts_matrix_. Wire
    // block order is destination major, source minor: a per-destination base plus a per-source prefix.
    // Both sweeps run SERIALLY here while S-1 partitions park, so narrowing them to the peers matters as
    // much as the message count does.
    auto size_staging_send_(size_t elem) -> void {
        zero_peer_slots_(col_sum_);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row_(u);
            for (const int b : peers_) {
                const size_t base = static_cast<size_t>(b) * static_cast<size_t>(s_);
                for (int t = 0; t < s_; ++t) {
                    col_sum_[base + static_cast<size_t>(t)] += row[base + static_cast<size_t>(t)];
                }
            }
        }
        // The [R] arrays stay fully zeroed -- MPI_Alltoallv reads every entry on the dense arm, and the
        // zeros are what make the prefix below equal the full 0..R one at the peers' positions.
        std::ranges::fill(mpi_send_counts_, 0);
        std::ranges::fill(mpi_send_displs_, 0);
        for (const int b : peers_) {
            const long long *col = col_sum_.data() + (static_cast<size_t>(b) * static_cast<size_t>(s_));
            long long send_sum = 0;
            for (int t = 0; t < s_; ++t) {
                send_sum += col[t];
            }
            mpi_send_counts_[static_cast<size_t>(b)] = checked_mpi_count(send_sum, "Per-rank send count");
        }
        // peers_ is ascending (dense is 0..R-1, sparse is a singleton), so a prefix over it takes the
        // same value at every peer as a prefix over all R: a non-peer contributes zero.
        long long send_running = 0;
        for (const int b : peers_) {
            mpi_send_displs_[static_cast<size_t>(b)] = checked_mpi_count(send_running, "Send displacement");
            send_running += mpi_send_counts_[static_cast<size_t>(b)];
        }
        const size_t total_send = static_cast<size_t>(checked_mpi_count(send_running, "Total send count"));
        for (const int b : peers_) {
            size_t cur = static_cast<size_t>(mpi_send_displs_[static_cast<size_t>(b)]);
            for (int t = 0; t < s_; ++t) {
                const size_t g = (static_cast<size_t>(b) * static_cast<size_t>(s_)) + static_cast<size_t>(t);
                base_send_[g] = cur;
                cur += static_cast<size_t>(col_sum_[g]);
            }
        }
        zero_peer_slots_(col_sum_);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row_(u);
            size_t *off = pack_off_.data() + pack_idx_(u, 0);
            for (const int b : peers_) {
                const size_t base = static_cast<size_t>(b) * static_cast<size_t>(s_);
                for (int t = 0; t < s_; ++t) {
                    const size_t g = base + static_cast<size_t>(t);
                    off[g] = base_send_[g] + static_cast<size_t>(col_sum_[g]);
                    col_sum_[g] += row[g];
                }
            }
        }
        // Grow-only, no zero-fill: pack_send_'s blocks tile [0, total_send) exactly.
        grow_(stage_send_, total_send * elem);
    }

    // recv_col_[a*S + t] = what partition t receives from rank a. `value(a, t)` reads it from the rows
    // published in Phase P0 (alltoallv) or from the count blocks just exchanged (the fused resolve).
    template <typename Value>
    auto fill_recv_col_(Value &&value) -> void {
        zero_peer_slots_(recv_col_);
        for (const int a : peers_) {
            for (int t = 0; t < s_; ++t) {
                recv_col_[(static_cast<size_t>(a) * static_cast<size_t>(s_)) + static_cast<size_t>(t)] = value(a, t);
            }
        }
    }

    // Partition 0's recv-side staging sizing, from recv_col_. Only the per-(rank, partition) base; the
    // post-B4 scatter re-derives the per-source offsets as it walks (a, su).
    auto size_staging_recv_(size_t elem) -> void {
        std::ranges::fill(mpi_recv_counts_, 0);
        std::ranges::fill(mpi_recv_displs_, 0);
        for (const int a : peers_) {
            long long recv_sum = 0;
            for (int t = 0; t < s_; ++t) {
                recv_sum += recv_col_[(static_cast<size_t>(a) * static_cast<size_t>(s_)) + static_cast<size_t>(t)];
            }
            mpi_recv_counts_[static_cast<size_t>(a)] = checked_mpi_count(recv_sum, "Per-rank recv count");
        }
        // Same ascending-peers prefix as the send side.
        long long recv_running = 0;
        for (const int a : peers_) {
            mpi_recv_displs_[static_cast<size_t>(a)] = checked_mpi_count(recv_running, "Recv displacement");
            recv_running += mpi_recv_counts_[static_cast<size_t>(a)];
        }
        const size_t total_recv = static_cast<size_t>(checked_mpi_count(recv_running, "Total recv count"));
        for (const int a : peers_) {
            size_t cur = static_cast<size_t>(mpi_recv_displs_[static_cast<size_t>(a)]);
            for (int t = 0; t < s_; ++t) {
                const size_t g = (static_cast<size_t>(a) * static_cast<size_t>(s_)) + static_cast<size_t>(t);
                base_recv_[g] = cur;
                cur += static_cast<size_t>(recv_col_[g]);
            }
        }
        grow_(stage_recv_, total_recv * elem);
    }

    auto pack_send_(int local_partition, size_t elem) -> void {
        const int u = local_partition;
        // Own slot only — no peer's published send buffer is read here, which is what lets every
        // partition pack concurrently in the B2→B3 window.
        const std::byte *src = slots_[static_cast<size_t>(u)].ptr;
        const int *my_send_counts = counts_row_(u);
        const int *my_send_displs = slots_[static_cast<size_t>(u)].send_displs;
        const size_t *off = pack_off_.data() + pack_idx_(u, 0);
        for (const int b : peers_) {
            const int base = b * s_;
            for (int t = 0; t < s_; ++t) {
                const int g = base + t;
                const int cnt = my_send_counts[g];
                if (cnt != 0) {
                    std::memcpy(stage_send_.data() + (off[g] * elem),
                                src + (static_cast<size_t>(my_send_displs[g]) * elem),
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
    }

    // Scatter each global source's contiguous run out of stage_recv_ to recv_displs[g] (all legs, incl.
    // self-rank, go through staging). Walks (a, su) in accumulation order, so `cur` re-derives the block
    // starts from base_recv_ and this partition's own counts.
    auto scatter_recv_(int local_partition, std::byte *dst, const int *recv_counts, const int *recv_displs, size_t elem)
        -> void {
        const int t = local_partition;
        for (const int a : peers_) {
            size_t cur = base_recv_[(static_cast<size_t>(a) * static_cast<size_t>(s_)) + static_cast<size_t>(t)];
            for (int su = 0; su < s_; ++su) {
                const int g = (a * s_) + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + (static_cast<size_t>(recv_displs[g]) * elem),
                                stage_recv_.data() + (cur * elem),
                                static_cast<size_t>(cnt) * elem);
                }
                cur += static_cast<size_t>(cnt);
            }
        }
    }

    // Resolved here and nowhere else, because this is the only point on the replay path that is
    // collective on parent_ and owned by the library rather than by the caller: the exported evolve_*
    // entry points take a graph and a communicator, so they cannot be trusted to have checked anything.
    // Ranks that resolve differently do not answer differently, they HANG -- each posts receives from the
    // peers its own bits imply -- so the check is an exact equality, not a probable one. One allreduce of
    // {v, ~v} pairs under MPI_MAX yields both max and min, and max == min == mine is certainty.
    auto agree_routing_(routing::Config mine) const -> bool {
        const std::array<uint64_t, 6> probe{mine.linear,
                                            ~mine.linear,
                                            mine.partitions,
                                            ~mine.partitions,
                                            mine.seed,
                                            ~mine.seed};
        std::array<uint64_t, 6> agreed{};
        MPI_Allreduce(probe.data(), agreed.data(), 6, MPI_UINT64_T, MPI_MAX, parent_);
        for (size_t i = 0; i < probe.size(); ++i) {
            if (agreed[i] != probe[i]) {
                throw routing::RoutingDisagreement(
                    std::format("routing configuration differs across the {} ranks (this one: {}). "
                                "monoprop_ROUTING / monoprop_ROUTE_SEED must reach every rank identically -- "
                                "a disagreement deadlocks the exchange rather than corrupting it.",
                                r_,
                                mine.describe()));
            }
        }
        return routing::routes_pairwise(mine, static_cast<size_t>(r_));
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

    static constexpr int kNoPeer = -1;
    static constexpr int kManyPeers = -2;

    MPI_Comm parent_;
    int s_;
    int r_ = 1;
    int mpi_rank_ = 0;
    bool pairwise_ = false;
    std::vector<Slot> slots_;
    // One per partition, written in Phase P0 by its owner and folded by partition 0 in B1->B2.
    std::vector<int> row_peer_;

    // Partition-0-managed, except the two Phase P0 tables at the end, whose rows each partition owns.
    // S*S per rank, the counts alltoall.
    std::vector<int> counts_send_;
    std::vector<int> counts_recv_;
    // Per-rank [R] counts/displs for the aggregated payload alltoallv.
    std::vector<int> mpi_send_counts_;
    std::vector<int> mpi_send_displs_;
    std::vector<int> mpi_recv_counts_;
    std::vector<int> mpi_recv_displs_;
    // [S x P] payload block starts in stage_send_, row u per global destination g. No scatter-side
    // table: see size_staging_recv_.
    std::vector<size_t> pack_off_;
    // [P] staging bases: base_send_[b*S+t] starts partition (b,t)'s region of the message to rank b,
    // base_recv_[a*S+t] starts local partition t's region of the message from rank a.
    std::vector<size_t> base_send_;
    std::vector<size_t> base_recv_;
    std::vector<long long> col_sum_;
    std::vector<long long> recv_col_;
    // [S x counts_stride_] send-count matrix, row u owned by partition u; padded so no two rows share a
    // 64-byte line. Lifetime rule: publish_counts_row_.
    std::vector<int> counts_matrix_store_;
    int *counts_matrix_ = nullptr;
    size_t counts_stride_ = 0;
    // [S x rows_stride_] per-partition per-rank recv totals, same ownership and padding rules.
    std::vector<long long> rows_store_;
    long long *rows_ = nullptr;
    size_t rows_stride_ = 0;
    // Aggregated MPI payload staging, HWM-sized.
    std::vector<std::byte> stage_send_;
    std::vector<std::byte> stage_recv_;
    double red_f64_ = 0.0;
    uint64_t red_u64_ = 0;
    std::vector<double> red_vec_;
    // Point-to-point request scratch for the sparse payload round; grown on demand, partition 0 only.
    std::vector<MPI_Request> reqs_;
    // The count round's own scratch, separate from reqs_ by construction and not merely by the current
    // ordering: it stays live across B2 and B3, and Pairwise.h's resize would move the buffer MPI holds
    // pointers into the moment a payload post ever preceded the count wait.
    std::vector<MPI_Request> count_reqs_;
    int count_posted_ = 0; // live requests in count_reqs_; always 0 on the dense (blocking) arm
    // This verb's peer ranks; see fill_peers_.
    std::vector<int> peers_;
    // alltoallv's wire plan, partition 0 only: written in B1->B2, read in B3->B4. See derived_wire_plan_.
    PeerPlan wire_plan_;

    PartitionBarrier barrier_;
};

} // namespace monoprop::mpi
