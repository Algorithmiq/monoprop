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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/HybridStaging.h"
#include "monoprop/detail/mpi/PairSlots.h"
#include "monoprop/detail/mpi/Pairwise.h"
#include "monoprop/detail/mpi/PartitionBarrier.h"

// Composes R MPI ranks x S in-process partitions into one flat P=R*S SPMD world. Global id is rank-major
// (g = mpi_rank*S + partition), keeping each rank's partitions contiguous in ascending-global order — the
// contract Resolve.h's positional pairing relies on. Only partition-0 masters call MPI (bracketed by
// intra-rank barriers ⇒ requires MPI_THREAD_SERIALIZED); ascending-order local sums + order-preserving
// MPI ⇒ bit-identical, repeatable results for fixed (R, S).
//
// The staging tables and the sweeps over them live in HybridStaging.h; this file owns the barrier
// discipline, the verbs and the reductions. Phase names used throughout: P0 is the pre-barrier publish
// each partition does into its own row, then B1..B4 are the four barriers a payload verb takes.

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
          pair_(n_local_partitions),
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
        staging_.emplace(parent_, mpi_rank_, r_, s_);
        const size_t ss = static_cast<size_t>(s_) * static_cast<size_t>(s_);
        pair_hdr_send_.resize(ss);
        pair_row_start_.resize(static_cast<size_t>(s_));
        pair_blocklens_.reserve(ss + 1);
        pair_displs_.reserve(ss + 1);
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    auto size() const -> int { return r_ * s_; }
    auto ranks() const -> int { return r_; }
    auto partitions() const -> int { return s_; }
    auto global_rank(int local_partition) const -> int { return mpi_rank_ * s_ + local_partition; }

    /*! @brief Bytes this rank's transport holds: the grow-only payload staging plus the fixed tables.
     *
     *  Diagnostic only. The two payload buffers are sized to the widest message the run has needed and
     *  never shrink, so on a run whose widest gate came early they are resident for the whole of it
     *  while nothing resting names them. `pair_recv_` is the one pair_exchange fills: it sends in place
     *  through a derived datatype, so it has no send-side staging at all. One HybridComm serves all S
     *  partitions of a rank, so this is a per-RANK figure: a caller summing over partitions must ask
     *  exactly one of them.
     */
    [[nodiscard]] auto staging_bytes() const -> size_t {
        const auto cap = [](const auto &v) {
            return v.capacity() * sizeof(typename std::decay_t<decltype(v)>::value_type);
        };
        return staging_->bytes() + slots_.capacity() * sizeof(Slot) + cap(red_vec_) + cap(pair_recv_)
               + cap(pair_hdr_send_) + cap(pair_blocklens_) + cap(pair_displs_) + cap(pair_row_start_)
               + pair_.staging_bytes();
    }

    auto alltoall_counts(int local_partition,
                         const int *send_counts /*[P]*/,
                         int *recv_counts /*[P]*/,
                         PeerPlan plan = {}) -> void {
        guard_partition0_(local_partition, "alltoall_counts", [this, local_partition, send_counts, recv_counts, plan] {
            alltoall_counts_impl_(local_partition, send_counts, recv_counts, plan);
        });
    }

    // See AlltoallvArgs for the send-buffer lifetime and the element-vs-byte convention; `dt` is the MPI
    // datatype whose extent is args.elem, and it stays a separate argument because the bundle is shared
    // with the non-MPI-capable transport.
    //
    // `derive_wire_bits` > 0 asks partition 0 to narrow the WIRE itself, to that many linear bits, from
    // the destination ranks the whole rank actually uses. A caller cannot supply that plan: only
    // partition 0 reaches MPI, and its own row may be the empty one while a sibling partition has the
    // rank's only traffic. Legal only for a SYMMETRIC layout (the recv counts are the send counts),
    // which is what lets the peer set be read off the published recv rows; asserted against the send rows.
    auto alltoallv(int local_partition,
                   const AlltoallvArgs &args,
                   MPI_Datatype dt,
                   PeerPlan plan = {},
                   int derive_wire_bits = 0) -> void {
        guard_partition0_(local_partition, "alltoallv", [this, local_partition, &args, dt, plan, derive_wire_bits] {
            alltoallv_impl_(local_partition, args, dt, plan, derive_wire_bits);
        });
    }

    // See AlltoallvResolveArgs: the recv side is an output, and args.recv is resized here.
    template <typename T>
    auto alltoallv_resolve(int local_partition,
                           const AlltoallvResolveArgs<T> &args,
                           MPI_Datatype dt,
                           PeerPlan plan = {}) -> void {
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

    //! @brief The gate exchange; PairExchange.h states the contract. Partition 0 alone reaches MPI.
    auto pair_exchange(int local_partition, int rank_shift, SubStreams send) -> PairRecv {
        return guard_partition0_(local_partition, "pair_exchange", [this, local_partition, rank_shift, send] {
            return pair_exchange_impl_(local_partition, rank_shift, send);
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
    // Counts travel through the staging tables, not through here.
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
        staging_->publish_counts_row(local_partition, send_counts);
        sync();
        if (local_partition == 0) {
            staging_->fill_peers(plan);
            staging_->pack_count_matrix(plan);
            staging_->exchange_count_blocks(plan);
        }
        sync();
        // Partition t extracts its row: recv from (rank a, partition su) is contiguous per source rank a.
        // Under a plan only the peer ranks were exchanged, so the rest of the row is zero by definition
        // (a non-peer cannot own the partner of any term this rank owns).
        const int t = local_partition;
        std::fill(recv_counts, recv_counts + static_cast<size_t>(r_) * static_cast<size_t>(s_), 0);
        for (const int a : staging_->peers()) {
            for (int su = 0; su < s_; ++su) {
                recv_counts[a * s_ + su] = staging_->count_from(a, t, su);
            }
        }
        // No trailing barrier: past the last sync only the count message is read, and partition 0 cannot
        // rewrite it until a future call's second barrier — unreachable until every extractor here has
        // arrived. send_counts is consumed before the first sync, so peers may free it on return.
    }

    // Flat variable all-to-all over caller-owned buffers; see AlltoallvArgs for the conventions.
    auto alltoallv_impl_(int local_partition,
                         const AlltoallvArgs &args,
                         MPI_Datatype dt,
                         PeerPlan plan,
                         int derive_wire_bits) -> void {
        Slot &me = slots_[static_cast<size_t>(local_partition)];
        me.ptr = args.send;
        me.send_displs = args.send_displs;
        staging_->publish_counts_row(local_partition, args.send_counts);
        staging_->publish_recv_rows(local_partition, args.recv_counts, plan);
        sync(); // B1

        // B2: partition 0 sizes/reallocates staging; must finish before any partition packs into it.
        if (local_partition == 0) {
            // Written here, read again in the B3->B4 window: partition 0 is this member's only toucher.
            wire_plan_ = plan;
            if (derive_wire_bits > 0) {
                wire_plan_ = staging_->derived_wire_plan(derive_wire_bits);
                // The recv rows it was read off against the send rows: the symmetry the parameter needs.
                assert(staging_->narrowing_is_lossless(wire_plan_));
            }
            staging_->fill_peers(wire_plan_);
            staging_->size_send(args.elem);
            staging_->fill_recv_col([this](int a, int t) { return staging_->published_recv(a, t); });
            staging_->size_recv(args.elem);
        }
        sync(); // B2

        // B3: each partition packs its own cross-rank blocks into the send staging (disjoint writes).
        pack_send_(local_partition, args.elem);
        sync(); // B3

        // B4: partition 0 moves the payload while peers park at the barrier.
        if (local_partition == 0) {
            staging_->exchange_payload(dt, args.elem, wire_plan_);
        }
        sync(); // B4

        staging_->scatter_recv(local_partition, args.recv, args.recv_counts, args.recv_displs, args.elem);
        // No trailing barrier: the recv bases are rewritten only in a later verb's B1→B2 window.
    }

    // Fused count-resolve + payload alltoallv: folds the standalone count exchange into this verb's
    // barriered windows (4 syncs instead of 6). recv_counts / recv_displs and `recv` (resized) are
    // outputs. Bit-identical to alltoall_counts + alltoallv.
    //
    // The count round is POSTED in B1→B2 and only waited on in B3→B4, so the whole B2→B3 packing runs
    // underneath it. Split, not fused, because nothing before fill_recv_col reads the count blocks: the
    // send side sizes from the locally published rows, and pack_send from that sizing. The dense arm is
    // MPI_Alltoall, blocking, and completes inside post_count_blocks regardless.
    template <typename T>
    auto alltoallv_resolve_impl_(int local_partition,
                                 const AlltoallvResolveArgs<T> &args,
                                 MPI_Datatype dt,
                                 PeerPlan plan) -> void {
        // Typed verb: element bytes are sizeof(T) by construction, so they are derived rather than passed.
        constexpr size_t elem = sizeof(T);
        Slot &me = slots_[static_cast<size_t>(local_partition)];
        // Typed here but byte-addressed in the slot: pack_send copies by (displ, count) in elements and
        // never reconstructs T, so the slot stays type-erased for the untyped alltoallv_impl_ above.
        me.ptr = reinterpret_cast<const std::byte *>(args.send);
        me.send_displs = args.send_displs;
        // Count row only: the recv counts do not exist until the count round is drained in B3→B4.
        staging_->publish_counts_row(local_partition, args.send_counts);
        sync(); // B1

        if (local_partition == 0) {
            staging_->fill_peers(plan);
            staging_->pack_count_matrix(plan);
            staging_->post_count_blocks(plan);
            staging_->size_send(elem);
        }
        sync(); // B2

        // B3: each partition packs its own blocks into the send staging, the count round in flight.
        pack_send_(local_partition, elem);
        sync(); // B3

        // B4: the counts land here -- fill_recv_col is their first reader -- then the payload moves.
        if (local_partition == 0) {
            staging_->wait_count_blocks();
            staging_->fill_recv_col([this](int a, int t) { return staging_->block_sum(a, t); });
            staging_->size_recv(elem);
            staging_->exchange_payload(dt, elem, plan);
        }
        sync(); // B4

        // Past B4 now, not before B3: the count blocks do not exist until the wait above. Partition 0
        // cannot rewrite them before a later verb's B1→B2 window, unreachable until every reader here
        // has arrived at that verb's B1.
        const int t = local_partition;
        long long total = 0;
        const size_t p = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        std::fill(args.recv_counts, args.recv_counts + p, 0);
        std::fill(args.recv_displs, args.recv_displs + p, 0);
        for (const int a : staging_->peers()) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int c = staging_->count_from(a, t, su);
                args.recv_counts[g] = c;
                args.recv_displs[g] = checked_mpi_count(total, "Recv displacement");
                total += c;
            }
        }
        args.recv.resize(static_cast<size_t>(checked_mpi_count(total, "Total recv count")));

        staging_->scatter_recv(local_partition,
                               reinterpret_cast<std::byte *>(args.recv.data()), // after the resize: it may realloc
                               args.recv_counts,
                               args.recv_displs,
                               elem);
        // No trailing barrier: same discipline as alltoallv_impl_.
    }

    auto pack_send_(int local_partition, size_t elem) -> void {
        const Slot &me = slots_[static_cast<size_t>(local_partition)];
        staging_->pack_send(local_partition, elem, me.ptr, me.send_displs);
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
            grow_to(red_vec_, len);
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

    /*! @brief The gate exchange's barrier discipline. Arguments are the verb's to validate
     *  (PairExchange.h), before any barrier; here they are asserted.
     */
    auto pair_exchange_impl_(int local_partition, int rank_shift, SubStreams send) -> PairRecv {
        assert(static_cast<int>(send.size()) == s_);
        assert(rank_shift >= 0 && (mpi_rank_ ^ rank_shift) < r_);
        pair_.publish(local_partition, send);
        if (rank_shift == 0) {
            sync(); // B1: descriptors published; the gather reads peers' buffers in place (PairExchange.h)
            return pair_.gather_in_rank(local_partition);
        }
        sync(); // B1: every partition's descriptors are published
        if (local_partition == 0) {
            pair_exchange_with_rank_(mpi_rank_ ^ rank_shift);
        }
        sync(); // B2: pair_recv_ holds the peer rank's header and payload, pair_row_start_ its row starts
        return pair_views_(local_partition);
    }

    /*! @brief Partition 0's one message each way, between B1 and B2.
     *
     *  The outgoing message is one hindexed datatype over absolute addresses: block 0 is the S*S
     *  word-count header, then the (t, u) sub-streams destination-major so that each receiving
     *  partition's run is contiguous; empty blocks are left out and the header carries their zero. The
     *  incoming message is sized by probing it -- its header is inside it -- and lands in one contiguous
     *  buffer, so neither side stages a byte. Isend before the blocking probe: both ranks do the same, so
     *  neither waits on a send the other has not posted.
     */
    auto pair_exchange_with_rank_(int peer) -> void {
        const size_t ss = static_cast<size_t>(s_) * static_cast<size_t>(s_);
        pair_blocklens_.clear();
        pair_displs_.clear();
        MPI_Aint addr = 0;
        MPI_Get_address(pair_hdr_send_.data(), &addr);
        pair_blocklens_.push_back(checked_mpi_count(static_cast<long long>(ss), "Pair header length"));
        pair_displs_.push_back(addr);
        for (int t = 0; t < s_; ++t) {
            for (int u = 0; u < s_; ++u) {
                const std::span<const size_t> sp = pair_.stream(/*reader=*/0, u, t);
                pair_hdr_send_[static_cast<size_t>(t) * static_cast<size_t>(s_) + static_cast<size_t>(u)] = sp.size();
                if (sp.empty()) {
                    continue;
                }
                MPI_Get_address(sp.data(), &addr);
                pair_blocklens_.push_back(
                    checked_mpi_count(static_cast<long long>(sp.size()), "Pair sub-stream length"));
                pair_displs_.push_back(addr);
            }
        }
        MPI_Datatype dt = MPI_DATATYPE_NULL;
        MPI_Type_create_hindexed(static_cast<int>(pair_blocklens_.size()),
                                 pair_blocklens_.data(),
                                 pair_displs_.data(),
                                 MPI_UINT64_T,
                                 &dt);
        MPI_Type_commit(&dt);
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Isend(MPI_BOTTOM, 1, dt, peer, kPairExchangeTag, parent_, &req);

        MPI_Message msg = MPI_MESSAGE_NULL;
        MPI_Status status;
        MPI_Mprobe(peer, kPairExchangeTag, parent_, &msg, &status);
        int total = 0;
        MPI_Get_count(&status, MPI_UINT64_T, &total);
        // Classic counts: a rank pair's gate payload above INT_MAX words is out of range and aborts here.
        if (total == MPI_UNDEFINED || static_cast<size_t>(total) < ss) {
            throw std::runtime_error(std::format("pair_exchange: peer rank {} sent {} words, below the {}-word header "
                                                 "or beyond the MPI count range",
                                                 peer,
                                                 total,
                                                 ss));
        }
        grow_to(pair_recv_, static_cast<size_t>(total)); // never shrunk: views into it outlive this gate
        MPI_Mrecv(pair_recv_.data(), total, MPI_UINT64_T, &msg, MPI_STATUS_IGNORE);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Type_free(&dt);

        // S*S serial adds here so that each partition's gather past B2 walks only its own S counts.
        size_t off = ss;
        for (int t = 0; t < s_; ++t) {
            pair_row_start_[static_cast<size_t>(t)] = off;
            for (int u = 0; u < s_; ++u) {
                off += pair_recv_[static_cast<size_t>(t) * static_cast<size_t>(s_) + static_cast<size_t>(u)];
            }
        }
        if (off != static_cast<size_t>(total)) {
            throw std::runtime_error(
                std::format("pair_exchange: peer rank {}'s header sums to {} words but its message holds {}",
                            peer,
                            off,
                            total));
        }
    }

    //! @brief Past B2: partition t's views into pair_recv_, one per source partition ascending.
    auto pair_views_(int local_partition) -> PairRecv {
        const size_t t = static_cast<size_t>(local_partition);
        auto row = pair_.from_row(local_partition);
        const size_t *counts = pair_recv_.data() + t * static_cast<size_t>(s_);
        size_t off = pair_row_start_[t];
        for (int u = 0; u < s_; ++u) {
            row[static_cast<size_t>(u)] = std::span<const size_t>(pair_recv_.data() + off, counts[u]);
            off += counts[u];
        }
        return PairRecv{row};
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
    // Sized from (R, S), which MPI_Comm_size only answers inside the constructor body.
    std::optional<HybridStaging> staging_;
    double red_f64_ = 0.0;
    uint64_t red_u64_ = 0;
    std::vector<double> red_vec_;
    // alltoallv's wire plan, partition 0 only: written in B1->B2, read in B3->B4. See derived_wire_plan.
    PeerPlan wire_plan_;

    // pair_exchange. The descriptor table has per-partition rows (see PairSlots); the rest is partition
    // 0's alone, written between B1 and B2 and read by every partition past B2.
    PairSlots pair_;
    std::vector<size_t> pair_hdr_send_; // [S*S] (t, u) word counts: the outgoing message's leading block
    std::vector<int> pair_blocklens_;   // hindexed block list scratch, S*S + 1 at most
    std::vector<MPI_Aint> pair_displs_;
    std::vector<size_t> pair_recv_;      // [S*S header][payload, (t, u)-major]; HWM-sized, views alias it
    std::vector<size_t> pair_row_start_; // [S] word offset of partition t's run in pair_recv_

    PartitionBarrier barrier_;
};

} // namespace monoprop::mpi
