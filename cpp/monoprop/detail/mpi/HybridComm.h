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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <print>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/HybridStaging.h"
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
    }

    HybridComm(const HybridComm &) = delete;
    auto operator=(const HybridComm &) -> HybridComm & = delete;

    auto size() const -> int { return r_ * s_; }
    auto ranks() const -> int { return r_; }
    auto partitions() const -> int { return s_; }
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
    auto alltoall_counts_impl_(int local_partition, const int *send_counts /*[P]*/, int *recv_counts /*[P]*/) -> void {
        staging_->publish_counts_row(local_partition, send_counts);
        sync();
        if (local_partition == 0) {
            staging_->pack_count_matrix();
            staging_->exchange_count_blocks();
        }
        sync();
        // Partition t extracts its row: recv from (rank a, partition su) is contiguous per source rank a.
        const int t = local_partition;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                recv_counts[a * s_ + su] = staging_->count_from(a, t, su);
            }
        }
        // No trailing barrier: past the last sync only the count message is read, and partition 0 cannot
        // rewrite it until a future call's second barrier — unreachable until every extractor here has
        // arrived. send_counts is consumed before the first sync, so peers may free it on return.
    }

    // Flat variable all-to-all over caller-owned buffers; see AlltoallvArgs for the conventions.
    auto alltoallv_impl_(int local_partition, const AlltoallvArgs &args, MPI_Datatype dt) -> void {
        Slot &me = slots_[static_cast<size_t>(local_partition)];
        me.ptr = args.send;
        me.send_displs = args.send_displs;
        staging_->publish_counts_row(local_partition, args.send_counts);
        staging_->publish_recv_rows(local_partition, args.recv_counts);
        sync(); // B1

        // B2: partition 0 sizes/reallocates staging; must finish before any partition packs into it.
        if (local_partition == 0) {
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
            staging_->exchange_payload(dt);
        }
        sync(); // B4

        staging_->scatter_recv(local_partition, args.recv, args.recv_counts, args.recv_displs, args.elem);
        // No trailing barrier: the recv bases are rewritten only in a later verb's B1→B2 window.
    }

    // Fused count-resolve + payload alltoallv: folds the standalone count exchange into this verb's
    // B1→B2 window (4 syncs instead of 6). recv_counts / recv_displs and `recv` (resized) are outputs.
    // Bit-identical to alltoall_counts + alltoallv.
    template <typename T>
    auto alltoallv_resolve_impl_(int local_partition, const AlltoallvResolveArgs<T> &args, MPI_Datatype dt) -> void {
        // Typed verb: element bytes are sizeof(T) by construction, so they are derived rather than passed.
        constexpr size_t elem = sizeof(T);
        Slot &me = slots_[static_cast<size_t>(local_partition)];
        // Typed here but byte-addressed in the slot: pack_send copies by (displ, count) in elements and
        // never reconstructs T, so the slot stays type-erased for the untyped alltoallv_impl_ above.
        me.ptr = reinterpret_cast<const std::byte *>(args.send);
        me.send_displs = args.send_displs;
        // Count row only: the recv counts do not exist until the count round in B1→B2.
        staging_->publish_counts_row(local_partition, args.send_counts);
        sync(); // B1

        if (local_partition == 0) {
            staging_->pack_count_matrix();
            staging_->exchange_count_blocks();
            staging_->size_send(elem);
            staging_->fill_recv_col([this](int a, int t) { return staging_->block_sum(a, t); });
            staging_->size_recv(elem);
        }
        sync(); // B2

        const int t = local_partition;
        long long total = 0;
        for (int a = 0; a < r_; ++a) {
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int c = staging_->count_from(a, t, su);
                args.recv_counts[g] = c;
                args.recv_displs[g] = checked_mpi_count(total, "Recv displacement");
                total += c;
            }
        }
        args.recv.resize(static_cast<size_t>(checked_mpi_count(total, "Total recv count")));

        pack_send_(local_partition, elem);
        sync(); // B3

        if (local_partition == 0) {
            staging_->exchange_payload(dt);
        }
        sync(); // B4

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

    PartitionBarrier barrier_;
};

} // namespace monoprop::mpi
