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
#include <cstring>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "monoprop/detail/mpi/CheckedCount.h"
#include "monoprop/detail/mpi/Comm.h"
#include "monoprop/detail/mpi/Pairwise.h"

// The staging half of HybridComm (see HybridComm.h for the barrier discipline the phase names refer to).
// Everything here is partition 0's, written and read inside a barriered window, EXCEPT the two published
// tables whose rows each partition owns -- see publish_counts_row / publish_recv_rows.

namespace monoprop::mpi {

//! @brief Grow-only resize: staging is high-water-mark sized and never shrinks within a run.
template <typename V>
auto grow_to(V &v, size_t need) -> void {
    if (v.size() < need) {
        v.resize(need);
    }
}

/*!
 * @brief HybridComm's payload staging: the count matrices, the per-rank wire layout, the two grow-only
 * payload buffers, and the sweeps that size, pack, move and scatter them.
 *
 * Wire block order is destination major, source minor, so one destination's region of a rank's message
 * is contiguous and one source partition's blocks within it are a prefix. Counts and displacements are
 * in ELEMENTS; `elem` scales them to bytes, and the buffers carry no element type to scale by.
 *
 * Held by one HybridComm and driven only from inside its barriered windows: this class takes no
 * barrier and asserts no ordering of its own.
 */
class HybridStaging {
public:
    HybridStaging(MPI_Comm parent, int mpi_rank, int ranks, int partitions)
        : parent_(parent),
          mpi_rank_(mpi_rank),
          r_(ranks),
          s_(partitions) {
        // Size all (R, S)-fixed scratch once so per-call paths never allocate; staging grows on demand.
        const size_t p = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        counts_send_.resize(p * static_cast<size_t>(s_));
        counts_recv_.resize(p * static_cast<size_t>(s_));
        mpi_send_counts_.resize(static_cast<size_t>(r_));
        mpi_recv_counts_.resize(static_cast<size_t>(r_));
        mpi_send_displs_.resize(static_cast<size_t>(r_));
        mpi_recv_displs_.resize(static_cast<size_t>(r_));
        pack_off_.resize(static_cast<size_t>(s_) * p); // S * R * S elements, indexed (u, g)
        base_send_.resize(p);
        base_recv_.resize(p);
        col_sum_.resize(p);
        recv_col_.resize(p);
        // Rounded up to whole 64-byte lines: each row has its own writer, so a shared line false-shares.
        counts_stride_ = round_up_(p, kIntsPerLine);
        rows_stride_ = round_up_(static_cast<size_t>(r_), kLongsPerLine);
        // One spare line each: the allocator guarantees alignof(T), not 64, so row 0 must be realigned.
        counts_matrix_store_.assign(static_cast<size_t>(s_) * counts_stride_ + kIntsPerLine, 0);
        counts_matrix_ = align_to_line_(counts_matrix_store_.data());
        rows_store_.assign(static_cast<size_t>(s_) * rows_stride_ + kLongsPerLine, 0LL);
        rows_ = align_to_line_(rows_store_.data());
        assert(reinterpret_cast<uintptr_t>(counts_matrix_) % kLineBytes == 0);
        assert(reinterpret_cast<uintptr_t>(rows_) % kLineBytes == 0);
        assert(counts_matrix_ + static_cast<size_t>(s_) * counts_stride_
               <= counts_matrix_store_.data() + counts_matrix_store_.size());
        assert(rows_ + static_cast<size_t>(s_) * rows_stride_ <= rows_store_.data() + rows_store_.size());
    }

    //! @brief Bytes these tables and buffers hold; see HybridComm::staging_bytes for what it is for.
    [[nodiscard]] auto bytes() const -> size_t {
        const auto cap = [](const auto &v) {
            return v.capacity() * sizeof(typename std::decay_t<decltype(v)>::value_type);
        };
        return cap(stage_send_) + cap(stage_recv_) + cap(counts_send_) + cap(counts_recv_) + cap(mpi_send_counts_)
               + cap(mpi_send_displs_) + cap(mpi_recv_counts_) + cap(mpi_recv_displs_) + cap(pack_off_)
               + cap(base_send_) + cap(base_recv_) + cap(col_sum_) + cap(recv_col_) + cap(counts_matrix_store_)
               + cap(rows_store_) + cap(reqs_) + cap(count_reqs_) + cap(peers_);
    }

    /*!
     * @brief Publish partition @a u's [P] send-count row (Phase P0).
     *
     * Partition u writes only its own row, so this pre-barrier write needs no barrier of its own. The
     * one cross-partition reader is partition 0 inside B1->B2, and no peer reaches verb k+1's P0 without
     * passing verb k's B2, so the rewrite always follows the read.
     */
    auto publish_counts_row(int u, const int *send_counts) -> void {
        std::memcpy(counts_row(u), send_counts, static_cast<size_t>(r_) * static_cast<size_t>(s_) * sizeof(int));
    }

    /*!
     * @brief Publish partition @a u's per-rank recv totals (Phase P0), same ownership rule as above.
     * `long long` because it sums S int counts.
     *
     * Masked through @a plan, symmetric with its one reader (fill_recv_col): a non-peer's row is zero by
     * definition, and a count left there would size staging for a block no receive is posted for. Every
     * entry is written, not just the peers', because derived_wire_plan reads the whole row.
     */
    auto publish_recv_rows(int u, const int *recv_counts, PeerPlan plan) -> void {
        long long *rr = row_recv_(u);
        std::fill_n(rr, r_, 0LL);
        const int f = plan.count(r_);
        for (int k = 0; k < f; ++k) {
            const int a = plan.peer(mpi_rank_, k);
            long long sum = 0;
            for (int su = 0; su < s_; ++su) {
                sum += recv_counts[a * s_ + su];
            }
            rr[a] = sum;
        }
    }

    /*!
     * @brief Materialise @a plan's peer ranks; every sweep below walks this set.
     *
     * Written by partition 0 in the B1->B2 window, like the recv bases, so every reader past B2 sees it.
     * The sizing sweeps index it S times each, which is why it is a list rather than a predicate.
     */
    auto fill_peers(PeerPlan plan) -> void {
        const int f = plan.count(r_);
        peers_.resize(static_cast<size_t>(f));
        for (int k = 0; k < f; ++k) {
            peers_[static_cast<size_t>(k)] = plan.peer(mpi_rank_, k);
        }
    }

    //! @brief The peer ranks fill_peers resolved, for the callers that extract a row per peer.
    [[nodiscard]] auto peers() const -> const std::vector<int> & { return peers_; }

    /*!
     * @brief The rank-level peer set read off the recv rows every partition published before B1 -- the
     * first point with a view wider than one partition's row.
     *
     * Under fanout-1 routing a layer's traffic is all on ONE rank, so this resolves to a shift; with
     * nothing occupied it resolves to the self peer, whose legs are then all zero, and that keeps the
     * collective-vs-pairwise branch a function of the caller's `bits` alone rather than of a rank's data
     * (a data-dependent branch straddles and hangs). A set wider than one rank contradicts the caller's
     * fanout claim, so it answers dense and nothing is dropped.
     */
    [[nodiscard]] auto derived_wire_plan(int bits) -> PeerPlan {
        int found = -1;
        for (int u = 0; u < s_; ++u) {
            const long long *rr = row_recv_(u);
            for (int a = 0; a < r_; ++a) {
                if (rr[a] != 0 && a != found) {
                    if (found >= 0) {
                        assert(false && "fanout claimed 1, but this rank's layer spans several peer ranks");
                        return PeerPlan{};
                    }
                    found = a;
                }
            }
        }
        // The plan is a boolean, so every rank bit is a linear bit and the mask is the rank index.
        return PeerPlan{.sparse = bits > 0, .shift = found < 0 ? 0 : (mpi_rank_ ^ found)};
    }

    //! @brief Do the published send rows put anything outside @a plan's peers? If so, narrowing to it
    //! would drop those blocks in silence -- the failure mode a wrong-but-agreed shift produces.
    [[nodiscard]] auto narrowing_is_lossless(PeerPlan plan) const -> bool {
        for (int su = 0; su < s_; ++su) {
            const int *row = counts_matrix_ + static_cast<size_t>(su) * counts_stride_;
            for (int g = 0; g < r_ * s_; ++g) {
                if (row[g] != 0 && !plan.contains(mpi_rank_, g / s_)) {
                    return false;
                }
            }
        }
        return true;
    }

    //! @brief What rank @a a's count block says (@a su -> @a t): the extraction the callers hand back.
    [[nodiscard]] auto count_from(int a, int t, int su) const -> int { return counts_recv_[counts_idx_(a, t, su)]; }

    /*!
     * @brief Transpose the published count rows into the count message, dest-major then source-minor.
     *
     * Source partition outer, so the peer-owned side streams. Only the peers' blocks are written, and
     * only they are ever read back, so no pre-zeroing.
     */
    auto pack_count_matrix([[maybe_unused]] PeerPlan plan) -> void {
        assert(narrowing_is_lossless(plan)); // a wrong shift every rank agrees on drops blocks silently
        for (int su = 0; su < s_; ++su) {
            const int *row = counts_row(su);
            for (const int b : peers_) {
                for (int t = 0; t < s_; ++t) {
                    counts_send_[counts_idx_(b, t, su)] = row[b * s_ + t];
                }
            }
        }
    }

    /*!
     * @brief Post the S*S-int count blocks: one MPI_Alltoall when dense, else a pair per peer.
     *
     * The sparse arm POSTS ONLY, so both count matrices must stay put until wait_count_blocks; the dense
     * MPI_Alltoall is blocking and has completed on return, which is why plan.dense() leaves nothing
     * live. The requests go in their own vector, never the payload one: see the member declaration.
     */
    auto post_count_blocks(PeerPlan plan) -> void {
        assert(count_posted_ == 0); // an un-drained round would be waited on twice
        const int block = s_ * s_;
        if (plan.dense()) {
            MPI_Alltoall(counts_send_.data(), block, MPI_INT, counts_recv_.data(), block, MPI_INT, parent_);
            return;
        }
        const PeerLayout blocks{.block = block};
        count_posted_ = sparse_pairwise(plan,
                                        mpi_rank_,
                                        r_,
                                        parent_,
                                        kHybridCountTag,
                                        MPI_INT,
                                        sizeof(int),
                                        reinterpret_cast<const std::byte *>(counts_send_.data()),
                                        blocks,
                                        reinterpret_cast<std::byte *>(counts_recv_.data()),
                                        blocks,
                                        count_reqs_);
    }

    //! @brief Drain post_count_blocks. A no-op on the dense arm, and on a plan whose only peer is this
    //! rank itself -- a count block is a fixed S*S ints, so no other peer can be skipped for a zero count.
    auto wait_count_blocks() -> void {
        if (count_posted_ != 0) {
            MPI_Waitall(count_posted_, count_reqs_.data(), MPI_STATUSES_IGNORE);
            count_posted_ = 0;
        }
    }

    //! @brief Post and drain in one step, for callers with no work to overlap.
    auto exchange_count_blocks(PeerPlan plan) -> void {
        post_count_blocks(plan);
        wait_count_blocks();
    }

    /*!
     * @brief Size the send side, in two sweeps of the published count rows (partition 0, B1->B2).
     *
     * Pass A takes the column sums over source partitions and lays the per-rank wire out from them;
     * pass B takes the exclusive prefix over source partitions, which is each partition's block start.
     */
    auto size_send(size_t elem) -> void {
        // Pass A: u outer so both sides sweep in address order.
        zero_peer_slots_(col_sum_);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row(u);
            for (const int b : peers_) {
                const size_t base = static_cast<size_t>(b) * static_cast<size_t>(s_);
                for (int t = 0; t < s_; ++t) {
                    col_sum_[base + static_cast<size_t>(t)] += row[base + static_cast<size_t>(t)];
                }
            }
        }
        const size_t total_send = layout_peers_(col_sum_, mpi_send_counts_, mpi_send_displs_, base_send_, kSendLabels);
        // Pass B: col_sum_ is free to be reused as the running prefix here.
        zero_peer_slots_(col_sum_);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row(u);
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
        // Grow-only, no zero-fill: pack_send's blocks tile [0, total_send) exactly.
        grow_to(stage_send_, total_send * elem);
    }

    /*!
     * @brief Fill the recv column: what local partition t receives from rank a.
     *
     * @a value reads it from the rows published in Phase P0 (alltoallv) or from the count blocks just
     * exchanged (the fused resolve), which is the only difference between the two verbs' recv sides.
     */
    template <typename Value>
    auto fill_recv_col(Value &&value) -> void {
        zero_peer_slots_(recv_col_);
        for (const int a : peers_) {
            for (int t = 0; t < s_; ++t) {
                recv_col_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)] = value(a, t);
            }
        }
    }

    //! @brief The Phase P0 rows, as fill_recv_col's `value`.
    [[nodiscard]] auto published_recv(int a, int t) -> long long { return row_recv_(t)[a]; }

    //! @brief Rank @a a's count block for partition @a t, summed over source partitions; the other one.
    [[nodiscard]] auto block_sum(int a, int t) const -> long long {
        const int *blk = counts_recv_.data() + counts_idx_(a, t, 0);
        long long sum = 0;
        for (int su = 0; su < s_; ++su) {
            sum += blk[su];
        }
        return sum;
    }

    //! @brief Size the recv side from the recv column. Only the per-(rank, partition) base: the scatter
    //! re-derives the per-source offsets as it walks (a, su).
    auto size_recv(size_t elem) -> void {
        const size_t total_recv = layout_peers_(recv_col_, mpi_recv_counts_, mpi_recv_displs_, base_recv_, kRecvLabels);
        grow_to(stage_recv_, total_recv * elem);
    }

    /*!
     * @brief Move the staged payload: one MPI_Alltoallv when dense, else a pair per peer over the same
     * per-rank counts and displacements. Blocking either way.
     *
     * A non-peer's count is zero, so the sparse arm drops nothing. @a elem is @a dt's extent: needed to
     * reach a block, and the same one the two sizing sweeps were given.
     */
    auto exchange_payload(MPI_Datatype dt, size_t elem, PeerPlan plan) -> void {
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
        const int posted =
            sparse_pairwise(plan,
                            mpi_rank_,
                            r_,
                            parent_,
                            kHybridPayloadTag,
                            dt,
                            elem,
                            stage_send_.data(),
                            PeerLayout{.counts = mpi_send_counts_.data(), .displs = mpi_send_displs_.data()},
                            stage_recv_.data(),
                            PeerLayout{.counts = mpi_recv_counts_.data(), .displs = mpi_recv_displs_.data()},
                            reqs_);
        MPI_Waitall(posted, reqs_.data(), MPI_STATUSES_IGNORE);
    }

    /*!
     * @brief Partition @a u copies its own blocks into the send staging (B2->B3).
     *
     * Own slot only -- no peer's published send buffer is read here, which is what lets every partition
     * pack concurrently.
     */
    auto pack_send(int u, size_t elem, const std::byte *src, const int *send_displs) -> void {
        const int *my_send_counts = counts_row(u);
        const size_t *off = pack_off_.data() + pack_idx_(u, 0);
        for (const int b : peers_) {
            const int base = b * s_;
            for (int t = 0; t < s_; ++t) {
                const int g = base + t;
                const int cnt = my_send_counts[g];
                if (cnt != 0) {
                    std::memcpy(stage_send_.data() + off[g] * elem,
                                src + static_cast<size_t>(send_displs[g]) * elem,
                                static_cast<size_t>(cnt) * elem);
                }
            }
        }
    }

    /*!
     * @brief Scatter each global source's contiguous run out of the recv staging to recv_displs[g].
     *
     * All legs, the self-rank one included, go through staging. Walks (a, su) in accumulation order, so
     * `cur` re-derives the block starts from the recv bases and this partition's own counts.
     */
    auto scatter_recv(int t, std::byte *dst, const int *recv_counts, const int *recv_displs, size_t elem) -> void {
        for (const int a : peers_) {
            size_t cur = base_recv_[static_cast<size_t>(a) * static_cast<size_t>(s_) + static_cast<size_t>(t)];
            for (int su = 0; su < s_; ++su) {
                const int g = a * s_ + su;
                const int cnt = recv_counts[g];
                if (cnt != 0) {
                    std::memcpy(dst + static_cast<size_t>(recv_displs[g]) * elem,
                                stage_recv_.data() + cur * elem,
                                static_cast<size_t>(cnt) * elem);
                }
                cur += static_cast<size_t>(cnt);
            }
        }
    }

private:
    // checked_mpi_count's message prefixes, so the shared layout sweep names the side it is laying out.
    struct CountLabels {
        const char *per_rank;
        const char *displ;
        const char *total;
    };
    static constexpr CountLabels kSendLabels{"Per-rank send count", "Send displacement", "Total send count"};
    static constexpr CountLabels kRecvLabels{"Per-rank recv count", "Recv displacement", "Total recv count"};

    /*!
     * @brief The per-rank wire layout both sides share: sum @a col over each rank's S slots into
     * @a counts, take the ascending prefix into @a displs, and lay each slot's block start into
     * @a base. Returns the total, in elements.
     */
    auto layout_peers_(const std::vector<long long> &col,
                       std::vector<int> &counts,
                       std::vector<int> &displs,
                       std::vector<size_t> &base,
                       CountLabels what) -> size_t {
        // The [R] arrays stay fully zeroed: MPI_Alltoallv reads every entry on the dense arm, and the
        // zeros are what make the peer-only prefix below equal the full 0..R one at the peers' positions.
        std::ranges::fill(counts, 0);
        std::ranges::fill(displs, 0);
        for (const int b : peers_) {
            const long long *slots = col.data() + static_cast<size_t>(b) * static_cast<size_t>(s_);
            long long sum = 0;
            for (int t = 0; t < s_; ++t) {
                sum += slots[t];
            }
            counts[static_cast<size_t>(b)] = checked_mpi_count(sum, what.per_rank);
        }
        // peers_ is ascending (dense is 0..R-1, sparse a singleton), so a prefix over it takes the same
        // value at every peer as a prefix over all R: a non-peer contributes zero.
        long long running = 0;
        for (const int b : peers_) {
            displs[static_cast<size_t>(b)] = checked_mpi_count(running, what.displ);
            running += counts[static_cast<size_t>(b)];
        }
        const auto total = static_cast<size_t>(checked_mpi_count(running, what.total));
        for (const int b : peers_) {
            size_t cur = static_cast<size_t>(displs[static_cast<size_t>(b)]);
            for (int t = 0; t < s_; ++t) {
                const size_t g = static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t);
                base[g] = cur;
                cur += static_cast<size_t>(col[g]);
            }
        }
        return total;
    }

    /*!
     * @brief Zero only the peer ranks' slots of a [P] table.
     *
     * Every sweep that writes one of these tables and every sweep that reads it walks the same peer set,
     * so the rest is never looked at -- and these run SERIALLY on partition 0 while S-1 partitions park,
     * so the width is the cost.
     */
    auto zero_peer_slots_(std::vector<long long> &table) -> void {
        for (const int b : peers_) {
            std::fill_n(table.begin() + (static_cast<std::ptrdiff_t>(b) * s_), s_, 0LL);
        }
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
        return p + pad / sizeof(T);
    }

    // (rank, dest partition, source partition) in the R*S*S count matrices: the count message's wire layout.
    [[nodiscard]] auto counts_idx_(int b, int t, int su) const -> size_t {
        return (static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t)) * static_cast<size_t>(s_)
               + static_cast<size_t>(su);
    }

    // [S x P] payload offset table: row u is source partition u's staging starts, one per destination g.
    [[nodiscard]] auto pack_idx_(int u, int g) const -> size_t {
        return static_cast<size_t>(u) * static_cast<size_t>(r_) * static_cast<size_t>(s_) + static_cast<size_t>(g);
    }

    [[nodiscard]] auto counts_row(int u) -> int * { return counts_matrix_ + static_cast<size_t>(u) * counts_stride_; }
    [[nodiscard]] auto row_recv_(int u) -> long long * { return rows_ + static_cast<size_t>(u) * rows_stride_; }

    MPI_Comm parent_;
    int mpi_rank_;
    int r_;
    int s_;

    // S*S per rank, the count message both ways.
    std::vector<int> counts_send_;
    std::vector<int> counts_recv_;
    // Per-rank [R] counts/displs for the aggregated payload alltoallv.
    std::vector<int> mpi_send_counts_;
    std::vector<int> mpi_send_displs_;
    std::vector<int> mpi_recv_counts_;
    std::vector<int> mpi_recv_displs_;
    // [S x P] payload block starts in stage_send_, row u per global destination g. No scatter-side
    // table: see size_recv.
    std::vector<size_t> pack_off_;
    // [P] staging bases: base_send_[b*S+t] starts partition (b,t)'s region of the message to rank b,
    // base_recv_[a*S+t] starts local partition t's region of the message from rank a.
    std::vector<size_t> base_send_;
    std::vector<size_t> base_recv_;
    std::vector<long long> col_sum_;
    std::vector<long long> recv_col_;
    // [S x counts_stride_] send-count matrix, row u owned by partition u; padded so no two rows share a
    // 64-byte line. Lifetime rule: publish_counts_row.
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
    // Point-to-point request scratch for the sparse payload round; grown on demand.
    std::vector<MPI_Request> reqs_;
    // The count round's own scratch, separate from reqs_ by construction and not merely by the current
    // ordering: it stays live across B2 and B3, and sparse_pairwise's resize would move the buffer MPI
    // holds pointers into the moment a payload post ever preceded the count wait.
    std::vector<MPI_Request> count_reqs_;
    int count_posted_ = 0; // live requests in count_reqs_; always 0 on the dense (blocking) arm
    // This verb's peer ranks; see fill_peers.
    std::vector<int> peers_;
};

} // namespace monoprop::mpi
