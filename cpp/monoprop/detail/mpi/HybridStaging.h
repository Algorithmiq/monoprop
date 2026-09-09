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
        : parent_(parent), mpi_rank_(mpi_rank), r_(ranks), s_(partitions) {
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
               + cap(rows_store_);
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

    //! @brief Publish partition @a u's per-rank recv totals (Phase P0), same ownership rule as above.
    //! `long long` because it sums S int counts.
    auto publish_recv_rows(int u, const int *recv_counts) -> void {
        long long *rr = row_recv_(u);
        for (int a = 0; a < r_; ++a) {
            long long sum = 0;
            for (int su = 0; su < s_; ++su) {
                sum += recv_counts[a * s_ + su];
            }
            rr[a] = sum;
        }
    }

    //! @brief What rank @a a's count block says (@a su -> @a t): the extraction the callers hand back.
    [[nodiscard]] auto count_from(int a, int t, int su) const -> int { return counts_recv_[counts_idx_(a, t, su)]; }

    /*!
     * @brief Transpose the published count rows into the count message, dest-major then source-minor.
     *
     * Source partition outer, so the peer-owned side streams. Every element is written, so no pre-zeroing.
     */
    auto pack_count_matrix() -> void {
        for (int su = 0; su < s_; ++su) {
            const int *row = counts_row(su);
            for (int b = 0; b < r_; ++b) {
                for (int t = 0; t < s_; ++t) {
                    counts_send_[counts_idx_(b, t, su)] = row[b * s_ + t];
                }
            }
        }
    }

    //! @brief Move the S*S-int count blocks. One MPI_Alltoall, blocking: complete on return.
    auto exchange_count_blocks() -> void {
        MPI_Alltoall(counts_send_.data(), s_ * s_, MPI_INT, counts_recv_.data(), s_ * s_, MPI_INT, parent_);
    }

    /*!
     * @brief Size the send side, in two sweeps of the published count rows (partition 0, B1->B2).
     *
     * Pass A takes the column sums over source partitions and lays the per-rank wire out from them;
     * pass B takes the exclusive prefix over source partitions, which is each partition's block start.
     */
    auto size_send(size_t elem) -> void {
        const size_t p = static_cast<size_t>(r_) * static_cast<size_t>(s_);
        // Pass A: u outer so both sides sweep in address order.
        std::ranges::fill(col_sum_, 0LL);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row(u);
            for (size_t g = 0; g < p; ++g) {
                col_sum_[g] += row[g];
            }
        }
        const size_t total_send = layout_peers_(col_sum_, mpi_send_counts_, mpi_send_displs_, base_send_, kSendLabels);
        // Pass B: col_sum_ is free to be reused as the running prefix here.
        std::ranges::fill(col_sum_, 0LL);
        for (int u = 0; u < s_; ++u) {
            const int *row = counts_row(u);
            size_t *off = pack_off_.data() + pack_idx_(u, 0);
            for (size_t g = 0; g < p; ++g) {
                off[g] = base_send_[g] + static_cast<size_t>(col_sum_[g]);
                col_sum_[g] += row[g];
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
        for (int a = 0; a < r_; ++a) {
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

    //! @brief Move the staged payload. One MPI_Alltoallv, blocking. `dt`'s extent must be the `elem`
    //! the two sizing sweeps were given.
    auto exchange_payload(MPI_Datatype dt) -> void {
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

    /*!
     * @brief Partition @a u copies its own blocks into the send staging (B2->B3).
     *
     * Own slot only -- no peer's published send buffer is read here, which is what lets every partition
     * pack concurrently.
     */
    auto pack_send(int u, size_t elem, const std::byte *src, const int *send_displs) -> void {
        const int *my_send_counts = counts_row(u);
        const size_t *off = pack_off_.data() + pack_idx_(u, 0);
        const int p = r_ * s_;
        for (int g = 0; g < p; ++g) {
            const int cnt = my_send_counts[g];
            if (cnt != 0) {
                std::memcpy(stage_send_.data() + off[g] * elem,
                            src + static_cast<size_t>(send_displs[g]) * elem,
                            static_cast<size_t>(cnt) * elem);
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
        for (int a = 0; a < r_; ++a) {
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
        for (int b = 0; b < r_; ++b) {
            const long long *slots = col.data() + static_cast<size_t>(b) * static_cast<size_t>(s_);
            long long sum = 0;
            for (int t = 0; t < s_; ++t) {
                sum += slots[t];
            }
            counts[static_cast<size_t>(b)] = checked_mpi_count(sum, what.per_rank);
        }
        long long running = 0;
        for (int b = 0; b < r_; ++b) {
            displs[static_cast<size_t>(b)] = checked_mpi_count(running, what.displ);
            running += counts[static_cast<size_t>(b)];
        }
        const auto total = static_cast<size_t>(checked_mpi_count(running, what.total));
        for (int b = 0; b < r_; ++b) {
            size_t cur = static_cast<size_t>(displs[static_cast<size_t>(b)]);
            for (int t = 0; t < s_; ++t) {
                const size_t g = static_cast<size_t>(b) * static_cast<size_t>(s_) + static_cast<size_t>(t);
                base[g] = cur;
                cur += static_cast<size_t>(col[g]);
            }
        }
        return total;
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
};

} // namespace monoprop::mpi
