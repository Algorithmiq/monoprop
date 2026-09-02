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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// The pair_exchange vocabulary and its in-rank descriptor table. Deliberately apart from PairExchange.h:
// ShmComm.h and HybridComm.h own an instance each, and the verb's header includes both of them.

namespace monoprop::mpi {

/*!
 * One partition's outgoing records for a gate: sub-stream `t` is destined for partition `t` of the peer
 * rank. Exactly S entries, empty ones included. Under flat-linear routing all but one are empty; the verb
 * does not care which.
 */
using SubStreams = std::span<const std::span<const size_t>>;

/*!
 * What one partition receives from a gate exchange. Ascending source order is part of the contract: the
 * caller's index assignment depends on it. The views alias memory the verb does not own for the caller --
 * a peer's send buffer in-rank, the rank master's receive buffer across ranks -- and are valid until this
 * participant's next pair_exchange on the same comm. PairExchange.h states the full lifetime rule.
 */
struct PairRecv {
    std::span<const std::span<const size_t>> from; //!< `from[u]`: from partition u of the peer rank, u ascending
};

/*!
 * The in-rank half of pair_exchange, shared by ShmComm and HybridComm: each of S partitions publishes
 * its S span descriptors, and after the owner's barrier every partition gathers its own column, or the
 * rank master reads the whole table. Holds no barrier and no payload: the descriptors alias the callers'
 * buffers, which is what makes the in-rank exchange copy-free.
 *
 * Descriptors are double-buffered by call parity because an in-rank gate takes ONE barrier: a partition
 * that has passed gate g's barrier may publish gate g+1 while a slower peer is still gathering gate g.
 * Parity puts the two gates in disjoint lines, and gate g+2 cannot reuse gate g's lines before the slow
 * peer has passed gate g+1's barrier -- which it reaches only after its gather of gate g returned. Every
 * participant makes every call, so the per-partition counters agree at each barrier.
 *
 * Each partition writes only its own descriptor rows and its own from-row, each padded to whole 64-byte
 * lines, so the pre-barrier writes need no ordering beyond the barrier's release.
 */
class PairSlots {
public:
    explicit PairSlots(int n)
        : n_(n),
          stride_(round_up_(static_cast<size_t>(n), kSpansPerLine)),
          // One spare line each: the allocator guarantees alignof(span), not 64, so row 0 must be realigned.
          desc_store_(2 * static_cast<size_t>(n) * stride_ + kSpansPerLine),
          from_store_(static_cast<size_t>(n) * stride_ + kSpansPerLine),
          calls_(static_cast<size_t>(n)) {
        desc_ = align_to_line_(desc_store_.data());
        from_ = align_to_line_(from_store_.data());
        assert(desc_ + 2 * static_cast<size_t>(n) * stride_ <= desc_store_.data() + desc_store_.size());
        assert(from_ + static_cast<size_t>(n) * stride_ <= from_store_.data() + from_store_.size());
    }

    PairSlots(const PairSlots &) = delete;
    auto operator=(const PairSlots &) -> PairSlots & = delete;

    //! Partition `u` stores its S descriptors for this call, before the owner's barrier. Own rows only.
    auto publish(int u, SubStreams send) -> void {
        assert(static_cast<int>(send.size()) == n_);
        ++calls_[static_cast<size_t>(u)].calls;
        std::span<const size_t> *row = desc_row_(parity_(u), u);
        for (int t = 0; t < n_; ++t) {
            row[t] = send[static_cast<size_t>(t)];
        }
    }

    //! After the barrier: partition `t` gathers descriptor (u -> t) of every u, ascending, into its from-row.
    auto gather_in_rank(int t) -> PairRecv {
        const unsigned par = parity_(t);
        auto row = from_row(t);
        for (int u = 0; u < n_; ++u) {
            row[static_cast<size_t>(u)] = desc_row_(par, u)[t];
        }
        return PairRecv{row};
    }

    /*!
     * Descriptor (u -> t) of the current call. `reader` names the partition asking, whose own call count
     * selects the parity: it equals u's, every participant having made the same calls.
     */
    [[nodiscard]] auto stream(int reader, int u, int t) const -> std::span<const size_t> {
        return desc_row_(parity_(reader), u)[t];
    }

    //! Partition t's own from-row, for a cross-rank path to fill with views into its receive buffer.
    [[nodiscard]] auto from_row(int t) -> std::span<std::span<const size_t>> {
        return {from_ + static_cast<size_t>(t) * stride_, static_cast<size_t>(n_)};
    }

private:
    struct alignas(64) Counter {
        uint64_t calls = 0; //!< this partition's publish count; its low bit is the descriptor parity
    };

    static constexpr size_t kLineBytes = 64;
    static constexpr size_t kSpansPerLine = kLineBytes / sizeof(std::span<const size_t>);
    static_assert(kLineBytes % sizeof(std::span<const size_t>) == 0);

    static auto round_up_(size_t n, size_t m) -> size_t { return ((n + m - 1) / m) * m; }

    template <typename T>
    static auto align_to_line_(T *p) -> T * {
        const auto addr = reinterpret_cast<uintptr_t>(p);
        const size_t pad = (kLineBytes - static_cast<size_t>(addr % kLineBytes)) % kLineBytes;
        return p + pad / sizeof(T);
    }

    [[nodiscard]] auto parity_(int u) const -> unsigned {
        return static_cast<unsigned>(calls_[static_cast<size_t>(u)].calls & 1U);
    }

    // Row (parity, u): partition u's S descriptors for the calls of that parity.
    [[nodiscard]] auto desc_row_(unsigned par, int u) const -> std::span<const size_t> * {
        return desc_ + (static_cast<size_t>(par) * static_cast<size_t>(n_) + static_cast<size_t>(u)) * stride_;
    }

    int n_;
    size_t stride_; // spans per row, whole lines
    std::vector<std::span<const size_t>> desc_store_;
    std::vector<std::span<const size_t>> from_store_;
    std::span<const size_t> *desc_ = nullptr;
    std::span<const size_t> *from_ = nullptr;
    std::vector<Counter> calls_;
};

} // namespace monoprop::mpi
