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
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
// For the per-row join key only: routing owns the fingerprint, and a second definition of it here would
// be a second thing to keep in step with BucketJoin's tag.
#include "monoprop/detail/mpi/Routing.h"

namespace monoprop::detail {

class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Operator-term store: entropy-packed position-list rows and nothing else. Row layout: slot 0 = popcount c
// (or kOverflowMarker if c > inline_width_), slots 1..c = ascending set-bit positions; stride_ is fixed
// for the container's life so row offsets stay stable. inline_width_ is a free parameter -- any width is
// correct, over-long rows spill losslessly to overflow. Single-writer: one partition, one thread;
// parallelism is cross-partition.
//
// There is deliberately no key -> row index here. Propagation never looks a term up by value against the
// whole store: a partner M ^ G, if tracked, anticommutes with G and so is inside the gate's own
// anticommuting set, which layer_build/BucketJoin.h joins per gate. The few out-of-gate lookups build a
// transient TermLookup over the rows they need.
//
// What IS kept per row is its 4-byte join key -- the top 32 bits of the mixed routing fingerprint, which
// is exactly BucketJoin's compare tag. Every gate stages its whole anticommuting set into that join, and
// folding the key there meant reading each row's positions once per gate; holding it costs 4 B/term once
// and turns pass 1 into a sequential read of keys_. It is NOT the 64-bit fingerprint: that is
// GF(2)-linear, the mixed key is not, so a partner's key still comes from the partner's own positions.
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = Monomial<NumModes>;
    using key_type = Monomial<NumModes>;
    using mapped_type = size_t;

    using PosT = std::
        conditional_t<(2 * NumModes <= 256), uint8_t, std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;

    static constexpr size_t kDefaultInlinePositions = 11;
    // A weight-w Pauli needs 2w positions; 32 covers the common case inline at the supported Pauli
    // cutoffs (2*cutoff <= 32 for cutoff <= 16).
    static constexpr size_t kMaxInlinePositions = 32;
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max();

    static_assert((2 * NumModes) - 1 <= std::numeric_limits<PosT>::max(),
                  "OperatorIndex PosT too narrow for 2*NumModes positions");
    static_assert(kMaxInlinePositions < std::numeric_limits<PosT>::max(),
                  "kOverflowMarker sentinel must not collide with a valid popcount");

    // Valid term indices are < kIndexCeiling (check_index_fits throws at the ceiling), so the all-ones
    // TermIndex is free as a sentinel wherever a row index is optional.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    // "Absent" result of a lookup; same value as detail::kMissingIndex (not included here — the operator
    // store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

    //! Row i's join key. Maintained by every writer, so a gate stages its join without reading a row.
    [[nodiscard]] auto key(size_t i) const -> uint32_t { return keys_[i]; }

    //! The compare tag of a 64-bit fingerprint: what a row's key is, and what BucketJoin tags a query with.
    [[nodiscard]] static auto join_tag(uint64_t fp) noexcept -> uint32_t {
        return static_cast<uint32_t>(routing::mix64(fp) >> 32U);
    }

    //! The join key of a term given as ascending positions, and of a dense one; the same map either way.
    [[nodiscard]] static auto key_of_positions(const PosT *pos, size_t k) noexcept -> uint32_t {
        return join_tag(routing::fingerprint_positions(labels(), pos, k));
    }
    [[nodiscard]] static auto key_of(const value_type &mono) noexcept -> uint32_t {
        return join_tag(routing::linear_hash<2 * NumModes>(mono));
    }

    explicit OperatorIndex(size_t inline_width = kDefaultInlinePositions)
        : inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)),
          stride_(1 + inline_width_) {}
    OperatorIndex(const OperatorIndex &) = delete;
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    OperatorIndex(OperatorIndex &&) = delete;
    OperatorIndex &operator=(OperatorIndex &&) = delete;

    // Called only on an idle store, so it needs no synchronization.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(inline_width_);
        out->rows_ = rows_;
        out->keys_ = keys_;
        out->size_ = size_;
        out->overflow_ = overflow_;
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    // Rows that exceeded inline_width_ and spilled; observable so a test can compare the two insert paths.
    [[nodiscard]] auto overflow_size() const -> size_t { return overflow_.size(); }

    auto reserve(size_t n) -> void { reserve_rows(n); }
    // Returns the pre-growth size (the caller's insert base). Growth is geometric (1.5×), never
    // exact-fit: an exact fit would realloc the whole operator every layer. Throws at the TermIndex
    // ceiling: every structure indexed by row (inverted index, graph endpoints) is TermIndex-wide.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        if (n != 0) {
            check_index_fits(base + n - 1);
        }
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve_rows(std::max(base + n, cap + (cap / 2) + 1));
        }
        // Default-init grow, not a zeroing resize: every freshly grown row is overwritten by set()
        // before any read, so a tail zero-fill would be wasted bandwidth.
        rows_.resize((base + n) * stride_);
        keys_.resize(base + n);
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so the row header is never pre-read
    // (freshly grown headers are indeterminate); a stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        const size_t c = mono.count();
        keys_[i] = key_of(mono);
        PosT *row = &rows_[i * stride_];
        if (c > inline_width_) {
            row[0] = kOverflowMarker;
            overflow_[i] = mono;
            return;
        }
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
        row[0] = static_cast<PosT>(c);
        PosT *out = row + 1;
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            *out++ = static_cast<PosT>(b);
        }
    }

    // set() from the row's own form: a row is an ascending position list. Same postcondition as set(),
    // including the dropped stale overflow entry.
    //
    // Precondition: `pos` strictly ascending, every entry < 2*NumModes. A violation is silent in release
    // -- an unsorted row simply never matches, and an out-of-range one decodes to a different term.
    auto set_positions(size_t i, std::span<const PosT> pos) -> void {
        const size_t count = pos.size();
        assert((count == 0 || static_cast<size_t>(pos[count - 1]) < 2 * NumModes) && "row position out of range");
        keys_[i] = key_of_positions(pos.data(), count);
        PosT *row = &rows_[i * stride_];
        if (count > inline_width_) {
            // The spill path has no position array, so build the dense form -- only here.
            row[0] = kOverflowMarker;
            value_type mono;
            for (size_t j = 0; j < count; ++j) {
                mono.set(pos[j]);
            }
            overflow_[i] = mono;
            return;
        }
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
        row[0] = static_cast<PosT>(count);
        std::copy_n(pos.data(), count, row + 1);
    }

    [[nodiscard]] auto row(size_t i) const -> value_type {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            return overflow_.at(i);
        }
        value_type mono;
        const PosT *pos = &rows_[(i * stride_) + 1];
        for (size_t j = 0; j < c; ++j) {
            mono.set(pos[j]);
        }
        return mono;
    }
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        const PosT *pos = &rows_[(i * stride_) + 1];
        for (size_t j = 0; j < c; ++j) {
            fn(static_cast<size_t>(pos[j]));
        }
    }
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        if (const PosT c = rows_[i * stride_]; c != kOverflowMarker) {
            return c;
        }
        return overflow_.at(i).count();
    }
    /*! @brief The row's stored ascending positions, empty for a spilled row. Invalidated by any insert. */
    struct RowPositions {
        std::span<const PosT> pos;
        //! A spilled row has no position array at all, which an empty inline row still does.
        [[nodiscard]] auto inlined() const -> bool { return pos.data() != nullptr; }
    };
    [[nodiscard]] auto row_positions(size_t i) const -> RowPositions {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            return {};
        }
        return {std::span<const PosT>(&rows_[(i * stride_) + 1], static_cast<size_t>(c))};
    }
    // The rows themselves. The join keys are reported separately (row_keys_bytes) so the breakdown can
    // price them on their own; neither figure includes the other.
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = rows_.capacity() * sizeof(PosT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }
    [[nodiscard]] auto row_keys_bytes() const -> size_t { return keys_.capacity() * sizeof(uint32_t); }

    // Every row in index order.
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (size_t i = 0; i < size_; ++i) {
            fn(row(i), i);
        }
    }

    // Compare row i against an ascending position list without materializing the row (the popcount byte
    // first, so a mismatch usually costs one compare); a spilled row falls back to a dense compare. This
    // is the confirm behind every fingerprint match (BucketJoin::run).
    [[nodiscard]] auto row_eq_positions(size_t i, std::span<const PosT> q) const -> bool {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            key_type mono;
            for (size_t j = 0; j < q.size(); ++j) {
                mono.set(q[j]);
            }
            return overflow_.at(i) == mono;
        }
        if (q.size() != static_cast<size_t>(c)) {
            return false;
        }
        return std::equal(q.begin(), q.end(), &rows_[(i * stride_) + 1]);
    }
    // Diagnostic: the part of memory_bytes() that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        return (rows_.capacity() * sizeof(PosT)) - (std::min(rows_.capacity(), size_ * stride_) * sizeof(PosT));
    }

private:
    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity() / stride_; }
    auto reserve_rows(size_t n) -> void {
        rows_.reserve(n * stride_);
        keys_.reserve(n);
    }

    // The label table, bound through a local static so the per-term path does not re-enter routing's guard.
    static auto labels() noexcept -> const uint64_t * {
        static const uint64_t *const table = routing::linear_basis<2 * NumModes>().data();
        return table;
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                                          "-Dmonoprop_WIDE_TERM_INDEX (this partition's term count exceeded ~2^32).");
        }
    }

    DefaultInitVector<PosT> rows_ = {};
    // One join key per row, parallel to rows_. Default-init like rows_: a grown row's key is
    // indeterminate until its set() writes both.
    DefaultInitVector<uint32_t> keys_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    // Lossless side-map for rows whose popcount exceeds inline_width_.
    std::unordered_map<size_t, value_type> overflow_ = {};
};

} // namespace monoprop::detail
