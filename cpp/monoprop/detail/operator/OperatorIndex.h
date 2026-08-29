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
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/RowHashTable.h"

namespace monoprop::detail {

// Operator-term store: entropy-packed position-list rows plus a RowHashTable keyed over them. The rows
// are this class's business and the index is not: nothing below reads a slot, and nothing in
// RowHashTable reads a row -- the two meet only through the hash and equality callables passed in.
// Row layout: slot 0 = popcount c (or kOverflowMarker if c > inline_width_), slots 1..c =
// ascending set-bit positions; stride_ is fixed for the container's life so row offsets stay stable.
// inline_width_ is a free parameter -- any width is correct, over-long rows spill losslessly to overflow.
// Single-writer: one partition, one thread; parallelism is cross-partition.
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

    // The table's row-index ceiling and its "absent" result, re-exported: a caller sizes or checks a
    // partition through this store and never sees the table underneath.
    static constexpr size_t kIndexCeiling = RowHashTable::kIndexCeiling;
    static constexpr size_t kNotFound = RowHashTable::kNotFound;

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
        out->size_ = size_;
        out->overflow_ = overflow_;
        out->reserve_index(table_.count());
        table_.for_each_slot([&out](TermIndex idx, uint32_t h) { out->table_.insert_distinct(idx, h); });
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    // Rows that exceeded inline_width_ and spilled; observable so a test can compare the two insert paths.
    [[nodiscard]] auto overflow_size() const -> size_t { return overflow_.size(); }

    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    // Returns the pre-growth size (the caller's insert base). Growth is geometric (1.5×), never
    // exact-fit: an exact fit would realloc the whole operator every layer.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        RowHashTable::check_append_fits(base, n);
        if (capacity() < base + n) {
            reserve_rows(geometric_row_capacity(base, n, capacity()));
        }
        // Default-init grow, not a zeroing resize: every freshly grown row is overwritten by set()
        // before any read, so a tail zero-fill would be wasted bandwidth.
        rows_.resize((base + n) * stride_);
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so the row header is never pre-read
    // (freshly grown headers are indeterminate); a stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        const size_t c = mono.count();
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
    [[nodiscard]] auto memory_bytes() const -> size_t {
        return (rows_.capacity() * sizeof(PosT)) + spilled_rows_bytes(overflow_);
    }

    auto find(const key_type &key) const -> std::optional<size_t> {
        return table_.find(fold_hash(key), [this, &key](size_t i) { return row_eq_key(i, key); });
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kNotFound. The prefetch the pipeline
    // is built around is the row prefetch below -- the table issues it between probe and confirm.
    auto find_batch(const key_type *keys, size_t n, size_t *out) const -> void {
        table_.find_batch(
            keys,
            n,
            out,
            [](const key_type &k) { return fold_hash(k); },
            [this](size_t i) { __builtin_prefetch(&rows_[i * stride_], 0, 0); },
            [this](size_t i, const key_type &k) { return row_eq_key(i, k); });
    }

    // find_batch over ascending position lists: query q is pos_flat[pos_off[q] .. pos_off[q] + k_of[q]).
    // Identical results to find_batch on the monomials those positions describe.
    auto find_batch_positions(std::span<const PosT> pos_flat,
                              std::span<const size_t> pos_off,
                              std::span<const uint32_t> k_of,
                              std::span<size_t> out,
                              std::span<uint32_t> hash_out = {}) const -> void {
        table_.find_batch_at(
            pos_off.size(),
            out.data(),
            [pos_flat, pos_off, k_of](size_t q) { return pos_flat.subspan(pos_off[q], k_of[q]); },
            [](std::span<const PosT> q) { return fold_hash_positions(q); },
            [this](size_t i) { __builtin_prefetch(&rows_[i * stride_], 0, 0); },
            [this](size_t i, std::span<const PosT> q) { return row_eq_positions(i, q); },
            hash_out);
    }

    // fold_hash of the monomial `pos` describes, through the same fold, so it is equal by construction.
    [[nodiscard]] static auto fold_hash_positions(std::span<const PosT> pos) noexcept -> uint32_t {
        key_type mono;
        for (size_t j = 0; j < pos.size(); ++j) {
            mono.set(pos[j]);
        }
        return fold_hash(mono);
    }

    // Insert-or-no-op. Row at `value` must already be written (the confirm reads dense rows).
    auto emplace(const key_type &key, mapped_type value) -> void {
        table_.emplace(fold_hash(key), value, [this, &key](size_t i) { return row_eq_key(i, key); });
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows must already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        table_.insert_distinct_range(base, n, [&key_at](size_t k) { return fold_hash(key_at(k)); });
    }
    // bulk_insert with the hashes already in hand: same precondition (n distinct rows, already written,
    // at consecutive indices) and the same slot assignment. `hash_at(k)` must be fold_hash of the key of
    // row base+k -- a wrong one leaves the row unfindable, which surfaces later as a duplicate insert.
    template <typename HashFn>
    auto bulk_insert_hashed(size_t n, mapped_type base, HashFn &&hash_at) -> void {
        table_.insert_distinct_range(base, n, std::forward<HashFn>(hash_at));
    }
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        table_.for_each_slot([this, &fn](TermIndex idx, uint32_t) { fn(row(idx), static_cast<size_t>(idx)); });
    }
    // Diagnostic: the part of memory_bytes() that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        return (rows_.capacity() * sizeof(PosT)) - (std::min(rows_.capacity(), size_ * stride_) * sizeof(PosT));
    }

    auto index_estimated_memory_bytes() const -> size_t { return sizeof(OperatorIndex) + table_.slot_bytes(); }

private:
    static auto fold_hash(const key_type &q) noexcept -> uint32_t {
        return RowHashTable::fold(MonomialHash<NumModes>{}(q));
    }

    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity() / stride_; }
    auto reserve_rows(size_t n) -> void { rows_.reserve(n * stride_); }
    auto reserve_index(size_t n) -> void { table_.reserve(n); }

    // Compare row i against key q without materializing the row (the find confirm). Reads the
    // popcount byte first, so a false h prefilter match usually costs one byte compare.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &q) const -> bool {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            return overflow_.at(i) == q;
        }
        if (q.count() != static_cast<size_t>(c)) {
            return false;
        }
        const PosT *pos = &rows_[(i * stride_) + 1];
        for (size_t j = 0; j < c; ++j) {
            if (!q.test(pos[j])) {
                return false;
            }
        }
        return true;
    }

    // Compare row i against an ascending position list; a spilled row falls back to a dense compare.
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

    DefaultInitVector<PosT> rows_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    // Lossless side-map for rows whose popcount exceeds inline_width_.
    std::unordered_map<size_t, value_type> overflow_ = {};
    RowHashTable table_ = {};
};

} // namespace monoprop::detail
