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

/*!
 * Operator-term store with packed position-list rows and a RowHashTable index.
 * Each row stores its popcount followed by ascending set-bit positions; longer rows use overflow storage.
 * Row stride is fixed for the container's lifetime. Each partition has one writer.
 */
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = Monomial<NumModes>; //!< Stored term type.
    using key_type = Monomial<NumModes>;   //!< Lookup key type.
    using mapped_type = size_t;            //!< Row index type.

    using PosT = std::
        conditional_t<(2 * NumModes <= 256), uint8_t, std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;
    //!< Smallest unsigned type covering all set-bit positions.

    static constexpr size_t kDefaultInlinePositions = 11; //!< Default inline row width.
    static constexpr size_t kMaxInlinePositions = 32;     //!< Maximum inline width; covers Pauli weights up to 16.
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max(); //!< Spilled-row header.

    static_assert((2 * NumModes) - 1 <= std::numeric_limits<PosT>::max(),
                  "OperatorIndex PosT too narrow for 2*NumModes positions");
    static_assert(kMaxInlinePositions < std::numeric_limits<PosT>::max(),
                  "kOverflowMarker sentinel must not collide with a valid popcount");

    static constexpr size_t kIndexCeiling = RowHashTable::kIndexCeiling; //!< Exclusive row-index upper bound.
    static constexpr size_t kNotFound = RowHashTable::kNotFound;         //!< Batch-lookup miss sentinel.

    //! Construct with an inline width clamped to [1, kMaxInlinePositions].
    explicit OperatorIndex(size_t inline_width = kDefaultInlinePositions)
        : inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)),
          stride_(1 + inline_width_) {}
    //! Copying is disabled; use clone().
    OperatorIndex(const OperatorIndex &) = delete;
    //! Copy assignment is disabled.
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    //! Moving is disabled.
    OperatorIndex(OperatorIndex &&) = delete;
    //! Move assignment is disabled.
    OperatorIndex &operator=(OperatorIndex &&) = delete;

    //! Copy rows and index. The store must be idle.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(inline_width_);
        out->rows_ = rows_;
        out->size_ = size_;
        out->overflow_ = overflow_;
        out->reserve_index(table_.count());
        table_.for_each_slot([&out](TermIndex idx, uint32_t h) { out->table_.insert_distinct(idx, h); });
        return out;
    }

    //! Number of stored rows.
    [[nodiscard]] auto size() const -> size_t { return size_; }

    //! Number of rows in overflow storage.
    [[nodiscard]] auto overflow_size() const -> size_t { return overflow_.size(); }

    //! Reserve row and index storage for n terms.
    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    //! Append n uninitialized rows with geometric capacity growth; return their starting index.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        RowHashTable::check_append_fits(base, n);
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve_rows(std::max(base + n, cap + (cap / 2) + 1));
        }
        // New rows must be written before any read.
        rows_.resize((base + n) * stride_);
        size_ = base + n;
        return base;
    }

    //! Append a term without indexing it.
    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    //! Write row i, which may be uninitialized, and update any overflow entry.
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

    //! Write row i from strictly ascending positions, each less than 2 * NumModes.
    auto set_positions(size_t i, std::span<const PosT> pos) -> void {
        const size_t count = pos.size();
        assert((count == 0 || static_cast<size_t>(pos[count - 1]) < 2 * NumModes) && "row position out of range");
        PosT *row = &rows_[i * stride_];
        if (count > inline_width_) {
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

    //! Decode row i to a monomial.
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
    //! Visit row i's set-bit positions in ascending order.
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
    //! Number of set bits in row i.
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        if (const PosT c = rows_[i * stride_]; c != kOverflowMarker) {
            return c;
        }
        return overflow_.at(i).count();
    }
    //! View of a row's ascending positions, invalidated by row-storage reallocation.
    struct RowPositions {
        std::span<const PosT> pos; //!< Null for a spilled row; may be empty for an inline row.
        //! Whether the row uses inline storage.
        [[nodiscard]] auto inlined() const -> bool { return pos.data() != nullptr; }
    };
    //! Return an inline position view or a null view for a spilled row.
    [[nodiscard]] auto row_positions(size_t i) const -> RowPositions {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            return {};
        }
        return {std::span<const PosT>(&rows_[(i * stride_) + 1], static_cast<size_t>(c))};
    }
    //! Estimated row and overflow storage in bytes, excluding the index.
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = rows_.capacity() * sizeof(PosT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    //! Return the matching row index, or std::nullopt.
    auto find(const key_type &key) const -> std::optional<size_t> {
        return table_.find(fold_hash(key), [this, &key](size_t i) { return row_eq_key(i, key); });
    }

    //! Write each key's row index or kNotFound to out.
    auto find_batch(const key_type *keys, size_t n, size_t *out) const -> void {
        table_.find_batch(
            keys,
            n,
            out,
            [](const key_type &k) { return fold_hash(k); },
            [this](size_t i) { __builtin_prefetch(&rows_[i * stride_], 0, 0); },
            [this](size_t i, const key_type &k) { return row_eq_key(i, k); });
    }

    /*!
     * Look up ascending position lists: query q starts at pos_off[q] and has k_of[q] entries.
     * Output spans must hold one entry per query; hash_out may be empty to skip hash output.
     */
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

    //! Fold the hash of the monomial represented by pos.
    [[nodiscard]] static auto fold_hash_positions(std::span<const PosT> pos) noexcept -> uint32_t {
        key_type mono;
        for (size_t j = 0; j < pos.size(); ++j) {
            mono.set(pos[j]);
        }
        return fold_hash(mono);
    }

    //! Index value unless key already exists. The row must already be written.
    auto emplace(const key_type &key, mapped_type value) -> void {
        table_.emplace(fold_hash(key), value, [this, &key](size_t i) { return row_eq_key(i, key); });
    }
    //! Index n distinct, initialized rows starting at base, using key_at(k) for row base + k.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        table_.insert_distinct_range(base, n, [&key_at](size_t k) { return fold_hash(key_at(k)); });
    }
    //! Index n distinct, initialized rows starting at base; hash_at(k) must equal fold_hash of row base + k.
    template <typename HashFn>
    auto bulk_insert_hashed(size_t n, mapped_type base, HashFn &&hash_at) -> void {
        table_.insert_distinct_range(base, n, std::forward<HashFn>(hash_at));
    }
    //! Visit indexed rows in table order as fn(monomial, row_index).
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        table_.for_each_slot([this, &fn](TermIndex idx, uint32_t) { fn(row(idx), static_cast<size_t>(idx)); });
    }
    //! Unused row capacity in bytes.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        return (rows_.capacity() * sizeof(PosT)) - (std::min(rows_.capacity(), size_ * stride_) * sizeof(PosT));
    }

    //! Estimated object and index-slot storage in bytes.
    auto index_estimated_memory_bytes() const -> size_t { return sizeof(OperatorIndex) + table_.slot_bytes(); }

private:
    //! Fold a monomial hash to the index's equality prefilter.
    static auto fold_hash(const key_type &q) noexcept -> uint32_t {
        return RowHashTable::fold(MonomialHash<NumModes>{}(q));
    }

    //! Allocated row capacity.
    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity() / stride_; }
    //! Reserve storage for n rows.
    auto reserve_rows(size_t n) -> void { rows_.reserve(n * stride_); }
    //! Reserve index storage for n rows.
    auto reserve_index(size_t n) -> void { table_.reserve(n); }

    //! Compare row i with q without decoding inline rows.
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

    //! Compare row i with ascending positions, using a dense comparison for spilled rows.
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

    DefaultInitVector<PosT> rows_ = {};                    //!< Packed row headers and positions.
    size_t size_ = 0;                                      //!< Stored row count.
    size_t inline_width_ = kMaxInlinePositions;            //!< Maximum inline positions per row.
    size_t stride_ = 1 + kMaxInlinePositions;              //!< Fixed row width including the header.
    std::unordered_map<size_t, value_type> overflow_ = {}; //!< Rows exceeding the inline width.
    RowHashTable table_ = {};                              //!< Row lookup index.
};

} // namespace monoprop::detail
