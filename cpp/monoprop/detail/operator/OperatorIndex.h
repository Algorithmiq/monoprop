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
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/RowHashTable.h"

namespace monoprop::detail {

// The requested monomial width needs bit positions wider than PosT can hold. Thrown rather than
// asserted: with the compile-time mode ceiling gone, the width is user data.
class OperatorIndexWidthUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Operator-term store: entropy-packed position-list rows plus a keyless open-addressing hash index over
// those rows. Row layout: slot 0 = popcount c (or kOverflowMarker if c > inline_width_),
// slots 1..c = ascending set-bit positions; stride_ is fixed for the container's life so row offsets stay
// stable, and so is the slot's width (see kNarrowPositions).
// inline_width_ is a free parameter -- any width is correct, over-long rows spill losslessly to overflow.
// Single-writer: one partition, one thread; parallelism is cross-partition.
class OperatorIndex {
public:
    // The store carries its width as data (num_bits_), so a row is a plain Bitset and the width is not
    // recoverable from the type.
    using value_type = Bitset;
    using key_type = Bitset;
    using mapped_type = size_t;

    static constexpr size_t kDefaultInlinePositions = 11;
    // A weight-w Pauli needs 2w positions; 32 covers the common case inline at the supported Pauli
    // cutoffs (2*cutoff <= 32 for cutoff <= 16).
    static constexpr size_t kMaxInlinePositions = 32;

    // A slot holds one bit position, so the narrowest integer that indexes num_bits_ is what a row costs
    // per slot: uint8_t at or below kNarrowPositions, uint16_t above it. The width is data, so it is a
    // member (narrow_) rather than the conditional_t on 2*NumModes it used to be, and the payload type is
    // bound per call by with_rows(). One fixed uint16_t is the obvious simplification and was rejected on
    // measurement: it doubles the entire operator's row footprint for every system at or below
    // kNarrowPositions/2 modes, which is where both shipping models sit.
    //
    // Row payload only -- never a hash input, never serialized, never an owner-routing input -- so the
    // width cannot change results, only footprint. That also means a term-and-energy baseline diff cannot
    // see a regression here; the operator_terms_bytes checks in the unit tests are the gate.
    static constexpr size_t kNarrowPositions = static_cast<size_t>(std::numeric_limits<uint8_t>::max()) + 1;
    static constexpr size_t kMaxPositions = static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;

    // The all-ones slot value, per row width. It only ever occupies slot 0, whose value is a popcount
    // bounded by inline_width_ <= kMaxInlinePositions, so it collides with no popcount at either width
    // -- asserted below on the narrower type, which covers both. A *position* slot may legitimately
    // hold the marker value (position 255 under uint8_t) and is never compared against it.
    template <typename P>
    static constexpr P kOverflowMarker = std::numeric_limits<P>::max();

    static_assert(kMaxInlinePositions < kOverflowMarker<uint8_t>,
                  "the overflow marker must not collide with a valid popcount at either row width");

    static constexpr size_t kIndexCeiling = RowHashTable::kIndexCeiling;
    static constexpr size_t kNotFound = RowHashTable::kNotFound;

    // num_bits is the storage bit width of every monomial this store will hold; row() reconstructs at
    // exactly that width, and a wrong one would change num_words() and therefore the hash, the probe
    // order and MPI owner routing. It also fixes the row payload width for the store's life, so a store
    // holds exactly one of the two row arrays and narrow_ is never reassigned. The check replaces the
    // static_assert that used to guard set()'s narrowing cast, which can no longer be a compile-time
    // assertion.
    explicit OperatorIndex(size_t num_bits, size_t inline_width = kDefaultInlinePositions)
        : num_bits_(num_bits),
          inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)),
          stride_(1 + inline_width_),
          narrow_(num_bits <= kNarrowPositions) {
        if (num_bits > kMaxPositions) {
            throw OperatorIndexWidthUnsupported(
                std::format("OperatorIndex supports at most {} bit positions ({} modes); got {} bits ({} modes).",
                            kMaxPositions,
                            kMaxPositions / 2,
                            num_bits,
                            num_bits / 2));
        }
    }

    [[nodiscard]] auto num_bits() const noexcept -> size_t { return num_bits_; }
    OperatorIndex(const OperatorIndex &) = delete;
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    OperatorIndex(OperatorIndex &&) = delete;
    OperatorIndex &operator=(OperatorIndex &&) = delete;

private:
    // The dispatcher sits here, mid-class, rather than with the other private members at the bottom: a
    // deduced (decltype(auto)) return type is not available to a caller that appears earlier in the class
    // body -- the body is a complete-class context for name lookup, but not for return-type deduction.
    //
    // Binds the row payload type for one call. narrow_ is fixed at construction, so the branch is a
    // load-and-test on a member that never changes: predicted, and amortized over the row loop inside f.
    // The dispatch is per call rather than hoisted into the store type on purpose -- the row width is not
    // part of the seam the scan is templated on (see the with_store note in MPOperator), and making it so
    // would double every downstream instantiation to save a predicted branch per row.
    template <typename F>
    [[gnu::always_inline]] auto with_rows(F &&f) const -> decltype(auto) {
        return narrow_ ? f(rows8_) : f(rows16_);
    }
    template <typename F>
    [[gnu::always_inline]] auto with_rows(F &&f) -> decltype(auto) {
        return narrow_ ? f(rows8_) : f(rows16_);
    }

    // Row i's address, for the prefetch hint, which discards the element type anyway.
    [[nodiscard]] auto row_addr(size_t i) const noexcept -> const void * {
        return with_rows([&](const auto &rows) -> const void * { return rows.data() + (i * stride_); });
    }

    // The two row arrays differ only in element size, so everything that counts bytes rather than
    // reading a slot is plain arithmetic off this and needs no type bound.
    [[nodiscard]] auto slot_bytes() const noexcept -> size_t { return narrow_ ? sizeof(uint8_t) : sizeof(uint16_t); }
    [[nodiscard]] auto row_slots_capacity() const -> size_t {
        return with_rows([&](const auto &rows) { return rows.capacity(); });
    }
    [[nodiscard]] auto row_bytes_capacity() const -> size_t { return row_slots_capacity() * slot_bytes(); }

public:
    // Called only on an idle store, so it needs no synchronization.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(num_bits_, inline_width_);
        // Both, not with_rows(): the dead one is empty, and copying it is cheaper than a dispatch.
        out->rows8_ = rows8_;
        out->rows16_ = rows16_;
        out->size_ = size_;
        out->overflow_ = overflow_;
        out->table_.reserve(table_.count());
        table_.for_each_slot([&](TermIndex idx, uint32_t h) { out->table_.insert_distinct(idx, h); });
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    // Returns the pre-growth size (the caller's insert base). Growth is geometric (1.5×), never
    // exact-fit: an exact fit would realloc the whole operator every layer.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve_rows(std::max(base + n, cap + (cap / 2) + 1));
        }
        // Default-init grow, not a zeroing resize: every freshly grown row is overwritten by set()
        // before any read, so a tail zero-fill would be wasted bandwidth.
        with_rows([&](auto &rows) { rows.resize((base + n) * stride_); });
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so the row header is never pre-read
    // (freshly grown headers are indeterminate); a stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        with_rows([&]<typename P>(DefaultInitVector<P> &rows) {
            const size_t c = mono.count();
            P *row = &rows[i * stride_];
            if (c > inline_width_) {
                row[0] = kOverflowMarker<P>;
                overflow_[i] = mono;
                return;
            }
            if (!overflow_.empty()) {
                overflow_.erase(i);
            }
            row[0] = static_cast<P>(c);
            P *out = row + 1;
            for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
                *out++ = static_cast<P>(b);
            }
        });
    }

    [[nodiscard]] auto row(size_t i) const -> value_type {
        return with_rows([&]<typename P>(const DefaultInitVector<P> &rows) -> value_type {
            const P c = rows[i * stride_];
            if (c == kOverflowMarker<P>) {
                return overflow_.at(i);
            }
            value_type mono(num_bits_);
            const P *pos = &rows[(i * stride_) + 1];
            for (size_t j = 0; j < c; ++j) {
                mono.set(pos[j]);
            }
            return mono;
        });
    }
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        with_rows([&]<typename P>(const DefaultInitVector<P> &rows) {
            const P c = rows[i * stride_];
            if (c == kOverflowMarker<P>) {
                const auto &m = overflow_.at(i);
                for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                    fn(b);
                }
                return;
            }
            const P *pos = &rows[(i * stride_) + 1];
            for (size_t j = 0; j < c; ++j) {
                fn(static_cast<size_t>(pos[j]));
            }
        });
    }
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        return with_rows([&]<typename P>(const DefaultInitVector<P> &rows) -> size_t {
            if (const P c = rows[i * stride_]; c != kOverflowMarker<P>) {
                return static_cast<size_t>(c);
            }
            return overflow_.at(i).count();
        });
    }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = row_bytes_capacity();
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    auto find(const key_type &key) const -> std::optional<size_t> {
        return table_.find(fold_hash(key), [&](size_t i) { return row_eq_key(i, key); });
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kNotFound. Same result as n
    // find() calls, but overlaps dram misses via a per-group hash/probe/confirm pipeline. An h
    // collision falls back to an exact find. must not run concurrently with inserts.
    // Templated on the key type rather than taking `const key_type *`: any Key that binds to
    // const key_type& works, since that is all fold_hash/row_eq_key need. SparseRowStore's twin takes
    // its own key type through the same signature, which is what lets Resolve.h call one spelling.
    template <typename Key>
    auto find_batch(const Key *keys, size_t n, size_t *out) const -> void {
        table_.find_batch(
            keys,
            n,
            out,
            [this](const Key &key) { return fold_hash(key); },
            [this](size_t i) { __builtin_prefetch(row_addr(i), 0, 0); },
            [this](size_t i, const Key &key) { return row_eq_key(i, key); });
    }

    // Insert-or-no-op. Row at `value` must already be written (the confirm reads dense rows).
    auto emplace(const key_type &key, mapped_type value) -> void {
        table_.emplace(fold_hash(key), value, [&](size_t i) { return row_eq_key(i, key); });
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows must already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        if (n == 0) {
            return;
        }
        RowHashTable::check_index_fits(base + n - 1);
        for (size_t k = 0; k < n; ++k) {
            table_.insert_distinct(static_cast<TermIndex>(base + k), fold_hash(key_at(k)));
        }
    }
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        table_.for_each_slot(
            [&](TermIndex idx, uint32_t) { fn(row(static_cast<size_t>(idx)), static_cast<size_t>(idx)); });
    }
    // Diagnostic: the part of memory_bytes() that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        const size_t cap = row_slots_capacity();
        return (cap - std::min(cap, size_ * stride_)) * slot_bytes();
    }

    auto index_estimated_memory_bytes() const -> size_t { return sizeof(OperatorIndex) + table_.slot_bytes(); }

private:
    // SplitmixHash directly, which is exactly what MonomialHash<NumModes> forwarded to -- the hash is
    // unchanged, and must stay so: it drives probe order and monomial_hash % rank_count owner routing
    // (plan invariant 2).
    static uint32_t fold_hash(const key_type &q) noexcept {
        const size_t full = SplitmixHash{}(q);
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }

    [[nodiscard]] auto capacity() const -> size_t { return row_slots_capacity() / stride_; }
    auto reserve_rows(size_t n) -> void {
        with_rows([&](auto &rows) { rows.reserve(n * stride_); });
    }
    auto reserve_index(size_t n) -> void { table_.reserve(n); }

    // Compare row i against key q without materializing the row (the find confirm). Reads the
    // popcount byte first, so a false h prefilter match usually costs one byte compare.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &q) const -> bool {
        return with_rows([&]<typename P>(const DefaultInitVector<P> &rows) {
            const P c = rows[i * stride_];
            if (c == kOverflowMarker<P>) {
                return overflow_.at(i) == q;
            }
            if (q.count() != static_cast<size_t>(c)) {
                return false;
            }
            const P *pos = &rows[(i * stride_) + 1];
            for (size_t j = 0; j < c; ++j) {
                if (!q.test(pos[j])) {
                    return false;
                }
            }
            return true;
        });
    }

    // Exactly one is ever non-empty, selected by narrow_ -- the same one-live-backend shape MPOperator
    // uses for its two stores.
    DefaultInitVector<uint8_t> rows8_ = {};
    DefaultInitVector<uint16_t> rows16_ = {};
    size_t num_bits_ = 0;
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    bool narrow_ = false;
    // Lossless side-map for rows whose popcount exceeds inline_width_.
    std::unordered_map<size_t, value_type> overflow_ = {};
    RowHashTable table_ = {};
};

} // namespace monoprop::detail
