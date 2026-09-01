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

namespace monoprop::detail {

class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Operator-term store: entropy-packed position-list rows plus a keyless open-addressing hash index over
// those rows. Row layout: slot 0 = popcount c (or kOverflowMarker if c > inline_width_), slots 1..c =
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

    // Valid term indices are < kIndexCeiling (check_index_fits throws at the ceiling), so the
    // all-ones TermIndex is free to mark an empty slot.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    static constexpr TermIndex kEmptySlot = std::numeric_limits<TermIndex>::max();
    // find_batch's "absent" result; same value as detail::kMissingIndex (not included here — the
    // operator store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

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
        out->reserve_index(table_.count);
        for (const Slot &e : table_.slots) {
            if (e.idx != kEmptySlot) {
                out->insert_slot_(e.idx, e.h);
            }
        }
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
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve_rows(std::max(base + n, cap + (cap / 2) + 1));
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
        size_t total = rows_.capacity() * sizeof(PosT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    auto find(const key_type &key) const -> std::optional<size_t> {
        const uint32_t h = fold_hash(key);
        if (table_.count == 0) {
            return std::nullopt;
        }
        size_t s = spread(h) & table_.mask;
        for (;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return std::nullopt;
            }
            if (e.h == h && row_eq_key(static_cast<size_t>(e.idx), key)) {
                return static_cast<size_t>(e.idx);
            }
        }
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kNotFound. Same result as n
    // find() calls, but overlaps dram misses via a per-group hash/probe/confirm pipeline. An h
    // collision falls back to an exact find. must not run concurrently with inserts.
    auto find_batch(const key_type *keys, size_t n, size_t *out) const -> void {
        static constexpr size_t G = 16; // keys prefetched together per pipeline pass
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = fold_hash(keys[base + j]);
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&table_.slots[sp[j] & table_.mask], 0, 0);
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kEmptySlot;
                if (table_.count == 0) {
                    continue;
                }
                cand[j] = probe_hash_match_(hh[j], sp[j] & table_.mask);
                if (cand[j] != kEmptySlot) {
                    __builtin_prefetch(&rows_[static_cast<size_t>(cand[j]) * stride_], 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                if (cand[j] != kEmptySlot && row_eq_key(static_cast<size_t>(cand[j]), keys[base + j])) {
                    out[base + j] = static_cast<size_t>(cand[j]);
                }
                else if (cand[j] != kEmptySlot) {
                    const auto v = find(keys[base + j]);
                    out[base + j] = v ? *v : kNotFound;
                }
                else {
                    out[base + j] = kNotFound;
                }
            }
        }
    }

    // find_batch over ascending position lists: query q is pos_flat[pos_off[q] .. pos_off[q] + k_of[q]).
    // Identical results to find_batch on the monomials those positions describe.
    auto find_batch_positions(std::span<const PosT> pos_flat,
                              std::span<const size_t> pos_off,
                              std::span<const uint32_t> k_of,
                              std::span<size_t> out,
                              std::span<uint32_t> hash_out = {}) const -> void {
        const size_t n = pos_off.size();
        static constexpr size_t G = 16;
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = fold_hash_positions(pos_flat.subspan(pos_off[base + j], k_of[base + j]));
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&table_.slots[sp[j] & table_.mask], 0, 0);
            }
            if (!hash_out.empty()) {
                std::copy_n(hh.begin(), g, hash_out.begin() + static_cast<std::ptrdiff_t>(base));
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kEmptySlot;
                if (table_.count == 0) {
                    continue;
                }
                cand[j] = probe_hash_match_(hh[j], sp[j] & table_.mask);
                if (cand[j] != kEmptySlot) {
                    __builtin_prefetch(&rows_[static_cast<size_t>(cand[j]) * stride_], 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                const size_t q = base + j;
                const std::span<const PosT> qpos = pos_flat.subspan(pos_off[q], k_of[q]);
                if (cand[j] == kEmptySlot) {
                    out[q] = kNotFound;
                }
                else if (row_eq_positions(static_cast<size_t>(cand[j]), qpos)) {
                    out[q] = static_cast<size_t>(cand[j]);
                }
                else {
                    // A 32-bit collision: rare enough to walk the chain from the top rather than resume it.
                    out[q] = find_positions_(hh[j], qpos);
                }
            }
        }
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
        check_index_fits(value);
        const uint32_t h = fold_hash(key);
        table_.rehash_if_needed();
        size_t s = spread(h) & table_.mask;
        while (table_.slots[s].idx != kEmptySlot) {
            if (table_.slots[s].h == h && row_eq_key(static_cast<size_t>(table_.slots[s].idx), key)) {
                return;
            }
            s = (s + 1) & table_.mask;
        }
        table_.slots[s] = Slot{static_cast<TermIndex>(value), h};
        ++table_.count;
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows must already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        if (n == 0) {
            return;
        }
        bulk_insert_hashed(n, base, [this, &key_at](size_t k) { return fold_hash(key_at(k)); });
    }
    // bulk_insert with the hashes already in hand: same precondition (n distinct rows, already written,
    // at consecutive indices) and the same slot assignment. `hashes[k]` must be fold_hash of the key of
    // row base+k -- a wrong one leaves the row unfindable, which surfaces later as a duplicate insert.
    template <typename HashFn>
    auto bulk_insert_hashed(size_t n, mapped_type base, HashFn &&hash_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        static constexpr size_t G = 16;
        std::array<uint32_t, G> hh;
        for (size_t b = 0; b < n; b += G) {
            const size_t g = std::min(G, n - b);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = hash_at(b + j);
                __builtin_prefetch(&table_.slots[spread(hh[j]) & table_.mask], /*rw=*/1, /*locality=*/0);
            }
            for (size_t j = 0; j < g; ++j) {
                insert_slot_(static_cast<TermIndex>(base + b + j), hh[j]);
            }
        }
    }
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (const Slot &e : table_.slots) {
            if (e.idx != kEmptySlot) {
                fn(row(static_cast<size_t>(e.idx)), static_cast<size_t>(e.idx));
            }
        }
    }
    // Diagnostic: the part of memory_bytes() that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        return (rows_.capacity() * sizeof(PosT)) - (std::min(rows_.capacity(), size_ * stride_) * sizeof(PosT));
    }

    auto index_estimated_memory_bytes() const -> size_t {
        return sizeof(OperatorIndex) + (table_.slots.capacity() * sizeof(Slot));
    }

private:
    struct Slot {
        TermIndex idx = kEmptySlot;
        uint32_t h = 0;
    };

    // First slot on h's probe chain whose stored hash matches, or kEmptySlot if the chain ends first.
    // Matches on h alone and leaves the dense-row comparison to the caller — that deferral is what lets
    // find_batch prefetch the row between probe and confirm, so do not fold row_eq_key in here (find()
    // deliberately keeps its own confirming variant). `start` must already be masked; the table must not
    // be mutated concurrently.
    [[gnu::always_inline]] auto probe_hash_match_(uint32_t h, size_t start) const -> TermIndex {
        for (size_t s = start;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return kEmptySlot;
            }
            if (e.h == h) {
                return e.idx;
            }
        }
    }

    static uint32_t fold_hash(const key_type &q) noexcept {
        const size_t full = MonomialHash<NumModes>{}(q);
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }
    // Avalanche the cached 32-bit fold into a full-width hash (splitmix64 finalizer): the stored h
    // is only an equality pre-filter, so it must be re-mixed before its low bits drive table bucketing.
    static size_t spread(uint32_t h) noexcept {
        uint64_t x = static_cast<uint64_t>(h) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }

    // One open-addressing table: power-of-2 slot count, linear probing, max load factor 0.7
    // (the group-prefetch win erodes at high load — longer probe chains add un-prefetched reads).
    struct Table {
        std::vector<Slot> slots = std::vector<Slot>(kMinSlots, Slot{});
        size_t mask = kMinSlots - 1;
        size_t count = 0;

        auto rehash_if_needed() -> void {
            if ((count + 1) * 10 >= slots.size() * 7) {
                rehash_to(slots.size() * 2);
            }
        }
        auto rehash_to(size_t new_cap) -> void {
            new_cap = std::bit_ceil(std::max<size_t>(new_cap, kMinSlots));
            if (new_cap <= slots.size()) {
                return;
            }
            std::vector<Slot> old = std::move(slots);
            slots.assign(new_cap, Slot{});
            mask = new_cap - 1;
            for (const Slot &e : old) {
                if (e.idx == kEmptySlot) {
                    continue;
                }
                size_t s = spread(e.h) & mask;
                while (slots[s].idx != kEmptySlot) {
                    s = (s + 1) & mask;
                }
                slots[s] = e;
            }
        }
    };
    static constexpr size_t kMinSlots = 16;
    // Slot count for `n` entries at ≤0.7 load.
    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, (n * 10 / 7) + 1)); }

    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity() / stride_; }
    auto reserve_rows(size_t n) -> void { rows_.reserve(n * stride_); }
    auto reserve_index(size_t n) -> void { table_.rehash_to(slots_for_(n + 1)); }

    // Insert (idx, h) into the table with no duplicate probe — callers on this path insert provably distinct
    // keys (⊕G-injective miss batches, clone re-insertion).
    auto insert_slot_(TermIndex idx, uint32_t h) -> void {
        table_.rehash_if_needed();
        size_t s = spread(h) & table_.mask;
        while (table_.slots[s].idx != kEmptySlot) {
            s = (s + 1) & table_.mask;
        }
        table_.slots[s] = Slot{idx, h};
        ++table_.count;
    }

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

    // find()'s chain walk for a position-list key, hash already folded; only the collision arm reaches it.
    [[nodiscard]] auto find_positions_(uint32_t h, std::span<const PosT> q) const -> size_t {
        if (table_.count == 0) {
            return kNotFound;
        }
        for (size_t s = spread(h) & table_.mask;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return kNotFound;
            }
            if (e.h == h && row_eq_positions(static_cast<size_t>(e.idx), q)) {
                return static_cast<size_t>(e.idx);
            }
        }
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: this partition's term count reached the 2^32 TermIndex "
                                          "ceiling; raise the partition or rank count to split it further.");
        }
    }

    DefaultInitVector<PosT> rows_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    // Lossless side-map for rows whose popcount exceeds inline_width_.
    std::unordered_map<size_t, value_type> overflow_ = {};
    Table table_ = {};
};

} // namespace monoprop::detail
