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
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

//! A partition exceeded the representable row-index range.
class TermIndexCeilingReached : public std::runtime_error {
public:
    //! Inherit constructors accepting an error message.
    using std::runtime_error::runtime_error;
};

/*!
 * Keyless row index with linear probing, power-of-two capacity, and maximum load factor 0.7.
 * Callers own the rows and supply hash, equality, and prefetch callbacks.
 * Slot order determines evolved-term iteration and floating-point accumulation order.
 * Each partition has one writer; lookups must not run concurrently with inserts.
 */
class RowHashTable {
public:
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    //!< Exclusive upper bound for row indices.
    static constexpr TermIndex kEmptySlot = std::numeric_limits<TermIndex>::max(); //!< Empty-slot sentinel.
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();        //!< Batch-lookup miss sentinel.

    //! Number of indexed rows.
    [[nodiscard]] auto count() const noexcept -> size_t { return count_; }
    //! Allocated slot storage in bytes.
    [[nodiscard]] auto slot_bytes() const -> size_t { return slots_.capacity() * sizeof(Slot); }

    //! Reserve for n rows without reaching the rehash threshold.
    auto reserve(size_t n) -> void { rehash_to(slots_for_(n + 1)); }

    //! Fold a full-width hash to the 32-bit equality prefilter.
    [[nodiscard]] static constexpr auto fold(size_t full) noexcept -> uint32_t {
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }

    //! Throw if appending n rows would exceed the index range. Requires base <= kIndexCeiling.
    static auto check_append_fits(size_t base, size_t n) -> void {
        if (n > kIndexCeiling - base) {
            throw TermIndexCeilingReached(
                std::format("appending {} rows to a partition holding {} would pass the 2^32 TermIndex "
                            "ceiling; raise the partition or rank count to split it further.",
                            n,
                            base));
        }
    }

    //! Throw if value is not a representable row index.
    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("this partition's row count reached the 2^32 TermIndex ceiling; "
                                          "raise the partition or rank count to split it further.");
        }
    }

    //! Find a row with hash h, confirming equality with eq(row_index).
    template <typename Eq>
    auto find(uint32_t h, Eq &&eq) const -> std::optional<size_t> {
        if (count_ == 0) {
            return std::nullopt;
        }
        size_t s = spread(h) & mask_;
        for (;; s = (s + 1) & mask_) {
            const Slot &e = slots_[s];
            if (e.idx == kEmptySlot) {
                return std::nullopt;
            }
            if (e.h == h && eq(static_cast<size_t>(e.idx))) {
                return static_cast<size_t>(e.idx);
            }
        }
    }

    //! Index value unless an equal row exists. The row must already be written.
    template <typename Eq>
    auto emplace(uint32_t h, size_t value, Eq &&eq) -> void {
        check_index_fits(value);
        rehash_if_needed();
        size_t s = spread(h) & mask_;
        while (slots_[s].idx != kEmptySlot) {
            if (slots_[s].h == h && eq(static_cast<size_t>(slots_[s].idx))) {
                return;
            }
            s = (s + 1) & mask_;
        }
        slots_[s] = Slot{static_cast<TermIndex>(value), h};
        ++count_;
    }

    //! Index distinct rows [base, base + n), using hash_at(k) for row base + k.
    template <typename HashFn>
    auto insert_distinct_range(size_t base, size_t n, HashFn &&hash_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        static constexpr size_t G = 16; // rows hashed together per prefetch pass
        std::array<uint32_t, G> hh;
        for (size_t b = 0; b < n; b += G) {
            const size_t g = std::min(G, n - b);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = hash_at(b + j);
                // A rehash inside the loop below invalidates these addresses; a stale prefetch is a
                // wasted hint, never a wrong insert.
                __builtin_prefetch(&slots_[spread(hh[j]) & mask_], /*rw=*/1, /*locality=*/0);
            }
            for (size_t j = 0; j < g; ++j) {
                insert_distinct(static_cast<TermIndex>(base + b + j), hh[j]);
            }
        }
    }

    //! Index a valid row index with hash h. The key must not already be indexed.
    auto insert_distinct(TermIndex idx, uint32_t h) -> void {
        rehash_if_needed();
        size_t s = spread(h) & mask_;
        while (slots_[s].idx != kEmptySlot) {
            s = (s + 1) & mask_;
        }
        slots_[s] = Slot{idx, h};
        ++count_;
    }

    //! Write each key's row index or kNotFound to out, prefetching rows before equality checks.
    template <typename Key, typename Hash, typename PrefetchRow, typename Eq>
    auto find_batch(const Key *keys, size_t n, size_t *out, Hash &&hash, PrefetchRow &&prefetch_row, Eq &&eq) const
        -> void {
        find_batch_at(n, out, [keys](size_t i) -> const Key & { return keys[i]; }, hash, prefetch_row, eq);
    }

    /*!
     * Batch lookup through key_at(i), which must be cheap and stable across repeated calls.
     * If non-empty, hash_out must hold n entries and receives each key's folded hash.
     */
    template <typename KeyAt, typename Hash, typename PrefetchRow, typename Eq>
    auto find_batch_at(size_t n,
                       size_t *out,
                       KeyAt &&key_at,
                       Hash &&hash,
                       PrefetchRow &&prefetch_row,
                       Eq &&eq,
                       std::span<uint32_t> hash_out = {}) const -> void {
        static constexpr size_t G = 16; // keys prefetched together per pipeline pass
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = hash(key_at(base + j));
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&slots_[sp[j] & mask_], 0, 0);
            }
            if (!hash_out.empty()) {
                std::copy_n(hh.begin(), g, hash_out.begin() + static_cast<std::ptrdiff_t>(base));
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kEmptySlot;
                if (count_ == 0) {
                    continue;
                }
                cand[j] = probe_hash_match_(hh[j], sp[j] & mask_);
                if (cand[j] != kEmptySlot) {
                    prefetch_row(static_cast<size_t>(cand[j]));
                }
            }
            for (size_t j = 0; j < g; ++j) {
                const size_t q = base + j;
                if (cand[j] == kEmptySlot) {
                    out[q] = kNotFound;
                }
                else if (eq(static_cast<size_t>(cand[j]), key_at(q))) {
                    out[q] = static_cast<size_t>(cand[j]);
                }
                else {
                    // A 32-bit collision: rare enough to walk the chain from the top rather than resume it.
                    const auto v = find(hh[j], [&eq, &key_at, q](size_t i) { return eq(i, key_at(q)); });
                    out[q] = v ? *v : kNotFound;
                }
            }
        }
    }

    //! Visit occupied slots in table order as fn(row_index, stored_hash).
    template <typename Fn>
    auto for_each_slot(Fn &&fn) const -> void {
        for (const Slot &e : slots_) {
            if (e.idx != kEmptySlot) {
                fn(e.idx, e.h);
            }
        }
    }

private:
    //! Row reference and equality prefilter.
    struct Slot {
        TermIndex idx = kEmptySlot; //!< Row index or kEmptySlot.
        uint32_t h = 0;             //!< Folded row hash.
    };

    static constexpr size_t kMinSlots = 16; //!< Minimum table capacity.

    //! Power-of-two capacity for n rows at a load factor below 0.7.
    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, (n * 10 / 7) + 1)); }

    //! Mix a folded hash with the splitmix64 finalizer for slot selection.
    static auto spread(uint32_t h) noexcept -> size_t {
        uint64_t x = static_cast<uint64_t>(h) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }

    /*!
     * Return the first hash-matching row or kEmptySlot. Requires a masked start slot.
     * Equality is deferred so batch lookup can prefetch the row before confirmation.
     */
    [[gnu::always_inline]] auto probe_hash_match_(uint32_t h, size_t start) const -> TermIndex {
        for (size_t s = start;; s = (s + 1) & mask_) {
            const Slot &e = slots_[s];
            if (e.idx == kEmptySlot) {
                return kEmptySlot;
            }
            if (e.h == h) {
                return e.idx;
            }
        }
    }

    //! Grow before the next insertion would reach the load threshold.
    auto rehash_if_needed() -> void {
        if ((count_ + 1) * 10 >= slots_.size() * 7) {
            rehash_to(slots_.size() * 2);
        }
    }

    //! Grow to at least new_cap slots, preserving stored hashes and row indices.
    auto rehash_to(size_t new_cap) -> void {
        new_cap = std::bit_ceil(std::max<size_t>(new_cap, kMinSlots));
        if (new_cap <= slots_.size()) {
            return;
        }
        std::vector<Slot> old = std::move(slots_);
        slots_.assign(new_cap, Slot{});
        mask_ = new_cap - 1;
        for (const Slot &e : old) {
            if (e.idx == kEmptySlot) {
                continue;
            }
            size_t s = spread(e.h) & mask_;
            while (slots_[s].idx != kEmptySlot) {
                s = (s + 1) & mask_;
            }
            slots_[s] = e;
        }
    }

    std::vector<Slot> slots_ = std::vector<Slot>(kMinSlots, Slot{}); //!< Hash-table slots.
    size_t mask_ = kMinSlots - 1;                                    //!< Slot-index mask.
    size_t count_ = 0;                                               //!< Occupied slot count.
};

} // namespace monoprop::detail
