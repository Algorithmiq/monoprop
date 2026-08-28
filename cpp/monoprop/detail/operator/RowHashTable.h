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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// The next row-array capacity for a geometric (1.5x) grow by `n` rows from `base` (the pre-growth size),
// given the current capacity. Never exact-fit: an exact fit would realloc the whole operator every layer.
// Shared by OperatorIndex and SparseRowStore, whose grow_rows_geometric() differ only in which arrays
// that capacity gets applied to.
[[nodiscard]] inline auto geometric_row_capacity(size_t base, size_t n, size_t capacity) noexcept -> size_t {
    return std::max(base + n, capacity + (capacity / 2) + 1);
}

// What a row store's spilled rows cost outside its own arrays: the map node per entry (key, mapped
// value and ~24 bytes of std::unordered_map node and bucket overhead) plus whatever each spilled
// monomial owns past its inline words. Shared for the same reason as the capacity rule above -- the
// node-overhead estimate is a single number that must not be corrected in one store and not the other,
// which would skew operator_memory_breakdown() for one backend only.
template <typename OverflowMap>
[[nodiscard]] inline auto spilled_rows_bytes(const OverflowMap &overflow) -> size_t {
    size_t total = overflow.size() * (sizeof(typename OverflowMap::mapped_type) + sizeof(size_t) + 24);
    for (const auto &[key, value] : overflow) {
        total += value.heap_bytes();
    }
    return total;
}

class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The keyless open-addressing index a row store puts over its rows: power-of-2 slot count, linear
// probing, max load factor 0.7 (the group-prefetch win erodes at higher load -- longer probe chains add
// un-prefetched reads). A slot holds a row index plus a 32-bit hash used only as an equality
// pre-filter, so the table never stores or compares a key itself.
//
// Keyless is why the hash and the equality test arrive as callables rather than as members: the whole
// point is that the caller owns the row representation. `eq(row_index)` confirms a pre-filter hit
// against the caller's rows, and no operation here reads a row.
//
// Shared by every row store, and that is load-bearing rather than tidiness: the layout this produces
// fixes the iteration order of for_each_slot(), which sets the order of a propagator's user-visible
// evolved-term list and therefore its floating-point accumulation order. Two stores that keyed rows
// through separate copies of this logic could diverge on that while both looking correct.
//
// Single-writer, matching its owners: one partition, one thread; parallelism is cross-partition.
class RowHashTable {
public:
    // Valid row indices are < kIndexCeiling (check_index_fits throws at the ceiling), so the all-ones
    // TermIndex is free to mark an empty slot.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    static constexpr TermIndex kEmptySlot = std::numeric_limits<TermIndex>::max();
    // find_batch's "absent" result; same value as detail::kMissingIndex (not included here -- the
    // operator store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

    [[nodiscard]] auto count() const noexcept -> size_t { return count_; }
    [[nodiscard]] auto slot_bytes() const -> size_t { return slots_.capacity() * sizeof(Slot); }

    // Slot capacity for `n` rows at <= 0.7 load. The +1 keeps a table reserved for exactly n rows off
    // the rehash threshold on the n-th insert.
    auto reserve(size_t n) -> void { rehash_to(slots_for_(n + 1)); }

    // The 32-bit form of a full-width row hash, which is what a slot stores as its equality
    // pre-filter. Each store supplies its own full-width hash and folds it here, so the two agree on
    // the pre-filter's format by construction.
    [[nodiscard]] static constexpr auto fold(size_t full) noexcept -> uint32_t {
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("operator index reached the TermIndex ceiling; rebuild with "
                                          "-Dmonoprop_WIDE_TERM_INDEX (this partition's term count exceeded ~2^32).");
        }
    }

    // eq(row_index) -> bool confirms a hash pre-filter hit.
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

    // Insert-or-no-op. The row at `value` must already be written -- eq reads it.
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

    // Insert with no duplicate probe -- callers on this path insert provably distinct keys
    // (+G-injective miss batches, clone re-insertion).
    // insert_distinct over consecutive row indices [base, base + n), hashing each through hash_at(k).
    // The stores' bulk_insert is this and nothing else, so it lives here rather than once per backend.
    template <typename HashFn>
    auto insert_distinct_range(size_t base, size_t n, HashFn &&hash_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        for (size_t k = 0; k < n; ++k) {
            insert_distinct(static_cast<TermIndex>(base + k), hash_at(k));
        }
    }

    auto insert_distinct(TermIndex idx, uint32_t h) -> void {
        rehash_if_needed();
        size_t s = spread(h) & mask_;
        while (slots_[s].idx != kEmptySlot) {
            s = (s + 1) & mask_;
        }
        slots_[s] = Slot{idx, h};
        ++count_;
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kNotFound. Same result as n find()
    // calls, but overlaps dram misses via a per-group hash/probe/confirm pipeline. An h collision falls
    // back to an exact find. Must not run concurrently with inserts.
    //
    // The three callables are what make the pipeline possible without the table knowing a row:
    // hash(key) -> uint32_t, prefetch_row(row_index) issued between probe and confirm (which is the
    // whole reason confirmation is deferred rather than folded into the probe), and
    // eq(row_index, key) -> bool.
    template <typename Key, typename Hash, typename PrefetchRow, typename Eq>
    auto find_batch(const Key *keys, size_t n, size_t *out, Hash &&hash, PrefetchRow &&prefetch_row, Eq &&eq) const
        -> void {
        static constexpr size_t G = 16; // keys prefetched together per pipeline pass
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = hash(keys[base + j]);
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&slots_[sp[j] & mask_], 0, 0);
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
                if (cand[j] != kEmptySlot && eq(static_cast<size_t>(cand[j]), keys[base + j])) {
                    out[base + j] = static_cast<size_t>(cand[j]);
                }
                else if (cand[j] != kEmptySlot) {
                    const auto v = find(hh[j], [&eq, &keys, &base, &j](size_t i) { return eq(i, keys[base + j]); });
                    out[base + j] = v ? *v : kNotFound;
                }
                else {
                    out[base + j] = kNotFound;
                }
            }
        }
    }

    // Occupied slots in table order, as fn(row_index, stored_hash). That order is the store's iteration
    // order; see the class comment on why it must not drift between stores.
    template <typename Fn>
    auto for_each_slot(Fn &&fn) const -> void {
        for (const Slot &e : slots_) {
            if (e.idx != kEmptySlot) {
                fn(e.idx, e.h);
            }
        }
    }

private:
    struct Slot {
        TermIndex idx = kEmptySlot;
        uint32_t h = 0;
    };

    static constexpr size_t kMinSlots = 16;

    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, (n * 10 / 7) + 1)); }

    // Avalanche the cached 32-bit fold into a full-width hash (splitmix64 finalizer): the stored h is
    // only an equality pre-filter, so it must be re-mixed before its low bits drive table bucketing.
    static auto spread(uint32_t h) noexcept -> size_t {
        uint64_t x = static_cast<uint64_t>(h) * 0x9E3779B97F4A7C15ULL;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }

    // First slot on h's probe chain whose stored hash matches, or kEmptySlot if the chain ends first.
    // Matches on h alone and leaves confirmation to the caller -- that deferral is what lets find_batch
    // prefetch the row between probe and confirm, so do not fold eq in here (find() deliberately keeps
    // its own confirming variant).  `start` must already be masked; the table must not be mutated
    // concurrently.
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

    auto rehash_if_needed() -> void {
        if ((count_ + 1) * 10 >= slots_.size() * 7) {
            rehash_to(slots_.size() * 2);
        }
    }

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

    std::vector<Slot> slots_ = std::vector<Slot>(kMinSlots, Slot{});
    size_t mask_ = kMinSlots - 1;
    size_t count_ = 0;
};

} // namespace monoprop::detail
