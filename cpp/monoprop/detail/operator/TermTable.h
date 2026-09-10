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

// The persistent join key -> row table over a partition's whole operator store.
//
// Slots are 4 bytes, not the 8 of an (index, hash) pair: a slot holds the row index in its low
// log2(slots) bits, and the bits above them -- 8 at 2^24 slots, i.e. up to ~11.7 M rows -- carry the
// next bits of the key's hash as a compare prefilter. A probe that walks past an unrelated occupant
// therefore costs one word compare in a line it has already fetched, and reads a row only once 32 bits
// of hash agree. The row is the confirm: every key match is checked against the query's positions, so
// a key collision costs a compare and can never produce a wrong partner. At the ceiling (2^32 slots)
// the prefilter is empty and every occupant is confirmed, which is slower and still exact.
//
// No key is resident: a row's key is folded off the row when the table indexes it
// (OperatorIndex::key_of_row), sequentially on a rebuild and hot on the append of a gate's mints.
// Linear probing at a load of at most 0.7; growth doubles and rebuilds from the store, so the old
// slots are released before the new ones are allocated. Single-writer: one partition, one thread.

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <vector>

#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/operator/RowKey.h"

namespace monoprop::detail {

class TermTable {
public:
    //! "No row holds this key". Same value as OperatorIndex::kNotFound and detail::kMissingIndex.
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();
    static constexpr size_t kMinSlots = 16;
    //! Keys probed together per pipeline pass; the depth the prefetches run ahead by.
    static constexpr size_t kGroup = 16;

    //! Rows indexed: rows [0, rows()) of the store this was last rebuilt from or appended with.
    [[nodiscard]] auto rows() const -> size_t { return rows_; }
    [[nodiscard]] auto slots() const -> size_t { return slots_.size(); }
    [[nodiscard]] auto memory_bytes() const -> size_t { return slots_.capacity() * sizeof(uint32_t); }

    //! Slot count for @a n rows at the load bound: a power of two, never below kMinSlots.
    [[nodiscard]] static auto slots_for(size_t n) -> size_t {
        const size_t want = std::bit_ceil(std::max<size_t>(kMinSlots, ((n * 10) / 7) + 1));
        if (want > kMaxSlots) {
            throw TermIndexCeilingReached(
                std::format("TermTable: {} rows need more than 2^32 slots; raise the partition or rank count "
                            "to split this partition further.",
                            n));
        }
        return want;
    }

    //! Rebuilds over rows [0, store.size()), folding each row's key once, in index order.
    template <size_t NumModes>
    auto rebuild(const OperatorIndex<NumModes> &store) -> void {
        const size_t n = store.size();
        reset_(slots_for(n));
        insert_range_(store, 0, n);
        rows_ = n;
    }

    /*! @brief Indexes the rows [base, base + n) the store has just grown by. @pre base == rows().
     *
     *  Grows first if the batch would pass the load bound, re-inserting the rows already indexed from
     *  the store; the batch itself is then inserted with the slot lines prefetched a group ahead.
     */
    template <size_t NumModes>
    auto append_rows(const OperatorIndex<NumModes> &store, size_t base, size_t n) -> void {
        assert(base == rows_ && "append_rows must continue where the table's rows end");
        assert(base + n <= store.size() && "the rows to index must already be written");
        if (n == 0) {
            return;
        }
        if (slots_.empty() || (base + n) * 10 > slots_.size() * 7) {
            reset_(slots_for(base + n));
            insert_range_(store, 0, base);
        }
        insert_range_(store, base, n);
        rows_ = base + n;
    }

    /*! @brief The row whose positions are @a pos under join key @a key, or kNotFound.
     *
     *  @pre key == key_of_positions<2 * NumModes>(pos): the key is the store's own map of the term,
     *  which is what indexing folds off a row.
     */
    template <size_t NumModes>
    [[nodiscard]] auto find(const OperatorIndex<NumModes> &store,
                            uint32_t key,
                            std::span<const typename OperatorIndex<NumModes>::PosT> pos) const -> size_t {
        if (rows_ == 0) {
            return kNotFound;
        }
        const uint64_t h = spread(key);
        return find_from_(store, h & mask_, prefilter_(h), pos);
    }

    //! find() for a term in hand densely: the out-of-gate lookups, which have no position list.
    template <size_t NumModes>
    [[nodiscard]] auto find(const OperatorIndex<NumModes> &store, const Monomial<NumModes> &mono) const -> size_t {
        using PosT = typename OperatorIndex<NumModes>::PosT;
        std::array<PosT, 2 * NumModes> pos{};
        size_t k = 0;
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            pos[k++] = static_cast<PosT>(b);
        }
        return find(store, key_of_positions<2 * NumModes>(pos.data(), k), std::span<const PosT>(pos.data(), k));
    }

    /*! @brief Probes @a n queries in order, out[q] = the confirmed row of query q, or @a not_found.
     *
     *  `key_of(q)` is query q's join key and `pos_of(q)` its ascending positions (the confirm).
     *  `on_hit(q, row)` runs at every confirm, in query order within a group.
     *
     *  Same result as n calls to find(), query by query. The group pipeline overlaps the three dependent
     *  misses of a probe (slot line, chain walk, row) across kGroup queries; a prefilter match that the
     *  row refutes resumes the chain synchronously, which a 32-bit collision is rare enough to afford.
     *
     *  @return How many queries confirmed a row.
     */
    template <size_t NumModes, typename KeyOf, typename PosOf, typename OnHit>
    auto find_batch(const OperatorIndex<NumModes> &store,
                    size_t n,
                    KeyOf &&key_of,
                    PosOf &&pos_of,
                    OnHit &&on_hit,
                    std::span<TermIndex> out,
                    TermIndex not_found) const -> size_t {
        assert(out.size() >= n && "one output slot per query");
        size_t hits = 0;
        if (rows_ == 0) {
            std::fill_n(out.begin(), n, not_found);
            return 0;
        }
        std::array<bool, kGroup> cand{};
        std::array<size_t, kGroup> at{};
        std::array<uint32_t, kGroup> pf{};
        const uint32_t *const slots = slots_.data();
        for (size_t base = 0; base < n; base += kGroup) {
            const size_t g = std::min(kGroup, n - base);
            for (size_t j = 0; j < g; ++j) {
                const uint64_t h = spread(key_of(base + j));
                at[j] = h & mask_;
                pf[j] = prefilter_(h);
                __builtin_prefetch(&slots[at[j]], 0, 0);
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = false;
                for (size_t s = at[j];; s = (s + 1) & mask_) {
                    const uint32_t e = slots[s];
                    if (e == kEmpty) {
                        break;
                    }
                    if ((static_cast<uint64_t>(e) >> shift_) == pf[j]) {
                        at[j] = s;
                        cand[j] = true;
                        store.prefetch_row(e & mask_);
                        break;
                    }
                }
            }
            for (size_t j = 0; j < g; ++j) {
                const size_t q = base + j;
                if (!cand[j]) {
                    out[q] = not_found;
                    continue;
                }
                const auto pos = pos_of(q);
                const size_t first = slots[at[j]] & mask_;
                const size_t row =
                    store.row_eq_positions(first, pos) ? first : find_from_(store, (at[j] + 1) & mask_, pf[j], pos);
                if (row == kNotFound) {
                    out[q] = not_found;
                    continue;
                }
                out[q] = static_cast<TermIndex>(row);
                ++hits;
                on_hit(q, row);
            }
        }
        return hits;
    }

    /*! @brief The hash a key's slot and prefilter are cut from: splitmix64's finalizer over the key.
     *
     *  The join key is a GF(2)-linear projection of mix64 labels, uniform but structured (key(S) ==
     *  key(T) iff key(S ^ T) == 0), so its bits are re-mixed before the low ones choose a slot.
     */
    [[nodiscard]] static auto spread(uint32_t key) noexcept -> uint64_t {
        uint64_t x = static_cast<uint64_t>(key) * 0x9E37'79B9'7F4A'7C15ULL;
        x ^= x >> 30;
        x *= 0xBF58'476D'1CE4'E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D0'49BB'1331'11EBULL;
        x ^= x >> 31;
        return x;
    }

private:
    static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();
    static constexpr size_t kMaxSlots = size_t{1} << 32;

    //! The hash bits above the slot index, as many as a 32-bit slot has left over: the compare prefilter.
    [[nodiscard]] auto prefilter_(uint64_t h) const noexcept -> uint32_t {
        return static_cast<uint32_t>((h >> shift_) & pf_mask_);
    }
    //! A slot's word: the row index below the prefilter. Never kEmpty, since idx < 0.7 * slots < mask_.
    [[nodiscard]] auto entry_(size_t idx, uint64_t h) const noexcept -> uint32_t {
        return static_cast<uint32_t>(static_cast<uint64_t>(idx) | (static_cast<uint64_t>(prefilter_(h)) << shift_));
    }

    //! An empty table of @a n_slots slots (a power of two). Releases the old storage first.
    auto reset_(size_t n_slots) -> void {
        assert(std::has_single_bit(n_slots) && n_slots >= kMinSlots && n_slots <= kMaxSlots);
        slots_ = std::vector<uint32_t>{};
        slots_.assign(n_slots, kEmpty);
        mask_ = n_slots - 1;
        shift_ = static_cast<unsigned>(std::countr_zero(n_slots));
        pf_mask_ = shift_ >= 32 ? 0U : (std::numeric_limits<uint32_t>::max() >> shift_);
        rows_ = 0;
    }

    //! Inserts rows [first, first + n) under their folded keys, slot lines prefetched a group ahead.
    template <size_t NumModes>
    auto insert_range_(const OperatorIndex<NumModes> &store, size_t first, size_t n) -> void {
        std::array<uint64_t, kGroup> hh{};
        uint32_t *const slots = slots_.data();
        for (size_t b = first; b < first + n; b += kGroup) {
            const size_t g = std::min(kGroup, first + n - b);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = spread(store.key_of_row(b + j));
                __builtin_prefetch(&slots[hh[j] & mask_], 1, 0);
            }
            for (size_t j = 0; j < g; ++j) {
                size_t s = hh[j] & mask_;
                while (slots[s] != kEmpty) {
                    s = (s + 1) & mask_;
                }
                slots[s] = entry_(b + j, hh[j]);
            }
        }
    }

    //! The chain walk from slot @a s with the row confirm: find() and find_batch's collision arm.
    template <size_t NumModes>
    [[nodiscard]] auto find_from_(const OperatorIndex<NumModes> &store,
                                  size_t s,
                                  uint32_t pf,
                                  std::span<const typename OperatorIndex<NumModes>::PosT> pos) const -> size_t {
        const uint32_t *const slots = slots_.data();
        for (;; s = (s + 1) & mask_) {
            const uint32_t e = slots[s];
            if (e == kEmpty) {
                return kNotFound;
            }
            if ((static_cast<uint64_t>(e) >> shift_) == pf && store.row_eq_positions(e & mask_, pos)) {
                return e & mask_;
            }
        }
    }

    std::vector<uint32_t> slots_; //!< row index | prefilter << shift_, or kEmpty
    size_t rows_ = 0;             //!< rows indexed: [0, rows_) of the store
    size_t mask_ = 0;             //!< slots - 1
    unsigned shift_ = 0;          //!< log2(slots): the row index's width inside a slot
    uint32_t pf_mask_ = 0;        //!< the prefilter's width, 32 - shift_ bits
};

} // namespace monoprop::detail
