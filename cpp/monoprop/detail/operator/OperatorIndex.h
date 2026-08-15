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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/EnvConfig.h"

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

    // set() from the row's own form. A row IS an ascending position list, so a caller that already holds
    // one -- the compact query record carries exactly this -- is otherwise made to pack it into a dense
    // bitset only for set() to unpack it again with a count() and a find_first/find_next walk over every
    // word. Same postcondition as set(), including the dropped stale overflow entry.
    //
    // `pos` must be strictly ascending and every entry < 2*NumModes; both are the record's own invariants
    // (CompactQuery::push asserts ascending on the way out) and both are asserted here on the way in,
    // because a violation is silent: an unsorted row still compares equal to nothing and simply never
    // matches, and an out-of-range position writes a row that row() would decode into a different term.
    auto set_positions(size_t i, const PosT *pos, size_t count) -> void {
        assert(std::is_sorted(pos, pos + count, std::less_equal<PosT>{}) && "row positions must be strictly ascending");
        assert((count == 0 || static_cast<size_t>(pos[count - 1]) < 2 * NumModes) && "row position out of range");
        PosT *row = &rows_[i * stride_];
        if (count > inline_width_) {
            // The spill path has no position array by construction, so the dense form has to be built --
            // but only here, on the ~1-in-20M rows that reach it.
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
        std::copy_n(pos, count, row + 1);
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
    // The row's ascending positions as they are stored, for callers that can consume the sparse form
    // directly instead of rebuilding a bitset from it. Empty (nullptr, 0) for a spilled row: the
    // overflow map holds a Monomial and has no position array to point at, so a caller must fall back
    // to for_each_position / row(). The pointer is invalidated by any insert, exactly like rows_.
    struct RowPositions {
        const PosT *pos;
        size_t count;
        [[nodiscard]] auto inlined() const -> bool { return pos != nullptr; }
    };
    [[nodiscard]] auto row_positions(size_t i) const -> RowPositions {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            return {nullptr, 0};
        }
        return {&rows_[(i * stride_) + 1], static_cast<size_t>(c)};
    }
    // Rows whose popcount exceeded inline_width_ and spilled. Diagnostic: every hot-path row read on
    // one of these is a hash-map lookup, so this fraction is what decides whether the width is right.
    [[nodiscard]] auto overflow_size() const -> size_t { return overflow_.size(); }
    [[nodiscard]] auto inline_width() const -> size_t { return inline_width_; }
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
    //
    // hash_out, when non-null, receives the folded hash of every key. It is not a diagnostic: at the
    // measured ~0 hit rate essentially every query becomes an insert, and the insert path recomputes
    // exactly this hash from exactly this key -- so without it the splitmix runs twice per term.
    // Feed it to bulk_insert_hashed.
    auto find_batch(const key_type *keys, size_t n, size_t *out, uint32_t *hash_out = nullptr) const -> void {
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
            if (hash_out != nullptr) {
                std::copy_n(hh.begin(), g, hash_out + base);
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

    // find_batch over ascending position lists instead of dense monomials: query q is
    // pos_flat[pos_off[q] .. pos_off[q] + k_of[q]). Identical results to find_batch on the monomials
    // those positions describe -- the hash is the same function of the same key, and the confirm is a
    // position-vs-position compare of the same two lists.
    //
    // WHY IT EXISTS. The caller already holds positions: that is what arrives on the wire under the
    // compact record. Handing find_batch a dense key means writing 64 B per query into an array sized
    // by the whole incoming batch, reading it back once to hash it, and then unpacking it a third time
    // at insert. This entry point lets the positions stay the currency end to end; the one dense
    // monomial still built per query is a stack temporary the hash consumes immediately, never a store.
    //
    // The three-stage pipeline is deliberately the same shape as find_batch's, including the row
    // prefetch that is dead weight at a ~0 hit rate: it is gated on a 32-bit hash match, which fires
    // ~21 times in 22M probes, so it costs nothing and stays honest on a hit-heavy workload.
    auto find_batch_positions(const PosT *pos_flat,
                              const size_t *pos_off,
                              const uint32_t *k_of,
                              size_t n,
                              size_t *out,
                              uint32_t *hash_out = nullptr) const -> void {
        static constexpr size_t G = 16;
        std::array<uint32_t, G> hh;
        std::array<size_t, G> sp;
        std::array<TermIndex, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                hh[j] = fold_hash_positions(pos_flat + pos_off[base + j], k_of[base + j]);
                sp[j] = spread(hh[j]);
                __builtin_prefetch(&table_.slots[sp[j] & table_.mask], 0, 0);
            }
            if (hash_out != nullptr) {
                std::copy_n(hh.begin(), g, hash_out + base);
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
                const PosT *qpos = pos_flat + pos_off[q];
                const size_t qk = k_of[q];
                if (cand[j] == kEmptySlot) {
                    out[q] = kNotFound;
                }
                else if (row_eq_positions(static_cast<size_t>(cand[j]), qpos, qk)) {
                    out[q] = static_cast<size_t>(cand[j]);
                }
                else {
                    // A 32-bit collision along the chain. Rare enough to be measured at ~0.01 events per
                    // run, so this walks the chain from the top rather than resuming it.
                    out[q] = find_positions_(hh[j], qpos, qk);
                }
            }
        }
    }

    // fold_hash of the monomial `pos` describes. Built through the same Monomial and the same fold, so
    // it is equal to fold_hash(mono) by construction rather than by an identity that has to be proved
    // and could drift when either side is edited. The bitset is a stack temporary: what this avoids is
    // the caller's per-query 64 B array, not the hash itself.
    [[nodiscard]] static auto fold_hash_positions(const PosT *pos, size_t count) noexcept -> uint32_t {
        key_type mono;
        for (size_t j = 0; j < count; ++j) {
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
        // Folding the key here and delegating means the two entry points share one insert loop, so the
        // prefetch pipeline below cannot end up on only one of them.
        bulk_insert_hashed(n, base, [&](size_t k) { return fold_hash(key_at(k)); });
    }
    // bulk_insert with the hashes already in hand. Same precondition (n distinct rows, already written,
    // at consecutive indices) and the same slot assignment; the only difference is that it does not
    // recompute a hash the caller's probe just computed from the same key. `hashes[k]` MUST be
    // fold_hash of the key of row base+k -- a wrong hash does not corrupt the row, it makes the row
    // unfindable, which reads later as a duplicate insert rather than as an error here.
    //
    // GROUP PREFETCH, and why it belongs here. find_batch hides its probe miss behind a 16-wide
    // prefetch pipeline; this loop is the same random access into the same table on the same workload
    // -- at the measured hit rate essentially every incoming query becomes an insert, so this runs ~22M
    // times per build_graph against a 4 MB table with two partitions sharing a 16 MB L3 -- and it had no
    // pipeline at all. Issuing the group's slot addresses before walking any of them puts G misses in
    // flight instead of one.
    //
    // Correctness does not depend on any of it: a prefetch is a hint, the insert re-reads the slot, and
    // a rehash mid-group only makes the remaining hints stale (the address is recomputed from the
    // current mask at insert time). hash_at is called exactly ONCE per element and buffered, because on
    // the key-taking path it is a splitmix over eight words, not an array read.
    //
    // This was briefly selectable, because three "obviously fewer instructions" changes on this branch
    // had measured SLOWER and a prefetch that misses its window is cache pollution rather than a no-op.
    // It measured 0.9008x and 0.9106x on incoming_s and 0.8786x/0.8763x on insert_s, 8/8 paired reps in
    // each of two independent cells at p=0.0078, so the unpipelined loop is gone. Note what separates it
    // from the three that lost: it removes no instructions, it overlaps a DRAM miss. The wins on this
    // branch have been memory-system wins and the losses instruction-count guesses.
    template <typename HashFn>
    auto bulk_insert_hashed(size_t n, mapped_type base, HashFn &&hash_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        static constexpr size_t G = 16; // same group width as find_batch, for the same reason
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

    // Compare row i against an ascending position list. Both sides are ascending position lists, so this
    // is the natural form of the confirm and the dense round-trip in row_eq_key exists only because the
    // key used to arrive dense. A spilled row has no position array, so it falls back to a dense compare
    // built from the query -- the one place this form has to materialise anything.
    [[nodiscard]] auto row_eq_positions(size_t i, const PosT *q, size_t qk) const -> bool {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            key_type mono;
            for (size_t j = 0; j < qk; ++j) {
                mono.set(q[j]);
            }
            return overflow_.at(i) == mono;
        }
        if (qk != static_cast<size_t>(c)) {
            return false;
        }
        return std::equal(q, q + qk, &rows_[(i * stride_) + 1]);
    }

    // find()'s chain walk for a position-list key, entered with the hash already folded. Only reachable
    // from find_batch_positions' 32-bit-collision arm.
    [[nodiscard]] auto find_positions_(uint32_t h, const PosT *q, size_t qk) const -> size_t {
        if (table_.count == 0) {
            return kNotFound;
        }
        for (size_t s = spread(h) & table_.mask;; s = (s + 1) & table_.mask) {
            const Slot &e = table_.slots[s];
            if (e.idx == kEmptySlot) {
                return kNotFound;
            }
            if (e.h == h && row_eq_positions(static_cast<size_t>(e.idx), q, qk)) {
                return static_cast<size_t>(e.idx);
            }
        }
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                                          "-Dmonoprop_WIDE_TERM_INDEX (this partition's term count exceeded ~2^32).");
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
