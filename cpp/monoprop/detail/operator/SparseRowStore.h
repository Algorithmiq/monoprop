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
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/operator/RowHashTable.h"

// Logical mode count at or above which the sparse rows are the cheaper backend. Build-time and not a
// runtime knob because what moves the crossover is the target ISA, which is fixed when the translation
// unit is compiled: dense costs one pass per storage word and sparse is flat in the width, so without a
// vector popcount the dense pass degrades an order of magnitude sooner. Set from CMake off
// monoprop_ARCH_MARCH (see the top-level CMakeLists for the measured values); the fallback here is the
// conservative baseline-x86-64 one, so a consumer compiling these headers without the project's
// definitions gets the value that suits the wheels rather than the developer's machine.
#if !defined(monoprop_SPARSE_ROW_MIN_MODES)
#define monoprop_SPARSE_ROW_MIN_MODES 256
#endif

namespace monoprop::detail {

// The requested width needs mode indices wider than ModeT can hold, or more slots than a codes word
// has room for. Thrown rather than asserted, for the same reason OperatorIndexWidthUnsupported is:
// with the compile-time mode ceiling gone, the width is user data.
class SparseRowStoreUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A sparse row's two storage types, at namespace scope rather than inside the store: the algebra that
// reads rows (CodesAlgebra.h) must not depend on the container that owns them.
using RowMode = uint16_t;
using RowCodes = uint64_t;

// Two bits per slot in one RowCodes. A wider row is representable in the mode lanes but would put the
// algebra back on a multi-word loop, which is the whole cost the support form removes.
inline constexpr size_t kRowMaxSlots = 32;
inline constexpr RowCodes kRowLoBits = 0x5555555555555555ULL; // bit 2j of every slot

// Bit 2j of each slot: set iff slot j is occupied at all. popcount is n, the support measure.
[[nodiscard]] inline constexpr auto row_occupied_bits(RowCodes codes) noexcept -> RowCodes {
    return (codes | (codes >> 1)) & kRowLoBits;
}
// Bit 2j of each slot: set iff slot j holds both of its positions. popcount is d.
[[nodiscard]] inline constexpr auto row_paired_bits(RowCodes codes) noexcept -> RowCodes {
    return codes & (codes >> 1) & kRowLoBits;
}
[[nodiscard]] inline constexpr auto row_slot_count(RowCodes codes) noexcept -> size_t {
    return static_cast<size_t>(std::popcount(row_occupied_bits(codes)));
}

// Non-owning view of one row: ascending mode lanes plus the codes word. The lane array is only read
// below num_slots(), which the codes word determines -- so a view stays valid over a padded row and
// carries no length of its own. It borrows the store's arrays, so it must not outlive them, and a row
// mutation invalidates it the way an iterator would.
struct SparseRow {
    const RowMode *modes = nullptr;
    RowCodes codes = 0;

    [[nodiscard]] auto num_slots() const noexcept -> size_t { return row_slot_count(codes); }
    [[nodiscard]] auto mode(size_t j) const noexcept -> size_t { return static_cast<size_t>(modes[j]); }
    // The 2-bit field of slot j: 0b01 is physical position 2*mode alone, 0b10 is 2*mode+1 alone, 0b11
    // the paired mode.
    [[nodiscard]] auto code(size_t j) const noexcept -> unsigned int {
        return static_cast<unsigned int>((codes >> (2 * j)) & 0b11U);
    }
};

// A row *key* that may be too wide for a codes word: `spilled` non-null means the key is that dense
// monomial and `row` is unread. These are the two shapes a stored row already has, and a query needs
// both for the same reason a stored row does -- a query is M ⊕ G, and a fully paired product escapes
// the cutoff, so nothing bounds its support.
//
// Deliberately not folded into SparseRow, which is what the per-term algebra reads: that one is a view
// of something that fits, and a branch on every read of it is the cost the support form exists to
// avoid. The branch belongs here, on the probe path, where it runs once per query.
struct SparseRowKey {
    SparseRow row;
    const Bitset *spilled = nullptr;

    [[nodiscard]] auto is_spilled() const noexcept -> bool { return spilled != nullptr; }
};

// Visits a *dense* monomial as (mode, code) slots, ascending: the same sequence a SparseRow over the
// same monomial yields. Positions arrive ascending, so a mode's two positions are adjacent and one pass
// closes each slot before opening the next.
template <typename Fn>
inline auto for_each_mode_slot(const Bitset &mono, Fn &&fn) -> void {
    size_t pos = mono.find_first();
    while (pos < mono.size()) {
        const size_t mode = pos >> 1;
        unsigned int code = 1U << (pos & 1U);
        pos = mono.find_next(pos);
        if (pos < mono.size() && (pos >> 1) == mode) {
            code |= 1U << (pos & 1U);
            pos = mono.find_next(pos);
        }
        fn(mode, code);
    }
}

// The row hash, as an accumulator over (mode, code) slots. One definition with two walkers -- a sparse
// row and a dense monomial -- because a keyed store must hold both and hash them identically: a fully
// paired term escapes the cutoff, so a row can occupy more modes than any codes word holds and has to
// spill to the dense side map, where it still needs to be findable.
//
// That requirement is why the mix is *sequential* rather than an XOR-fold of slot-indexed terms, which
// is what the plan's "fixed-width mix over the padded row" would have been. Sequential mixing is
// positional without packing a slot index into the mixed word, so it does not care how many slots there
// are -- and it depends on neither the row capacity nor the padding, so two stores tuned to different
// capacities agree, which matters because the hash decides probe order.
class SparseRowHasher {
public:
    auto add(size_t mode, unsigned int code) noexcept -> void {
        h_ = SplitmixHash::mix(h_ ^ ((static_cast<uint64_t>(mode) << 2) | code));
    }
    [[nodiscard]] auto value() const noexcept -> size_t { return static_cast<size_t>(h_); }

private:
    // Nonzero, so an empty row does not hash to zero and every slot count starts from a mixed state.
    uint64_t h_ = 0x9E3779B97F4A7C15ULL;
};

[[nodiscard]] inline auto sparse_row_hash(const SparseRow &row) noexcept -> size_t {
    SparseRowHasher hasher;
    const size_t n = row.num_slots();
    for (size_t j = 0; j < n; ++j) {
        hasher.add(row.mode(j), row.code(j));
    }
    return hasher.value();
}

[[nodiscard]] inline auto sparse_row_hash(const Bitset &mono) noexcept -> size_t {
    SparseRowHasher hasher;
    for_each_mode_slot(mono, [&](size_t mode, unsigned int code) { hasher.add(mode, code); });
    return hasher.value();
}

// Dispatches to whichever shape the key holds, so a batch of keys hashes identically whether or not any
// of them spilled. The two arms must agree with the store's own row hash, which is what makes a spilled
// row findable by either form.
[[nodiscard]] inline auto sparse_row_hash(const SparseRowKey &key) noexcept -> size_t {
    return key.is_spilled() ? sparse_row_hash(*key.spilled) : sparse_row_hash(key.row);
}

// Whether a dense monomial and a sparse row hold the same slots, without materializing either. Used
// where one side is a spilled row (no codes word) and the other is a query.
[[nodiscard]] inline auto dense_row_equals(const Bitset &mono, const SparseRow &row) -> bool {
    const size_t n = row.num_slots();
    size_t j = 0;
    bool equal = true;
    for_each_mode_slot(mono, [&](size_t mode, unsigned int code) {
        if (!equal) {
            return;
        }
        if (j >= n || row.mode(j) != mode || row.code(j) != code) {
            equal = false;
            return;
        }
        ++j;
    });
    return equal && j == n;
}

// Operator-term store in support form: each row is a fixed-width list of the *modes* it occupies plus
// one word holding two bits per occupied mode. It is the third backend behind the four TypeAliases.h
// row accessors, alongside std::vector<Bitset> and OperatorIndex, and agrees with both through them
// (cpp/tests/row_accessor_tests.cpp).
//
// Layout, structure-of-arrays: modes_ is `slots_per_row_` ModeT lanes per row, ascending, padded with
// kPadLane; codes_ is one CodesT per row. The two live in separate arrays on purpose -- the cutoff and
// pairing algebra reads only codes_, so evaluating it over a run of rows is a sequential walk that
// never touches a mode list.
//
// codes_ packs slot j (the j-th occupied mode, ascending) into bits 2j and 2j+1: bit 2j marks physical
// position 2*mode, bit 2j+1 marks 2*mode+1. The whole cutoff algebra follows from that one word --
// with occupied = (codes | codes>>1) & 0x5555..., paired = codes & (codes>>1) & 0x5555...,
// n = popcount(occupied) and d = popcount(paired) give or_sum = n, popcount_sum = n + d and
// xor_sum = n - d, independent of the storage width. popcount() below is the first consumer; the rest
// arrives with the algebra port.
//
// Rows wider than slots_per_row_ spill losslessly to a side map. They are not a corner case to be
// ruled out by sizing: a fully-paired term escapes the cutoff (xor_sum == 0 is kept unconditionally),
// so support is genuinely unbounded no matter what the cutoff is. They are rare -- ~0.07% of rows on
// production models, per MPOperator.h -- and an all-0b11 row needs only its mode list, so a second
// cheap row kind is available if that ever stops being true.
//
// The keyless index over the rows is the shared RowHashTable, the same one OperatorIndex uses, so both
// stores produce the same slot layout for the same insertion sequence. What differs is only what a key
// is: rows hash through sparse_row_hash and confirm through a codes compare plus a lane memcmp, where
// OperatorIndex hashes a whole Bitset. That hash is *not* the dense one, so a store swap changes probe
// order, MPI owner routing and therefore floating-point accumulation order -- the deliberate
// re-baseline, not a regression.
//
// static_assert cannot express it, so: SparseRowStore is interchangeable with OperatorIndex through the
// TypeAliases.h accessors and through find/emplace/bulk_insert/find_batch, and cpp/tests are what hold
// that. It is not a subclass of anything and nothing dispatches on it.
//
// Single-writer, like OperatorIndex: one partition, one thread; parallelism is cross-partition.
class SparseRowStore {
public:
    using value_type = Bitset;
    using key_type = Bitset;
    using mapped_type = size_t;
    using ModeT = RowMode;
    using CodesT = RowCodes;

    static constexpr size_t kMaxSlots = kRowMaxSlots;
    static constexpr size_t kDefaultSlots = 8;

    // The top two ModeT values are markers, so a valid mode index is at most kPadLane - 2. kPadLane
    // fills the unused lanes of a short row (fixed, so two equal rows have equal lanes); kOverflowLane
    // sits in lane 0 of a spilled row, where it cannot be confused with the empty row's kPadLane.
    static constexpr ModeT kPadLane = std::numeric_limits<ModeT>::max();
    static constexpr ModeT kOverflowLane = static_cast<ModeT>(kPadLane - 1);
    static constexpr size_t kMaxModes = static_cast<size_t>(kOverflowLane); // exclusive bound

    static constexpr size_t kIndexCeiling = RowHashTable::kIndexCeiling;
    static constexpr size_t kNotFound = RowHashTable::kNotFound;

    // The Stage 3 crossover, as a predicate rather than a bare number so the rule has one home.
    static constexpr size_t kMinModes = monoprop_SPARSE_ROW_MIN_MODES;
    [[nodiscard]] static constexpr auto preferred_for_modes(size_t num_modes) noexcept -> bool {
        return num_modes >= kMinModes;
    }

    // num_bits is the storage bit width of every monomial this store will hold, exactly as for
    // OperatorIndex: row() reconstructs at that width, and a wrong one changes num_words() and with it
    // the hash, the probe order and MPI owner routing. slots_per_row is the per-row mode capacity --
    // any value is correct, since over-long rows spill; size it from CutoffEvaluator::max_mode_bound().
    explicit SparseRowStore(size_t num_bits, size_t slots_per_row = kDefaultSlots)
        : num_bits_(num_bits),
          slots_per_row_(std::clamp<size_t>(slots_per_row, 1, kMaxSlots)) {
        if (((num_bits + 1) / 2) > kMaxModes) {
            throw SparseRowStoreUnsupported(
                std::format("SparseRowStore supports at most {} modes ({} bits); got {} bits ({} modes).",
                            kMaxModes,
                            2 * kMaxModes,
                            num_bits,
                            (num_bits + 1) / 2));
        }
    }

    SparseRowStore(const SparseRowStore &) = delete;
    SparseRowStore &operator=(const SparseRowStore &) = delete;
    SparseRowStore(SparseRowStore &&) = delete;
    SparseRowStore &operator=(SparseRowStore &&) = delete;

    [[nodiscard]] auto num_bits() const noexcept -> size_t { return num_bits_; }
    [[nodiscard]] auto slots_per_row() const noexcept -> size_t { return slots_per_row_; }
    [[nodiscard]] auto size() const noexcept -> size_t { return size_; }

    // Called only on an idle store, so it needs no synchronization.
    [[nodiscard]] auto clone() const -> std::unique_ptr<SparseRowStore> {
        auto out = std::make_unique<SparseRowStore>(num_bits_, slots_per_row_);
        out->modes_ = modes_;
        out->codes_ = codes_;
        out->size_ = size_;
        out->overflow_ = overflow_;
        out->table_.reserve(table_.count());
        table_.for_each_slot([&](TermIndex idx, uint32_t h) { out->table_.insert_distinct(idx, h); });
        return out;
    }

    auto reserve(size_t n) -> void {
        reserve_rows_(n);
        table_.reserve(n);
    }

    // Returns the pre-growth size (the caller's insert base). Growth is geometric (1.5x), never
    // exact-fit: an exact fit would realloc the whole operator every layer. Rows only -- the table grows
    // on its own load factor, and pre-sizing it per layer would rehash for nothing.
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve_rows_(std::max(base + n, cap + (cap / 2) + 1));
        }
        // Default-init grow, not a zeroing resize: every freshly grown row is overwritten by set()
        // before any read, so a tail zero-fill would be wasted bandwidth.
        modes_.resize((base + n) * slots_per_row_);
        codes_.resize(base + n);
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows_geometric(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so nothing in the row is pre-read; a
    // stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        ModeT *lanes = &modes_[i * slots_per_row_];
        CodesT codes = 0;
        size_t used = 0;
        bool overflows = false;
        // The slot walk is shared with the hash, so the two cannot disagree about what a row's slots are.
        // Lanes come out ascending because the walk is.
        for_each_mode_slot(mono, [&](size_t mode, unsigned int code) {
            if (overflows) {
                return;
            }
            if (used == slots_per_row_) {
                overflows = true;
                return;
            }
            lanes[used] = static_cast<ModeT>(mode);
            codes |= static_cast<CodesT>(code) << (2 * used);
            ++used;
        });
        if (overflows) {
            lanes[0] = kOverflowLane;
            codes_[i] = 0;
            overflow_[i] = mono;
            return;
        }
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
        for (size_t j = used; j < slots_per_row_; ++j) {
            lanes[j] = kPadLane;
        }
        codes_[i] = codes;
    }

    // The row form of set(), and the write the support form exists for: the lanes are already ascending
    // and the codes word already says what each holds, so this copies `n` lanes and one word where the
    // dense overload walks the monomial's storage words.
    //
    // A row wider than this store's capacity still has to spill, and a spilled row is held densely, so
    // that arm materializes. It cannot be asserted away: the capacity is sized from the cutoff and a
    // fully paired term escapes the cutoff.
    auto set(size_t i, const SparseRow &row) -> void {
        const size_t n = row.num_slots();
        // Contiguity from slot 0 is the representation's invariant -- num_slots() counts occupied slots
        // and the lanes are read from 0 -- so a row with a hole would silently lose its high slots here.
        assert((n >= kRowMaxSlots || (row.codes >> (2 * n)) == 0) && "SparseRow slots must be contiguous from slot 0");
        ModeT *lanes = &modes_[i * slots_per_row_];
        if (n > slots_per_row_) {
            lanes[0] = kOverflowLane;
            codes_[i] = 0;
            overflow_[i] = to_monomial_(row);
            return;
        }
        // Hygiene, not correctness: spilled() reads lane 0, so a stale entry here is already unreachable
        // -- it would just keep a monomial alive for the store's lifetime.
        if (!overflow_.empty()) {
            overflow_.erase(i);
        }
        std::memcpy(lanes, row.modes, n * sizeof(ModeT));
        // Padding is load-bearing for the empty row and only for it: with n == 0 nothing above writes a
        // lane, so lane 0 would keep a previous occupant's kOverflowLane and the row would read as spilled.
        for (size_t j = n; j < slots_per_row_; ++j) {
            lanes[j] = kPadLane;
        }
        codes_[i] = row.codes;
    }

    // Whichever shape the key holds. The spilled arm is the dense set(), so a key that arrived too wide
    // for a codes word lands in the side map exactly as the dense path would have put it.
    auto set(size_t i, const SparseRowKey &key) -> void {
        if (key.is_spilled()) {
            set(i, *key.spilled);
            return;
        }
        set(i, key.row);
    }

    [[nodiscard]] auto row(size_t i) const -> value_type {
        if (spilled(i)) {
            return overflow_.at(i);
        }
        value_type mono(num_bits_);
        for_each_slot(i, [&](size_t mode, unsigned int code) {
            if ((code & 1U) != 0U) {
                mono.set(2 * mode);
            }
            if ((code & 2U) != 0U) {
                mono.set((2 * mode) + 1);
            }
        });
        return mono;
    }

    // Ascending, matching the dense backends: slots are stored ascending in the mode, and within a mode
    // position 2*mode precedes 2*mode+1.
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        if (spilled(i)) {
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        for_each_slot(i, [&](size_t mode, unsigned int code) {
            if ((code & 1U) != 0U) {
                fn(2 * mode);
            }
            if ((code & 2U) != 0U) {
                fn((2 * mode) + 1);
            }
        });
    }

    // Visits (mode, code) per occupied slot, ascending in the mode; code is the 2-bit field, so 0b01 is
    // position 2*mode alone, 0b10 is 2*mode+1 alone and 0b11 is the paired mode. Row i must not be
    // spilled -- a spilled row has no slots, and its lane 0 marker would read as a mode.
    template <typename Fn>
    auto for_each_slot(size_t i, Fn &&fn) const -> void {
        assert(!spilled(i) && "SparseRowStore::for_each_slot on a spilled row");
        const ModeT *lanes = &modes_[i * slots_per_row_];
        const CodesT codes = codes_[i];
        for (size_t j = 0; j < slots_per_row_ && lanes[j] != kPadLane; ++j) {
            fn(static_cast<size_t>(lanes[j]), static_cast<unsigned int>((codes >> (2 * j)) & 0b11U));
        }
    }

    // The row's codes word. Meaningless for a spilled row -- ask spilled(i) first; the algebra port
    // will need the same guard, which is why the spill is kept rare rather than made general.
    [[nodiscard]] auto codes(size_t i) const -> CodesT { return codes_[i]; }

    // What the codes algebra reads. Borrows this store's arrays, so it is invalidated by anything that
    // reallocates them (grow_rows_geometric, reserve) or rewrites row i; row i must not be spilled.
    [[nodiscard]] auto view(size_t i) const -> SparseRow {
        assert(!spilled(i) && "SparseRowStore::view on a spilled row");
        return SparseRow{&modes_[i * slots_per_row_], codes_[i]};
    }

    [[nodiscard]] auto spilled(size_t i) const -> bool { return modes_[i * slots_per_row_] == kOverflowLane; }

    // Occupied modes -- the support measure, or_sum.
    [[nodiscard]] auto slot_count(size_t i) const -> size_t {
        if (spilled(i)) {
            return occupied_modes(overflow_.at(i));
        }
        return row_slot_count(codes_[i]);
    }

    // Set bits -- the length measure, popcount_sum = n + d straight off the codes word.
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        if (spilled(i)) {
            return overflow_.at(i).count();
        }
        const CodesT codes = codes_[i];
        return row_slot_count(codes) + static_cast<size_t>(std::popcount(row_paired_bits(codes)));
    }

    // --- the keyless index over those rows ---------------------------------------------------------
    //
    // A key is a SparseRow or a Bitset, and both hash through sparse_row_hash, so the two are
    // interchangeable at a call site. Prefer the row: it is what the scan holds, and it is the only form
    // that needs no slot walk to hash. The Bitset form is what a caller still holding a monomial uses.

    auto find(const SparseRow &key) const -> std::optional<size_t> { return find_hashed_(key); }
    auto find(const key_type &key) const -> std::optional<size_t> { return find_hashed_(key); }
    auto find(const SparseRowKey &key) const -> std::optional<size_t> { return find_hashed_(key); }

    // Insert-or-no-op. The row at `value` must already be written -- the confirm reads it.
    template <typename Key>
    auto emplace(const Key &key, mapped_type value) -> void {
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

    // out[i] = row index of keys[i], or kNotFound. Same result as n find() calls; see
    // RowHashTable::find_batch for why the row prefetch sits between probe and confirm. Both arrays are
    // prefetched: the confirm reads the codes word first and the lanes only if it matches, but they are
    // separate allocations and so separate cache misses.
    template <typename Key>
    auto find_batch(const Key *keys, size_t n, size_t *out) const -> void {
        table_.find_batch(
            keys,
            n,
            out,
            [this](const Key &key) { return fold_hash(key); },
            [this](size_t i) {
                __builtin_prefetch(&codes_[i], 0, 0);
                __builtin_prefetch(&modes_[i * slots_per_row_], 0, 0);
            },
            [this](size_t i, const Key &key) { return row_eq_key(i, key); });
    }

    // Rows in index order, as fn(row_index). Not the row itself: a spilled row has no view, so what a
    // caller wants off the index is the index.
    template <typename Fn>
    auto for_each_index(Fn &&fn) const -> void {
        table_.for_each_slot([&](TermIndex idx, uint32_t) { fn(static_cast<size_t>(idx)); });
    }

    // OperatorIndex's signature, fn(monomial, row_index), so the two stores are interchangeable at the
    // one call site that wants both. Materializes each row, which for_each_index does not -- prefer that
    // where the index alone will do.
    template <typename Fn>
    auto for_each(Fn &&fn) const -> void {
        for_each_index([&](size_t i) { fn(row(i), i); });
    }

    [[nodiscard]] auto indexed_count() const noexcept -> size_t { return table_.count(); }

    [[nodiscard]] auto index_estimated_memory_bytes() const -> size_t {
        return sizeof(SparseRowStore) + table_.slot_bytes();
    }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = modes_.capacity() * sizeof(ModeT);
        total += codes_.capacity() * sizeof(CodesT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    // Diagnostic: the part of memory_bytes() that is unused geometric-growth capacity.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        const size_t lanes = modes_.capacity() - std::min(modes_.capacity(), size_ * slots_per_row_);
        const size_t words = codes_.capacity() - std::min(codes_.capacity(), size_);
        return (lanes * sizeof(ModeT)) + (words * sizeof(CodesT));
    }

    // Slot count for a cutoff bound in modes (CutoffEvaluator::max_mode_bound()), clamped to what one
    // codes word holds. A bound above kMaxSlots is not an error: the rows that exceed it spill.
    [[nodiscard]] static auto slots_for_bound(size_t mode_bound) noexcept -> size_t {
        return std::clamp<size_t>(mode_bound, 1, kMaxSlots);
    }

    // Slot count for a row in flight rather than a row at rest: a product occupies up to the term's modes
    // plus the generator's, so a scan scratch row and a wire record both need the cutoff's mode bound plus
    // the widest generator's locality. Every rank derives this from the same circuit and cutoff, so they
    // agree on it without communication -- which is what lets it fix a wire stride.
    [[nodiscard]] static auto scratch_slots_for(size_t mode_bound, size_t max_generator_modes) noexcept -> size_t {
        return std::clamp<size_t>(mode_bound + max_generator_modes, 1, kMaxSlots);
    }

private:
    // Spilled rows have no codes word, so their support is counted the dense way.
    [[nodiscard]] static auto occupied_modes(const value_type &mono) -> size_t {
        size_t n = 0;
        for_each_mode_slot(mono, [&](size_t, unsigned int) { ++n; });
        return n;
    }

    // The 32-bit fold the table stores as its equality pre-filter, over the full-width row hash.
    template <typename Key>
    static auto fold_hash(const Key &key) noexcept -> uint32_t {
        const size_t full = sparse_row_hash(key);
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }

    template <typename Key>
    auto find_hashed_(const Key &key) const -> std::optional<size_t> {
        return table_.find(fold_hash(key), [&](size_t i) { return row_eq_key(i, key); });
    }

    // The find confirm, against a row query. Codes first: one word compare rejects nearly every
    // pre-filter false positive, and it is what fixes the lane compare's length -- equal codes means
    // equal slot counts, so only the occupied lanes can differ. Comparing the padded width instead (as
    // the plan sketched) would be the same cost but would silently mismatch any query a caller left
    // unpadded, and the padding is capacity-dependent where a key must not be.
    [[nodiscard]] auto row_eq_key(size_t i, const SparseRow &key) const -> bool {
        if (spilled(i)) {
            return dense_row_equals(overflow_.at(i), key);
        }
        if (codes_[i] != key.codes) {
            return false;
        }
        return std::memcmp(&modes_[i * slots_per_row_], key.modes, row_slot_count(key.codes) * sizeof(ModeT)) == 0;
    }

    // The same against a monomial query, which is the only form that can match a spilled row exactly.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &key) const -> bool {
        if (spilled(i)) {
            return overflow_.at(i) == key;
        }
        return dense_row_equals(key, view(i));
    }

    [[nodiscard]] auto row_eq_key(size_t i, const SparseRowKey &key) const -> bool {
        return key.is_spilled() ? row_eq_key(i, *key.spilled) : row_eq_key(i, key.row);
    }

    // A row at this store's width. Only the spill arms need it: everything else reads slots in place.
    [[nodiscard]] auto to_monomial_(const SparseRow &row) const -> value_type {
        value_type mono(num_bits_);
        const size_t n = row.num_slots();
        for (size_t j = 0; j < n; ++j) {
            const unsigned int code = row.code(j);
            if ((code & 1U) != 0U) {
                mono.set(2 * row.mode(j));
            }
            if ((code & 2U) != 0U) {
                mono.set((2 * row.mode(j)) + 1);
            }
        }
        return mono;
    }

    auto reserve_rows_(size_t n) -> void {
        modes_.reserve(n * slots_per_row_);
        codes_.reserve(n);
    }

    [[nodiscard]] auto capacity() const -> size_t { return codes_.capacity(); }

    DefaultInitVector<ModeT> modes_ = {};
    DefaultInitVector<CodesT> codes_ = {};
    size_t num_bits_ = 0;
    size_t size_ = 0;
    size_t slots_per_row_ = kDefaultSlots;
    // Lossless side-map for rows occupying more than slots_per_row_ modes.
    std::unordered_map<size_t, value_type> overflow_ = {};
    RowHashTable table_ = {};
};

} // namespace monoprop::detail
