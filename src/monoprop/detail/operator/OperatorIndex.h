#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

/**
 * @brief Operator-term store: entropy-packed position-list rows PLUS a keyless hash index over
 * those rows, in one self-contained object.
 *
 * ROWS (byte layout identical to the former PackedMajoranaVector):
 *   slot 0      : popcount c (or kOverflowMarker if c > inline_width_)
 *   slots 1..c  : the c set-bit positions, ascending (PosT each)
 * stride_ = 1 + inline_width_, fixed for the container's life so the parallel disjoint miss-fill
 * stays lock-free. inline_width_ is a CONSTRUCTION INVARIANT (the constructor's only argument,
 * default = kDefaultInlinePositions), clamped to kMaxInlinePositions; re-init with a different width
 * by assigning a fresh store. Callers derive the width from the cutoff (see packed_inline_width_): a
 * weight-w Pauli occupies up to 2w positions, so an under-sized width would spill the COMMON case to
 * overflow — the arena is meant for pathological high-weight terms, not the bulk. Rows whose popcount
 * exceeds the width spill LOSSLESSLY to a dense-bitset overflow map (mutex-guarded; touched only on
 * genuine overflow transitions).
 *
 * INDEX: a single OPEN-ADDRESSING table of Slot{TermIndex idx, uint32_t h} (power-of-2 capacity,
 * linear probing, max load 0.7). The 32-bit folded hash is cached at insert from the in-hand key
 * (never from a row reconstruction), so insert/rehash are gather-free; find pre-filters on h and
 * confirms a hit by reading THIS store's own row (row_eq_key). The hand-rolled table exists for one
 * capability boost::unordered_flat_set cannot expose: **find_batch**, a group-prefetch pipelined
 * lookup that overlaps the DRAM misses of many probes, which the latency-bound resolve phases need.
 *
 * The store is non-copyable/non-movable and heap-owned via unique_ptr: owners share stable pointers
 * to it across the codebase, and clone() is the single named deep-copy.
 */
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = Monomial<NumModes>;
    using key_type = Monomial<NumModes>;
    using mapped_type = size_t;

    // Position element: u8 when 2N<=256 (byte-identical to the original packed layout), widening
    // only for larger mode counts so positions never truncate.
    using PosT = std::
        conditional_t<(2 * NumModes <= 256), uint8_t, std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;

    // Default inline width when no cutoff-derived bound is supplied (e.g. Schrödinger state rows).
    // Kept at the historical value so default-constructed stores are byte-identical.
    static constexpr size_t kDefaultInlinePositions = 11;
    // Ceiling on the caller-requested inline width. A weight-w Pauli needs 2w positions; at the
    // supported Pauli cutoffs this covers the common case inline (2*cutoff <= 32 for cutoff <= 16)
    // so the bulk of terms stay out of the overflow arena. Beyond it, rows spill losslessly.
    static constexpr size_t kMaxInlinePositions = 32;
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max();

    static_assert(2 * NumModes - 1 <= std::numeric_limits<PosT>::max(),
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
    static bool would_overflow(size_t value) noexcept { return value >= kIndexCeiling; }

    // ---- index element + hashing ------------------------------------------------------------
    struct Slot {
        TermIndex idx = kEmptySlot;
        uint32_t h = 0;
    };

    static uint32_t fold_hash(const key_type &q) noexcept {
        const size_t full = MonomialHash<NumModes>{}(q);
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }
    // Avalanche the cached 32-bit fold into a full-width hash (splitmix64 finalizer): the stored h
    // is only an equality pre-filter, so it must be re-mixed before its low bits drive table bucketing.
    static size_t spread(uint32_t h) noexcept {
        uint64_t x = static_cast<uint64_t>(h) * 0x9E3779B97F4A7C15ull;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBull;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }

    // The index is a SINGLE lock-free open-addressing table, filled serially within one shard.
    // Operator sharding across cores is handled a level up by ShardGroup (one OperatorIndex per shard),
    // so this table never needs internal partitioning.
    //
    // ---- ctors --------------------------------------------------------------------------------
    // The inline width (hence stride) is a CONSTRUCTION INVARIANT, fixed here and never mutated.
    // It is purely a memory/overflow trade: rows longer than the width spill to overflow losslessly,
    // so any width is correct -- callers pass the cutoff that bounds the common-case popcount.
    explicit OperatorIndex(size_t inline_width = kDefaultInlinePositions)
        : inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)), stride_(1 + inline_width_) {}
    OperatorIndex(const OperatorIndex &) = delete;
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    OperatorIndex(OperatorIndex &&) = delete;
    OperatorIndex &operator=(OperatorIndex &&) = delete;

    // ---- deep copy ----------------------------------------------------------------------------
    // Single named deep-copy (enables the simulator's __deepcopy__). Entries are re-inserted into the
    // clone's table rather than copied verbatim. Returns by unique_ptr because owners hold the store
    // that way.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(inline_width_);
        {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            out->rows_ = rows_;
            out->size_ = size_;
            out->overflow_ = overflow_;
        }
        out->reserve_index(index_size());
        for (const Slot &e : table_.slots) {
            if (e.idx != kEmptySlot) {
                out->insert_slot_(e.idx, e.h);
            }
        }
        return out;
    }

    // ---- sizing -------------------------------------------------------------------------------
    [[nodiscard]] auto size() const -> size_t { return size_; }
    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity() / stride_; }
    [[nodiscard]] auto inline_width() const -> size_t { return inline_width_; }

    // Capacity hints only -- width is a construction invariant, never touched here.
    // reserve_rows vs reserve_index are kept SEPARATE on purpose: the builder's per-layer geometric
    // growth grows ROW capacity, while the index is right-sized to its element count by bulk_insert.
    auto reserve_rows(size_t n) -> void { rows_.reserve(n * stride_); }
    auto reserve_index(size_t n) -> void { table_.rehash_to(slots_for_(n + 1)); }
    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    // Grow the row store to hold `n` additional rows, returning the pre-growth size — the insert base
    // the caller then writes into. Growth is GEOMETRIC at 1.5×, never an exact-fit reserve: an exact
    // fit would realloc the whole persistent operator every layer, whereas 1.5× (vs 2×) halves the
    // transient realloc overshoot on the 100M-term row array at the cost of ~log₁.₅ vs log₂ reallocs.
    // The reserve-then-resize split is load-bearing: reserve grows capacity geometrically, resize sets
    // the logical size. Shared by the two per-layer partner-insert sites (build_layer).
    auto grow_rows_geometric(size_t n) -> size_t {
        const size_t base = size_;
        if (capacity() < base + n) {
            const size_t cap = capacity();
            reserve_rows(std::max(base + n, cap + cap / 2 + 1));
        }
        // Default-init grow, NOT a zeroing resize: every freshly grown row [base, base+n) is overwritten
        // by the disjoint miss-fill scatter (set_fresh) before any read, so a serial tail zero-fill would
        // be pure wasted bandwidth on the ~100M-term row array.
        rows_.resize((base + n) * stride_);
        size_ = base + n;
        return base;
    }
    auto clear() -> void {
        rows_.clear();
        overflow_.clear();
        size_ = 0;
        table_.slots.assign(table_.slots.size(), Slot{});
        table_.count = 0;
    }

    // ---- row writes ---------------------------------------------------------------------------
    auto push_back(const value_type &maj) -> void {
        const size_t idx = size_;
        rows_.resize((idx + 1) * stride_, 0);
        size_ = idx + 1;
        write_row(idx, maj);
    }
    // Overwrite a pre-sized slot. Test-support only: the production miss-fill uses set_fresh; kept for
    // the operator_index_tests fixtures that build rows directly.
    auto set(size_t i, const value_type &maj) -> void { write_row(i, maj); }
    // Overwrite a FRESHLY-GROWN (default-init) slot on the parallel disjoint miss-fill scatter; skips
    // the overflow pre-read/erase that set()/write_row do for possibly-existing rows (see write_row_fresh).
    auto set_fresh(size_t i, const value_type &maj) -> void { write_row_fresh(i, maj); }

    // ---- row reads ----------------------------------------------------------------------------
    [[nodiscard]] auto row(size_t i) const -> value_type {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            return overflow_.at(i);
        }
        value_type maj;
        const PosT *pos = &rows_[i * stride_ + 1];
        for (size_t j = 0; j < c; ++j) {
            maj.set(pos[j]);
        }
        return maj;
    }
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        const PosT *pos = &rows_[i * stride_ + 1];
        for (size_t j = 0; j < c; ++j) {
            fn(static_cast<size_t>(pos[j]));
        }
    }
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        const PosT c = rows_[i * stride_];
        if (c != kOverflowMarker) {
            return c;
        }
        std::lock_guard<std::mutex> lock(overflow_mutex_);
        return overflow_.at(i).count();
    }
    // Test-support only: lets operator_index_tests assert how many rows spilled past the inline width.
    [[nodiscard]] auto overflow_count() const -> size_t { return overflow_.size(); }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = rows_.capacity() * sizeof(PosT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    // Compare row i against key q without materializing the row (the find confirm). Reads the
    // popcount byte first, so a false h prefilter match usually costs one byte compare.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &q) const -> bool {
        const PosT c = rows_[i * stride_];
        if (c == kOverflowMarker) {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            return overflow_.at(i) == q;
        }
        if (q.count() != static_cast<size_t>(c)) {
            return false;
        }
        const PosT *pos = &rows_[i * stride_ + 1];
        for (size_t j = 0; j < c; ++j) {
            if (!q.test(pos[j])) {
                return false;
            }
        }
        return true;
    }

    // ---- index API ----------------------------------------------------------------------------
    // Returns the dense row index for `key`, or nullopt if absent. Usage: `if (auto i = find(k)) ...`.
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

    // GROUP-PREFETCH batch find: out[i] = row index of keys[i], or kMissingIndex. Semantically
    // identical to n independent find() calls on an unchanging table; the point is memory-level
    // parallelism — per group of G keys, stage 1 hashes and prefetches every home slot, stage 2
    // probes on the h prefilter and prefetches the candidate row, stage 3 confirms against the
    // row bytes. An h collision (wrong candidate) falls back to the exact single find.
    // MUST NOT run concurrently with inserts (the resolve phases are probe-only by construction).
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
                size_t s = sp[j] & table_.mask;
                for (;; s = (s + 1) & table_.mask) {
                    const Slot &e = table_.slots[s];
                    if (e.idx == kEmptySlot) {
                        break;
                    }
                    if (e.h == hh[j]) {
                        cand[j] = e.idx;
                        break;
                    }
                }
                if (cand[j] != kEmptySlot) {
                    __builtin_prefetch(&rows_[static_cast<size_t>(cand[j]) * stride_], 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                if (cand[j] != kEmptySlot && row_eq_key(static_cast<size_t>(cand[j]), keys[base + j])) {
                    out[base + j] = static_cast<size_t>(cand[j]);
                }
                else if (cand[j] != kEmptySlot) {
                    // h collision: the first h-match wasn't the key — resolve exactly.
                    const auto v = find(keys[base + j]);
                    out[base + j] = v ? *v : kNotFound;
                }
                else {
                    out[base + j] = kNotFound;
                }
            }
        }
    }

    // Insert-or-no-op. Row at `value` MUST already be written (the confirm reads dense rows).
    auto emplace(const key_type &key, mapped_type value) -> void {
        check_index_fits(value);
        const uint32_t h = fold_hash(key);
        table_.rehash_if_needed();
        size_t s = spread(h) & table_.mask;
        while (table_.slots[s].idx != kEmptySlot) {
            if (table_.slots[s].h == h && row_eq_key(static_cast<size_t>(table_.slots[s].idx), key)) {
                return; // key already present — no-op (matches the former set semantics)
            }
            s = (s + 1) & table_.mask;
        }
        table_.slots[s] = Slot{static_cast<TermIndex>(value), h};
        ++table_.count;
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows MUST already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        // Serial insert: probe the lock-free table for each of the n distinct keys.
        for (size_t k = 0; k < n; ++k) {
            const uint32_t h = fold_hash(key_at(k));
            table_.rehash_if_needed();
            insert_into_(table_, static_cast<TermIndex>(base + k), h);
        }
    }
    auto index_size() const -> size_t { return table_.count; }
    // Test-support only: visits every indexed (row, index) pair. Production reads rows by index via
    // row()/popcount()/for_each_position(); this whole-index walk exists for simulator_copy_tests.
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (const Slot &e : table_.slots) {
            if (e.idx != kEmptySlot) {
                fn(row(static_cast<size_t>(e.idx)), static_cast<size_t>(e.idx));
            }
        }
    }
    auto index_estimated_memory_bytes() const -> size_t {
        return sizeof(OperatorIndex) + table_.slots.capacity() * sizeof(Slot);
    }

private:
    // One open-addressing table: power-of-2 slot count, linear probing, max load factor 0.7
    // (the group-prefetch win erodes at high load — longer probe chains add un-prefetched reads).
    struct Shard {
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
    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, n * 10 / 7 + 1)); }

    // Insert (idx, h) into `shard` with NO dup probe — callers on this path insert provably
    // distinct keys (⊕G-injective miss batches, clone re-insertion). Caller ensures capacity.
    auto insert_into_(Shard &shard, TermIndex idx, uint32_t h) -> void {
        size_t s = spread(h) & shard.mask;
        while (shard.slots[s].idx != kEmptySlot) {
            s = (s + 1) & shard.mask;
        }
        shard.slots[s] = Slot{idx, h};
        ++shard.count;
    }
    auto insert_slot_(TermIndex idx, uint32_t h) -> void {
        table_.rehash_if_needed();
        insert_into_(table_, idx, h);
    }

    auto write_row(size_t i, const value_type &maj) -> void {
        const size_t c = maj.count();
        PosT *row = &rows_[i * stride_];
        const bool was_overflow = (row[0] == kOverflowMarker);
        if (c > inline_width_) {
            row[0] = kOverflowMarker;
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            overflow_[i] = maj;
            return;
        }
        if (was_overflow) {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            overflow_.erase(i);
        }
        row[0] = static_cast<PosT>(c);
        PosT *out = row + 1;
        for (size_t b = maj.find_first(); b < maj.size(); b = maj.find_next(b)) {
            *out++ = static_cast<PosT>(b);
        }
    }
    // Fresh-row variant of write_row for the parallel disjoint miss-fill scatter into
    // grow_rows_geometric'd slots. Those rows are DEFAULT-INITIALIZED (indeterminate row[0]) and are
    // provably never in overflow_ (freshly grown, never previously written), so write_row's
    // `was_overflow = (row[0] == kOverflowMarker)` pre-read is BOTH unnecessary AND unsafe here: a
    // garbage row[0] could spuriously equal kOverflowMarker and take overflow_mutex_ inside the
    // PARALLEL scatter (data race / lock churn). This path writes row[0] unconditionally. The
    // c>inline_width_ branch is KEPT: a genuinely long fresh row still must spill losslessly to overflow_.
    auto write_row_fresh(size_t i, const value_type &maj) -> void {
        const size_t c = maj.count();
        PosT *row = &rows_[i * stride_];
        if (c > inline_width_) {
            row[0] = kOverflowMarker;
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            overflow_[i] = maj;
            return;
        }
        row[0] = static_cast<PosT>(c);
        PosT *out = row + 1;
        for (size_t b = maj.find_first(); b < maj.size(); b = maj.find_next(b)) {
            *out++ = static_cast<PosT>(b);
        }
    }
    static auto check_index_fits(size_t value) -> void {
        if (would_overflow(value)) {
            throw std::runtime_error("OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                                     "-Dmonoprop_WIDE_TERM_INDEX (term count exceeded ~2^32).");
        }
    }

    // DefaultInitVector: grow_rows_geometric skips the serial tail zero-fill; every freshly grown row
    // is overwritten by set_fresh (the disjoint miss-fill scatter) before any read. push_back still
    // zero-fills its one cold-path row (resize(..., 0)) so its write_row sees a defined row[0].
    DefaultInitVector<PosT> rows_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    mutable std::unordered_map<size_t, value_type> overflow_ = {};
    mutable std::mutex overflow_mutex_ = {};
    // Single open-addressing index table (see the Shard doc-comment: one shard per core lives a level
    // up in ShardGroup, so this store never partitions internally).
    Shard table_ = {};
};

} // namespace monoprop::detail
