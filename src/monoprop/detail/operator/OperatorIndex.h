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
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>
#include <tbb/parallel_for.h>

#include "monoprop/Threading.h"
#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

/**
 * @brief Operator-term store: entropy-packed position-list rows PLUS a keyless hash index over
 * those rows, in one self-contained object. Replaces detail::PackedMajoranaVector + IndexMap.
 *
 * ROWS (byte layout identical to the former PackedMajoranaVector):
 *   slot 0      : popcount c (or kOverflowMarker if c > inline_width_)
 *   slots 1..c  : the c set-bit positions, ascending (PosT each)
 * stride_ = 1 + inline_width_, fixed for the container's life so the parallel disjoint miss-fill
 * stays lock-free. inline_width_ is a CONSTRUCTION INVARIANT (the constructor's only argument,
 * default = kMaxInlinePositions); re-init with a different width by assigning a fresh store. Rows
 * whose popcount exceeds the width spill LOSSLESSLY to a dense-bitset overflow map (mutex-guarded;
 * touched only on genuine overflow transitions).
 *
 * INDEX (element identical to the former IndexMap): a boost::unordered_flat_set of
 * Entry{TermIndex idx, uint32_t h32}. The 32-bit folded hash is cached at insert from the in-hand
 * key (never from a row reconstruction), so insert/rehash are gather-free; find pre-filters on h32
 * and confirms a hit by reconstructing dense(idx) from THIS store's own rows.
 *
 * BACK-REFERENCE: RowEq caches a plain `const OperatorIndex*` (= `this`, fixed at construction)
 * and reads dense(idx) through it. The store is therefore NON-MOVABLE -- moving it would dangle that
 * pointer. Owners that relocate a store (MPOperator, SchroGeneratorBasis) hold it by unique_ptr, so
 * the heap address is stable and no move-repair machinery (the former SelfRef cell) is needed.
 */
template <size_t NumModes>
class OperatorIndex {
public:
    using value_type = MajoranaSet<NumModes>;
    using key_type = MajoranaSet<NumModes>;
    using mapped_type = size_t;

    // Position element: u8 when 2N<=256 (byte-identical to the original packed layout), widening
    // only for larger mode counts so positions never truncate.
    using PosT = std::conditional_t<(2 * NumModes <= 256), uint8_t,
                                    std::conditional_t<(2 * NumModes <= 65536), uint16_t, uint32_t>>;

    static constexpr size_t kMaxInlinePositions = 11;
    static constexpr PosT kOverflowMarker = std::numeric_limits<PosT>::max();

    static_assert(2 * NumModes - 1 <= std::numeric_limits<PosT>::max(),
                  "OperatorIndex PosT too narrow for 2*NumModes positions");
    static_assert(kMaxInlinePositions < std::numeric_limits<PosT>::max(),
                  "kOverflowMarker sentinel must not collide with a valid popcount");

    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    static bool would_overflow(size_t value) noexcept { return value >= kIndexCeiling; }

    // ---- index element + functors ----------------------------------------------------------
    struct Entry {
        TermIndex idx;
        uint32_t h;
    };

    static uint32_t fold_hash(const key_type &q) noexcept {
        const size_t full = MPHash<NumModes>{}(q);
        return static_cast<uint32_t>(full ^ (static_cast<uint64_t>(full) >> 32));
    }
    // Avalanche the cached 32-bit fold into a full-width bucket hash (splitmix64 finalizer): the
    // stored h is only an equality pre-filter, so it must be re-mixed before it drives bucketing.
    static size_t spread(uint32_t h) noexcept {
        uint64_t x = static_cast<uint64_t>(h) * 0x9E3779B97F4A7C15ull;
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27;
        x *= 0x94D049BB133111EBull;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }
    struct RowHash {
        using is_transparent = void;
        size_t operator()(const Entry &e) const noexcept { return spread(e.h); }
        size_t operator()(const key_type &q) const noexcept { return spread(fold_hash(q)); }
    };
    struct RowEq {
        const OperatorIndex *store = nullptr; // reads store->dense(idx) only to CONFIRM a hash match
        using is_transparent = void;
        bool operator()(const Entry &a, const Entry &b) const noexcept {
            return a.h == b.h
                   && store->dense(static_cast<size_t>(a.idx)) == store->dense(static_cast<size_t>(b.idx));
        }
        bool operator()(const Entry &a, const key_type &q) const noexcept {
            return a.h == fold_hash(q) && store->dense(static_cast<size_t>(a.idx)) == q;
        }
        bool operator()(const key_type &q, const Entry &a) const noexcept { return (*this)(a, q); }
    };
    using Set = boost::unordered_flat_set<Entry, RowHash, RowEq>;

    // The index is SHARDED into independent sets so the per-layer build insert -- a serial probe loop on
    // one giant table, latency-bound and non-scaling -- becomes a parallel disjoint-shard insert. The
    // shard COUNT is chosen from the available parallelism at construction (choose_shard_count): an MPI
    // rank with a small thread pool gets few shards, a 1-thread build gets a single table (no overhead),
    // a 20-thread build gets enough shards for load balance. Routing uses the HIGH bits of the avalanched
    // hash, leaving the low/mid bits (in-shard bucketing + the SIMD tag byte) full-entropy. Sharding only
    // relocates entries; it never changes membership or what find() returns, so it is BIT-EXACT, and
    // power-of-2 shard counts keep total bucket_count -- hence memory -- ~neutral.
    static constexpr size_t kMaxShards = 64;                    // cap (bounds per-shard overhead)
    static constexpr size_t kBulkInsertParallelMin = 1U << 12U; // small batches insert serially

    // ~2x the worker count, rounded up to a power of two and capped: each worker gets a couple of shards
    // for load balance without excessive per-shard overhead. 1 worker -> 1 shard (single table).
    static auto choose_shard_count() -> size_t {
        const size_t workers = threading::effective_parallelism();
        if (workers <= 1) {
            return 1;
        }
        return std::min(kMaxShards, std::bit_ceil(workers * 2));
    }
    // Route by the top log2(shard_count_) bits of the avalanched hash; the mask makes shard_count_==1 -> 0.
    auto shard_of(uint32_t h) const noexcept -> size_t { return (spread(h) >> shard_shift_) & shard_mask_; }

    // ---- ctors (immovable: RowEq holds a fixed back-pointer to this store) ------------------
    // The inline width (hence stride) is a CONSTRUCTION INVARIANT, fixed here and never mutated.
    // It is purely a memory/overflow trade: rows longer than the width spill to overflow losslessly,
    // so any width is correct -- callers pass the cutoff that bounds the common-case popcount. To
    // re-init a store with a different width, assign a freshly-constructed one (that is what the old
    // clear()+reserve(n,width) pair did). No setter, no "set width before the first push" footgun.
    //
    // The store is NON-MOVABLE: RowEq confirms hash hits by reading dense(idx) back from THIS store,
    // so it caches `this`, which must stay valid for the index's life. Owners that need to relocate a
    // store (MPOperator, SchroGeneratorBasis) hold it by std::unique_ptr -- the heap address is stable,
    // so no move-repair (the former SelfRef cell + move ctor/assign) is needed.
    explicit OperatorIndex(size_t inline_width = kMaxInlinePositions)
        : inline_width_(std::clamp<size_t>(inline_width, 1, kMaxInlinePositions)), stride_(1 + inline_width_),
          shard_count_(choose_shard_count()),
          shard_shift_(shard_count_ <= 1 ? 0 : 64U - static_cast<size_t>(std::countr_zero(shard_count_))),
          shard_mask_(shard_count_ - 1) {
        // Each shard is an independent set whose RowEq back-references THIS store (confirms hash hits via
        // dense(idx)). Built here so every shard carries the correct back-pointer.
        shards_.reserve(shard_count_);
        for (size_t i = 0; i < shard_count_; ++i) {
            shards_.emplace_back(0, RowHash{}, RowEq{this});
        }
    }
    OperatorIndex(const OperatorIndex &) = delete;
    OperatorIndex &operator=(const OperatorIndex &) = delete;
    OperatorIndex(OperatorIndex &&) = delete;
    OperatorIndex &operator=(OperatorIndex &&) = delete;

    // ---- deep copy --------------------------------------------------------------------------
    // The store stays non-copyable/non-movable (RowEq caches a `this` back-pointer), so clone() is
    // the single named deep-copy: it returns a fresh heap store whose index back-pointer is repaired
    // to the CLONE. Row data is copied verbatim; the index is rebuilt by re-inserting every {idx,h}
    // Entry into a set bound to the new store. The cloned rows are byte-identical, so each hash hit
    // confirms against them exactly as in the source. Returns by unique_ptr because owners (MPOperator,
    // SchroGeneratorBasis) already hold the store that way -- the heap address is what RowEq needs stable.
    [[nodiscard]] auto clone() const -> std::unique_ptr<OperatorIndex> {
        auto out = std::make_unique<OperatorIndex>(inline_width_);
        {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            out->rows_ = rows_;
            out->size_ = size_;
            out->overflow_ = overflow_;
        }
        // Re-route each entry through the clone's own shard_of (its shard_count_ may differ if the worker
        // count changed); out's RowEq{out} confirms each against the cloned rows.
        out->reserve_index(index_size());
        for (const Set &shard : shards_) {
            for (const Entry &e : shard) {
                out->shards_[out->shard_of(e.h)].insert(e);
            }
        }
        return out;
    }

    // ---- sizing -----------------------------------------------------------------------------
    [[nodiscard]] auto size() const -> size_t { return size_; }
    [[nodiscard]] auto empty() const -> bool { return size_ == 0; }
    [[nodiscard]] auto capacity() const -> size_t { return rows_.capacity() / stride_; }
    [[nodiscard]] auto inline_width() const -> size_t { return inline_width_; }

    // Capacity hints only -- width is a construction invariant, never touched here.
    // reserve_rows vs reserve_index are kept SEPARATE on purpose: the builder's per-layer geometric
    // growth doubles ROW capacity, while the index is right-sized to its element count by bulk_insert.
    // Coupling an index reserve to that 2x row growth would over-reserve the hash table to ~2x its
    // elements (cache-sparse probes). Use reserve(n) only when sizing once to a known final count.
    auto reserve_rows(size_t n) -> void { rows_.reserve(n * stride_); }
    auto reserve_index(size_t n) -> void {
        const size_t per = n / shard_count_ + 1;
        for (Set &shard : shards_) {
            shard.reserve(per);
        }
    }
    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    auto resize(size_t n) -> void {
        rows_.resize(n * stride_, 0);
        if (n < size_) {
            std::lock_guard<std::mutex> lock(overflow_mutex_);
            for (auto it = overflow_.begin(); it != overflow_.end();) {
                it = (it->first >= n) ? overflow_.erase(it) : std::next(it);
            }
        }
        size_ = n;
    }
    auto clear() -> void {
        rows_.clear();
        overflow_.clear();
        size_ = 0;
        for (Set &shard : shards_) {
            shard.clear();
        }
    }

    // ---- row writes -------------------------------------------------------------------------
    auto push_back(const value_type &maj) -> void {
        const size_t idx = size_;
        rows_.resize((idx + 1) * stride_, 0);
        size_ = idx + 1;
        write_row(idx, maj);
    }
    // Overwrite a pre-sized slot (parallel disjoint miss-fill writes distinct indices).
    auto set(size_t i, const value_type &maj) -> void { write_row(i, maj); }

    // ---- row reads --------------------------------------------------------------------------
    [[nodiscard]] auto dense(size_t i) const -> value_type {
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
    [[nodiscard]] auto overflow_count() const -> size_t { return overflow_.size(); }

    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = rows_.capacity() * sizeof(PosT);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    // ---- index API --------------------------------------------------------------------------
    // Returns the dense row index for `key`, or nullopt if absent. Usage: `if (auto i = find(k)) ...`.
    std::optional<size_t> find(const key_type &key) const {
        const Set &shard = shards_[shard_of(fold_hash(key))];
        const auto it = shard.find(key);
        if (it == shard.end()) {
            return std::nullopt;
        }
        return static_cast<size_t>(it->idx);
    }

    // Insert-or-no-op. Row at `value` MUST already be written (find's confirm reads dense(value)).
    void emplace(const key_type &key, mapped_type value) {
        check_index_fits(value);
        const uint32_t h = fold_hash(key);
        shards_[shard_of(h)].insert(Entry{static_cast<TermIndex>(value), h});
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows MUST already be written.
    template <typename KeyFn>
    void bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        // Small batch (or unsharded single-thread build): serial insert -- parallel_for + per-shard
        // bucketing overhead would exceed the work.
        if (shard_count_ == 1 || n < kBulkInsertParallelMin) {
            for (size_t k = 0; k < n; ++k) {
                const uint32_t h = fold_hash(key_at(k));
                shards_[shard_of(h)].insert(Entry{static_cast<TermIndex>(base + k), h});
            }
            return;
        }
        // Bucket the n DISTINCT keys by shard (serial, hash-only -- no table probe), then insert each
        // shard's entries in parallel: shards are disjoint, so the probe-heavy inserts never contend.
        std::vector<std::vector<Entry>> per_shard(shard_count_);
        for (size_t k = 0; k < n; ++k) {
            const uint32_t h = fold_hash(key_at(k));
            per_shard[shard_of(h)].push_back(Entry{static_cast<TermIndex>(base + k), h});
        }
        tbb::parallel_for(size_t{0}, shard_count_, [&](size_t s) {
            std::vector<Entry> &batch = per_shard[s];
            if (batch.empty()) {
                return;
            }
            Set &shard = shards_[s];
            shard.reserve(shard.size() + batch.size());
            for (const Entry &e : batch) {
                shard.insert(e);
            }
        });
    }
    size_t index_size() const {
        size_t total = 0;
        for (const Set &shard : shards_) {
            total += shard.size();
        }
        return total;
    }
    template <typename Func>
    void for_each(Func &&fn) const {
        for (const Set &shard : shards_) {
            for (const Entry &e : shard) {
                fn(dense(static_cast<size_t>(e.idx)), static_cast<size_t>(e.idx));
            }
        }
    }
    size_t index_estimated_memory_bytes() const {
        size_t buckets = 0;
        for (const Set &shard : shards_) {
            buckets += shard.bucket_count();
        }
        return sizeof(OperatorIndex) + buckets * (sizeof(Entry) + sizeof(unsigned char));
    }

private:
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
    static void check_index_fits(size_t value) {
        if (would_overflow(value)) {
            throw std::runtime_error(
                "OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                "-DMONOPROP_WIDE_TERM_INDEX (term count exceeded ~2^32).");
        }
    }

    std::vector<PosT> rows_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    mutable std::unordered_map<size_t, value_type> overflow_ = {};
    mutable std::mutex overflow_mutex_ = {};
    // Sharded index: each shard's RowEq holds a fixed `this` pointer, so the store must never move (see
    // ctors). Count/shift/mask are set once at construction from the worker count (choose_shard_count).
    size_t shard_count_ = 1;
    size_t shard_shift_ = 0;
    size_t shard_mask_ = 0;
    std::vector<Set> shards_ = {};
};

} // namespace monoprop::detail
