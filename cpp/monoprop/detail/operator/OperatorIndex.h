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
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

class TermIndexCeilingReached : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Rows per chunk of the row store. Fixed rather than geometric: the index -> (chunk, offset) split is a
// shift and a mask on the hottest path in propagate, and a geometric ladder would make it a countl_zero
// plus a branch there to save capacity only on operators small enough for their absolute footprint not
// to matter. Trades one whole chunk of quantized slack per partition for never copying a row on growth.
//
// 4096 is measured, not guessed. The cost of chunking is the chunk TABLE's L1 footprint, not the extra
// dependent load: sweeping 1024/2048/4096/8192 moves Hubbard `propagate` 1.0248/1.0228/1.0158/1.0155x,
// i.e. it plateaus once the table stops competing for L1 (3.2 KB at 4096 on Hubbard's 1.66M rows per
// partition). Past 4096 the curve is flat and only the quantized slack grows, which is why this is the
// operating point and not 8192.
//
// There is a third term the sweep did not isolate: allocator packing. Going 2048 -> 4096 took kicked
// Ising c14's peak RSS 2,657,232 -> 2,576,992 KB, 78 MB BETTER, where the quantized slack predicted
// ~1.4 MB worse. Chunks land under glibc's 128 KB mmap threshold, so they come off the heap and the
// smaller ones fragment it harder. Do not reason about this constant from the slack alone.
//
// Deliberately NOT a CMake cache variable, unlike monoprop_INVIDX_CHUNK_WORDS: 4096 and 8192 differ by
// 0.03% and everything below is strictly worse, so there is no workload-dependent trade for a user to
// tune. It stays overridable here only so the sweep can be re-run against a future machine.
#ifndef monoprop_OPINDEX_CHUNK_ROWS
#define monoprop_OPINDEX_CHUNK_ROWS 4096
#endif
inline constexpr size_t kOpIndexChunkRows = static_cast<size_t>(monoprop_OPINDEX_CHUNK_ROWS);
static_assert(std::has_single_bit(kOpIndexChunkRows), "chunk rows must be a power of two (shift/mask addressing)");

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

    // Valid term indices are < kIndexCeiling (check_index_fits throws at the ceiling). Emptiness is
    // carried by the slot's fingerprint byte, not by a reserved index, so the whole TermIndex range
    // below the ceiling is usable.
    static constexpr size_t kIndexCeiling = static_cast<size_t>(std::numeric_limits<TermIndex>::max());
    // find_batch's "absent" result; same value as detail::kMissingIndex (not included here — the
    // operator store must not depend on evolution headers).
    static constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

    // Row-store chunk geometry. Counted in rows, not bytes: stride_ is a runtime value, so only a
    // row-granular chunk keeps every row entirely inside one allocation.
    static constexpr size_t kChunkRows = kOpIndexChunkRows;
    static constexpr size_t kChunkShift = static_cast<size_t>(std::countr_zero(kChunkRows));
    static constexpr size_t kChunkMask = kChunkRows - 1;

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
        out->reserve_rows(size_);
        for (size_t i = 0; i < size_; i += kChunkRows) {
            const size_t n = std::min(kChunkRows, size_ - i);
            std::memcpy(out->chunks_[i >> kChunkShift].get(), chunks_[i >> kChunkShift].get(),
                        n * stride_ * sizeof(PosT));
        }
        out->size_ = size_;
        out->overflow_ = overflow_;
        out->reserve_index(table_.count);
        // The stored fingerprint is not a hash, so the clone rebuilds each entry's hash from the row
        // it already copied. Clone runs off a quiescent store and is rare; this is not a hot path.
        for (size_t s = 0; s < table_.slots; ++s) {
            if (slot_fp_(s) != 0) {
                const size_t i = slot_idx_(s);
                out->insert_slot_(static_cast<TermIndex>(i), spread(fold_hash(row(i))));
            }
        }
        return out;
    }

    [[nodiscard]] auto size() const -> size_t { return size_; }

    auto reserve(size_t n) -> void {
        reserve_rows(n);
        reserve_index(n);
    }
    // Returns the pre-growth size (the caller's insert base). Growth appends whole chunks and never
    // moves an existing row, so there is no overshoot to carry and no copy to amortise: the 1.5x
    // geometric rule this replaces left 4.1-6.0 B/term of dead capacity and memcpied the whole row
    // array on every step (~3x the final size over a build).
    auto grow_rows(size_t n) -> size_t {
        const size_t base = size_;
        if (capacity() < base + n) {
            reserve_rows(base + n);
        }
        size_ = base + n;
        return base;
    }

    auto push_back(const value_type &mono) -> void { set(grow_rows(1), mono); }

    // Row i may be grown-but-uninitialized or hold a prior value, so the row header is never pre-read
    // (freshly grown headers are indeterminate); a stale overflow entry at i, if any, is dropped.
    auto set(size_t i, const value_type &mono) -> void {
        const size_t c = mono.count();
        PosT *row = row_ptr_mut_(i);
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

    [[nodiscard]] auto row(size_t i) const -> value_type {
        const PosT *r = row_ptr_(i);
        const PosT c = r[0];
        if (c == kOverflowMarker) {
            return overflow_.at(i);
        }
        value_type mono;
        const PosT *pos = r + 1;
        for (size_t j = 0; j < c; ++j) {
            mono.set(pos[j]);
        }
        return mono;
    }
    template <typename Fn>
    auto for_each_position(size_t i, Fn &&fn) const -> void {
        const PosT *r = row_ptr_(i);
        const PosT c = r[0];
        if (c == kOverflowMarker) {
            const auto &m = overflow_.at(i);
            for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                fn(b);
            }
            return;
        }
        const PosT *pos = r + 1;
        for (size_t j = 0; j < c; ++j) {
            fn(static_cast<size_t>(pos[j]));
        }
    }
    [[nodiscard]] auto popcount(size_t i) const -> size_t {
        if (const PosT c = row_ptr_(i)[0]; c != kOverflowMarker) {
            return c;
        }
        return overflow_.at(i).count();
    }
    [[nodiscard]] auto memory_bytes() const -> size_t {
        size_t total = capacity() * stride_ * sizeof(PosT);
        total += chunks_.capacity() * sizeof(std::unique_ptr<PosT[]>);
        total += overflow_.size() * (sizeof(value_type) + sizeof(size_t) + 24);
        return total;
    }

    auto find(const key_type &key) const -> std::optional<size_t> {
        if (table_.count == 0) {
            return std::nullopt;
        }
        const size_t sp = spread(fold_hash(key));
        const uint8_t f = fingerprint(sp);
        for (size_t s = sp & table_.mask;; s = (s + 1) & table_.mask) {
            const uint8_t e = slot_fp_(s);
            if (e == 0) {
                return std::nullopt;
            }
            if (e == f) {
                const size_t i = slot_idx_(s);
                if (row_eq_key(i, key)) {
                    return i;
                }
            }
        }
    }

    // Group-prefetch batch find: out[i] = row index of keys[i], or kNotFound. Same result as n
    // find() calls, but overlaps dram misses via a per-group hash/probe/confirm pipeline. A
    // fingerprint collision falls back to an exact find. must not run concurrently with inserts.
    auto find_batch(const key_type *keys, size_t n, size_t *out) const -> void {
        static constexpr size_t G = 16; // keys prefetched together per pipeline pass
        std::array<size_t, G> sp;
        std::array<size_t, G> cand;
        for (size_t base = 0; base < n; base += G) {
            const size_t g = std::min(G, n - base);
            for (size_t j = 0; j < g; ++j) {
                sp[j] = spread(fold_hash(keys[base + j]));
                // One prefetch covers the probe and the confirm: the fingerprint and the index it
                // guards are consecutive bytes of the same record.
                __builtin_prefetch(slot_addr_(sp[j] & table_.mask), 0, 0);
            }
            for (size_t j = 0; j < g; ++j) {
                cand[j] = kNoSlot;
                if (table_.count == 0) {
                    continue;
                }
                cand[j] = probe_fp_match_(fingerprint(sp[j]), sp[j] & table_.mask);
                if (cand[j] != kNoSlot) {
                    // One extra load (the chunk pointer) now sits in front of this address. The chunk
                    // table is one pointer per kChunkRows rows -- a few KB at 26M terms -- so it is an
                    // L1 hit and the lead time this prefetch buys is preserved.
                    __builtin_prefetch(row_ptr_(slot_idx_(cand[j])), 0, 0);
                }
            }
            for (size_t j = 0; j < g; ++j) {
                if (cand[j] == kNoSlot) {
                    out[base + j] = kNotFound;
                    continue;
                }
                const size_t i = slot_idx_(cand[j]);
                if (row_eq_key(i, keys[base + j])) {
                    out[base + j] = i;
                }
                else {
                    const auto v = find(keys[base + j]);
                    out[base + j] = v ? *v : kNotFound;
                }
            }
        }
    }

    // Insert-or-no-op. Row at `value` must already be written: the confirm reads dense rows, and since
    // the table stores a fingerprint rather than a hash, so does every rehash this insert may trigger.
    auto emplace(const key_type &key, mapped_type value) -> void {
        check_index_fits(value);
        const size_t sp = spread(fold_hash(key));
        const uint8_t f = fingerprint(sp);
        rehash_if_needed_();
        size_t s = sp & table_.mask;
        while (slot_fp_(s) != 0) {
            if (slot_fp_(s) == f && row_eq_key(slot_idx_(s), key)) {
                return;
            }
            s = (s + 1) & table_.mask;
        }
        write_slot_(s, f, static_cast<TermIndex>(value));
        ++table_.count;
    }
    // Insert n distinct rows with consecutive indices [base, base+n). Rows must already be written.
    template <typename KeyFn>
    auto bulk_insert(size_t n, mapped_type base, KeyFn &&key_at) -> void {
        if (n == 0) {
            return;
        }
        check_index_fits(base + n - 1);
        for (size_t k = 0; k < n; ++k) {
            insert_slot_(static_cast<TermIndex>(base + k), spread(fold_hash(key_at(k))));
        }
    }
    template <typename Func>
    auto for_each(Func &&fn) const -> void {
        for (size_t s = 0; s < table_.slots; ++s) {
            if (slot_fp_(s) != 0) {
                const size_t i = slot_idx_(s);
                fn(row(i), i);
            }
        }
    }
    // Diagnostic: the unused tail of the last chunk, which is all the dead capacity a chunked store can
    // hold. Bounded by kChunkRows-1 rows regardless of operator size, where the 1.5x flat vector this
    // replaced carried up to half of live (measured 4.1-6.0 B/term across the benchmark workloads).
    //
    // There is deliberately no shrink_rows_to_fit() any more. It existed to hand back geometric
    // overshoot at a quiescent point, and a gated shrink could never win: any rule leaving low slack at
    // rest has to fire near the end of a build, and "the end" is not observable to the library -- it
    // measured 1.030x on Hubbard propagate (6/6, p=0.031) and was left uncalled. Bounding the overshoot
    // at the source removes both the slack and the mechanism.
    [[nodiscard]] auto slack_bytes() const -> size_t {
        return (capacity() - std::min(capacity(), size_)) * stride_ * sizeof(PosT);
    }

    auto index_estimated_memory_bytes() const -> size_t { return sizeof(OperatorIndex) + table_.buf.capacity(); }

    // Diagnostics on the dedup table's realised sizing: the slot count is what indexing_bytes is
    // really measuring, and occupancy is entries/slots. Exposed so the load-factor ceiling can be
    // pinned by a test rather than asserted in a comment.
    [[nodiscard]] auto index_slot_count() const -> size_t { return table_.slots; }
    [[nodiscard]] auto index_entry_count() const -> size_t { return table_.count; }

private:
    // No slot on the chain matched, as distinct from a slot index that happens to be valid.
    static constexpr size_t kNoSlot = std::numeric_limits<size_t>::max();

    // First slot on the chain whose fingerprint matches, or kNoSlot if the chain ends first. Matches on
    // the prefilter byte alone and leaves the dense-row comparison to the caller — that deferral is what
    // lets find_batch prefetch the row between probe and confirm, so do not fold row_eq_key in here
    // (find() deliberately keeps its own confirming variant). `start` must already be masked; the table
    // must not be mutated concurrently.
    [[gnu::always_inline]] auto probe_fp_match_(uint8_t f, size_t start) const -> size_t {
        for (size_t s = start;; s = (s + 1) & table_.mask) {
            const uint8_t e = slot_fp_(s);
            if (e == 0) {
                return kNoSlot;
            }
            if (e == f) {
                return s;
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
    //
    // The power of 2 is deliberate and was re-measured, because it looks like pure waste: `bit_ceil`
    // against a 0.7 threshold leaves the realised load anywhere in [0.35, 0.7], and this table is the
    // largest structure in the operator, so the overshoot is worth ~4 B/term. Sizing the table exactly
    // and range-reducing with a multiply-shift instead of this mask was built and measured: it does
    // deliver the memory (index 20.18 → 16.32 B/term on Hubbard, 18.51 → 14.91 on the 29M-term
    // Majorana point, 7% of the whole operator) and it costs **`propagate` 1.015× on Hubbard and
    // 1.019× on kicked Ising, both 6/6, p=0.031**.
    //
    // The cost is not the multiply, it is the density the memory win is made of: linear-probe chain
    // length goes as (1 + 1/(1-α))/2, so moving Hubbard's load from 0.396 to 0.490 buys 7% of the
    // operator with 11% more probe steps on hits and 29% more on misses.
    //
    // It was then tried a second time STACKED on the narrow slot below, where the two levers are
    // orthogonal — one sets slots per term, the other bytes per slot — and it reached index 20.18 →
    // 9.12 B/term on Hubbard, 21.4% of the whole operator, for `propagate` 1.024×. It still loses, on
    // the other model: kicked Ising `propagate` **1.056×** and `build_graph` **1.072×**, 6/6, while its
    // load barely moved (0.633 → 0.623) because `bit_ceil` had already dealt that workload a good
    // number. All of the cost, none of the win — the same row that went backwards the first time.
    //
    // The cost is carried by the **1.5× growth factor**, which is what bounds the load band to
    // [0.467, 0.7] and is therefore inseparable from the win: it takes rehash work from 2n to 3n and
    // the number of resizes from ~18 to ~30, each re-materialising every live row AND reallocating a
    // multi-megabyte buffer. Dropping back to 2× growth to pay for it was measured too (exact sizing
    // and multiply-shift retained) and collects **nothing**: load came back 0.396 → 0.396, 0.633 →
    // 0.633, 0.426 → 0.426, and 0.432 → 0.363 on the Majorana point — i.e. once growth doubles, exact
    // sizing lands in the same band `bit_ceil` does, and can land worse. The occupancy win lives
    // entirely in the growth factor, not in the sizing arithmetic, and it is not affordable.
    //
    // Narrowing the slot is the opposite trade — fewer bytes at constant α — and it is what shipped.
    //
    // The slot is 1 + sizeof(TermIndex) bytes — a fingerprint byte in place of the 4-byte hash — and the
    // two fields are **interleaved**, one record per slot, not split into parallel arrays.
    //
    // Splitting them was built and measured first, on the Swiss-table reasoning that a miss should walk
    // a 1-byte-per-slot array (64 slots to a line instead of 8). It delivers exactly the predicted
    // memory — index 20.18 → 12.61 B/term on Hubbard, 18.51 → 11.57 on the 29M-term Majorana point —
    // and it costs **`propagate` 1.036× on Hubbard (6/6, p=0.031) and `build_graph` 1.036× on kicked
    // Ising (6/6)**. The reason is that the fp scan has nothing to amortise here: this table is sized
    // to ≤0.7 load and realises α ∈ [0.35, 0.7], where chains are 1–2 slots, so a probe never reads a
    // second fp byte — while every hit now touches two cache lines, fp's and idx's, where the 8-byte
    // slot touched one. The wide-scan win needs the α ≈ 0.875 a Swiss table runs at; at this density
    // the split is pure added traffic on the operator's hottest random-access structure.
    //
    // Interleaved, a slot's fingerprint and index are 5 consecutive bytes, so the probe and the confirm
    // land on one line (a record straddles a boundary 4 times in 64) — the base layout's line count at
    // 5/8 of its bytes. The unaligned index is read and written by memcpy from a raw byte array rather
    // than through a packed struct member: GCC lowers packed-member access to byte-wise reads on
    // aarch64 and this project builds for both partitions of Deucalion, whereas the memcpy form is a
    // single load on both. (Confirmed on x86-64 GCC 14.3: both forms emit one `movl`.)
    struct Table {
        // One interleaved record per slot: [fingerprint][TermIndex]. fp == 0 marks an empty slot, so a
        // zero-filled buffer is an empty table and the probe never needs a separate index sentinel.
        std::vector<uint8_t> buf = std::vector<uint8_t>(kMinSlots * kSlotBytes, 0);
        size_t slots = kMinSlots;
        size_t mask = kMinSlots - 1;
        size_t count = 0;
    };
    static constexpr size_t kMinSlots = 16;
    static constexpr size_t kSlotBytes = 1 + sizeof(TermIndex);
    // Slot count for `n` entries at ≤0.7 load.
    static auto slots_for_(size_t n) -> size_t { return std::bit_ceil(std::max<size_t>(kMinSlots, (n * 10 / 7) + 1)); }

    // Prefilter byte for a spread hash. The low log2(slots) bits pick the bucket, so the top byte is
    // independent of it. 0 is reserved for "empty", so the value is folded into 1..255; the resulting
    // 1/255 false-prefilter rate costs one row_eq_key, which reads the popcount byte first and almost
    // always exits on it.
    static auto fingerprint(size_t sp) noexcept -> uint8_t {
        const auto f = static_cast<uint8_t>(sp >> 56);
        return f == 0 ? uint8_t{1} : f;
    }

    // Slot record access. The index is unaligned by construction, so it goes through memcpy — see the
    // Table comment for why that and not a packed struct member.
    [[gnu::always_inline]] auto slot_fp_(size_t s) const noexcept -> uint8_t { return table_.buf[s * kSlotBytes]; }
    [[gnu::always_inline]] auto slot_idx_(size_t s) const noexcept -> size_t {
        TermIndex v = 0;
        std::memcpy(&v, table_.buf.data() + (s * kSlotBytes) + 1, sizeof(TermIndex));
        return static_cast<size_t>(v);
    }
    [[gnu::always_inline]] auto write_slot_(size_t s, uint8_t f, TermIndex v) noexcept -> void {
        table_.buf[s * kSlotBytes] = f;
        std::memcpy(table_.buf.data() + (s * kSlotBytes) + 1, &v, sizeof(TermIndex));
    }
    // The one line a probe of slot s touches.
    [[gnu::always_inline]] auto slot_addr_(size_t s) const noexcept -> const uint8_t * {
        return table_.buf.data() + (s * kSlotBytes);
    }

    auto rehash_if_needed_() -> void {
        if ((table_.count + 1) * 10 >= table_.slots * 7) {
            rehash_to_(table_.slots * 2);
        }
    }
    // An 8-bit fingerprint cannot survive a resize: the new home bucket needs a hash bit the slot does
    // not carry, so every live row's hash is recomputed from the row itself. That recompute is not a
    // rounding error and it is where nearly all of this layout's cost lives. A null-control arm that
    // stored the hash and skipped it — a 9-byte slot, wider than the 8-byte baseline and therefore
    // unshippable — measured **1.004×** against the baseline where the same layout with a slot-order
    // recompute measured 1.024×. Four fifths of the narrowing penalty was this one loop.
    //
    // Both sides of the move have to be overlapped, and each was got wrong once:
    //
    //   * Reads. Visiting live entries in old-slot order scatters the row reads at random across a row
    //     store far larger than L3. Term indices are dense in [0, size_), so the same rows are
    //     visited in index order instead and the read side becomes one sequential stream. Chunking the
    //     row store does not break that: ascending indices walk each chunk start to finish, so the
    //     stream restarts once per kChunkRows rather than being scattered. The live set
    //     rides in a bitmap — one bit per term, 3.3 MB at 26.6M terms, cache-resident — filled by a
    //     sequential walk of the old table.
    //   * Writes. Old-slot order was not arbitrary: under doubling a slot's new home is its old home
    //     with at most one bit set above the old mask, so walking old slots in order wrote the new
    //     table in two interleaved ascending runs, i.e. nearly sequentially. Index order gives that up,
    //     and on Hubbard that trade ALONE measured 1.037× against 1.024× — the scattered writes cost
    //     more than the scattered reads saved. So the hash is computed a group ahead and each entry's
    //     destination line is prefetched for write.
    //
    // With both overlapped the penalty is 1.017×, against the 1.004× floor; what remains is arithmetic,
    // not stalls — materialising each row and hashing it, ~53M times over a Hubbard build.
    //
    // Rebuilding in index order changes insertion order into the table. That is legal: the determinism
    // contract is bit-identity at fixed (R, S), which no table ordering participates in, and nothing
    // serializes or iterates this table for output.
    auto rehash_to_(size_t new_cap) -> void {
        new_cap = std::bit_ceil(std::max<size_t>(new_cap, kMinSlots));
        if (new_cap <= table_.slots) {
            return;
        }
        const std::vector<uint8_t> old_buf = std::move(table_.buf);
        const size_t old_slots = table_.slots;
        const size_t old_count = table_.count;
        table_.buf.assign(new_cap * kSlotBytes, uint8_t{0});
        table_.slots = new_cap;
        table_.mask = new_cap - 1;
        table_.count = 0;
        if (old_count == 0) {
            return;
        }

        std::vector<uint64_t> live((size_ / 64) + 1, 0);
        for (size_t s = 0; s < old_slots; ++s) {
            if (old_buf[s * kSlotBytes] == 0) {
                continue;
            }
            TermIndex v = 0;
            std::memcpy(&v, old_buf.data() + (s * kSlotBytes) + 1, sizeof(TermIndex));
            const size_t i = static_cast<size_t>(v);
            live[i / 64] |= uint64_t{1} << (i % 64);
        }

        static constexpr size_t G = 16;
        std::array<TermIndex, G> idxb{};
        std::array<size_t, G> spb{};
        size_t m = 0;
        const auto flush = [&] {
            for (size_t j = 0; j < m; ++j) {
                place_slot_(idxb[j], spb[j]);
            }
            m = 0;
        };
        for (size_t w = 0; w < live.size(); ++w) {
            for (uint64_t bits = live[w]; bits != 0; bits &= bits - 1) {
                const size_t i = (w * 64) + static_cast<size_t>(std::countr_zero(bits));
                idxb[m] = static_cast<TermIndex>(i);
                spb[m] = spread(fold_hash(row(i)));
                __builtin_prefetch(&table_.buf[(spb[m] & table_.mask) * kSlotBytes], 1, 0);
                if (++m == G) {
                    flush();
                }
            }
        }
        flush();
    }

    // Place a provably-absent entry without a growth check. Only rehash_to_ may use this: it has
    // already sized the table for every entry it is about to place, and calling the growing variant
    // there would recurse.
    auto place_slot_(TermIndex idx, size_t sp) -> void {
        size_t s = sp & table_.mask;
        while (slot_fp_(s) != 0) {
            s = (s + 1) & table_.mask;
        }
        write_slot_(s, fingerprint(sp), idx);
        ++table_.count;
    }

    [[nodiscard]] auto capacity() const -> size_t { return chunks_.size() * kChunkRows; }

    // Allocate whole chunks up to n rows. make_unique_for_overwrite, not make_unique: every row is
    // written by set() before it is read, so zeroing a chunk is the same wasted bandwidth the
    // default_init_allocator on the old flat vector existed to avoid.
    auto reserve_rows(size_t n) -> void {
        const size_t want = (n + kChunkRows - 1) / kChunkRows;
        chunks_.reserve(want);
        while (chunks_.size() < want) {
            chunks_.push_back(std::make_unique_for_overwrite<PosT[]>(kChunkRows * stride_));
        }
    }

    // The whole point of the chunked layout: one shift, one mask, one load from a table small enough to
    // stay in L1 (one pointer per kChunkRows rows). Callers that walk a run of consecutive rows should
    // hoist the chunk base instead of calling this per row -- see rehash_to_.
    [[gnu::always_inline]] [[nodiscard]] auto row_ptr_(size_t i) const noexcept -> const PosT * {
        return chunks_[i >> kChunkShift].get() + ((i & kChunkMask) * stride_);
    }
    [[gnu::always_inline]] [[nodiscard]] auto row_ptr_mut_(size_t i) noexcept -> PosT * {
        return chunks_[i >> kChunkShift].get() + ((i & kChunkMask) * stride_);
    }
    auto reserve_index(size_t n) -> void { rehash_to_(slots_for_(n + 1)); }

    // Insert (idx, spread hash) into the table with no duplicate probe — callers on this path insert
    // provably distinct keys (⊕G-injective miss batches, clone re-insertion).
    auto insert_slot_(TermIndex idx, size_t sp) -> void {
        rehash_if_needed_();
        place_slot_(idx, sp);
    }

    // Compare row i against key q without materializing the row (the find confirm). Reads the
    // popcount byte first, so a false h prefilter match usually costs one byte compare.
    [[nodiscard]] auto row_eq_key(size_t i, const key_type &q) const -> bool {
        const PosT *r = row_ptr_(i);
        const PosT c = r[0];
        if (c == kOverflowMarker) {
            return overflow_.at(i) == q;
        }
        if (q.count() != static_cast<size_t>(c)) {
            return false;
        }
        const PosT *pos = r + 1;
        for (size_t j = 0; j < c; ++j) {
            if (!q.test(pos[j])) {
                return false;
            }
        }
        return true;
    }

    static auto check_index_fits(size_t value) -> void {
        if (value >= kIndexCeiling) {
            throw TermIndexCeilingReached("OperatorIndex: operator index reached the TermIndex ceiling; rebuild with "
                                          "-Dmonoprop_WIDE_TERM_INDEX (this partition's term count exceeded ~2^32).");
        }
    }

    // One allocation per kChunkRows rows, never reallocated once handed out. The chunk table itself does
    // double, but at one pointer per kChunkRows*stride_ bytes of payload it is ~0.04% of the store.
    std::vector<std::unique_ptr<PosT[]>> chunks_ = {};
    size_t size_ = 0;
    size_t inline_width_ = kMaxInlinePositions;
    size_t stride_ = 1 + kMaxInlinePositions;
    // Lossless side-map for rows whose popcount exceeds inline_width_.
    std::unordered_map<size_t, value_type> overflow_ = {};
    Table table_ = {};
};

} // namespace monoprop::detail
