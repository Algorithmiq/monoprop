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
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

inline constexpr size_t kColumnBlockWords = 1024; // 8 KB block ≈ L1-resident (bench knee)

// Chunk height, in fold words. The kernel's chunk loop nests inside its block loop, so a chunk may be
// any divisor of kColumnBlockWords; it does not have to equal it.
//
// The optimum is interior and the trade is not the obvious one: what scales with the height is the tail
// buffer's capacity, not the unsealed rows, because seal_chunk_ sizes the next tail from the chunk it
// just sealed. So tail ~ C against directory ~ 1/C, and the sum bottoms out at 512 at the production
// per-partition size. Smaller C also costs fold time, ~28 ns per extra (column, chunk) visit, which is
// why the minimum is not at the smallest height. Measured table in docs/content/docs/building.mdx.
#ifndef monoprop_INVIDX_CHUNK_WORDS
#define monoprop_INVIDX_CHUNK_WORDS 512
#endif
inline constexpr size_t kChunkWords = static_cast<size_t>(monoprop_INVIDX_CHUNK_WORDS);
inline constexpr size_t kChunkRows = kChunkWords * 64;
inline constexpr size_t kChunkBitmapBytes = kChunkWords * sizeof(uint64_t);

// A chunk-local row id must fit the u16 the containers and the directory both store, and the kernel's
// chunk loop assumes chunks tile a block exactly.
static_assert(kChunkRows <= 65536, "chunk-local row ids are 16-bit");
static_assert(kColumnBlockWords % kChunkWords == 0, "chunks must tile a fold block");

// There is deliberately no bitmap-promotion threshold, and no density constant anywhere in this file.
// `monoprop_INVIDX_BITMAP_M_STAR` used to force Bitmap past a posting count, buying fold speed with
// memory; it was measured and deleted because 1.7x the index bought 4% of the fold once 16 partitions
// competed for the memory system, and the fold is itself a minority of `propagate`. Numbers in
// docs/content/docs/building.mdx.

// Which container holds one (column, chunk) cell, chosen per cell as whichever is smallest. A cell is
// re-chosen only when its chunk seals, so there is no in-place promotion and no one-way transition.
enum class ChunkTag : uint8_t {
    Empty = 0,   // no postings; occupies no arena bytes
    Bitmap = 1,  // kChunkBitmapBytes of chunk-height bits, 8-byte aligned so the XOR arm vectorises
    U8Delta = 2, // 1 byte per gap, 0xFF + 2 bytes little-endian for a gap of 255 or more
};

// A fourth container, U16 (2 bytes per posting, chunk-local absolute row ids), was built and deleted.
// It beats the delta stream only below density 1/368, where the argmin picked it for 5 KB of 92.8 MB;
// one fewer arm in the kernel's switch is worth more than that, and is not free in Pauli where the tags
// are genuinely mixed. A workload that does go that sparse pays 3 bytes per posting instead of 2 --
// bounded, and visible in `d_invidx_delta_bytes`.

// One directory cell, 8 bytes, in a flat kNumColumns-major array per chunk.
//
// `count` is the decode length for U8Delta, and the smallest-container rule caps it at
// kChunkRows/8 postings (past that a bitmap is strictly smaller), so it never saturates where it is
// load-bearing. For Bitmap the payload is fixed-size and self-describing, so `count` is diagnostic only;
// it can saturate there, but only for a fully-set column at the largest permitted chunk height.
struct ChunkDir {
    uint32_t offset = 0; // payload byte offset within this chunk's arena segment
    uint16_t count = 0;
    uint8_t tag = static_cast<uint8_t>(ChunkTag::Empty);
};
static_assert(sizeof(ChunkDir) == 8, "directory cells are sized so d_invidx_dir_bytes stays 8 per cell");

// The U8Delta gap encoding, in one place: one byte per gap, or kGapEscape plus two little-endian bytes
// for a gap of 255 or more.
//
// The escape stays a branch on both sides. Decoding it branchlessly -- read the wide gap
// unconditionally, select with a cmov, advance the pointer by 1 or 3 -- measured 1.015x to 1.04x SLOWER
// (4/4 and 4/4 paired, identical sink): the cmov puts the pointer advance on the loop-carried dependency
// chain, and at ~4.8% taken the branch predicts well enough that there was little to recover. It would
// also need padding past every stream.
inline constexpr uint32_t kGapEscape = 0xFFU;

// What `gap` costs in the stream, without encoding it.
inline constexpr auto gap_bytes(uint32_t gap) -> size_t {
    return gap < kGapEscape ? 1U : 3U;
}

// Append `gap` to any byte sink, so the arena, the tail and a scratch buffer share one encoder.
template <typename Push>
[[gnu::always_inline]] inline auto encode_gap(uint32_t gap, Push &&push) -> void {
    if (gap < kGapEscape) {
        push(static_cast<uint8_t>(gap));
        return;
    }
    push(static_cast<uint8_t>(kGapEscape));
    push(static_cast<uint8_t>(gap & 0xFFU));
    push(static_cast<uint8_t>(gap >> 8U));
}

// Read one gap and advance `p` past it.
[[gnu::always_inline]] inline auto decode_gap(const uint8_t *&p) -> uint32_t {
    uint32_t gap = *p++;
    if (gap == kGapEscape) [[unlikely]] {
        gap = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8U);
        p += 2;
    }
    return gap;
}

// Visit the `count` chunk-local row ids of a U8Delta byte stream, ascending. Length-free by design: the
// decode is driven by the posting count, so a payload may be followed by alignment padding.
template <typename RowOp>
[[gnu::always_inline]] inline auto for_each_delta_row(const uint8_t *p, size_t count, RowOp op) -> void {
    uint32_t next = 0;
    for (size_t j = 0; j < count; ++j) {
        const uint32_t row = next + decode_gap(p);
        op(row);
        next = row + 1;
    }
}

// Visit a bitmap payload's set bits, ascending, as ids local to the payload.
template <typename RowOp>
[[gnu::always_inline]] inline auto for_each_bitmap_row(const uint64_t *w, size_t nwords, RowOp op) -> void {
    for (size_t wi = 0; wi < nwords; ++wi) {
        for (uint64_t b = w[wi]; b != 0; b &= b - 1) {
            op(wi * 64 + static_cast<size_t>(std::countr_zero(b)));
        }
    }
}

// Lazy transposed operator storage: one bit-vector per column (bit position), bit r set iff term r
// touches that column. XOR-combining a generator G's columns yields |M ∩ G| mod 2 per term M -- the
// anticommutation bit for an even generator; odd generators add a per-row parity(|M|) correction, so both
// parities are served.
//
// Storage is one arena of sealed kChunkRows-row chunks plus a growing tail. Each (column, chunk) cell
// independently takes whichever container is smallest, which is bit-identical to all-dense because every
// container holds the same row set. The tail obeys the same rule, restricted to the two containers that
// can still be appended to: a delta stream (appending IS the encoding) and a row-proportional bitmap
// (appending is one bit). So append_rows stays a push_back and sealing is usually a memcpy.
template <size_t NumModes>
class InvertedIndex {
public:
    static constexpr size_t kNumColumns = Monomial<NumModes>::size();
    static constexpr size_t kNoColumnSkip = static_cast<size_t>(-1);

    auto rows() const -> size_t { return row_count_; }
    auto words() const -> size_t { return (row_count_ + 63) / 64; }
    auto sealed_chunks() const -> size_t { return chunk_base_.size(); }
    auto chunk_count() const -> size_t { return (row_count_ + kChunkRows - 1) / kChunkRows; }

    // A column with no postings anywhere: the scan's zero-postings early-out, which must stay O(1)
    // because it runs per fold column per gate.
    auto column_is_empty(size_t c) const -> bool { return col_postings_[c] == 0; }
    auto column_postings(size_t c) const -> size_t { return col_postings_[c]; }

    // One (column, chunk) cell. `count` drives a U8Delta decode; `bitmap_bytes` bounds a Bitmap read and
    // is zero for every other tag, so neither field can be read for a container it does not describe.
    //
    // A sealed bitmap spans the whole chunk, but a tail bitmap is only as tall as the rows appended so
    // far -- and shorter still if the column went quiet before the last row, since it grows only on a
    // push. Words past it hold no set rows, so stopping there is exact rather than an approximation.
    struct ChunkView {
        const uint8_t *data = nullptr;
        size_t count = 0;
        size_t bitmap_bytes = 0;
        ChunkTag tag = ChunkTag::Empty;
    };

    auto chunk(size_t c, size_t k) const -> ChunkView {
        assert(k < chunk_count());
        if (k < chunk_base_.size()) {
            const ChunkDir &e = dir_[k * kNumColumns + c];
            const ChunkTag tag = static_cast<ChunkTag>(e.tag);
            return ChunkView{arena_.data() + chunk_base_[k] + e.offset,
                             e.count,
                             tag == ChunkTag::Bitmap ? kChunkBitmapBytes : 0,
                             tag};
        }
        const std::vector<uint8_t> &t = tail_[c];
        const size_t m = tail_count_[c];
        const bool is_bitmap = tail_is_bitmap_[c];
        const ChunkTag tag = is_bitmap ? ChunkTag::Bitmap : (m == 0 ? ChunkTag::Empty : ChunkTag::U8Delta);
        return ChunkView{t.data(), m, is_bitmap ? t.size() : 0, tag};
    }

    auto container_tag(size_t c, size_t k) const -> ChunkTag { return chunk(c, k).tag; }

    // Bitmap payloads are placed on an 8-byte boundary at seal precisely so this cast is well-defined and
    // the XOR arm keeps auto-vectorising to vpxor.
    static auto bitmap_words(const ChunkView &v) -> const uint64_t * {
        assert(reinterpret_cast<uintptr_t>(v.data) % alignof(uint64_t) == 0);
        return reinterpret_cast<const uint64_t *>(v.data);
    }

    auto row_parity_words() const -> const uint64_t * {
        if (row_parity_.empty() && row_count_ != 0) {
            build_row_parity_();
        }
        return row_parity_.data();
    }

    // Every set row of a column, ascending, as an absolute row index. The container-independent view two
    // differently-stored indices can be compared through.
    template <typename RowOp>
    auto for_each_row_in_column(size_t c, RowOp op) const -> void {
        const size_t nk = chunk_count();
        for (size_t k = 0; k < nk; ++k) {
            const size_t base = k * kChunkRows;
            const ChunkView v = chunk(c, k);
            switch (v.tag) {
                case ChunkTag::Empty:
                    break;
                case ChunkTag::Bitmap:
                    for_each_bitmap_row(bitmap_words(v), v.bitmap_bytes / sizeof(uint64_t), [&](size_t local) {
                        op(base + local);
                    });
                    break;
                case ChunkTag::U8Delta:
                    for_each_delta_row(v.data, v.count, [&](uint32_t local) { op(base + local); });
                    break;
            }
        }
    }

    // Bytes the sealed arena holds, for tests and sizing checks that need it without the whole split.
    auto arena_bytes() const -> size_t { return arena_.size(); }

private:
    // Callers gate on row_count_ != 0, so the bitmap is never sized to zero words here. Bitmap cells are
    // walked bit by bit rather than word-XORed: the correction applies only to odd-|G| Majorana
    // generators, and the Majorana regime has no bitmap cells at all.
    auto build_row_parity_() const -> void {
        row_parity_.assign(words(), 0);
        for (size_t c = 0; c < kNumColumns; ++c) {
            if (col_postings_[c] == 0) {
                continue;
            }
            for_each_row_in_column(c, [this](size_t r) { row_parity_[r >> 6] ^= uint64_t{1} << (r & 63U); });
        }
    }

    // A tail bitmap can go bad in one way: the column falls quiet, a posting arrives far later, and the
    // bitmap stretches to cover the gap that a delta stream would have spent three bytes on. A delta
    // stream never exceeds 3 bytes per posting, so a bitmap past 3*m is provably the worse of the two.
    // Converting back at 4*m leaves a band either side of the crossover, so neither direction can thrash.
    static constexpr size_t kTailBitmapGiveUpFactor = 4;

    // Every growth path in this file states its own slack, because std::vector's doubling is exactly what
    // left the layout this replaced carrying 1.4x of unused capacity. Denominators are reciprocals: a
    // buffer grows by capacity/kTailGrowthDen, and so on.
    static constexpr size_t kTailGrowthDen = 4; // reallocation slack on an appending tail
    static constexpr size_t kTailSizingDen = 8; // headroom over the previous chunk's delta cost
    static constexpr size_t kTailShrinkDen = 4; // only reshape a tail once it is this far oversized
    static constexpr size_t kTailSlackPad = 16; // absorbs the first few appends on a small buffer

    // Grow the tail by a quarter rather than letting push_back double it. The tail is the whole index
    // until the first chunk seals, so its slack is not a rounding error there -- and at ~1 KB per column
    // the extra reallocations are free.
    static auto tail_reserve_(std::vector<uint8_t> &t, size_t needed) -> void {
        if (needed <= t.capacity()) {
            return;
        }
        t.reserve(std::max(needed, t.capacity() + t.capacity() / kTailGrowthDen + kTailSlackPad));
    }

    // Bytes a row-proportional bitmap needs to cover chunk-local rows [0, rows).
    static auto bitmap_bytes_for_rows_(size_t rows) -> size_t { return ((rows + 63) / 64) * sizeof(uint64_t); }

    auto tail_to_bitmap_(size_t c, uint32_t rows) -> void {
        const size_t want = bitmap_bytes_for_rows_(rows);
        std::vector<uint8_t> bits;
        // Sized to what it holds, not to what it may grow into: an index whose partition never fills a
        // chunk keeps this buffer for its whole life, so headroom here is permanent. tail_reserve_ grows
        // it geometrically from here as rows arrive.
        bits.reserve(want + kTailSlackPad);
        bits.resize(want, 0);
        uint64_t *w = reinterpret_cast<uint64_t *>(bits.data());
        for_each_delta_row(tail_[c].data(), tail_count_[c], [w](uint32_t row) {
            w[row >> 6U] |= uint64_t{1} << (row & 63U);
        });
        tail_[c].swap(bits);
        tail_is_bitmap_[c] = true;
    }

    auto tail_to_delta_(size_t c) -> void {
        std::vector<uint8_t> gaps;
        // Exact, at the cost of one extra scan on a rare path.
        gaps.reserve(tail_delta_bytes_(c) + kTailSlackPad);
        uint32_t next = 0;
        tail_for_each_row_(c, [&](uint32_t row) {
            encode_gap(row - next, [&gaps](uint8_t b) { gaps.push_back(b); });
            next = row + 1;
        });
        tail_[c].swap(gaps);
        tail_is_bitmap_[c] = false;
    }

    // Append one posting to the growing chunk. `local` is chunk-relative and must be strictly increasing
    // per column, which the row-ordered fill guarantees.
    auto tail_push_(size_t c, uint32_t local) -> void {
        std::vector<uint8_t> &t = tail_[c];
        const uint32_t gap = local - tail_next_[c];
        // The bookkeeping lands before the encoding, not after: tail_to_bitmap_ below re-encodes the
        // stream by decoding tail_count_ postings from it, so a count that trails by one silently drops
        // the very posting that triggered the switch.
        tail_next_[c] = local + 1;
        ++tail_count_[c];
        ++col_postings_[c];
        if (tail_is_bitmap_[c]) {
            const size_t want = bitmap_bytes_for_rows_(local + 1);
            if (t.size() < want) {
                tail_reserve_(t, want);
                t.resize(want, 0);
            }
            reinterpret_cast<uint64_t *>(t.data())[local >> 6U] |= uint64_t{1} << (local & 63U);
            if (t.size() > kTailBitmapGiveUpFactor * tail_count_[c]) {
                tail_to_delta_(c);
            }
            return;
        }
        tail_reserve_(t, t.size() + gap_bytes(gap));
        encode_gap(gap, [&t](uint8_t b) { t.push_back(b); });
        if (t.size() > bitmap_bytes_for_rows_(local + 1)) {
            tail_to_bitmap_(c, local + 1);
        }
    }

    // The tail's rows, ascending, whichever of the two forms holds them.
    template <typename RowOp>
    auto tail_for_each_row_(size_t c, RowOp op) const -> void {
        if (!tail_is_bitmap_[c]) {
            for_each_delta_row(tail_[c].data(), tail_count_[c], op);
            return;
        }
        for_each_bitmap_row(reinterpret_cast<const uint64_t *>(tail_[c].data()),
                            tail_[c].size() / sizeof(uint64_t),
                            [&op](size_t local) { op(static_cast<uint32_t>(local)); });
    }

    // What the tail's rows would cost as a delta stream. Free when the tail already is one; a scan when
    // it is a bitmap, which is what lets the seal below take an honest argmin either way.
    auto tail_delta_bytes_(size_t c) const -> size_t {
        if (!tail_is_bitmap_[c]) {
            return tail_[c].size();
        }
        size_t bytes = 0;
        uint32_t next = 0;
        tail_for_each_row_(c, [&](uint32_t row) {
            bytes += gap_bytes(row - next);
            next = row + 1;
        });
        return bytes;
    }

    static constexpr size_t kArenaGrowthDen = 8;      // bounded 12.5% slack, amortised O(1) copies
    static constexpr size_t kArenaProjectionDen = 32; // margin on the one-shot whole-arena projection
    static constexpr size_t kArenaAlign = alignof(uint64_t);

    // std::vector's own doubling is what left the layout this replaced carrying 1.43x of unused
    // capacity, so the arena never uses it.
    auto arena_reserve_(size_t needed) -> void {
        if (needed <= arena_.capacity()) {
            return;
        }
        const size_t cap = arena_.capacity();
        arena_.reserve(std::max(needed, cap + cap / kArenaGrowthDen));
    }

    // The whole tiering policy: per (column, chunk), take whichever container is smallest. Ties go to the
    // faster fold (Bitmap over postings), and the delta stream the tail already built is usually the
    // winner, in which case sealing is a memcpy.
    static auto choose_tag_(size_t postings, size_t delta_bytes) -> ChunkTag {
        if (postings == 0) {
            return ChunkTag::Empty;
        }
        return kChunkBitmapBytes <= delta_bytes ? ChunkTag::Bitmap : ChunkTag::U8Delta;
    }

    static auto tag_bytes_(ChunkTag tag, size_t delta_bytes) -> size_t {
        switch (tag) {
            case ChunkTag::Empty:
                return 0;
            case ChunkTag::Bitmap:
                return kChunkBitmapBytes;
            case ChunkTag::U8Delta:
                return delta_bytes;
        }
        return 0;
    }

    static auto align_up_(size_t n) -> size_t { return (n + kArenaAlign - 1) & ~(kArenaAlign - 1); }

    using TagArray = std::array<ChunkTag, kNumColumns>;
    using DeltaArray = std::array<uint32_t, kNumColumns>;

    // Price every column of the completed chunk and pick its container, returning the segment's payload
    // size. Costed against what a delta stream WOULD take, not against what the tail happens to hold: a
    // tail that switched to a bitmap mid-chunk must still be able to seal as postings if the column went
    // quiet afterwards, or a brief dense spell would cost a full bitmap for the whole chunk.
    auto size_chunk_(TagArray &tags, DeltaArray &deltas) const -> size_t {
        size_t total = 0;
        for (size_t c = 0; c < kNumColumns; ++c) {
            deltas[c] = static_cast<uint32_t>(tail_delta_bytes_(c));
            const ChunkTag tag = choose_tag_(tail_count_[c], deltas[c]);
            if (tag == ChunkTag::Bitmap) {
                total = align_up_(total);
            }
            tags[c] = tag;
            total += tag_bytes_(tag, deltas[c]);
        }
        return total;
    }

    // Reserve the arena for a segment of `total` payload bytes and return its aligned base. The segment
    // base is aligned too, so every offset computed against it keeps its alignment.
    auto open_segment_(size_t total) -> size_t {
        const size_t needed = align_up_(arena_.size()) + total;
        // A bulk rebuild knows its final row count, so the very first seal projects the finished size
        // from the chunk it just measured and reserves it in one step. Worth 0.81 B/term at 5M rows.
        //
        // Projecting once, rather than at every seal, is load-bearing. Chunk sizes wobble by ~0.02%, and
        // re-projecting turns each wobble into a request one byte over capacity -- which
        // arena_reserve_'s geometric floor then rounds up by the full 12.5%, the exact slack the
        // projection is here to avoid. The margin covers the drift so the floor never fires; if it is
        // ever short, the geometric fallback is still correct, just fatter.
        const size_t chunks_total = row_count_ / kChunkRows; // floor: a partial trailing chunk never seals
        if (chunk_base_.empty() && chunks_total > 1) {
            const size_t projected = needed * chunks_total;
            arena_.reserve(projected + projected / kArenaProjectionDen);
        }
        arena_reserve_(needed);
        arena_.resize(align_up_(arena_.size()), 0);
        return arena_.size();
    }

    // Write one column's payload at the arena's current end and fill in its directory cell.
    auto write_column_(size_t c, ChunkTag tag, uint32_t delta_bytes, size_t base, ChunkDir &cell) -> void {
        if (tag == ChunkTag::Bitmap) {
            arena_.resize(base + align_up_(arena_.size() - base), 0);
        }
        cell.offset = static_cast<uint32_t>(arena_.size() - base);
        cell.count = static_cast<uint16_t>(std::min<size_t>(tail_count_[c], 0xFFFFU));
        cell.tag = static_cast<uint8_t>(tag);
        switch (tag) {
            case ChunkTag::Empty:
                break;
            case ChunkTag::U8Delta: {
                // A delta tail seals as a memcpy; a bitmap tail is re-encoded, which is the case
                // size_chunk_ priced.
                if (!tail_is_bitmap_[c]) {
                    arena_.insert(arena_.end(), tail_[c].begin(), tail_[c].end());
                }
                else {
                    uint32_t next = 0;
                    tail_for_each_row_(c, [&](uint32_t local) {
                        encode_gap(local - next, [this](uint8_t b) { arena_.push_back(b); });
                        next = local + 1;
                    });
                }
                delta_bytes_ += delta_bytes;
                break;
            }
            case ChunkTag::Bitmap: {
                const size_t at = arena_.size();
                arena_.resize(at + kChunkBitmapBytes, 0);
                uint64_t *w = reinterpret_cast<uint64_t *>(arena_.data() + at);
                tail_for_each_row_(c, [w](uint32_t local) { w[local >> 6U] |= uint64_t{1} << (local & 63U); });
                bitmap_bytes_ += kChunkBitmapBytes;
                ++bitmap_chunks_;
                break;
            }
        }
    }

    // Empty a column's tail and size it for the next chunk from the delta cost of the one just sealed,
    // rather than leaving std::vector's growth to settle there. Chunks are the same height and a column's
    // density barely moves between them, so the previous size plus an eighth is both a good guess and a
    // bounded one. Measured worth 2.2 B/term at 234k rows, where the tail is 16% of the index.
    //
    // Only reshape when the buffer is actually the wrong size: reshaping on every seal frees and
    // re-mallocs kNumColumns buffers per chunk to move a capacity that barely moved, and that churn shows
    // up as resident fragmentation rather than as anything this counts.
    auto reset_tail_(size_t c, uint32_t delta_bytes) -> void {
        const size_t want = delta_bytes + delta_bytes / kTailSizingDen + kTailSlackPad;
        tail_[c].clear();
        if (tail_[c].capacity() > want + want / kTailShrinkDen) {
            tail_[c].shrink_to_fit();
        }
        if (tail_[c].capacity() < want) {
            tail_[c].reserve(want);
        }
        tail_is_bitmap_[c] = false;
        tail_next_[c] = 0;
        tail_count_[c] = 0;
    }

    // Move the completed growing chunk into the arena and reset the tail. Sizing runs first so the arena
    // is reserved once and no payload write can reallocate under a pointer taken into it.
    auto seal_chunk_() -> void {
        TagArray tags{};
        DeltaArray deltas{};
        const size_t total = size_chunk_(tags, deltas);

        const size_t base = open_segment_(total);
        chunk_base_.push_back(base);
        dir_.resize(dir_.size() + kNumColumns);
        ChunkDir *row = dir_.data() + (dir_.size() - kNumColumns);

        for (size_t c = 0; c < kNumColumns; ++c) {
            write_column_(c, tags[c], deltas[c], base, row[c]);
            reset_tail_(c, deltas[c]);
        }
        assert(arena_.size() - base == total);
        sealed_rows_ += kChunkRows;
    }

    // Encode the new rows [base, base+n) of `op` into the growing chunk, sealing whenever one fills.
    // row_count_ must already cover [0, base+n) and `base` must be the first unencoded row.
    template <typename Rows>
    auto fill_rows(const Rows &op, size_t base, size_t n) -> void {
        assert(base + n == row_count_);
        const size_t end = base + n;
        size_t r = base;
        while (r < end) {
            const size_t chunk_end = sealed_rows_ + kChunkRows;
            const size_t stop = std::min(end, chunk_end);
            for (; r < stop; ++r) {
                const uint32_t local = static_cast<uint32_t>(r - sealed_rows_);
                for_each_row_position<NumModes>(op, r, [this, local](size_t bit) { tail_push_(bit, local); });
            }
            if (r == chunk_end) {
                seal_chunk_();
            }
        }
    }

public:
    // Clear, then append everything: one code path, so the from-scratch and the incremental build cannot
    // disagree about anything, container choice included.
    template <typename Rows>
    auto rebuild(const Rows &op) -> void {
        const size_t size = op.size();
        arena_.clear();
        arena_.shrink_to_fit();
        chunk_base_.clear();
        chunk_base_.shrink_to_fit();
        dir_.clear();
        dir_.shrink_to_fit();
        for (auto &t : tail_) {
            t.clear();
            t.shrink_to_fit();
        }
        tail_next_.fill(0);
        tail_count_.fill(0);
        tail_is_bitmap_.fill(false);
        col_postings_.fill(0);
        row_parity_.clear();
        bitmap_bytes_ = 0;
        delta_bytes_ = 0;
        bitmap_chunks_ = 0;
        sealed_rows_ = 0;
        row_count_ = size;
        if (size == 0) {
            return;
        }
        fill_rows(op, 0, size);
    }

    // Bulk-append the contiguous new terms op[base .. base+n), which must equal rows [row_count_ ..
    // row_count_+n).
    template <typename Rows>
    auto append_rows(const Rows &op, size_t base, size_t n) -> void {
        if (n == 0) {
            return;
        }
        row_count_ = base + n;
        fill_rows(op, base, n);
        if (!row_parity_.empty()) {
            row_parity_.resize((row_count_ + 63) / 64, 0);
            for (size_t j = 0; j < n; ++j) {
                const size_t r = base + j;
                if (row_popcount<NumModes>(op, r) & 1U) {
                    row_parity_[r >> 6] |= (uint64_t{1} << (r & 63));
                }
            }
        }
    }

    auto memory_bytes() const -> size_t {
        size_t total = arena_.capacity();
        total += chunk_base_.capacity() * sizeof(size_t);
        total += dir_.capacity() * sizeof(ChunkDir);
        for (const auto &t : tail_) {
            total += t.capacity();
        }
        total += row_parity_.capacity() * sizeof(uint64_t);
        return total;
    }

    // Diagnostic split of memory_bytes(). The arena's slack is reported so it stays visible and can never
    // quietly become the 1.43x of std::vector capacity this layout replaced.
    struct Stats {
        size_t bitmap_bytes = 0;
        size_t delta_bytes = 0; // U8Delta payloads, tail included
        size_t dir_bytes = 0;
        size_t tail_bytes = 0;
        size_t arena_slack_bytes = 0;
        size_t bitmap_chunks = 0;
    };

    auto stats() const -> Stats {
        Stats s;
        s.bitmap_bytes = bitmap_bytes_;
        s.bitmap_chunks = bitmap_chunks_;
        size_t tail_bitmap = 0;
        for (size_t c = 0; c < kNumColumns; ++c) {
            const size_t cap = tail_[c].capacity();
            s.tail_bytes += cap;
            if (tail_is_bitmap_[c]) {
                tail_bitmap += cap;
                ++s.bitmap_chunks;
            }
        }
        s.bitmap_bytes += tail_bitmap;
        s.delta_bytes = delta_bytes_ + (s.tail_bytes - tail_bitmap);
        s.dir_bytes = dir_.capacity() * sizeof(ChunkDir) + chunk_base_.capacity() * sizeof(size_t);
        s.arena_slack_bytes = arena_.capacity() - arena_.size();
        return s;
    }

private:
    std::vector<uint8_t> arena_;     // sealed payloads, chunk-major then column-major
    std::vector<size_t> chunk_base_; // arena offset of each sealed chunk's segment; size == sealed chunks
    std::vector<ChunkDir> dir_;      // flat [chunk][column], kNumColumns entries per sealed chunk

    // The growing chunk, which the smallest-container rule applies to as well. Exactly two containers can
    // still be appended to: a delta stream, which is nothing but appends, and a bitmap, where an append
    // is one bit. So the tail starts as a delta stream and switches to a row-proportional bitmap the
    // moment the bitmap is smaller. Leaving the tail delta-only was measured at 1.34x WORSE than the
    // two-tier layout it replaces on 127-qubit kicked Ising, where every partition holds fewer rows than
    // a chunk and columns run to density 0.44 -- a byte per posting against a bit per row.
    std::array<std::vector<uint8_t>, kNumColumns> tail_{};
    std::array<uint32_t, kNumColumns> tail_next_{};  // next chunk-local row that would encode as gap 0
    std::array<uint32_t, kNumColumns> tail_count_{}; // postings held by the tail, per column
    std::array<bool, kNumColumns> tail_is_bitmap_{}; // tail_[c] holds bitmap words, not a delta stream
    std::array<size_t, kNumColumns> col_postings_{}; // lifetime postings, per column; makes emptiness O(1)

    size_t row_count_ = 0;
    size_t sealed_rows_ = 0; // always a multiple of kChunkRows, and the growing chunk's first row

    // Container split of the arena, maintained at seal so the diagnostics never walk the directory.
    size_t bitmap_bytes_ = 0;
    size_t delta_bytes_ = 0;
    size_t bitmap_chunks_ = 0;

    // Parity of |M| per row, packed 1 bit/row: bit r = popcount(row r) & 1. Built on first use and only
    // by odd-|G| generators, so even-parity workloads never allocate it.
    mutable std::vector<uint64_t> row_parity_; // empty == not built
};

// Reusable fold blocks (thread_local: each partition master owns its copy). Two independent scratches
// because the build scan needs the generator fold and the pivot column expanded simultaneously -- the
// tag parameter is what keeps them independent, since each instantiation gets its own thread_local.
template <int Slot>
inline auto block_scratch_() -> std::vector<uint64_t> & {
    static thread_local std::vector<uint64_t> blk;
    if (blk.size() < kColumnBlockWords) {
        blk.assign(kColumnBlockWords, 0);
    }
    return blk;
}
inline auto column_block_scratch() -> std::vector<uint64_t> & {
    return block_scratch_<0>();
}
inline auto pivot_column_block_scratch() -> std::vector<uint64_t> & {
    return block_scratch_<1>();
}

// XOR one (column, chunk) cell into blk over the absolute fold words [wlo, whi), which lie inside chunk
// k. Postings are ascending, so a range narrower than the chunk stops early rather than scanning on.
//
// `whole` says [wlo, whi) covers chunk k entirely, which every production call site satisfies for every
// chunk but the last: the scan and both replay paths block from 0 in kColumnBlockWords steps, and
// kChunkWords divides that. It is worth a separate loop because the two range tests are per *posting*,
// and at Majorana density a posting costs only ~2.2 cycles to begin with -- the tests are a fifth of
// the arm. `whole` also makes cw >= bb, so the block base folds into a pointer once instead of being
// re-added per posting.
template <size_t NumModes>
[[gnu::always_inline]] inline auto xor_column_chunk(const InvertedIndex<NumModes> &sc,
                                                    size_t c,
                                                    size_t k,
                                                    uint64_t *blk,
                                                    size_t bb,
                                                    size_t wlo,
                                                    size_t whi,
                                                    bool whole) -> void {
    const auto v = sc.chunk(c, k);
    const size_t cw = k * kChunkWords;
    switch (v.tag) {
        case ChunkTag::Empty:
            return;
        case ChunkTag::Bitmap: {
            const uint64_t *w = InvertedIndex<NumModes>::bitmap_words(v);
            // A tail bitmap is only as tall as the rows pushed into it; words past it hold no set rows,
            // so stopping short of `whi` is exact, not an approximation.
            const size_t wend = std::min(whi, cw + v.bitmap_bytes / sizeof(uint64_t));
            for (size_t wi = wlo; wi < wend; ++wi) {
                blk[wi - bb] ^= w[wi - cw];
            }
            return;
        }
        case ChunkTag::U8Delta: {
            const uint8_t *p = v.data;
            uint32_t next = 0;
            // The two arms below decode identically and are kept apart on purpose: the whole-chunk arm
            // drops the two per-posting range tests, which are a fifth of the arm at Majorana density.
            if (whole) {
                // A plain read-modify-write per posting, deliberately. Roughly half of all postings share
                // a 64-bit word with their predecessor (the mean gap is 1/d ~ 84 rows), so holding the
                // word in a register and storing only on a word change looks like it should pay -- it
                // measured 2.4x SLOWER, because the word-changed test is a ~50%-taken branch and a
                // mispredict here costs ~15 cycles against a ~3.5-cycle posting. This loop is bound by
                // branches, not by store-to-load forwarding.
                uint64_t *out = blk + (cw - bb);
                for (size_t j = 0; j < v.count; ++j) {
                    next += decode_gap(p);
                    out[next >> 6U] ^= uint64_t{1} << (next & 63U);
                    ++next;
                }
                return;
            }
            const uint32_t lo = static_cast<uint32_t>((wlo - cw) * 64);
            const uint32_t hi = static_cast<uint32_t>((whi - cw) * 64);
            for (size_t j = 0; j < v.count; ++j) {
                const uint32_t local = next + decode_gap(p);
                next = local + 1;
                if (local >= hi) {
                    return;
                }
                if (local >= lo) {
                    blk[cw + (local >> 6U) - bb] ^= uint64_t{1} << (local & 63U);
                }
            }
            return;
        }
    }
}

// The fold kernel. `Seed` writes blk before accumulating (combine_columns_block); otherwise blk is XORed
// into (xor_columns_block). `skip` drops one entry of `cols` by index, which is how the scan folds a
// generator whose pivot column it has already expanded separately.
//
// Seeding is per chunk over that chunk's own word sub-range only. The chunks partition [bb, be), so this
// is exactly equivalent to one seed over the whole block -- and it lets each chunk pick its own bitmap to
// memcpy from, which one block-wide seed cannot.
template <bool Seed, size_t NumModes>
[[gnu::always_inline]] inline auto fold_columns_block(const InvertedIndex<NumModes> &sc,
                                                      std::span<const size_t> cols,
                                                      uint64_t *blk,
                                                      size_t bb,
                                                      size_t be,
                                                      size_t skip) -> void {
    if (be <= bb) {
        return;
    }
    const size_t k0 = bb / kChunkWords;
    const size_t k1 = (be - 1) / kChunkWords;
    for (size_t k = k0; k <= k1; ++k) {
        const size_t cw = k * kChunkWords;
        const size_t wlo = std::max(bb, cw);
        const size_t whi = std::min(be, cw + kChunkWords);
        const bool whole = (wlo == cw) && (whi == cw + kChunkWords);
        size_t seed_ci = cols.size();
        if constexpr (Seed) {
            uint64_t *out = blk + (wlo - bb);
            const size_t nw = whi - wlo;
            // Only a bitmap that spans the whole sub-range can be copied wholesale: a tail bitmap that
            // stops short would leave the rest of the block holding whatever the caller had there.
            for (size_t ci = 0; ci < cols.size(); ++ci) {
                if (ci == skip) {
                    continue;
                }
                const auto v = sc.chunk(cols[ci], k);
                if (v.tag == ChunkTag::Bitmap && v.bitmap_bytes >= (whi - cw) * sizeof(uint64_t)) {
                    // XOR is commutative, so seeding from a bitmap is bit-identical to memset + XOR-all
                    // while saving one pass over the chunk.
                    std::memcpy(out, InvertedIndex<NumModes>::bitmap_words(v) + (wlo - cw), nw * sizeof(uint64_t));
                    seed_ci = ci;
                    break;
                }
            }
            if (seed_ci == cols.size()) {
                std::memset(out, 0, nw * sizeof(uint64_t));
            }
        }
        for (size_t ci = 0; ci < cols.size(); ++ci) {
            if (ci == skip || ci == seed_ci) {
                continue;
            }
            // Deliberately no software prefetch of the next column's payload, though the arena is
            // chunk-major and that stream is a genuinely cold line: `__builtin_prefetch` won 4% on one
            // core and 10% with 16 partitions competing, and still moved nothing end to end -- slightly
            // the wrong way on kicked Ising. Do not re-add it on the strength of a kernel microbenchmark.
            xor_column_chunk<NumModes>(sc, cols[ci], k, blk, bb, wlo, whi, whole);
        }
    }
}

// XOR a generator's inverted-index columns for fold words [bb, be) into blk[0 .. be-bb), seeding rather
// than accumulating. XOR associativity means any block decomposition reproduces the full-width fold
// bit-for-bit, and every container holds the same row set, so the result never depends on which
// containers the index happened to pick.
template <size_t NumModes>
[[gnu::always_inline]] inline auto combine_columns_block(const InvertedIndex<NumModes> &sc,
                                                         std::span<const size_t> cols,
                                                         uint64_t *blk,
                                                         size_t bb,
                                                         size_t be) -> void {
    fold_columns_block<true, NumModes>(sc, cols, blk, bb, be, InvertedIndex<NumModes>::kNoColumnSkip);
}

// combine_columns_block's accumulating form: XOR into an already-seeded blk, optionally skipping one
// entry of `cols` (by index into `cols`, not by column number).
template <size_t NumModes>
[[gnu::always_inline]] inline auto xor_columns_block(const InvertedIndex<NumModes> &sc,
                                                     std::span<const size_t> cols,
                                                     uint64_t *blk,
                                                     size_t bb,
                                                     size_t be,
                                                     size_t skip = InvertedIndex<NumModes>::kNoColumnSkip) -> void {
    fold_columns_block<false, NumModes>(sc, cols, blk, bb, be, skip);
}

} // namespace monoprop::detail
