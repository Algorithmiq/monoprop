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

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace monoprop::detail {

// The compact query record: a monomial as its ascending set-bit positions instead of its dense words.
//
// WHY. At the 29M-term configuration the dense record is kWords + 1 = 9 words = 72 B carrying a mean of
// 5.33 set bits and a sign, and the two alltoallv legs that move it are 68.2% of layer time. The record
// is measured at 72.0 B and 1.48 GiB per build_graph. Positions cost 2 B each, so the same information
// is 16 B: 4.5x fewer bytes through a funnel that is bytes-bound, not latency-bound.
//
// WHAT IS FIXED AND WHY IT HAS TO BE. Reading the wire path pins three constraints, and this layout is
// what is left after all three:
//   1. There is no byte datatype. datatype<T> (MPICompat.h) supports int, double, uint32_t, uint64_t,
//      size_t and static_asserts on anything else, so a record must be a whole number of 64-bit words
//      riding VecZ. Adding a uint8_t specialisation would also drop checked_mpi_count's per-peer ceiling
//      from ~17 GiB to ~2 GiB against a measured 1.48 GiB aggregate -- too close to be comfortable.
//   2. Fixed stride is load-bearing. Engine.h derives the SECOND alltoallv's recv counts as
//      bytes / stride precisely to avoid a count-exchange round; two more sites divide by the stride.
//      Varint, delta and run-length encodings are therefore out -- they would cost four extra
//      collectives per gate, charged straight to the 68.2%.
//   3. Record order IS the floating-point accumulation order. Resolve.h mints each miss's term index in
//      (sender, record) order and, at the measured ~0 hit rate, essentially every record is a miss. So
//      records may not be reordered, sorted or coalesced to compress better.
//
// THE ESCAPE. k is NOT bounded by the cutoff: a fully paired term is kept unconditionally
// (AlgebraCommon.h), so k = 2d can exceed the six positions that fit beside the header. This is rare --
// 94 fully-paired rows in 20,860,967, and one row at k=8 -- but rare is not never, and getting it wrong
// is silent: a truncated monomial is a different, valid-looking term that inserts as a duplicate row.
//
// The escape is INLINE CONTINUATION RECORDS, not a side region. A header record with kFlagContinued set
// is followed by ceil((k - 6) / 8) records that are pure positions. This keeps the fixed stride, adds no
// collective, and preserves record order. What it changes is that the number of QUERIES is no longer
// buf.size() / stride -- that ratio now counts RECORDS, which is exactly what MPI needs for its counts,
// while the query count comes from a local scan (count_queries below). The alternative considered and
// rejected was a flag carrying an index into a per-rank overflow region announced in a header record:
// that needs a side channel, a second length to keep consistent, and a rule for the empty case.
//
// LAYOUT (little-endian by explicit shift, not by struct punning, so the bytes do not depend on how the
// compiler lays out a bitfield):
//   header word 0: pos0 | pos1<<16 | pos2<<32 | pos3<<48
//   header word 1: pos4 | pos5<<16 | k<<32 (16 bits) | phase<<48 | flags<<56
//   continuation:  eight more positions, four per word, same packing
// Unused position slots are written as zero rather than left indeterminate, so identical inputs produce
// identical bytes and a wire dump is comparable between runs.
template <size_t NumModes>
struct CompactQuery {
    using PosT = uint16_t;

    static constexpr size_t kBits = 2 * NumModes;
    // 65535 and not 65536: k is a 16-bit field and k <= kBits, so allowing kBits == 65536 would let a
    // fully occupied monomial write a k that wraps to zero.
    static_assert(kBits <= 65535, "a physical bit position and the popcount must both fit a uint16_t");

    static constexpr size_t kWordsPerRecord = 2;    // 16 B; the fixed stride
    static constexpr size_t kInlinePositions = 6;   // 12 B of positions + 4 B of header
    static constexpr size_t kContPositions = 8;     // a continuation record is all positions
    static constexpr uint8_t kFlagContinued = 0x01; // more position records follow this one

    // k gets a full 16 bits, and d is NOT on the wire.
    //
    // k had one byte in the first cut of this format, on the reasoning that 255 positions is far above
    // anything that reaches this path. That reasoning is wrong: k is bounded by the CUTOFF only for
    // unpaired terms, and a fully paired term is kept unconditionally, so at MAX_NUM_MODES = 1024 a
    // paired term can carry k = 2048. One byte would have truncated it silently into a different,
    // valid-looking monomial. Widening k needed a byte, and d was the one to give up: its only consumer
    // recomputes the packed (k,d) digest at insert, and d is ~k comparisons over positions that are
    // already in hand and already known to be ascending. Carrying it saved nothing measurable.
    static constexpr size_t kMaxPositions = 65535;

    // bit_cast, not a conversion, so the coefficient arrives bit-identical. Defined here rather than
    // reused from Common.h because Common.h includes this header, not the other way round.
    static_assert(sizeof(size_t) == sizeof(double), "the fused value word assumes a 64-bit VecZ element");
    [[nodiscard]] static auto encode_value_bits(double v) noexcept -> size_t { return std::bit_cast<size_t>(v); }
    [[nodiscard]] static auto decode_value_bits(size_t w) noexcept -> double { return std::bit_cast<double>(w); }

    [[nodiscard]] static constexpr auto pos_in(uint64_t w, size_t j) noexcept -> PosT {
        return static_cast<PosT>((w >> (16U * j)) & 0xFFFFU);
    }
    [[nodiscard]] static constexpr auto pack_pos(uint64_t w, size_t j, PosT p) noexcept -> uint64_t {
        return w | (static_cast<uint64_t>(p) << (16U * j));
    }

    // Records occupied by a term of k positions: the header, plus ceil((k-6)/8) continuations.
    [[nodiscard]] static constexpr auto records_for(size_t k) noexcept -> size_t {
        return (k <= kInlinePositions) ? 1U : 1U + ((k - kInlinePositions + kContPositions - 1U) / kContPositions);
    }
    // ...and the WORDS it occupies. Offsets into a query buffer are counted in words, not records,
    // because the fused stream interleaves a single-word value after each query: a record index cannot
    // name the position of the query that follows one. Everything below therefore takes and returns a
    // word offset, and kWordsPerRecord survives only as the shape of one record.
    [[nodiscard]] static constexpr auto words_for(size_t k) noexcept -> size_t {
        return records_for(k) * kWordsPerRecord;
    }

    // pos must be ASCENDING physical bit positions, k of them; d = number of modes carrying both
    // Majoranas. Ascending is not decorative: OperatorIndex rows are ascending, row_eq_key compares
    // element-wise, and the digest overloads assume it.
    // Returns the WORDS written, so a caller keeping a byte accounting does not have to re-derive
    // the record width. LayerProfile's qbytes did exactly that against a hardcoded dense stride,
    // which reports 72 B/record whatever the codec actually wrote -- an instrument that cannot see
    // the change it exists to measure.
    static auto push(VecZ &buf, const PosT *pos, size_t k, int phase) -> size_t {
        assert(k <= kMaxPositions && "term has more positions than the record's k field can hold");
        assert(phase >= -128 && phase <= 127 && "the phase byte holds a rotation sign, not a magnitude");
        for (size_t j = 1; j < k; ++j) {
            assert(pos[j] > pos[j - 1] && "positions must be strictly ascending");
        }

        const size_t n_inline = (k < kInlinePositions) ? k : kInlinePositions;
        uint64_t w0 = 0;
        for (size_t j = 0; j < n_inline && j < 4; ++j) {
            w0 = pack_pos(w0, j, pos[j]);
        }
        uint64_t w1 = 0;
        for (size_t j = 4; j < n_inline; ++j) {
            w1 = pack_pos(w1, j - 4, pos[j]);
        }
        const uint8_t flags = (k > kInlinePositions) ? kFlagContinued : uint8_t{0};
        w1 |= static_cast<uint64_t>(static_cast<uint16_t>(k)) << 32U;
        w1 |= static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(phase))) << 48U;
        w1 |= static_cast<uint64_t>(flags) << 56U;
        buf.push_back(static_cast<size_t>(w0));
        buf.push_back(static_cast<size_t>(w1));

        for (size_t base = kInlinePositions; base < k; base += kContPositions) {
            uint64_t c0 = 0;
            uint64_t c1 = 0;
            for (size_t j = 0; j < kContPositions && base + j < k; ++j) {
                if (j < 4) {
                    c0 = pack_pos(c0, j, pos[base + j]);
                }
                else {
                    c1 = pack_pos(c1, j - 4, pos[base + j]);
                }
            }
            buf.push_back(static_cast<size_t>(c0));
            buf.push_back(static_cast<size_t>(c1));
        }
        return words_for(k);
    }

    [[nodiscard]] static auto flags_at(const VecZ &buf, size_t off) noexcept -> uint8_t {
        return static_cast<uint8_t>((static_cast<uint64_t>(buf[off + 1]) >> 56U) & 0xFFU);
    }
    [[nodiscard]] static auto k_at(const VecZ &buf, size_t off) noexcept -> size_t {
        return static_cast<size_t>((static_cast<uint64_t>(buf[off + 1]) >> 32U) & 0xFFFFU);
    }
    [[nodiscard]] static auto phase_at(const VecZ &buf, size_t off) noexcept -> int {
        const auto raw = static_cast<uint8_t>((static_cast<uint64_t>(buf[off + 1]) >> 48U) & 0xFFU);
        return static_cast<int>(static_cast<int8_t>(raw));
    }

    // Writes the term's positions to out (which must hold k_at(buf, rec)); returns the record index of
    // the next term. Reading positions rather than a Monomial is what lets the resolve side skip the
    // dense round-trip once the store speaks positions too; read_mono wraps this for the consumers that
    // still want a bitset.
    //
    // OutT is a template parameter rather than PosT because the STORE's position type is narrower than
    // the WIRE's below 129 modes: OperatorIndex picks uint8_t for 2*NumModes <= 256, while a record is
    // always uint16_t (its k field must hold a fully paired term at MAX_NUM_MODES). Decoding straight
    // into the store's width is what keeps the resolve path free of a second copy. The narrowing is safe
    // because every position is < 2*NumModes, which is the same bound OperatorIndex::PosT is chosen
    // against -- and it is re-asserted where the positions become a row, not only here.
    template <typename OutT>
    static auto read_positions(const VecZ &buf, size_t off, OutT *out) -> size_t {
        const size_t k = k_at(buf, off);
        const auto w0 = static_cast<uint64_t>(buf[off]);
        const auto w1 = static_cast<uint64_t>(buf[off + 1]);
        const size_t n_inline = (k < kInlinePositions) ? k : kInlinePositions;
        for (size_t j = 0; j < n_inline; ++j) {
            out[j] = static_cast<OutT>((j < 4) ? pos_in(w0, j) : pos_in(w1, j - 4));
        }
        size_t next = off + kWordsPerRecord;
        for (size_t base = kInlinePositions; base < k; base += kContPositions) {
            const auto c0 = static_cast<uint64_t>(buf[next]);
            const auto c1 = static_cast<uint64_t>(buf[next + 1]);
            for (size_t j = 0; j < kContPositions && base + j < k; ++j) {
                out[base + j] = static_cast<OutT>((j < 4) ? pos_in(c0, j) : pos_in(c1, j - 4));
            }
            next += kWordsPerRecord;
        }
        assert(next == off + words_for(k) && "continuation walk disagrees with words_for");
        assert(check_header(buf, off, out) && "record header is inconsistent with its own positions");
        return next;
    }

    // Every field on the wire must be checkable from the rest of the record, or it rots: nothing in the
    // 2a wire change reads the flag bit, so without this it would be write-only and a wrong value would
    // ship undetected until a later change started trusting it. Debug-only -- it walks the positions.
    template <typename OutT>
    [[nodiscard]] static auto check_header(const VecZ &buf, size_t off, const OutT *pos) -> bool {
        const size_t k = k_at(buf, off);
        const bool continued = (flags_at(buf, off) & kFlagContinued) != 0U;
        if (continued != (k > kInlinePositions)) {
            return false;
        }
        if ((flags_at(buf, off) & ~kFlagContinued) != 0U) {
            return false; // an unknown flag bit is a format the reader does not understand
        }
        for (size_t j = 0; j + 1 < k; ++j) {
            if (pos[j] >= pos[j + 1]) {
                return false; // positions must arrive strictly ascending
            }
        }
        return true;
    }

    // d, recomputed where it is wanted rather than carried. Ascending order (checked above) makes a
    // complete mode pair exactly an even position immediately followed by its successor.
    template <typename OutT>
    [[nodiscard]] static auto pair_count(const OutT *pos, size_t k) noexcept -> size_t {
        size_t d = 0;
        for (size_t j = 0; j + 1 < k; ++j) {
            if ((pos[j] % 2 == 0) && (pos[j + 1] == pos[j] + 1)) {
                ++d;
            }
        }
        return d;
    }

    // Dense reconstruction, for every consumer that still speaks Monomial. A zero plus k set bits
    // against the 64 B memcpy it replaces -- roughly a wash, which is the point: the whole win is on the
    // wire, so the receiver is allowed to stay dense.
    static auto read_mono(const VecZ &buf, size_t off, Monomial<NumModes> &mono_out, int &phase_out) -> size_t {
        const size_t k = k_at(buf, off);
        phase_out = phase_at(buf, off);
        mono_out = Monomial<NumModes>{};
        const auto w0 = static_cast<uint64_t>(buf[off]);
        const auto w1 = static_cast<uint64_t>(buf[off + 1]);
        const size_t n_inline = (k < kInlinePositions) ? k : kInlinePositions;
        for (size_t j = 0; j < n_inline; ++j) {
            mono_out.set(static_cast<size_t>((j < 4) ? pos_in(w0, j) : pos_in(w1, j - 4)));
        }
        size_t next = off + kWordsPerRecord;
        for (size_t base = kInlinePositions; base < k; base += kContPositions) {
            const auto c0 = static_cast<uint64_t>(buf[next]);
            const auto c1 = static_cast<uint64_t>(buf[next + 1]);
            for (size_t j = 0; j < kContPositions && base + j < k; ++j) {
                mono_out.set(static_cast<size_t>((j < 4) ? pos_in(c0, j) : pos_in(c1, j - 4)));
            }
            next += kWordsPerRecord;
        }
        assert(mono_out.count() == k
               && "decoded popcount disagrees with the record's k -- duplicate or out-of-range position");
#ifndef NDEBUG
        {
            // read_positions carries the header consistency check; running it here too means the dense
            // reader validates d, the flag bit and ascending order without paying for them in Release.
            std::vector<PosT> dbg(k);
            (void)read_positions(buf, off, dbg.data());
        }
#endif
        return next;
    }

    // The query count, which is NOT buf.size() / kWordsPerRecord once continuations exist. Cheap: one
    // pass over the header words, no position decoding. Called once per (gate, peer), not per record.
    [[nodiscard]] static auto count_queries(const VecZ &buf) -> size_t {
        size_t off = 0;
        size_t n = 0;
        while (off < buf.size()) {
            off += words_for(k_at(buf, off));
            ++n;
        }
        assert(off == buf.size() && "a continuation ran past the end of the buffer");
        return n;
    }

    // Encode straight from a dense monomial. This is what the scan calls today: the partner is built
    // densely for the cutoff decision anyway, so the positions have to be recovered from it.
    //
    // The walk is ~k iterations of find_next, which is the same shape of data-dependent loop that made
    // a position-hash routing experiment a regression -- and here it is affordable, which is worth stating with the
    // arithmetic rather than by assertion. That regression cost +8.3 ns/push over 22.0M pushes = +0.18
    // partition-s. The bytes this record saves are projected to take exchange from 43.91 to ~17.1
    // partition-s. A 0.18 tax on a 26.8 saving is 0.7%, so the walk is not worth avoiding here; the
    // place to remove it, if ever, is to take the positions from xor_gen's merge output instead.
    //
    // Stack for anything realistic (k is 4 or 6 for all but 94 rows in 20.9M), heap only for a fully
    // paired term wider than that, so the encoder below stays the single tested implementation.
    static constexpr size_t kStackPositions = 64;
    static auto push_mono(VecZ &buf, const Monomial<NumModes> &mono, int phase) -> size_t {
        const size_t k = mono.count();
        if (k <= kStackPositions) {
            PosT scratch[kStackPositions];
            size_t j = 0;
            for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
                scratch[j++] = static_cast<PosT>(b);
            }
            assert(j == k && "find_first/find_next walk disagrees with count()");
            return push(buf, scratch, k, phase);
        }
        std::vector<PosT> scratch(k);
        size_t j = 0;
        for (size_t b = mono.find_first(); b < mono.size(); b = mono.find_next(b)) {
            scratch[j++] = static_cast<PosT>(b);
        }
        assert(j == k && "find_first/find_next walk disagrees with count()");
        return push(buf, scratch.data(), k, phase);
    }

    // Advance past one query without decoding it. One word read, so a caller that must keep a cursor in
    // step while skipping (a follower already matched by a leader) pays almost nothing for it.
    [[nodiscard]] static auto skip(const VecZ &buf, size_t off) noexcept -> size_t {
        return off + words_for(k_at(buf, off));
    }

    // The fused stream appends the source's pre-cos coefficient as one word after the query's records,
    // so a query plus its value is words_for(k) + 1 words. This is why offsets are words: with a record
    // index there is no way to name where the next query starts.
    static auto push_value(VecZ &buf, double v) -> void { buf.push_back(encode_value_bits(v)); }
    [[nodiscard]] static auto value_at(const VecZ &buf, size_t value_off) -> double {
        return decode_value_bits(buf[value_off]);
    }
};

} // namespace monoprop::detail
