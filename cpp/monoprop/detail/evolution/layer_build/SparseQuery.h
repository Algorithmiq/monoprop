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

// A variable-width query record: one word-aligned record per term holding the term's ascending set-bit
// positions, gap-coded. Record order is preserved everywhere, because it is the floating-point
// accumulation order (Resolve.h mints misses in it).
//
// Header in the low bits of word 0, then the payload LSB-first, both by explicit shift, never punning:
//   [0..1] phase+1 (emit_phase is TERNARY)  [2..6] k, 31 escaping to a following kLongKBits-wide k
//   then [kGwBits] gw  then pos[0] raw at kPosBits, then k-1 gaps of gw bits.
//
// ONE form, no mode field and no per-record argmin. Gap coding is never wider than raw lanes, because
// gw = bit_width(max gap) <= bit_width(kBits - 1) = kPosBits, hence kPosBits + (k-1)*gw <= k*kPosBits. A
// raw kBits mask is narrower only once k*kPosBits > kBits, i.e. from k = 34 at 128 modes, where this
// record costs one word more: measured on 0 of 106,368 captured records (max k = 23, at pauli cutoff
// 12), and that is the price of one code path. Dropping the argmin also dropped the stack array it
// sized -- gap coding emits bits monotonically, so the encoder streams straight into `buf` and no longer
// zeroes 48 B per push.
template <size_t NumModes>
struct SparseQuery {
    using PosT = uint16_t;

    static constexpr size_t kBits = 2 * NumModes;
    static_assert(kBits <= 65535, "a physical bit position and the popcount must both fit a uint16_t");

    //: Bits for one raw position in [0, 2*NumModes); compile-time, so the lane width is free.
    static constexpr size_t kPosBits = static_cast<size_t>(std::bit_width(kBits - 1));

    static constexpr size_t kPhaseBits = 2;
    static constexpr size_t kKBits = 5;
    //: k is a popcount of a kBits bitset, so the escape can never need more than this.
    static constexpr size_t kLongKBits = static_cast<size_t>(std::bit_width(kBits));
    static constexpr size_t kGwBits = 4;
    static constexpr size_t kKEscape = (1U << kKBits) - 1U;
    static constexpr size_t kHeaderBits = kPhaseBits + kKBits + kGwBits;
    static_assert(kPosBits <= (1U << kGwBits) - 1U, "gw <= kPosBits must fit the header's gap-width field");
    static_assert(kHeaderBits + kLongKBits <= 64, "the widest header must be readable from word 0 alone");

    //: A popcount cannot exceed the width, which the old 65535 never said.
    static constexpr size_t kMaxPositions = kBits;

    // ---- bit stream -------------------------------------------------------------------------------

    // Streams into `buf`: one accumulator, flushed when a word fills, in place of an array sized by the
    // worst case of three encodings.
    struct Writer {
        VecZ &buf;
        uint64_t cur = 0;
        size_t nbits = 0; // bits held in cur, always < 64
        size_t words = 0;

        [[gnu::always_inline]] auto put(uint64_t v, size_t width) noexcept -> void {
            if (width == 0) {
                assert(v == 0 && "a zero-width field cannot carry a value");
                return;
            }
            // Assert BEFORE masking: masking alone turns an overflow into a different well-formed record.
            assert((width >= 64 || (v >> width) == 0) && "field value does not fit its width");
            if (width < 64) {
                v &= (uint64_t{1} << width) - 1U;
            }
            cur |= v << nbits;
            if (nbits + width < 64) {
                nbits += width;
                return;
            }
            buf.push_back(static_cast<size_t>(cur));
            ++words;
            // nbits == 0 only at width == 64, where every bit is already in cur; `v >> 64` would be UB.
            cur = (nbits == 0) ? 0 : (v >> (64U - nbits));
            nbits = nbits + width - 64U;
        }

        auto flush() noexcept -> void {
            if (nbits != 0) {
                buf.push_back(static_cast<size_t>(cur));
                ++words;
                cur = 0;
                nbits = 0;
            }
        }
    };

    struct Reader {
        const VecZ &buf;
        size_t base; // word offset of the record start
        size_t nbits = 0;

        [[nodiscard]] auto get(size_t width) noexcept -> uint64_t {
            if (width == 0) {
                return 0;
            }
            const size_t word = nbits >> 6U;
            const size_t off = nbits & 63U;
            uint64_t v = static_cast<uint64_t>(buf[base + word]) >> off;
            if (off + width > 64) {
                v |= static_cast<uint64_t>(buf[base + word + 1]) << (64U - off);
            }
            nbits += width;
            return (width < 64) ? (v & ((uint64_t{1} << width) - 1U)) : v;
        }
    };

    // ---- header -----------------------------------------------------------------------------------

    struct Header {
        int phase = 0;
        size_t k = 0;
        size_t gw = 0;
        size_t bits = 0; // header width, i.e. where the payload begins
    };

    // One word load and a few masks; deliberately does NOT touch the payload -- the cursor walks call it.
    [[nodiscard]] static auto header_at(const VecZ &buf, size_t off) noexcept -> Header {
        const auto w0 = static_cast<uint64_t>(buf[off]);
        Header h;
        h.phase = static_cast<int>(w0 & 0x3U) - 1;
        h.k = static_cast<size_t>((w0 >> kPhaseBits) & kKEscape);
        h.bits = kPhaseBits + kKBits;
        if (h.k == kKEscape) {
            h.k = static_cast<size_t>((w0 >> h.bits) & ((uint64_t{1} << kLongKBits) - 1U));
            h.bits += kLongKBits;
        }
        h.gw = static_cast<size_t>((w0 >> h.bits) & ((uint64_t{1} << kGwBits) - 1U));
        h.bits += kGwBits;
        return h;
    }

    [[nodiscard]] static constexpr auto header_bits_for(size_t k) noexcept -> size_t {
        return kHeaderBits + ((k >= kKEscape) ? kLongKBits : 0U);
    }

    [[nodiscard]] static constexpr auto gap_bits(size_t k, size_t gw) noexcept -> size_t {
        return header_bits_for(k) + ((k == 0) ? 0U : kPosBits + (k - 1U) * gw);
    }
    [[nodiscard]] static constexpr auto words_of(size_t bits) noexcept -> size_t { return (bits + 63U) / 64U; }

    //: The record's word count, from the header alone: k does not determine it, gw does too.
    [[nodiscard]] static constexpr auto words_of_header(const Header &h) noexcept -> size_t {
        return words_of(gap_bits(h.k, h.gw));
    }

    [[nodiscard]] static auto words_at(const VecZ &buf, size_t off) noexcept -> size_t {
        return words_of_header(header_at(buf, off));
    }

    [[nodiscard]] static auto k_at(const VecZ &buf, size_t off) noexcept -> size_t { return header_at(buf, off).k; }
    [[nodiscard]] static auto phase_at(const VecZ &buf, size_t off) noexcept -> int {
        return header_at(buf, off).phase;
    }

    // ---- encode -----------------------------------------------------------------------------------

    //: gw = bit_width(max gap). Folded into the caller's single pass in push(), never a second walk.
    template <typename PosU>
    [[nodiscard]] static auto gap_width(const PosU *pos, size_t k) noexcept -> size_t {
        size_t g = 0;
        for (size_t j = 1; j < k; ++j) {
            const size_t d = static_cast<size_t>(pos[j] - pos[j - 1] - 1U);
            const auto b = static_cast<size_t>(std::bit_width(d));
            g = (b > g) ? b : g;
        }
        return g;
    }

    // Precondition: k STRICTLY ASCENDING physical bit positions in [0, kBits). A violation is silent in
    // release -- gap coding is meaningless without it and an out-of-range position decodes to a different
    // valid-looking monomial. Returns the WORDS written; PosU is generic because the store's position
    // type is narrower than the wire's below 129 modes, and the encoding does not depend on it.
    template <typename PosU>
    static auto push(VecZ &buf, const PosU *pos, size_t k, int phase) -> size_t {
        assert(k <= kMaxPositions && "term has more positions than the record's width admits");
        assert(phase >= -1 && phase <= 1 && "emit_phase is ternary: rotation_sign, or REAL_PARTS entry");
        for (size_t j = 1; j < k; ++j) {
            assert(pos[j] > pos[j - 1] && "positions must be strictly ascending");
        }

        const size_t gw = gap_width(pos, k);
        Writer w{buf};
        w.put(static_cast<uint64_t>(phase + 1), kPhaseBits);
        if (k >= kKEscape) {
            w.put(kKEscape, kKBits);
            w.put(static_cast<uint64_t>(k), kLongKBits);
        }
        else {
            w.put(static_cast<uint64_t>(k), kKBits);
        }
        w.put(static_cast<uint64_t>(gw), kGwBits);
        if (k != 0) {
            w.put(static_cast<uint64_t>(pos[0]), kPosBits);
            for (size_t j = 1; j < k; ++j) {
                w.put(static_cast<uint64_t>(pos[j] - pos[j - 1] - 1U), gw);
            }
        }
        w.flush();
        assert(w.words == words_of(gap_bits(k, gw)) && "encoder wrote a different width than it costed");
        return w.words;
    }

    // ---- decode -----------------------------------------------------------------------------------

    // OutT is generic so the resolve path decodes straight into the store's (narrower) position width.
    template <typename OutT>
    static auto read_positions(const VecZ &buf, size_t off, OutT *out) -> size_t {
        const Header h = header_at(buf, off);
        Reader r{buf, off, h.bits};
        if (h.k != 0) {
            auto prev = static_cast<size_t>(r.get(kPosBits));
            out[0] = static_cast<OutT>(prev);
            for (size_t j = 1; j < h.k; ++j) {
                prev += static_cast<size_t>(r.get(h.gw)) + 1U;
                out[j] = static_cast<OutT>(prev);
            }
        }
        const size_t next = off + words_of_header(h);
        assert(check_header(buf, off, out) && "record header is inconsistent with its own positions");
        return next;
    }

    // Debug-only: every wire field must be checkable from the rest of the record, or it rots.
    template <typename OutT>
    [[nodiscard]] static auto check_header(const VecZ &buf, size_t off, const OutT *pos) -> bool {
        const Header h = header_at(buf, off);
        if (h.phase < -1 || h.phase > 1) {
            return false;
        }
        for (size_t j = 0; j + 1 < h.k; ++j) {
            if (static_cast<size_t>(pos[j]) >= static_cast<size_t>(pos[j + 1])) {
                return false; // positions must arrive strictly ascending
            }
        }
        if (h.k != 0 && static_cast<size_t>(pos[h.k - 1]) >= kBits) {
            return false;
        }
        // gw is the MAXIMUM gap width: too small truncates a gap silently, too large wastes bits.
        size_t g = 0;
        for (size_t j = 1; j < h.k; ++j) {
            const auto b = static_cast<size_t>(std::bit_width(static_cast<size_t>(pos[j] - pos[j - 1] - 1)));
            g = (b > g) ? b : g;
        }
        return g == h.gw;
    }

    // d, recomputed rather than carried: ascending order makes a pair an even position then its successor.
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

    static constexpr size_t kStackPositions = 64;

    static auto read_mono(const VecZ &buf, size_t off, Monomial<NumModes> &mono_out, int &phase_out) -> size_t {
        const Header h = header_at(buf, off);
        phase_out = h.phase;
        mono_out = Monomial<NumModes>{};
        if (h.k <= kStackPositions) {
            PosT scratch[kStackPositions];
            const size_t next = read_positions(buf, off, scratch);
            for (size_t j = 0; j < h.k; ++j) {
                mono_out.set(static_cast<size_t>(scratch[j]));
            }
            assert(mono_out.count() == h.k && "decoded popcount disagrees with the record's k");
            return next;
        }
        std::vector<PosT> scratch(h.k);
        const size_t next = read_positions(buf, off, scratch.data());
        for (size_t j = 0; j < h.k; ++j) {
            mono_out.set(static_cast<size_t>(scratch[j]));
        }
        assert(mono_out.count() == h.k && "decoded popcount disagrees with the record's k");
        return next;
    }

    // Encode from a dense monomial, for callers that hold only a bitset; the emit path merges positions.
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
};

} // namespace monoprop::detail
