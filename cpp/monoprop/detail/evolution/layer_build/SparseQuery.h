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
// positions. push() picks the argmin of three closed forms -- FIXED (k lanes of kPosBits), GAP (first
// position raw, then k-1 gaps of gw bits) and BITMAP (a raw kBits mask) -- so the record is never larger
// than any of the three, which is also what bounds Writer's kMaxWords. Record order is preserved
// everywhere, because it is the floating-point accumulation order (Resolve.h mints misses in it).
//
// Header in the low bits of word 0, then the payload LSB-first, both by explicit shift, never punning:
//   [0..1] mode (FIXED/GAP/BITMAP)  [2..3] phase+1 (emit_phase is TERNARY)  [4..9] k, 63 escaping to a
//   following 16-bit k  then [4 bits] gw, in GAP mode only  then payload.
// Header width is a design variable, not overhead: it decides which side of a 64-bit line a term falls
// on.
template <size_t NumModes>
struct SparseQuery {
    using PosT = uint16_t;

    static constexpr size_t kBits = 2 * NumModes;
    static_assert(kBits <= 65535, "a physical bit position and the popcount must both fit a uint16_t");

    //: Bits for one raw position in [0, 2*NumModes); compile-time, so the lane width is free.
    static constexpr size_t kPosBits = static_cast<size_t>(std::bit_width(kBits - 1));

    static constexpr uint64_t kModeFixed = 0;
    static constexpr uint64_t kModeGap = 1;
    static constexpr uint64_t kModeBitmap = 2;

    static constexpr size_t kModeBits = 2;
    static constexpr size_t kPhaseBits = 2;
    static constexpr size_t kKBits = 6;
    static constexpr size_t kLongKBits = 16;
    static constexpr size_t kGwBits = 4;
    static constexpr size_t kKEscape = (1U << kKBits) - 1U;
    static constexpr size_t kBaseHeaderBits = kModeBits + kPhaseBits + kKBits;
    static_assert(kPosBits <= (1U << kGwBits) - 1U, "gw <= kPosBits must fit the header's 4-bit gap-width field");

    static constexpr size_t kMaxPositions = 65535;
    //: Words in a full occupancy mask. ceil, NOT kBits/64: a C++ caller may pick any width (LiH: 24).
    static constexpr size_t kMaskWords = (kBits + 63U) / 64U;
    //: BITMAP bounds every mode from above, because the encoder takes the minimum of the three.
    static constexpr size_t kMaxWords = (kBits + kBaseHeaderBits + kLongKBits + kGwBits + 63U) / 64U + 1U;

    // ---- bit stream -------------------------------------------------------------------------------

    struct Writer {
        uint64_t w[kMaxWords] = {};
        size_t nbits = 0;

        constexpr auto put(uint64_t v, size_t width) noexcept -> void {
            if (width == 0) {
                assert(v == 0 && "a zero-width field cannot carry a value");
                return;
            }
            // Assert BEFORE masking: masking alone turns an overflow into a different well-formed record.
            assert((width >= 64 || (v >> width) == 0) && "field value does not fit its width");
            if (width < 64) {
                v &= (uint64_t{1} << width) - 1U;
            }
            const size_t word = nbits >> 6U;
            const size_t off = nbits & 63U;
            assert(word < kMaxWords && "record overran its worst-case word bound");
            w[word] |= v << off;
            // off > 0 is implied here (width <= 64), so the shift is in [1, 63]; `v >> 64` would be UB.
            if (off + width > 64) {
                w[word + 1] |= v >> (64U - off);
            }
            nbits += width;
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
        uint64_t mode = 0;
        int phase = 0;
        size_t k = 0;
        size_t gw = 0;
        size_t bits = 0; // header width, i.e. where the payload begins
    };

    // One word load and a few masks; deliberately does NOT touch the payload -- the cursor walks call it.
    [[nodiscard]] static auto header_at(const VecZ &buf, size_t off) noexcept -> Header {
        const auto w0 = static_cast<uint64_t>(buf[off]);
        Header h;
        h.mode = w0 & 0x3U;
        h.phase = static_cast<int>((w0 >> kModeBits) & 0x3U) - 1;
        h.k = static_cast<size_t>((w0 >> (kModeBits + kPhaseBits)) & kKEscape);
        h.bits = kBaseHeaderBits;
        if (h.k == kKEscape) {
            h.k = static_cast<size_t>((w0 >> h.bits) & 0xFFFFU);
            h.bits += kLongKBits;
        }
        if (h.mode == kModeGap) {
            h.gw = static_cast<size_t>((w0 >> h.bits) & 0xFU);
            h.bits += kGwBits;
        }
        return h;
    }

    [[nodiscard]] static constexpr auto header_bits_for(size_t k, uint64_t mode) noexcept -> size_t {
        return kBaseHeaderBits + ((k >= kKEscape) ? kLongKBits : 0U) + ((mode == kModeGap) ? kGwBits : 0U);
    }

    [[nodiscard]] static constexpr auto fixed_bits(size_t k) noexcept -> size_t {
        return header_bits_for(k, kModeFixed) + k * kPosBits;
    }
    [[nodiscard]] static constexpr auto gap_bits(size_t k, size_t gw) noexcept -> size_t {
        return header_bits_for(k, kModeGap) + ((k == 0) ? 0U : kPosBits + (k - 1U) * gw);
    }
    [[nodiscard]] static constexpr auto bitmap_bits(size_t k) noexcept -> size_t {
        return header_bits_for(k, kModeBitmap) + kBits;
    }
    [[nodiscard]] static constexpr auto words_of(size_t bits) noexcept -> size_t { return (bits + 63U) / 64U; }

    //: The record's word count, from the header alone: k does not determine it, mode and gw do too.
    [[nodiscard]] static constexpr auto words_of_header(const Header &h) noexcept -> size_t {
        switch (h.mode) {
            case kModeGap:
                return words_of(gap_bits(h.k, h.gw));
            case kModeBitmap:
                return words_of(bitmap_bits(h.k));
            default:
                return words_of(fixed_bits(h.k));
        }
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
        assert(k <= kMaxPositions && "term has more positions than the record's k field can hold");
        assert(phase >= -1 && phase <= 1 && "emit_phase is ternary: rotation_sign, or REAL_PARTS entry");
        for (size_t j = 1; j < k; ++j) {
            assert(pos[j] > pos[j - 1] && "positions must be strictly ascending");
        }

        const size_t gw = gap_width(pos, k);
        const size_t wf = words_of(fixed_bits(k));
        const size_t wg = words_of(gap_bits(k, gw));
        const size_t wb = words_of(bitmap_bits(k));

        uint64_t mode = kModeFixed;
        size_t want = wf;
        if (wg < want) {
            mode = kModeGap;
            want = wg;
        }
        if (wb < want) {
            mode = kModeBitmap;
            want = wb;
        }

        Writer w;
        w.put(mode, kModeBits);
        w.put(static_cast<uint64_t>(phase + 1), kPhaseBits);
        if (k >= kKEscape) {
            w.put(kKEscape, kKBits);
            w.put(static_cast<uint64_t>(k), kLongKBits);
        }
        else {
            w.put(static_cast<uint64_t>(k), kKBits);
        }

        if (mode == kModeGap) {
            w.put(static_cast<uint64_t>(gw), kGwBits);
            if (k != 0) {
                w.put(static_cast<uint64_t>(pos[0]), kPosBits);
                for (size_t j = 1; j < k; ++j) {
                    w.put(static_cast<uint64_t>(pos[j] - pos[j - 1] - 1U), gw);
                }
            }
        }
        else if (mode == kModeBitmap) {
            // The trailing partial word is NOT optional: at kBits 24 a `kBits / 64` loop writes nothing
            // while bitmap_bits() still charges 24 bits, and the decoder then recovers no positions.
            uint64_t mask[kMaskWords] = {};
            for (size_t j = 0; j < k; ++j) {
                const auto p = static_cast<size_t>(pos[j]);
                mask[p >> 6U] |= uint64_t{1} << (p & 63U);
            }
            for (size_t done = 0, i = 0; done < kBits; ++i) {
                const size_t chunk = (kBits - done < 64U) ? (kBits - done) : 64U;
                w.put(mask[i], chunk);
                done += chunk;
            }
        }
        else {
            for (size_t j = 0; j < k; ++j) {
                w.put(static_cast<uint64_t>(pos[j]), kPosBits);
            }
        }

        assert(words_of(w.nbits) == want && "encoder wrote a different width than it costed");
        for (size_t i = 0; i < want; ++i) {
            buf.push_back(static_cast<size_t>(w.w[i]));
        }
        return want;
    }

    // ---- decode -----------------------------------------------------------------------------------

    // OutT is generic so the resolve path decodes straight into the store's (narrower) position width.
    template <typename OutT>
    static auto read_positions(const VecZ &buf, size_t off, OutT *out) -> size_t {
        const Header h = header_at(buf, off);
        Reader r{buf, off, h.bits};
        if (h.mode == kModeGap) {
            if (h.k != 0) {
                auto prev = static_cast<size_t>(r.get(kPosBits));
                out[0] = static_cast<OutT>(prev);
                for (size_t j = 1; j < h.k; ++j) {
                    prev += static_cast<size_t>(r.get(h.gw)) + 1U;
                    out[j] = static_cast<OutT>(prev);
                }
            }
        }
        else if (h.mode == kModeBitmap) {
            // Symmetric with the encoder, trailing partial word included -- see the note there.
            size_t n = 0;
            for (size_t done = 0, i = 0; done < kBits; ++i) {
                const size_t chunk = (kBits - done < 64U) ? (kBits - done) : 64U;
                uint64_t word = r.get(chunk);
                while (word != 0) {
                    const auto b = static_cast<size_t>(std::countr_zero(word));
                    out[n++] = static_cast<OutT>(done + b);
                    word &= word - 1U;
                }
                done += chunk;
            }
            assert(n == h.k && "bitmap popcount disagrees with the record's k");
        }
        else {
            for (size_t j = 0; j < h.k; ++j) {
                out[j] = static_cast<OutT>(r.get(kPosBits));
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
        if (h.mode > kModeBitmap) {
            return false; // an unknown mode is a format the reader does not understand
        }
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
        if (h.mode == kModeGap && h.k > 1) {
            // gw is the MAXIMUM gap width: too small truncates a gap silently, too large wastes bits.
            size_t g = 0;
            for (size_t j = 1; j < h.k; ++j) {
                const auto b = static_cast<size_t>(std::bit_width(static_cast<size_t>(pos[j] - pos[j - 1] - 1)));
                g = (b > g) ? b : g;
            }
            if (g != h.gw) {
                return false;
            }
        }
        return true;
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

    // Encode from a dense monomial, which is what the scan holds: the partner is built densely anyway.
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
