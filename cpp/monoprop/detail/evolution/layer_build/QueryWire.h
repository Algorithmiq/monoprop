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
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"

namespace monoprop::detail {

// One term's wire record for a cross-rank query: its ascending set-bit positions, gap-coded, plus a
// phase. It replaces a fixed dense stride, which would spend one word per 64 modes no matter how few
// bits a term actually sets.
// Layout, LSB-first in word 0 onward: [2b phase][5b k, 31 escapes to a wider k][4b gap width gw]
// [kPosBits first position][k-1 gaps of gw bits].

// `fused` records carry a trailing value word after the positions; `plain` ones do not. A named enum,
// not a bare bool, so a form argument cannot read as plausibly correct either way.
enum class QueryForm { Plain, Fused };

template <size_t NumModes>
struct QueryWire {
    using PosT = uint16_t;

    static constexpr size_t kBits = 2 * NumModes;
    static_assert(kBits <= 65535, "a physical bit position and the popcount must both fit a uint16_t");

    // Bits for one raw position in [0, 2*NumModes); compile-time, so the lane width is free.
    static constexpr size_t kPosBits = static_cast<size_t>(std::bit_width(kBits - 1));

    static constexpr size_t kPhaseBits = 2;
    static constexpr size_t kKBits = 5;
    // k is a popcount of a kBits bitset, so the escape field never needs more bits than this.
    static constexpr size_t kLongKBits = static_cast<size_t>(std::bit_width(kBits));
    static constexpr size_t kGwBits = 4;
    static constexpr size_t kKEscape = (1U << kKBits) - 1U;
    static constexpr size_t kHeaderBits = kPhaseBits + kKBits + kGwBits;
    static_assert(kPosBits <= (1U << kGwBits) - 1U, "gw <= kPosBits must fit the header's gap-width field");
    static_assert(kHeaderBits + kLongKBits <= 64, "the widest header must be readable from word 0 alone");

    static constexpr size_t kMaxPositions = kBits;

    // Reserve hint only, for a caller batching many records into one flat position buffer.
    static constexpr size_t kReservePositionsPerQuery = 6;

    // Bit-packs fields into `buf`, one word at a time.
    struct Writer {
        VecZ &buf;
        uint64_t cur = 0;
        size_t nbits = 0; // bits held in cur, always less than 64
        size_t words = 0;

        [[gnu::always_inline]] auto put(uint64_t v, size_t width) noexcept -> void {
            if (width == 0) {
                assert(v == 0 && "a zero-width field cannot carry a value");
                return;
            }
            assert(width < 64 && "no field in this record reaches a full word");
            // Assert before masking: masking alone turns an overflow into a different well-formed record.
            assert((v >> width) == 0 && "field value does not fit its width");
            v &= (uint64_t{1} << width) - 1U;
            cur |= v << nbits;
            if (nbits + width < 64) {
                nbits += width;
                return;
            }
            buf.push_back(static_cast<size_t>(cur));
            ++words;
            cur = v >> (64U - nbits);
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

    // Unpacks fields out of `buf`, starting at word `base`.
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

    struct Header {
        int phase = 0;
        size_t k = 0;
        size_t gw = 0;
        size_t bits = 0; // header width, i.e. where the payload begins
    };

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

    // The record's word count, derived from the header alone: k does not determine it since gw varies too.
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

    // gw = bit_width(max gap), folded into push()'s own pass over the positions.
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

    // Precondition: k strictly ascending positions in [0, kBits). A violation is silent in release and
    // decodes a different, still valid-looking monomial. Returns the words written; PosU is generic
    // because the store's position type is narrower than the wire's below 129 modes.
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

    // Decodes one record's positions; returns the offset just past them. OutT is generic so the resolve
    // path decodes straight into the store's (narrower) position width.
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

    // Debug-only: every wire field must be checkable from the rest of the record, or it rots unnoticed.
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
        // gw is the maximum gap width: too small truncates a gap silently, too large wastes bits.
        size_t g = 0;
        for (size_t j = 1; j < h.k; ++j) {
            const auto b = static_cast<size_t>(std::bit_width(static_cast<size_t>(pos[j] - pos[j - 1] - 1)));
            g = (b > g) ? b : g;
        }
        return g == h.gw;
    }

    // d, recomputed rather than carried: in ascending order a pair is an even position then its successor.
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

    // Offset of the next record in a stream; `off` always names the start of one.
    [[nodiscard]] static auto next_off(const VecZ &buf, QueryForm form, size_t off) -> size_t {
        return off + words_at(buf, off) + (form == QueryForm::Fused ? 1U : 0U);
    }

    // Decodes one record's positions and phase from a query stream, whose form says whether a value
    // word follows; returns the offset of the next record.
    template <typename OutT>
    static auto read_query(const VecZ &buf, QueryForm form, size_t off, OutT *out, int &phase_out) -> size_t {
        phase_out = phase_at(buf, off);
        return read_positions(buf, off, out) + (form == QueryForm::Fused ? 1U : 0U);
    }

    // The fused value word, a bit_cast that follows a record's positions.
    [[nodiscard]] static auto value_at(const VecZ &buf, [[maybe_unused]] QueryForm form, size_t off) -> double {
        assert(form == QueryForm::Fused && "there is no value word in a plain query stream");
        return decode_value(buf[off + words_at(buf, off)]);
    }

    static auto push_value(VecZ &buf, double v) -> void { buf.push_back(encode_value(v)); }

    // Number of records in the stream: widths vary, so this walks rather than divides.
    [[nodiscard]] static auto count_queries(const VecZ &buf, QueryForm form) -> size_t {
        size_t off = 0;
        size_t n = 0;
        while (off < buf.size()) {
            off = next_off(buf, form, off);
            ++n;
        }
        assert(off == buf.size() && "a compact query stream ran past the end of the buffer");
        return n;
    }

    // Interleaves a plain query stream with its parallel value array into one fused stream.
    static auto build_fused(const VecZ &queries, const std::vector<double> &vals, VecZ &out) -> void {
        out.clear();
        out.reserve(queries.size() + vals.size());
        size_t off = 0;
        size_t i = 0;
        while (off < queries.size()) {
            const size_t n = words_at(queries, off);
            out.insert(out.end(),
                       queries.begin() + static_cast<std::ptrdiff_t>(off),
                       queries.begin() + static_cast<std::ptrdiff_t>(off + n));
            assert(i < vals.size() && "fused build needs exactly one value per query");
            out.push_back(encode_value(vals[i]));
            off += n;
            ++i;
        }
        assert(i == vals.size() && "fused build needs exactly one value per query");
    }

    // Copies the record at src_off (and its value word, if fused) to dst_off; returns the words moved.
    static auto move_query(VecZ &buf, QueryForm form, size_t src_off, size_t dst_off) -> size_t {
        const size_t n = words_at(buf, src_off) + (form == QueryForm::Fused ? 1U : 0U);
        if (src_off != dst_off) {
            assert(dst_off < src_off && "compaction only ever moves a query earlier");
            std::copy(buf.begin() + static_cast<std::ptrdiff_t>(src_off),
                      buf.begin() + static_cast<std::ptrdiff_t>(src_off + n),
                      buf.begin() + static_cast<std::ptrdiff_t>(dst_off));
        }
        return n;
    }
};

} // namespace monoprop::detail
