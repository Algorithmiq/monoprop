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
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/layer_build/Common.h"

namespace monoprop::detail {

/*! @brief `Fused` records carry a trailing value word after the positions; `Plain` ones do not. */
enum class QueryForm { Plain, Fused };

/*! @brief One term's wire record in the gate exchange: the partner's ascending set-bit positions,
 *  gap-coded, plus the sender's emit phase and its rotation bit.
 *
 *  The record is what a term ν sends to the owner of its partner μ = ν ⊕ G (Engine.h has the
 *  protocol): the key is μ's positions, `phase` is φ_ν = A::emit_phase(ν, G), and `rot` is the
 *  sender's emission predicate E(ν) -- whether ν alone would rotate the pair. A `Fused` record
 *  carries one trailing word with ν's pre-cos coefficient (`push_value`), bit-cast so it arrives
 *  exactly.
 *
 *  It replaces a fixed dense stride, which would spend one word per 64 modes however few bits a
 *  term actually sets. Fields pack LSB-first from word 0: a 2-bit phase biased to unsigned, the
 *  1-bit @p rot, a 5-bit popcount @p k whose all-ones value escapes to a wider field, a 4-bit gap
 *  width @p gw, then the first position in `kPosBits` bits and the remaining k-1 positions as gaps
 *  of @p gw bits to their predecessor. Both k and gw are per-record, so a two-bit term and a
 *  full-support one each cost what they are.
 *
 *  At `NumModes = 128` (`kBits = 256`, so `kPosBits = 8`) a term at positions {3, 7, 8, 40} with
 *  phase +1 and rot set has k = 4 and gaps {3, 0, 31}, hence `gw = bit_width(31) = 5`:
 *
 *  @code
 *  bits  0..1   phase   2   +1, biased by 1
 *  bit   2      rot     1
 *  bits  3..7   k       4   below kKEscape, so no wide-k field follows
 *  bits  8..11  gw      5
 *  bits 12..19  first   3
 *  bits 20..24  gap     3   7 - 3 - 1
 *  bits 25..29  gap     0   8 - 7 - 1
 *  bits 30..34  gap     31  40 - 8 - 1
 *  @endcode
 *
 *  35 bits, so one word, against the four a 256-bit dense stride spends on the same term.
 */
template <size_t NumModes>
struct QueryWire {
    using PosT = uint16_t;
    using WireView = std::span<const size_t>; //!< a serialized record stream, read-only

    static constexpr size_t kBits = 2 * NumModes;
    static_assert(kBits <= 65535, "a physical bit position and the popcount must both fit a uint16_t");

    //! Bits for one raw position in [0, 2*NumModes); compile-time, so the lane width is free.
    static constexpr size_t kPosBits = static_cast<size_t>(std::bit_width(kBits - 1));

    static constexpr size_t kPhaseBits = 2;
    static constexpr size_t kRotBits = 1;
    static constexpr size_t kKBits = 5;
    //! k is a popcount of a kBits bitset, so the escape field never needs more bits than this.
    static constexpr size_t kLongKBits = static_cast<size_t>(std::bit_width(kBits));
    static constexpr size_t kGwBits = 4;
    static constexpr size_t kKEscape = (1U << kKBits) - 1U;
    static constexpr size_t kHeaderBits = kPhaseBits + kRotBits + kKBits + kGwBits;
    static_assert(kPosBits <= (1U << kGwBits) - 1U, "gw <= kPosBits must fit the header's gap-width field");
    static_assert(kHeaderBits + kLongKBits <= 64, "the widest header must be readable from word 0 alone");

    static constexpr size_t kMaxPositions = kBits;

    //! Reserve hint only, for a caller batching many records into one flat position buffer.
    static constexpr size_t kReservePositionsPerQuery = 6;

    /*! @brief What one decode call read and consumed. */
    struct Decoded {
        size_t next; //!< word offset just past what this call consumed
        int phase;   //!< the record's ternary phase
        bool rot;    //!< the sender's rotation bit
    };

    /*! @brief Bit-packs variable-width fields into `buf`, one 64-bit word at a time. */
    struct Writer {
        VecZ &buf;
        uint64_t cur = 0;
        size_t nbits = 0; //!< bits held in cur, always less than 64
        size_t words = 0;

        [[gnu::always_inline]] auto put(uint64_t v, size_t width) noexcept -> void {
            if (width == 0) {
                return;
            }
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

    /*! @brief Unpacks variable-width fields out of `buf`, starting at word `base`. */
    struct Reader {
        WireView buf;
        size_t base; //!< word offset of the record start
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

    /*! @brief A record's decoded header fields. */
    struct Header {
        int phase = 0;
        bool rot = false;
        size_t k = 0;
        size_t gw = 0;
        size_t bits = 0; //!< header width, i.e. where the payload begins
    };

    [[nodiscard]] static auto header_at(WireView buf, size_t off) noexcept -> Header {
        const auto w0 = static_cast<uint64_t>(buf[off]);
        Header h;
        h.phase = static_cast<int>(w0 & 0x3U) - 1;
        h.rot = ((w0 >> kPhaseBits) & 1U) != 0;
        h.k = static_cast<size_t>((w0 >> (kPhaseBits + kRotBits)) & kKEscape);
        h.bits = kPhaseBits + kRotBits + kKBits;
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

    //! The record's word count, derived from the header alone: k does not determine it since gw varies too.
    [[nodiscard]] static constexpr auto words_of_header(const Header &h) noexcept -> size_t {
        return words_of(gap_bits(h.k, h.gw));
    }

    [[nodiscard]] static auto words_at(WireView buf, size_t off) noexcept -> size_t {
        return words_of_header(header_at(buf, off));
    }

    [[nodiscard]] static auto k_at(WireView buf, size_t off) noexcept -> size_t { return header_at(buf, off).k; }
    [[nodiscard]] static auto phase_at(WireView buf, size_t off) noexcept -> int { return header_at(buf, off).phase; }
    [[nodiscard]] static auto rot_at(WireView buf, size_t off) noexcept -> bool { return header_at(buf, off).rot; }

    //! gw = bit_width(max gap), folded into push()'s own pass over the positions.
    template <std::ranges::contiguous_range Pos>
    [[nodiscard]] static auto gap_width(const Pos &pos) noexcept -> size_t {
        const size_t k = std::ranges::size(pos);
        size_t g = 0;
        for (size_t j = 1; j < k; ++j) {
            const size_t d = static_cast<size_t>(pos[j] - pos[j - 1] - 1U);
            const auto b = static_cast<size_t>(std::bit_width(d));
            g = (b > g) ? b : g;
        }
        return g;
    }

    /*! @brief Appends one record for `pos`, `phase` and `rot`, and returns the words written.
     *
     *  `pos` must be strictly ascending with every position below `kBits`; a violation encodes a
     *  different, still well-formed record rather than failing. The element type is deduced because
     *  the store's position width is narrower than the wire's below 129 modes.
     */
    template <std::ranges::contiguous_range Pos>
    static auto push(VecZ &buf, const Pos &pos, int phase, bool rot = false) -> size_t {
        const size_t k = std::ranges::size(pos);
        assert(k <= kMaxPositions && "term has more positions than the record's width admits");
        assert(phase >= -1 && phase <= 1 && "emit_phase is ternary: rotation_sign, or REAL_PARTS entry");

        const size_t gw = gap_width(pos);
        Writer w{buf};
        w.put(static_cast<uint64_t>(phase + 1), kPhaseBits);
        w.put(static_cast<uint64_t>(rot), kRotBits);
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
        return w.words;
    }

    /*! @brief Decodes the record at `off` into `out`, whose element type is deduced so the resolve
     *  path can decode straight into the store's narrower position width.
     *
     *  `Decoded::next` names the word just past the positions, excluding any value word.
     */
    template <std::ranges::contiguous_range Out>
    static auto read_positions(WireView buf, size_t off, Out &&out) -> Decoded {
        using OutT = std::ranges::range_value_t<Out>;
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
        return {off + words_of_header(h), h.phase, h.rot};
    }

    //! d, recomputed rather than carried: in ascending order a pair is an even position then its successor.
    template <std::ranges::contiguous_range Pos>
    [[nodiscard]] static auto pair_count(const Pos &pos) noexcept -> size_t {
        const size_t k = std::ranges::size(pos);
        size_t d = 0;
        for (size_t j = 0; j + 1 < k; ++j) {
            if ((pos[j] % 2 == 0) && (pos[j + 1] == pos[j] + 1)) {
                ++d;
            }
        }
        return d;
    }

    //! Offset of the next record in a stream; `off` always names the start of one.
    [[nodiscard]] static auto next_off(WireView buf, QueryForm form, size_t off) -> size_t {
        return off + words_at(buf, off) + (form == QueryForm::Fused ? 1U : 0U);
    }

    /*! @brief Decodes one record from a query stream, whose `form` says whether a value word
     *  follows. `Decoded::next` names the following record.
     */
    template <std::ranges::contiguous_range Out>
    static auto read_query(WireView buf, QueryForm form, size_t off, Out &&out) -> Decoded {
        Decoded d = read_positions(buf, off, std::forward<Out>(out));
        d.next += (form == QueryForm::Fused ? 1U : 0U);
        return d;
    }

    //! The fused value word, a bit_cast that follows a record's positions.
    [[nodiscard]] static auto value_at(WireView buf, size_t off) -> double {
        return decode_value(buf[off + words_at(buf, off)]);
    }

    static auto push_value(VecZ &buf, double v) -> void { buf.push_back(encode_value(v)); }

    //! Number of records in the stream: widths vary, so this walks rather than divides.
    [[nodiscard]] static auto count_queries(WireView buf, QueryForm form) -> size_t {
        size_t off = 0;
        size_t n = 0;
        while (off < buf.size()) {
            off = next_off(buf, form, off);
            ++n;
        }
        return n;
    }
};

} // namespace monoprop::detail
