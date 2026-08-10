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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>

namespace monoprop {

// std::bitset replacement over contiguous uint64_t words: zero-copy MPI, word-wise hashing,
// portable std::countr_zero scanning, memcpy-safe.
template <size_t NumBits>
class Bitset {
    static_assert(NumBits > 0, "Bitset requires at least 1 bit");

    using word_type = uint64_t;
    static constexpr auto word_width = sizeof(word_type) * 8;

    static constexpr auto kNumWords = (NumBits + word_width - 1) / word_width;
    static constexpr auto kTopBits = NumBits % word_width;
    static constexpr auto kTopMask = kTopBits ? ((word_type{1} << kTopBits) - 1) : ~word_type{0};

    std::array<word_type, kNumWords> words_{};

    constexpr auto sanitize_top() noexcept -> void {
        if constexpr (kTopBits != 0) {
            words_[kNumWords - 1] &= kTopMask;
        }
    }

public:
    constexpr Bitset() noexcept = default;

    constexpr explicit(false) Bitset(uint64_t val) noexcept : words_{val} { sanitize_top(); }

    [[nodiscard]] constexpr auto count() const noexcept -> size_t {
        size_t c = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            c += static_cast<size_t>(std::popcount(words_[i]));
        return c;
    }

    [[nodiscard]] constexpr auto test(size_t pos) const noexcept -> bool {
        return (words_[pos / word_width] >> (pos % word_width)) & 1;
    }

    [[nodiscard]] constexpr auto any() const noexcept -> bool {
        for (size_t i = 0; i < kNumWords; ++i)
            if (words_[i])
                return true;
        return false;
    }

    [[nodiscard]] constexpr auto none() const noexcept -> bool { return !any(); }

    [[nodiscard]] static constexpr auto size() noexcept -> size_t { return NumBits; }

    // popcount(*this & other) without materializing the temporary.
    [[nodiscard]] constexpr auto count_and(const Bitset &o) const noexcept -> size_t {
        size_t c = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            c += static_cast<size_t>(std::popcount(words_[i] & o.words_[i]));
        return c;
    }

    [[nodiscard]] constexpr auto parity_and(const Bitset &o) const noexcept -> bool {
        word_type parity_word = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            parity_word ^= words_[i] & o.words_[i];
        return (std::popcount(parity_word) & 1U) != 0;
    }

    // Every quantity a caller building `*this ^ gen` typically also needs alongside it: the XORed
    // result, popcount(*this & gen) (overlap), and popcount(result). Composing this from operator^
    // and count_and() costs two full passes over the words; once the width is a runtime value
    // rather than a compile-time NumBits (see the NumModes-NTTP-removal plan), each pass also pays
    // its own loop prologue/tail, so the separate-ops cost keeps growing where this stays one pass.
    // Kept alongside the existing composable ops -- a cold path that only needs one of the three
    // should keep using them. result needs no sanitize_top(): XOR of two already-sanitized operands
    // never sets a bit above NumBits.
    struct FusedXor {
        Bitset result;
        size_t overlap;
        size_t result_count;
    };

    [[nodiscard]] constexpr auto fused_xor(const Bitset &gen) const noexcept -> FusedXor {
        Bitset result;
        size_t overlap = 0;
        size_t result_count = 0;
        for (size_t i = 0; i < kNumWords; ++i) {
            const word_type n = words_[i] ^ gen.words_[i];
            result.words_[i] = n;
            overlap += static_cast<size_t>(std::popcount(words_[i] & gen.words_[i]));
            result_count += static_cast<size_t>(std::popcount(n));
        }
        return {result, overlap, result_count};
    }

    constexpr auto set(size_t pos) noexcept -> Bitset & {
        words_[pos / word_width] |= uint64_t(1) << (pos % word_width);
        return *this;
    }

    constexpr auto operator&=(const Bitset &rhs) noexcept -> Bitset & {
        for (auto i = 0uz; i < kNumWords; ++i)
            words_[i] &= rhs.words_[i];
        return *this;
    }

    constexpr auto operator|=(const Bitset &rhs) noexcept -> Bitset & {
        for (auto i = 0uz; i < kNumWords; ++i)
            words_[i] |= rhs.words_[i];
        return *this;
    }

    constexpr auto operator^=(const Bitset &rhs) noexcept -> Bitset & {
        for (auto i = 0uz; i < kNumWords; ++i)
            words_[i] ^= rhs.words_[i];
        return *this;
    }

    [[nodiscard]] constexpr auto operator~() const noexcept -> Bitset {
        Bitset r = *this;
        for (auto i = 0uz; i < kNumWords; ++i)
            r.words_[i] = ~r.words_[i];
        r.sanitize_top();
        return r;
    }

    [[nodiscard]] friend constexpr auto operator&(const Bitset &lhs, const Bitset &rhs) noexcept -> Bitset {
        Bitset r = lhs;
        r &= rhs;
        return r;
    }

    [[nodiscard]] friend constexpr auto operator|(const Bitset &lhs, const Bitset &rhs) noexcept -> Bitset {
        Bitset r = lhs;
        r |= rhs;
        return r;
    }

    [[nodiscard]] friend constexpr auto operator^(const Bitset &lhs, const Bitset &rhs) noexcept -> Bitset {
        Bitset r = lhs;
        r ^= rhs;
        return r;
    }

    constexpr auto operator>>=(size_t pos) noexcept -> Bitset & {
        if (pos >= NumBits) {
            words_.fill(0);
            return *this;
        }
        if constexpr (kNumWords == 1) {
            words_[0] >>= pos;
        }
        else {
            const size_t word_shift = pos / word_width;
            const size_t limit = kNumWords - word_shift;
            if (const size_t bit_shift = pos % word_width; bit_shift == 0) {
                for (size_t i = 0; i < limit; ++i)
                    words_[i] = words_[i + word_shift];
            }
            else {
                const size_t inv_shift = word_width - bit_shift;
                for (size_t i = 0; i + 1 < limit; ++i) {
                    words_[i] = (words_[i + word_shift] >> bit_shift) | (words_[i + word_shift + 1] << inv_shift);
                }
                words_[limit - 1] = words_[kNumWords - 1] >> bit_shift;
            }
            for (size_t i = limit; i < kNumWords; ++i)
                words_[i] = 0;
        }
        return *this;
    }

    [[nodiscard]] constexpr auto operator>>(size_t pos) const noexcept -> Bitset {
        Bitset r = *this;
        r >>= pos;
        return r;
    }

    [[nodiscard]] constexpr auto operator==(const Bitset &o) const noexcept -> bool {
        for (size_t i = 0; i < kNumWords; ++i)
            if (words_[i] != o.words_[i])
                return false;
        return true;
    }

    [[nodiscard]] static constexpr auto num_words() noexcept -> size_t { return kNumWords; }
    [[nodiscard]] constexpr auto data() const noexcept -> const uint64_t * { return words_.data(); }
    [[nodiscard]] constexpr auto data() noexcept -> uint64_t * { return words_.data(); }
    [[nodiscard]] constexpr auto word(size_t i) const noexcept -> uint64_t { return words_[i]; }

    [[nodiscard]] constexpr auto find_first() const noexcept -> size_t { // NumBits if none
        for (size_t i = 0; i < kNumWords; ++i) {
            if (words_[i])
                return (i * word_width) + static_cast<size_t>(std::countr_zero(words_[i]));
        }
        return NumBits;
    }

    [[nodiscard]] constexpr auto find_next(size_t pos) const noexcept -> size_t { // NumBits if none
        if (++pos >= NumBits)
            return NumBits;
        if constexpr (kNumWords == 1) {
            if (const uint64_t w = words_[0] >> pos; w)
                return pos + static_cast<size_t>(std::countr_zero(w));
            return NumBits;
        }
        else {
            size_t wi = pos / word_width;
            if (const uint64_t w = words_[wi] >> (pos % word_width); w)
                return pos + static_cast<size_t>(std::countr_zero(w));
            for (++wi; wi < kNumWords; ++wi) {
                if (words_[wi])
                    return (wi * word_width) + static_cast<size_t>(std::countr_zero(words_[wi]));
            }
            return NumBits;
        }
    }

    // Stream output MSB→LSB (std::bitset convention).
    friend auto operator<<(std::ostream &os, const Bitset &bs) -> std::ostream & {
        for (size_t i = NumBits; i-- > 0;)
            os << (bs.test(i) ? '1' : '0');
        return os;
    }
};
} // namespace monoprop

template <typename T>
struct SplitmixHash;

template <size_t NumBits>
struct SplitmixHash<monoprop::Bitset<NumBits>> {
    static constexpr auto mix(uint64_t x) noexcept -> uint64_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    auto operator()(const monoprop::Bitset<NumBits> &bs) const noexcept -> size_t {
        constexpr size_t W = monoprop::Bitset<NumBits>::num_words();
        if constexpr (W == 1) {
            return static_cast<size_t>(mix(bs.word(0)));
        }
        else {
            uint64_t h = 0;
            for (size_t i = 0; i < W; ++i) {
                h ^= mix(bs.word(i) + static_cast<uint64_t>(i));
            }
            return static_cast<size_t>(h);
        }
    }
};

namespace std {
template <size_t NumBits>
struct hash<monoprop::Bitset<NumBits>> {
    auto operator()(const monoprop::Bitset<NumBits> &bs) const noexcept -> size_t {
        return SplitmixHash<monoprop::Bitset<NumBits>>{}(bs);
    }
};
} // namespace std
