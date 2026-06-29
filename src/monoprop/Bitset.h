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

/**
 * @brief Fixed-size bitset with direct word access for MPI transmission.
 *
 * Drop-in replacement for std::bitset<NumModes> that stores data as contiguous
 * uint64_t words. This enables:
 *   - Zero-copy MPI send/recv via data() pointer
 *   - O(1) hash computation on raw words
 *   - Portable bit scanning via std::countr_zero
 *   - Trivially copyable (memcpy-safe)
 *
 * @tparam NumModes Total number of bits in the bitset.
 */
template <size_t NumModes>
class Bitset {
    static_assert(NumModes > 0, "Bitset requires at least 1 bit");

    using word_type = uint64_t;
    static constexpr auto word_width = sizeof(word_type) * 8;

    static constexpr auto kNumWords = (NumModes + word_width - 1) / word_width;
    static constexpr auto kTopBits = NumModes % word_width;
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

    // --- Query ---
    [[nodiscard]] constexpr auto count() const noexcept -> size_t {
        size_t c = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            c += static_cast<size_t>(std::popcount(words_[i]));
        return c;
    }

    [[nodiscard]] constexpr auto parity() const noexcept -> bool {
        word_type parity_word = 0;
        for (size_t i = 0; i < kNumWords; ++i)
            parity_word ^= words_[i];
        return (std::popcount(parity_word) & 1U) != 0;
    }

    [[nodiscard]] constexpr auto test(size_t pos) const noexcept -> bool {
        return (words_[pos / word_width] >> (pos % word_width)) & 1;
    }

    [[nodiscard]] constexpr auto operator[](size_t pos) const noexcept -> bool { return test(pos); }

    [[nodiscard]] constexpr auto any() const noexcept -> bool {
        for (size_t i = 0; i < kNumWords; ++i)
            if (words_[i])
                return true;
        return false;
    }

    [[nodiscard]] constexpr auto none() const noexcept -> bool { return !any(); }

    [[nodiscard]] static constexpr auto size() noexcept -> size_t { return NumModes; }

    /// Count of bits set in (this & other) without creating a temporary.
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

    /// Count of bits set after left-shifting by n, without creating a temporary.
    /// Equivalent to (*this << n).count() but avoids temporary + sanitize_top.
    [[nodiscard]] constexpr auto shifted_count(size_t n) const noexcept -> size_t {
        if (n >= NumModes)
            return 0;
        if constexpr (kNumWords == 1) {
            // Shift and mask to NumModes — cheaper than creating a Bitset and sanitizing
            return static_cast<size_t>(std::popcount((words_[0] << n) & kTopMask));
        }
        else {
            // Fall back to the general shift + count path
            return (*this << n).count();
        }
    }

    // --- Modification ---

    constexpr auto set(size_t pos) noexcept -> Bitset & {
        words_[pos / word_width] |= uint64_t(1) << (pos % word_width);
        return *this;
    }

    constexpr auto reset(size_t pos) noexcept -> Bitset & {
        words_[pos / word_width] &= ~(uint64_t(1) << (pos % word_width));
        return *this;
    }

    constexpr auto flip(size_t pos) noexcept -> Bitset & {
        words_[pos / word_width] ^= uint64_t(1) << (pos % word_width);
        return *this;
    }

    // --- Bitwise operators ---
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

    constexpr auto operator<<=(size_t pos) noexcept -> Bitset & {
        if (pos >= NumModes) {
            words_.fill(0);
            return *this;
        }
        if constexpr (kNumWords == 1) {
            words_[0] <<= pos;
        }
        else {
            const size_t word_shift = pos / word_width;
            if (const size_t bit_shift = pos % word_width; bit_shift == 0) {
                for (size_t i = kNumWords; i-- > word_shift;)
                    words_[i] = words_[i - word_shift];
            }
            else {
                const size_t inv_shift = word_width - bit_shift;
                for (size_t i = kNumWords - 1; i > word_shift; --i) {
                    words_[i] = (words_[i - word_shift] << bit_shift) | (words_[i - word_shift - 1] >> inv_shift);
                }
                words_[word_shift] = words_[0] << bit_shift;
            }
            for (size_t i = 0; i < word_shift; ++i)
                words_[i] = 0;
        }
        sanitize_top();
        return *this;
    }

    [[nodiscard]] constexpr auto operator<<(size_t pos) const noexcept -> Bitset {
        Bitset r = *this;
        r <<= pos;
        return r;
    }

    constexpr auto operator>>=(size_t pos) noexcept -> Bitset & {
        if (pos >= NumModes) {
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

    // --- Comparison ---

    [[nodiscard]] constexpr auto operator==(const Bitset &o) const noexcept -> bool {
        for (size_t i = 0; i < kNumWords; ++i)
            if (words_[i] != o.words_[i])
                return false;
        return true;
    }

    // --- Word access (MPI, hashing, serialization) ---

    [[nodiscard]] static constexpr auto num_words() noexcept -> size_t { return kNumWords; }
    [[nodiscard]] constexpr auto data() const noexcept -> const uint64_t * { return words_.data(); }
    [[nodiscard]] constexpr auto data() noexcept -> uint64_t * { return words_.data(); }
    [[nodiscard]] constexpr auto word(size_t i) const noexcept -> uint64_t { return words_[i]; }

    // --- Bit scanning ---

    /// Find the position of the first set bit, or NumModes if none.
    [[nodiscard]] constexpr auto find_first() const noexcept -> size_t {
        for (size_t i = 0; i < kNumWords; ++i) {
            if (words_[i])
                return i * word_width + static_cast<size_t>(std::countr_zero(words_[i]));
        }
        return NumModes;
    }

    /// Find the next set bit after pos, or NumModes if none.
    [[nodiscard]] constexpr auto find_next(size_t pos) const noexcept -> size_t {
        if (++pos >= NumModes)
            return NumModes;
        if constexpr (kNumWords == 1) {
            if (const uint64_t w = words_[0] >> pos; w)
                return pos + static_cast<size_t>(std::countr_zero(w));
            return NumModes;
        }
        else {
            size_t wi = pos / word_width;
            if (const uint64_t w = words_[wi] >> (pos % word_width); w)
                return pos + static_cast<size_t>(std::countr_zero(w));
            for (++wi; wi < kNumWords; ++wi) {
                if (words_[wi])
                    return wi * word_width + static_cast<size_t>(std::countr_zero(words_[wi]));
            }
            return NumModes;
        }
    }

    /// Stream output: prints bits from MSB to LSB (matches std::bitset convention).
    friend auto operator<<(std::ostream &os, const Bitset &bs) -> std::ostream & {
        for (size_t i = NumModes; i-- > 0;)
            os << (bs.test(i) ? '1' : '0');
        return os;
    }
};
} // namespace monoprop

template <typename T>
struct SplitmixHash;

template <size_t NumModes>
struct SplitmixHash<monoprop::Bitset<NumModes>> {
    static constexpr auto mix(uint64_t x) noexcept -> uint64_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    auto operator()(const monoprop::Bitset<NumModes> &bs) const noexcept -> size_t {
        constexpr size_t W = monoprop::Bitset<NumModes>::num_words();
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

// std::hash specialization for Bitset — enables use with std:: containers.
namespace std {
template <size_t NumModes>
struct hash<monoprop::Bitset<NumModes>> {
    auto operator()(const monoprop::Bitset<NumModes> &bs) const noexcept -> size_t {
        return SplitmixHash<monoprop::Bitset<NumModes>>{}(bs);
    }
};
} // namespace std
