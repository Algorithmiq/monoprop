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
#include <type_traits>

namespace monoprop::detail {

// Dispatches a runtime word count in [0, 8] to a fully-unrolled arm (W known at compile time inside
// `f`, so a per-word loop written against it has no back-edge and no trip-count prologue/tail -- see
// the NumModes-NTTP-removal plan's Stage 2b). Callers gate on `n <= Bitset::kInlineWords` themselves and
// fall back to a plain runtime loop above that; `default` is an unreachable safety net, not a ninth arm.
template <typename F>
[[gnu::always_inline]] inline auto with_nwords(size_t n, F &&f) -> decltype(auto) {
    switch (n) {
        case 0:
            return f(std::integral_constant<size_t, 0>{});
        case 1:
            return f(std::integral_constant<size_t, 1>{});
        case 2:
            return f(std::integral_constant<size_t, 2>{});
        case 3:
            return f(std::integral_constant<size_t, 3>{});
        case 4:
            return f(std::integral_constant<size_t, 4>{});
        case 5:
            return f(std::integral_constant<size_t, 5>{});
        case 6:
            return f(std::integral_constant<size_t, 6>{});
        case 7:
            return f(std::integral_constant<size_t, 7>{});
        default:
            return f(std::integral_constant<size_t, 8>{});
    }
}

} // namespace monoprop::detail

namespace monoprop {

// std::bitset replacement over contiguous uint64_t words, with a *runtime* width: zero-copy MPI (via
// data()/word()), word-wise hashing, portable std::countr_zero scanning. The first kInlineWords words
// live inline (covers the shipped default ceiling, monoprop_MAX_NUM_MODES=250 -> 500 bits -> 8 words,
// so the whole shipped range never allocates); a wider bitset spills the *entire* word array to the
// heap, keeping data()/word(i) a single contiguous view regardless of which storage is active. Every
// per-word loop routes through detail::with_nwords for n <= kInlineWords (the hot regime) and a plain
// loop above it -- see the NumModes-NTTP-removal plan's Stage 2b.
//
// Unlike the Bitset<NumBits> template this replaces, one object is sized for the *widest* supported
// bitset rather than exactly for its own width (72 bytes vs the old 8/16/32/64 for 32/64/128/250
// modes). That is inherent to owning the bits inline at a runtime width, and it is paid per monomial
// wherever monomials are held by value in bulk. Keep hot paths off by-value temporaries because of
// it: prefer a word loop over `a & b` chains, and hoist masks out of per-term code. The plan's route
// out is Stage 6's arena, where a Bitset becomes a non-owning {pointer, width} view over
// exactly-sized storage and the size question disappears.
class Bitset {
    using word_type = uint64_t;
    static constexpr auto word_width = sizeof(word_type) * 8;
    static constexpr size_t kInlineWords = 8;

    // The inline words and the heap pointer are never both live -- nwords_ alone selects which -- so
    // they share storage. A std::vector member instead costs 24 bytes on *every* monomial at *every*
    // width, and monomials are stored by value in bulk (MonomialList, OperatorIndex::overflow_,
    // IncomingProbe::mono), so that overhead is multiplied by the term count. Sizing the object for
    // the widest supported bitset already costs enough; see the Stage 2b follow-up note below.
    union Storage {
        std::array<word_type, kInlineWords> inline_;
        word_type *heap_;
        constexpr Storage() noexcept : inline_{} {}
    };

    Storage s_{};
    uint32_t nwords_ = 0;
    uint32_t top_bits_ = 0; // bits used in the last word; 0 means "all word_width bits used"

    // Selects the active union member. Must be consulted *before* nwords_ is overwritten by an
    // assignment, and *after* it is set by a constructor.
    [[nodiscard]] auto spilled() const noexcept -> bool { return nwords_ > kInlineWords; }

    [[nodiscard]] auto top_mask() const noexcept -> word_type {
        return top_bits_ ? ((word_type{1} << top_bits_) - 1) : ~word_type{0};
    }

    auto sanitize_top() noexcept -> void {
        if (nwords_ != 0) {
            data()[nwords_ - 1] &= top_mask();
        }
    }

public:
    Bitset() noexcept = default;

    // num_bits: the logical width. Zero-initialized.
    explicit Bitset(size_t num_bits) noexcept
        : nwords_(static_cast<uint32_t>((num_bits + word_width - 1) / word_width)),
          top_bits_(static_cast<uint32_t>(num_bits % word_width)) {
        if (spilled()) {
            s_.heap_ = new word_type[nwords_]{};
        }
    }

    // num_bits plus a value packed into word 0. Width can no longer be implied by the type (unlike the
    // old Bitset<NumBits>(uint64_t) implicit conversion), so this stays explicit and two-argument.
    explicit Bitset(size_t num_bits, uint64_t val) noexcept : Bitset(num_bits) {
        if (nwords_ != 0) {
            data()[0] = val;
            sanitize_top();
        }
    }

    Bitset(const Bitset &o) noexcept : nwords_(o.nwords_), top_bits_(o.top_bits_) {
        if (spilled()) {
            s_.heap_ = new word_type[nwords_];
            std::memcpy(s_.heap_, o.s_.heap_, nwords_ * sizeof(word_type));
        }
        else {
            s_.inline_ = o.s_.inline_;
        }
    }

    // Copies the union's object representation, which steals the pointer in the spilled case. Zeroing
    // the source's nwords_ makes it inline-empty, so its destructor frees nothing.
    Bitset(Bitset &&o) noexcept : s_(o.s_), nwords_(o.nwords_), top_bits_(o.top_bits_) {
        o.nwords_ = 0;
        o.top_bits_ = 0;
    }

    auto operator=(const Bitset &o) noexcept -> Bitset & {
        if (this == &o) {
            return *this;
        }
        // Same width is the overwhelmingly common case (a monomial assigned from another monomial of
        // the same operator), and it needs no reallocation at all.
        if (nwords_ == o.nwords_) {
            top_bits_ = o.top_bits_;
            if (spilled()) {
                std::memcpy(s_.heap_, o.s_.heap_, nwords_ * sizeof(word_type));
            }
            else {
                s_.inline_ = o.s_.inline_;
            }
            return *this;
        }
        if (spilled()) {
            delete[] s_.heap_;
        }
        nwords_ = o.nwords_;
        top_bits_ = o.top_bits_;
        if (spilled()) {
            s_.heap_ = new word_type[nwords_];
            std::memcpy(s_.heap_, o.s_.heap_, nwords_ * sizeof(word_type));
        }
        else {
            s_.inline_ = o.s_.inline_;
        }
        return *this;
    }

    auto operator=(Bitset &&o) noexcept -> Bitset & {
        if (this == &o) {
            return *this;
        }
        if (spilled()) {
            delete[] s_.heap_;
        }
        s_ = o.s_;
        nwords_ = o.nwords_;
        top_bits_ = o.top_bits_;
        o.nwords_ = 0;
        o.top_bits_ = 0;
        return *this;
    }

    ~Bitset() noexcept {
        if (spilled()) {
            delete[] s_.heap_;
        }
    }

    [[nodiscard]] auto data() const noexcept -> const word_type * { return spilled() ? s_.heap_ : s_.inline_.data(); }
    [[nodiscard]] auto data() noexcept -> word_type * { return spilled() ? s_.heap_ : s_.inline_.data(); }
    [[nodiscard]] auto word(size_t i) const noexcept -> uint64_t { return data()[i]; }

    [[nodiscard]] auto num_words() const noexcept -> size_t { return nwords_; }
    [[nodiscard]] auto size() const noexcept -> size_t {
        if (nwords_ == 0) {
            return 0;
        }
        return top_bits_ != 0 ? (static_cast<size_t>(nwords_ - 1) * word_width + top_bits_)
                              : static_cast<size_t>(nwords_) * word_width;
    }

    [[nodiscard]] auto count() const noexcept -> size_t {
        const word_type *w = data();
        if (nwords_ <= kInlineWords) {
            return detail::with_nwords(nwords_, [w]<size_t W>(std::integral_constant<size_t, W>) {
                size_t c = 0;
                for (size_t i = 0; i < W; ++i)
                    c += static_cast<size_t>(std::popcount(w[i]));
                return c;
            });
        }
        size_t c = 0;
        for (size_t i = 0; i < nwords_; ++i)
            c += static_cast<size_t>(std::popcount(w[i]));
        return c;
    }

    [[nodiscard]] auto test(size_t pos) const noexcept -> bool {
        return (data()[pos / word_width] >> (pos % word_width)) & 1;
    }

    [[nodiscard]] auto any() const noexcept -> bool {
        const word_type *w = data();
        if (nwords_ <= kInlineWords) {
            return detail::with_nwords(nwords_, [w]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i)
                    if (w[i])
                        return true;
                return false;
            });
        }
        for (size_t i = 0; i < nwords_; ++i)
            if (w[i])
                return true;
        return false;
    }

    [[nodiscard]] auto none() const noexcept -> bool { return !any(); }

    // popcount(*this & other) without materializing the temporary.
    [[nodiscard]] auto count_and(const Bitset &o) const noexcept -> size_t {
        const word_type *a = data();
        const word_type *b = o.data();
        if (nwords_ <= kInlineWords) {
            return detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                size_t c = 0;
                for (size_t i = 0; i < W; ++i)
                    c += static_cast<size_t>(std::popcount(a[i] & b[i]));
                return c;
            });
        }
        size_t c = 0;
        for (size_t i = 0; i < nwords_; ++i)
            c += static_cast<size_t>(std::popcount(a[i] & b[i]));
        return c;
    }

    [[nodiscard]] auto parity_and(const Bitset &o) const noexcept -> bool {
        const word_type *a = data();
        const word_type *b = o.data();
        word_type parity_word = 0;
        if (nwords_ <= kInlineWords) {
            parity_word = detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                word_type p = 0;
                for (size_t i = 0; i < W; ++i)
                    p ^= a[i] & b[i];
                return p;
            });
        }
        else {
            for (size_t i = 0; i < nwords_; ++i)
                parity_word ^= a[i] & b[i];
        }
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
    //
    // Forward-declared here, defined below: a member holding Bitset by value can't be nested inside
    // Bitset's own (still-incomplete) definition -- unlike the old Bitset<NumBits>, a template, this
    // is no longer a template instantiated as one unit, so the usual incomplete-type rule applies.
    struct FusedXor;

    [[nodiscard]] auto fused_xor(const Bitset &gen) const noexcept -> FusedXor;

    auto set(size_t pos) noexcept -> Bitset & {
        data()[pos / word_width] |= uint64_t(1) << (pos % word_width);
        return *this;
    }

    auto operator&=(const Bitset &rhs) noexcept -> Bitset & {
        word_type *a = data();
        const word_type *b = rhs.data();
        if (nwords_ <= kInlineWords) {
            detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i)
                    a[i] &= b[i];
            });
        }
        else {
            for (size_t i = 0; i < nwords_; ++i)
                a[i] &= b[i];
        }
        return *this;
    }

    auto operator|=(const Bitset &rhs) noexcept -> Bitset & {
        word_type *a = data();
        const word_type *b = rhs.data();
        if (nwords_ <= kInlineWords) {
            detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i)
                    a[i] |= b[i];
            });
        }
        else {
            for (size_t i = 0; i < nwords_; ++i)
                a[i] |= b[i];
        }
        return *this;
    }

    auto operator^=(const Bitset &rhs) noexcept -> Bitset & {
        word_type *a = data();
        const word_type *b = rhs.data();
        if (nwords_ <= kInlineWords) {
            detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i)
                    a[i] ^= b[i];
            });
        }
        else {
            for (size_t i = 0; i < nwords_; ++i)
                a[i] ^= b[i];
        }
        return *this;
    }

    [[nodiscard]] auto operator~() const noexcept -> Bitset {
        Bitset r = *this;
        word_type *w = r.data();
        if (nwords_ <= kInlineWords) {
            detail::with_nwords(nwords_, [w]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i)
                    w[i] = ~w[i];
            });
        }
        else {
            for (size_t i = 0; i < nwords_; ++i)
                w[i] = ~w[i];
        }
        r.sanitize_top();
        return r;
    }

    [[nodiscard]] friend auto operator&(const Bitset &lhs, const Bitset &rhs) noexcept -> Bitset {
        Bitset r = lhs;
        r &= rhs;
        return r;
    }

    [[nodiscard]] friend auto operator|(const Bitset &lhs, const Bitset &rhs) noexcept -> Bitset {
        Bitset r = lhs;
        r |= rhs;
        return r;
    }

    [[nodiscard]] friend auto operator^(const Bitset &lhs, const Bitset &rhs) noexcept -> Bitset {
        Bitset r = lhs;
        r ^= rhs;
        return r;
    }

    auto operator>>=(size_t pos) noexcept -> Bitset & {
        const size_t num_bits = size();
        word_type *w = data();
        if (pos >= num_bits) {
            for (size_t i = 0; i < nwords_; ++i)
                w[i] = 0;
            return *this;
        }
        if (nwords_ <= 1) {
            if (nwords_ == 1) {
                w[0] >>= pos;
            }
            return *this;
        }
        const size_t word_shift = pos / word_width;
        const size_t limit = nwords_ - word_shift;
        if (const size_t bit_shift = pos % word_width; bit_shift == 0) {
            for (size_t i = 0; i < limit; ++i)
                w[i] = w[i + word_shift];
        }
        else {
            const size_t inv_shift = word_width - bit_shift;
            for (size_t i = 0; i + 1 < limit; ++i) {
                w[i] = (w[i + word_shift] >> bit_shift) | (w[i + word_shift + 1] << inv_shift);
            }
            w[limit - 1] = w[nwords_ - 1] >> bit_shift;
        }
        for (size_t i = limit; i < nwords_; ++i)
            w[i] = 0;
        return *this;
    }

    [[nodiscard]] auto operator>>(size_t pos) const noexcept -> Bitset {
        Bitset r = *this;
        r >>= pos;
        return r;
    }

    // Width first, or equality is asymmetric: the loops below run this->nwords_, so without it a
    // default-constructed (width-0) bitset compares equal to everything while nothing compares equal
    // to it. Unreachable while every bitset is built at a real width, but Monomial keys live in a
    // boost::unordered_flat_map, and an asymmetric operator== there is a silent corruption rather
    // than a crash -- and de-templating the wire readers introduces exactly the
    // default-construct-then-assign pattern that produces a width-0 operand.
    [[nodiscard]] auto operator==(const Bitset &o) const noexcept -> bool {
        if (nwords_ != o.nwords_) {
            return false;
        }
        const word_type *a = data();
        const word_type *b = o.data();
        if (nwords_ <= kInlineWords) {
            return detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i)
                    if (a[i] != b[i])
                        return false;
                return true;
            });
        }
        for (size_t i = 0; i < nwords_; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    [[nodiscard]] auto find_first() const noexcept -> size_t { // size() if none
        const word_type *w = data();
        if (nwords_ <= kInlineWords) {
            const size_t hit = detail::with_nwords(nwords_, [w]<size_t W>(std::integral_constant<size_t, W>) {
                for (size_t i = 0; i < W; ++i) {
                    if (w[i])
                        return (i * word_width) + static_cast<size_t>(std::countr_zero(w[i]));
                }
                return static_cast<size_t>(-1);
            });
            return hit == static_cast<size_t>(-1) ? size() : hit;
        }
        for (size_t i = 0; i < nwords_; ++i) {
            if (w[i])
                return (i * word_width) + static_cast<size_t>(std::countr_zero(w[i]));
        }
        return size();
    }

    [[nodiscard]] auto find_next(size_t pos) const noexcept -> size_t { // size() if none
        const size_t num_bits = size();
        if (++pos >= num_bits) {
            return num_bits;
        }
        const word_type *w = data();
        if (nwords_ <= 1) {
            if (const uint64_t x = w[0] >> pos; x)
                return pos + static_cast<size_t>(std::countr_zero(x));
            return num_bits;
        }
        size_t wi = pos / word_width;
        if (const uint64_t x = w[wi] >> (pos % word_width); x)
            return pos + static_cast<size_t>(std::countr_zero(x));
        for (++wi; wi < nwords_; ++wi) {
            if (w[wi])
                return (wi * word_width) + static_cast<size_t>(std::countr_zero(w[wi]));
        }
        return num_bits;
    }

    // Stream output MSB→LSB (std::bitset convention).
    friend auto operator<<(std::ostream &os, const Bitset &bs) -> std::ostream & {
        for (size_t i = bs.size(); i-- > 0;)
            os << (bs.test(i) ? '1' : '0');
        return os;
    }
};

struct Bitset::FusedXor {
    Bitset result;
    size_t overlap;
    size_t result_count;
};

inline auto Bitset::fused_xor(const Bitset &gen) const noexcept -> FusedXor {
    Bitset result(size());
    const word_type *a = data();
    const word_type *b = gen.data();
    word_type *out = result.data();
    size_t overlap = 0;
    size_t result_count = 0;
    if (nwords_ <= kInlineWords) {
        detail::with_nwords(nwords_, [&]<size_t W>(std::integral_constant<size_t, W>) {
            for (size_t i = 0; i < W; ++i) {
                const word_type n = a[i] ^ b[i];
                out[i] = n;
                overlap += static_cast<size_t>(std::popcount(a[i] & b[i]));
                result_count += static_cast<size_t>(std::popcount(n));
            }
        });
    }
    else {
        for (size_t i = 0; i < nwords_; ++i) {
            const word_type n = a[i] ^ b[i];
            out[i] = n;
            overlap += static_cast<size_t>(std::popcount(a[i] & b[i]));
            result_count += static_cast<size_t>(std::popcount(n));
        }
    }
    return {result, overlap, result_count};
}

// Bit-identical to the old per-width SplitmixHash<Bitset<NumBits>>: same mix(), same per-word fold
// order, same "+i" per-word offset -- only the num_words()==1 dispatch moved from `if constexpr` to a
// runtime check (see the NumModes-NTTP-removal plan's invariant on SplitmixHash not changing).
struct SplitmixHash {
    static constexpr auto mix(uint64_t x) noexcept -> uint64_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    auto operator()(const Bitset &bs) const noexcept -> size_t {
        const size_t w = bs.num_words();
        if (w == 1) {
            return static_cast<size_t>(mix(bs.word(0)));
        }
        uint64_t h = 0;
        for (size_t i = 0; i < w; ++i) {
            h ^= mix(bs.word(i) + static_cast<uint64_t>(i));
        }
        return static_cast<size_t>(h);
    }
};

} // namespace monoprop

namespace std {
template <>
struct hash<monoprop::Bitset> {
    auto operator()(const monoprop::Bitset &bs) const noexcept -> size_t { return monoprop::SplitmixHash{}(bs); }
};
} // namespace std
