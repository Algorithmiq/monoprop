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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <type_traits>

namespace monoprop::detail {

// Dispatches a runtime word count in [0, 8] to a fully-unrolled arm: W is known at compile time inside
// `f`, so a per-word loop written against it has no back-edge and no trip-count prologue/tail. Callers
// gate on `n <= Bitset::kInlineWords` themselves and fall back to a plain runtime loop above that;
// `default` is an unreachable safety net, not a ninth arm.
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

// The two popcounts a fused XOR reports, without its result. Nested in Bitset as FusedCounts, and
// named here because the word pass that produces them is declared before the class.
struct FusedWordCounts {
    size_t overlap;      // popcount(a & b)
    size_t result_count; // popcount(a ^ b)
};

// The word passes shared by Bitset's own inline arms and by the per-gate WordKernel below. They live
// here, ahead of both, so each is defined once: Bitset reaches them through with_nwords and the kernel
// through its bound W, and the two must not be able to drift -- one of them decides emitted term signs
// and the other feeds a cutoff.
//
// W is the exact word count of every operand. Correct at W == 0 (an empty fold), which is the arm
// with_nwords hands a zero-width bitset.
template <size_t W>
[[gnu::always_inline]] inline auto fused_xor_words(const uint64_t *a, const uint64_t *b, uint64_t *out) noexcept
    -> FusedWordCounts {
    size_t overlap = 0;
    size_t result_count = 0;
    for (size_t i = 0; i < W; ++i) {
        const uint64_t n = a[i] ^ b[i];
        out[i] = n;
        overlap += static_cast<size_t>(std::popcount(a[i] & b[i]));
        result_count += static_cast<size_t>(std::popcount(n));
    }
    return {overlap, result_count};
}

// XOR-fold of a & b into one word. Folding first and popcounting once is what makes the caller's
// parity that of the whole AND rather than of any per-word rounding.
template <size_t W>
[[gnu::always_inline]] inline auto and_fold_words(const uint64_t *a, const uint64_t *b) noexcept -> uint64_t {
    uint64_t folded = 0;
    for (size_t i = 0; i < W; ++i) {
        folded ^= a[i] & b[i];
    }
    return folded;
}

} // namespace monoprop::detail

namespace monoprop {

// std::bitset replacement over contiguous uint64_t words, with a *runtime* width: zero-copy MPI (via
// data()/word()), word-wise hashing, portable std::countr_zero scanning. The first kInlineWords words
// live inline (250 modes -> 500 bits -> 8 words, which was the compile-time ceiling before it was
// removed and is still where the interesting models sit); above that a bitset spills the *entire* word
// array to the heap, keeping data()/word(i) a single contiguous view regardless of which storage is
// active -- so wider systems are correct, but pay an allocation per by-value monomial. Every
// per-word loop routes through detail::with_nwords for n <= kInlineWords (the hot regime) and a plain
// loop above it.
//
// Unlike the Bitset<NumBits> template this replaces, one object is sized for the *widest* supported
// bitset rather than exactly for its own width (72 bytes vs the old 8/16/32/64 for 32/64/128/250
// modes). That is inherent to owning the bits inline at a runtime width, and it is paid per monomial
// wherever monomials are held by value in bulk. Keep hot paths off by-value temporaries because of
// it: prefer a word loop over `a & b` chains, and hoist masks out of per-term code. The plan's route
// out is Stage 6's arena, where a Bitset becomes a non-owning {pointer, width} view over
// exactly-sized storage and the size question disappears.
class Bitset {
public:
    // The word vocabulary is public because callers reason in words: data() already hands out a
    // word_type*, and a per-gate kernel that binds the word count needs the same three names to say
    // what it binds (see detail::WordKernel).
    using word_type = uint64_t;
    static constexpr auto word_width = sizeof(word_type) * 8;
    static constexpr size_t kInlineWords = 8;

private:
    // The inline words and the heap pointer are never both live -- nwords_ alone selects which -- so
    // they share storage. A std::vector member instead costs 24 bytes on *every* monomial at *every*
    // width, and monomials are stored by value in bulk (MonomialList, OperatorIndex::overflow_,
    // IncomingProbe::mono), so that overhead is multiplied by the term count. Sizing the object for
    // the widest inline width already costs enough; see the class comment above.
    //
    // Deliberately *not* initializing: a value-initialized inline_ zero-fills all kInlineWords words,
    // and since a default member initializer also runs before a copy constructor's body, every copy
    // paid that fill before overwriting it. Each constructor writes exactly the words it owns.
    union Storage {
        std::array<word_type, kInlineWords> inline_;
        word_type *heap_;
        Storage() noexcept {}
    };

    Storage s_;
    uint32_t nwords_ = 0;
    uint32_t top_bits_ = 0; // bits used in the last word; 0 means "all word_width bits used"

    // Invariant: only words [0, nwords_) hold a value. The inline tail above nwords_ is indeterminate,
    // and no reader may touch it -- every word loop, operator==, and SplitmixHash run nwords_, and the
    // MPI readers memcpy num_words() words. That is what lets a copy cost the operand's own width
    // instead of the widest supported one (16 bytes at 64 modes rather than 64), which is paid per
    // element wherever monomials are held by value in bulk.

    // Copies the live words only. Precondition: !spilled() -- with_nwords caps at kInlineWords, so a
    // spilled width would silently copy the first 8 words and drop the rest; those paths memcpy.
    auto copy_inline_from(const Bitset &o) noexcept -> void {
        word_type *d = s_.inline_.data();
        const word_type *s = o.data();
        detail::with_nwords(nwords_, [d, s]<size_t W>(std::integral_constant<size_t, W>) {
            for (size_t i = 0; i < W; ++i) {
                d[i] = s[i];
            }
        });
    }

    // Zeroes the live words only, same precondition and same reason as copy_inline_from. An unrolled
    // store loop rather than std::memset: the length is a runtime value, so memset would be an out-of-line
    // call on a path that is one or two stores wide.
    auto zero_inline() noexcept -> void {
        word_type *d = s_.inline_.data();
        detail::with_nwords(nwords_, [d]<size_t W>(std::integral_constant<size_t, W>) {
            for (size_t i = 0; i < W; ++i) {
                d[i] = 0;
            }
        });
    }

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

    // num_bits: the logical width, zeroed.
    explicit Bitset(size_t num_bits) noexcept
        : nwords_(static_cast<uint32_t>((num_bits + word_width - 1) / word_width)),
          top_bits_(static_cast<uint32_t>(num_bits % word_width)) {
        if (spilled()) {
            s_.heap_ = new word_type[nwords_]{};
        }
        else {
            zero_inline();
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
            copy_inline_from(o);
        }
    }

    // Steals the pointer in the spilled case and copies the live words otherwise, not the union's whole
    // object representation. Zeroing the source's nwords_ makes it inline-empty, so its destructor
    // frees nothing.
    Bitset(Bitset &&o) noexcept : nwords_(o.nwords_), top_bits_(o.top_bits_) {
        if (spilled()) {
            s_.heap_ = o.s_.heap_;
        }
        else {
            copy_inline_from(o);
        }
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
                copy_inline_from(o);
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
            copy_inline_from(o);
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
        // spilled() reads nwords_, so the two tests below straddle the assignment on purpose: the first
        // frees against the old width, the second selects the union member for the new one.
        nwords_ = o.nwords_;
        top_bits_ = o.top_bits_;
        if (spilled()) {
            s_.heap_ = o.s_.heap_;
        }
        else {
            copy_inline_from(o);
        }
        o.nwords_ = 0;
        o.top_bits_ = 0;
        return *this;
    }

    ~Bitset() noexcept {
        if (spilled()) {
            delete[] s_.heap_;
        }
    }

    // Bytes this bitset owns *outside* its own object, so a container counting sizeof(Bitset) per element
    // can add what a spilled element points at. Zero for an inline width, which is why a container that
    // omits it looks correct until someone runs past 8 words.
    [[nodiscard]] auto heap_bytes() const noexcept -> size_t { return spilled() ? nwords_ * sizeof(word_type) : 0; }

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

    // Every binary op below loops *this*'s word count and indexes the other operand unchecked, so a
    // narrower operand is read past its own width. That is not merely wrong-but-harmless: it only
    // reads zeros from the inline array while the *result* fits inline, and once *this* is spilled
    // (> kInlineWords) it reads off the end of the narrower operand's array. Widths must match at
    // every call site, so this is asserted rather than handled -- Release keeps the loops bare.
    //
    // popcount(*this & other) without materializing the temporary.
    [[nodiscard]] auto count_and(const Bitset &o) const noexcept -> size_t {
        assert(nwords_ == o.nwords_ && "Bitset::count_and width mismatch");
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
        assert(nwords_ == o.nwords_ && "Bitset::parity_and width mismatch");
        const word_type *a = data();
        const word_type *b = o.data();
        word_type parity_word = 0;
        if (nwords_ <= kInlineWords) {
            parity_word = detail::with_nwords(nwords_, [a, b]<size_t W>(std::integral_constant<size_t, W>) {
                return detail::and_fold_words<W>(a, b);
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
    // rather than a compile-time width, each pass also pays
    // its own loop prologue/tail, so the separate-ops cost keeps growing where this stays one pass.
    // Kept alongside the existing composable ops -- a cold path that only needs one of the three
    // should keep using them. result needs no sanitize_top(): XOR of two already-sanitized operands
    // never sets a bit above NumBits.
    //
    // Forward-declared here, defined below: a member holding Bitset by value can't be nested inside
    // Bitset's own (still-incomplete) definition -- unlike the old Bitset<NumBits>, a template, this
    // is no longer a template instantiated as one unit, so the usual incomplete-type rule applies.
    struct FusedXor;

    // Returned by value so the pass that computes the two counts can write the XOR straight into a
    // caller-owned destination.
    using FusedCounts = detail::FusedWordCounts;

    // fused_xor's one word pass, writing the XOR into `out` instead of into a fresh Bitset. The hot
    // path wants this form: `out` is a scratch monomial that lives for the whole gate, so a term
    // costs the word pass alone -- no width recomputation, no spill test, no allocation, and none of
    // the copies a by-value result goes through on its way to the caller's variable.
    auto fused_xor_into(const Bitset &gen, Bitset &out) const noexcept -> FusedCounts;

    [[nodiscard]] auto fused_xor(const Bitset &gen) const noexcept -> FusedXor;

    auto set(size_t pos) noexcept -> Bitset & {
        data()[pos / word_width] |= uint64_t(1) << (pos % word_width);
        return *this;
    }

    // Clear every bit, keeping the width. Not the same as assigning a default-constructed Bitset,
    // which drops the width to 0 -- the distinction matters wherever code needs "a zero monomial the
    // same shape as this one", which is copy-then-reset and nothing shorter (see change_basis).
    auto reset() noexcept -> Bitset & {
        std::memset(data(), 0, nwords_ * sizeof(word_type));
        return *this;
    }

    auto operator&=(const Bitset &rhs) noexcept -> Bitset & {
        assert(nwords_ == rhs.nwords_ && "Bitset::operator&= width mismatch");
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
        assert(nwords_ == rhs.nwords_ && "Bitset::operator|= width mismatch");
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
        assert(nwords_ == rhs.nwords_ && "Bitset::operator^= width mismatch");
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
    // to it. Unreachable while every bitset is built at a real width, but monomial keys live in a
    // boost::unordered_flat_map, and an asymmetric operator== there is a silent corruption rather
    // than a crash -- and the de-templated wire readers use exactly the
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

inline auto Bitset::fused_xor_into(const Bitset &gen, Bitset &dst) const noexcept -> FusedCounts {
    assert(num_words() == gen.num_words() && "Bitset::fused_xor_into width mismatch");
    assert(num_words() == dst.num_words() && "Bitset::fused_xor_into destination width mismatch");
    const word_type *a = data();
    const word_type *b = gen.data();
    word_type *out = dst.data();
    if (nwords_ <= kInlineWords) {
        return detail::with_nwords(nwords_, [a, b, out]<size_t W>(std::integral_constant<size_t, W>) {
            return detail::fused_xor_words<W>(a, b, out);
        });
    }
    // Spilled: no compile-time trip count to bind, so the plain loop.
    size_t overlap = 0;
    size_t result_count = 0;
    for (size_t i = 0; i < nwords_; ++i) {
        const word_type n = a[i] ^ b[i];
        out[i] = n;
        overlap += static_cast<size_t>(std::popcount(a[i] & b[i]));
        result_count += static_cast<size_t>(std::popcount(n));
    }
    return {overlap, result_count};
}

inline auto Bitset::fused_xor(const Bitset &gen) const noexcept -> FusedXor {
    Bitset result(size());
    const auto counts = fused_xor_into(gen, result);
    return {result, counts.overlap, counts.result_count};
}

// Bit-identical to the old per-width SplitmixHash<Bitset<NumBits>>: same mix(), same per-word fold
// order, same "+i" per-word offset -- only the num_words()==1 dispatch moved from `if constexpr` to a
// runtime check. The values must not change: they drive MPI owner routing (see monomial_hash).
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

namespace detail {

// The per-term word ops with the word count supplied by the caller instead of read off the operand.
//
// The arithmetic is identical to Bitset's own methods; what differs is what the compiler knows. A
// Bitset method must load nwords_, compare it against the inline capacity and select a storage
// pointer on every call, and none of those three can be hoisted out of a loop the optimizer cannot
// see through -- which, on the per-term path, is every call. Handing a kernel a compile-time W and
// the word pointers the caller resolved once leaves a straight-line unrolled loop, which is what the
// per-width Bitset<NumBits> got for free.
//
// Preconditions, none of them checkable here: every pointer is a Bitset::data() of a bitset of
// exactly W words, and W <= kInlineWords so no operand is spilled. The only legal caller is one that
// bound W from a width it owns for the whole loop -- see DenseTermProductsW, which is the seam that
// binds it once per gate.
//
// Standing in for a Bitset method is also what decides membership: the scan's other bound-width word
// pass, fully_paired_words, answers a question about the algebra rather than the storage, so it lives
// beside its own oracle in AlgebraCommon.h instead.
template <size_t W>
struct WordKernel {
    static_assert(W >= 1 && W <= Bitset::kInlineWords,
                  "the kernel covers the inline regime; above it the runtime loop already wins");

    using word_type = Bitset::word_type;

    static auto clear(word_type *a) noexcept -> void {
        for (size_t i = 0; i < W; ++i) {
            a[i] = 0;
        }
    }

    // Bitset::fused_xor_into with W fixed -- the same pass, reached without the nwords_ load and
    // storage-pointer select the method does per call.
    static auto fused_xor_into(const word_type *a, const word_type *b, word_type *out) noexcept -> Bitset::FusedCounts {
        return fused_xor_words<W>(a, b, out);
    }

    // Bitset::parity_and with W fixed, likewise.
    [[nodiscard]] static auto parity_and(const word_type *a, const word_type *b) noexcept -> bool {
        return (std::popcount(and_fold_words<W>(a, b)) & 1U) != 0;
    }

    // SplitmixHash with W fixed. Must stay bit-identical to it: this value routes MPI ownership, so a
    // divergence would move terms between ranks rather than merely run slower. Hence the W == 1 arm
    // reproducing the same special case rather than folding into the loop.
    [[nodiscard]] static auto splitmix(const word_type *a) noexcept -> size_t {
        if constexpr (W == 1) {
            return static_cast<size_t>(SplitmixHash::mix(a[0]));
        }
        else {
            uint64_t h = 0;
            for (size_t i = 0; i < W; ++i) {
                h ^= SplitmixHash::mix(a[i] + static_cast<uint64_t>(i));
            }
            return static_cast<size_t>(h);
        }
    }
};

} // namespace detail

} // namespace monoprop

namespace std {
template <>
struct hash<monoprop::Bitset> {
    auto operator()(const monoprop::Bitset &bs) const noexcept -> size_t { return monoprop::SplitmixHash{}(bs); }
};
} // namespace std
