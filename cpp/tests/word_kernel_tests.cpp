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

// The scan's bound-width word passes against the computations they restate: detail::WordKernel<W>'s
// four Bitset methods, and detail::fully_paired_words<W>, whose oracle is cutoff_sums rather than a
// Bitset method (which is why it lives in AlgebraCommon.h and is tested here anyway -- one sweep over
// the inline regime, one set of word patterns). Every one of them is a second copy of an existing
// computation, so the only thing worth testing is that the two copies agree -- at every W in the
// inline regime, on the word patterns that distinguish a per-word fold from a whole-register one.
//
// splitmix carries the strongest obligation and gets the strictest test: that value is monomial_hash,
// which routes MPI ownership, so a divergence would move terms between ranks rather than merely run
// slower. It is asserted equal to SplitmixHash for every W, not merely well-distributed.

#include <boost/test/unit_test.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "monoprop/Bitset.h"
// AlgebraCommon.h before Monomial.h and without Utilities.h: Utilities.h is not self-contained (it is
// reached through TypeAliases.h, which pulls MPOperator.h in ahead of the free functions that header
// calls), so even_bits arrives transitively here the same way term_product_tests.cpp gets it.
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"

#include "InlineWidths.h"

using monoprop::Bitset;
using monoprop::detail::WordKernel;
using test_utils::for_each_inline_width;

namespace {

// A bitset of exactly W words with the given words written straight in. Going through data() rather
// than set() because these tests are about the words: a pattern like "every odd bit of word 3" is a
// word, not a bit list.
template <size_t W>
auto from_words(const std::array<uint64_t, W> &words) -> Bitset {
    Bitset bs(W * Bitset::word_width);
    BOOST_REQUIRE(bs.num_words() == W);
    for (size_t w = 0; w < W; ++w) {
        bs.data()[w] = words[w];
    }
    return bs;
}

// The patterns a per-word fold can get wrong where a whole-register one cannot: all-zero and all-ones
// (parity of an even count of set bits), the two single-mode halves (a fold that dropped a word would
// still see one of them), and the paired/unpaired even-odd patterns fully_paired keys on. Randomized
// words follow in every case; these are the ones worth naming.
template <size_t W>
auto interesting_words(std::mt19937_64 &rng) -> std::vector<std::array<uint64_t, W>> {
    std::vector<std::array<uint64_t, W>> out;
    const auto fill = [&](uint64_t v) {
        std::array<uint64_t, W> a{};
        a.fill(v);
        out.push_back(a);
    };
    fill(0);
    fill(~uint64_t{0});
    fill(0x5555555555555555ULL); // every even bit: every occupied mode singly occupied
    fill(0xAAAAAAAAAAAAAAAAULL); // every odd bit: likewise, the other Majorana
    fill(0xFFFFFFFFFFFFFFFFULL); // every mode fully paired
    // One word at a time set, so a fold that skipped word w fails on exactly one entry.
    for (size_t w = 0; w < W; ++w) {
        std::array<uint64_t, W> a{};
        a[w] = 0x0123456789ABCDEFULL;
        out.push_back(a);
        std::array<uint64_t, W> b{};
        b[w] = uint64_t{1} << (w % Bitset::word_width);
        out.push_back(b);
    }
    for (size_t t = 0; t < 64; ++t) {
        std::array<uint64_t, W> a{};
        for (auto &word : a) {
            word = rng();
        }
        out.push_back(a);
        // ...and the same words sparsified, since real monomials are sparse and a dense random word
        // never exercises the "one set bit in the whole register" shapes.
        std::array<uint64_t, W> sparse{};
        for (size_t k = 0; k < 3; ++k) {
            const size_t pos = rng() % (W * Bitset::word_width);
            sparse[pos / Bitset::word_width] |= uint64_t{1} << (pos % Bitset::word_width);
        }
        out.push_back(sparse);
    }
    return out;
}

} // namespace

// The owner-routing value. Equality with SplitmixHash (and hence monomial_hash) at every W, including
// the W == 1 arm both sides special-case.
BOOST_AUTO_TEST_CASE(word_kernel_splitmix_is_the_owner_routing_hash) {
    std::mt19937_64 rng(20260821U);
    for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        for (const auto &words : interesting_words<W>(rng)) {
            const Bitset bs = from_words<W>(words);
            const size_t expected = monoprop::SplitmixHash{}(bs);
            BOOST_TEST(WordKernel<W>::splitmix(bs.data()) == expected);
            // The name that marks the value as pinned, asserted separately from the hash functor so a
            // future indirection between them cannot pass this test by tautology.
            BOOST_TEST(WordKernel<W>::splitmix(bs.data()) == monoprop::monomial_hash(bs));
        }
    });
}

// The product and its two counts, in the same destination the scan writes.
BOOST_AUTO_TEST_CASE(word_kernel_fused_xor_into_matches_bitset) {
    std::mt19937_64 rng(31337U);
    for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        const auto lhs_words = interesting_words<W>(rng);
        const auto rhs_words = interesting_words<W>(rng);
        for (size_t i = 0; i < lhs_words.size(); ++i) {
            const Bitset lhs = from_words<W>(lhs_words[i]);
            const Bitset rhs = from_words<W>(rhs_words[i]);

            Bitset reference_out(W * Bitset::word_width);
            const auto reference = lhs.fused_xor_into(rhs, reference_out);

            Bitset candidate_out(W * Bitset::word_width);
            const auto candidate = WordKernel<W>::fused_xor_into(lhs.data(), rhs.data(), candidate_out.data());

            BOOST_TEST(candidate.overlap == reference.overlap);
            BOOST_TEST(candidate.result_count == reference.result_count);
            BOOST_TEST((candidate_out == reference_out));
        }
    });
}

// The Majorana rotation sign's parity, which is of the whole AND and not of any per-word rounding --
// so the all-ones patterns above matter: they make the per-word popcounts even and the total even too,
// where a wrong fold would still agree.
BOOST_AUTO_TEST_CASE(word_kernel_parity_and_matches_bitset) {
    std::mt19937_64 rng(4242U);
    for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        const auto lhs_words = interesting_words<W>(rng);
        const auto rhs_words = interesting_words<W>(rng);
        for (size_t i = 0; i < lhs_words.size(); ++i) {
            const Bitset lhs = from_words<W>(lhs_words[i]);
            const Bitset rhs = from_words<W>(rhs_words[i]);
            BOOST_TEST(WordKernel<W>::parity_and(lhs.data(), rhs.data()) == lhs.parity_and(rhs));
        }
    });
}

// The cutoff's fully-paired clause. The oracle is cutoff_sums' xor_sum over the whole register, which
// is the only window the pass is allowed to answer for (a narrower one keeps going through the
// evaluator -- see DenseTermProductsW). It carries the even-bit pattern as a literal, so the mask built
// here is also the check that the literal is what even_bits<LSb0> would have produced.
BOOST_AUTO_TEST_CASE(fully_paired_words_matches_cutoff_sums) {
    std::mt19937_64 rng(5150U);
    for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        const size_t num_bits = W * Bitset::word_width;
        const Bitset mask = monoprop::even_bits<monoprop::LSb0>(num_bits);
        for (size_t w = 0; w < W; ++w) {
            BOOST_TEST(mask.word(w) == 0x5555555555555555ULL);
        }
        size_t paired = 0;
        size_t unpaired = 0;
        for (const auto &words : interesting_words<W>(rng)) {
            const Bitset bs = from_words<W>(words);
            const bool expected = monoprop::cutoff_sums(bs, num_bits / 2).xor_sum == 0;
            BOOST_TEST(monoprop::detail::fully_paired_words<W>(bs.data()) == expected);
            paired += expected ? 1 : 0;
            unpaired += expected ? 0 : 1;
        }
        // Both answers occur, so neither is passing by always returning the same one.
        BOOST_TEST(paired > 0U);
        BOOST_TEST(unpaired > 0U);
    });
}

// clear() zeroes exactly W words. The word above is checked because the kernel's whole contract is
// "the caller bound the width": writing one word too many would corrupt an unrelated monomial's
// storage, and no other test reads that word.
BOOST_AUTO_TEST_CASE(word_kernel_clear_zeroes_exactly_its_width) {
    for_each_inline_width([&]<size_t W>(std::integral_constant<size_t, W>) {
        std::array<uint64_t, W + 1> buffer{};
        buffer.fill(~uint64_t{0});
        WordKernel<W>::clear(buffer.data());
        for (size_t w = 0; w < W; ++w) {
            BOOST_TEST(buffer[w] == 0U);
        }
        BOOST_TEST(buffer[W] == ~uint64_t{0});
    });
}
