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

// Bitset.h in isolation (inline and heap-spilled widths) against a std::bitset oracle, so a
// regression in the hand-rolled shift / scan / mask / trampoline surfaces here rather than as a
// distant energy drift. N stays a *template* parameter of the test helpers purely so std::bitset<N>
// (the oracle) can be spelled; the Bitset under test is always constructed with a runtime width.

#include <boost/test/unit_test.hpp>

#include <bitset>
#include <cstdint>
#include <random>
#include <vector>

#include "monoprop/Bitset.h"

using monoprop::Bitset;

namespace {

template <size_t N>
auto make_pair(const std::vector<size_t> &positions) -> std::pair<Bitset, std::bitset<N>> {
    Bitset bs(N);
    std::bitset<N> ref;
    for (size_t p : positions) {
        bs.set(p);
        ref.set(p);
    }
    return {bs, ref};
}

template <size_t N>
auto expect_equal(const Bitset &bs, const std::bitset<N> &ref) -> void {
    BOOST_TEST(bs.size() == N);
    for (size_t i = 0; i < N; ++i) {
        BOOST_TEST(bs.test(i) == ref.test(i), "bit " << i);
    }
    BOOST_TEST(bs.count() == ref.count());
}

} // namespace

// The ctor masks off bits beyond the requested width, so a partial top word never leaks stray high bits.
BOOST_AUTO_TEST_CASE(bitset_ctor_sanitizes_top) {
    const Bitset b(10, 0xFFFFULL);
    BOOST_TEST(b.count() == 10U);
    BOOST_TEST(b.word(0) == 0x3FFULL);
    const Bitset full(64, ~uint64_t{0});
    BOOST_TEST(full.count() == 64U);
}

BOOST_AUTO_TEST_CASE(bitset_set_test_word_boundaries) {
    auto [bs, ref] = make_pair<100>({0, 63, 64, 99});
    expect_equal<100>(bs, ref);
    BOOST_TEST(bs.test(63));
    BOOST_TEST(bs.test(64));
    BOOST_TEST(!bs.test(62));
    BOOST_TEST(!bs.test(65));
    BOOST_TEST(bs.count() == 4U);
}

BOOST_AUTO_TEST_CASE(bitset_count_and_parity_and_cross_word) {
    auto [a, ra] = make_pair<192>({1, 63, 64, 130, 191});
    auto [b, rb] = make_pair<192>({63, 64, 65, 130});
    const size_t expected = (ra & rb).count();
    BOOST_TEST(a.count_and(b) == expected);
    BOOST_TEST(a.parity_and(b) == ((expected & 1U) != 0U));
    auto [c, rc] = make_pair<192>({0, 2, 4});
    auto [d, rd] = make_pair<192>({1, 3, 5});
    BOOST_TEST(c.count_and(d) == 0U);
    BOOST_TEST(!c.parity_and(d));
}

// fused_xor must agree with the composed operator^ / count_and it replaces in the hot path
// (Scan.h's emit_term_products) -- same operands, same three quantities, one pass instead of two.
BOOST_AUTO_TEST_CASE(bitset_fused_xor_matches_composed_ops) {
    auto [a, ra] = make_pair<192>({1, 63, 64, 130, 191});
    auto [b, rb] = make_pair<192>({63, 64, 65, 130});
    const auto fused = a.fused_xor(b);
    expect_equal<192>(fused.result, ra ^ rb);
    BOOST_TEST(fused.overlap == a.count_and(b));
    BOOST_TEST(fused.result_count == (ra ^ rb).count());

    // Single-word path (num_words() == 1) takes the same loop body, just one iteration.
    auto [c, rc] = make_pair<64>({0, 7, 31, 63});
    auto [d, rd] = make_pair<64>({7, 8, 31});
    const auto fused_sw = c.fused_xor(d);
    expect_equal<64>(fused_sw.result, rc ^ rd);
    BOOST_TEST(fused_sw.overlap == c.count_and(d));
    BOOST_TEST(fused_sw.result_count == (rc ^ rd).count());
}

BOOST_AUTO_TEST_CASE(bitset_not_respects_top_mask) {
    BOOST_TEST((~Bitset(100)).count() == 100U);
    BOOST_TEST((~Bitset(64)).count() == 64U);
    BOOST_TEST((~Bitset(10)).count() == 10U);
    auto [bs, ref] = make_pair<100>({3, 70, 99});
    BOOST_TEST((~~bs) == bs);
    (void)ref;
}

// Shift amounts cover exact word multiples, sub-word crossings, and >= size() (which must zero the set).
BOOST_AUTO_TEST_CASE(bitset_shift_right_cross_word) {
    const std::vector<size_t> pos{0, 5, 63, 64, 65, 130, 191};
    for (size_t s : {size_t{0},
                     size_t{1},
                     size_t{37},
                     size_t{63},
                     size_t{64},
                     size_t{65},
                     size_t{128},
                     size_t{191},
                     size_t{192},
                     size_t{300}}) {
        auto [bs, ref] = make_pair<192>(pos);
        bs >>= s;
        const std::bitset<192> expected = ref >> s;
        expect_equal<192>(bs, expected);
    }
    // Single-word path (num_words() == 1) takes a separate branch.
    auto [bs, ref] = make_pair<64>({0, 7, 31, 63});
    bs >>= 8;
    expect_equal<64>(bs, ref >> 8);
}

BOOST_AUTO_TEST_CASE(bitset_find_first_next_chain) {
    auto [bs, ref] = make_pair<192>({5, 63, 64, 130, 191});
    (void)ref;
    BOOST_TEST(bs.find_first() == 5U);
    BOOST_TEST(bs.find_next(5) == 63U);
    BOOST_TEST(bs.find_next(63) == 64U);
    BOOST_TEST(bs.find_next(64) == 130U);
    BOOST_TEST(bs.find_next(130) == 191U);
    BOOST_TEST(bs.find_next(191) == 192U); // past the last set bit -> size()
    BOOST_TEST(Bitset(192).find_first() == 192U);
    // Single-word find_next branch.
    auto [sb, sref] = make_pair<64>({0, 40});
    (void)sref;
    BOOST_TEST(sb.find_first() == 0U);
    BOOST_TEST(sb.find_next(0) == 40U);
    BOOST_TEST(sb.find_next(40) == 64U);
}

// The multi-word hash must depend on which word carries a bit (the +i mix guard), and be deterministic.
BOOST_AUTO_TEST_CASE(bitset_splitmix_hash_position_sensitive) {
    Bitset low(128);
    low.set(0);
    Bitset high(128);
    high.set(64); // bit 0 of word 1 — same intra-word position as `low`'s bit
    const std::hash<Bitset> h;
    BOOST_TEST(h(low) != h(high));
    BOOST_TEST(h(low) == h(low));
    Bitset low_copy(128);
    low_copy.set(0);
    BOOST_TEST(h(low) == h(low_copy));
}

BOOST_AUTO_TEST_CASE(bitset_random_differential) {
    constexpr size_t N = 128;
    std::mt19937_64 rng(0xB175E7ULL);
    std::uniform_int_distribution<size_t> bit(0, N - 1);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<size_t> pa;
        std::vector<size_t> pb;
        for (int k = 0; k < 12; ++k) {
            pa.push_back(bit(rng));
            pb.push_back(bit(rng));
        }
        auto [a, ra] = make_pair<N>(pa);
        auto [b, rb] = make_pair<N>(pb);
        expect_equal<N>(a & b, ra & rb);
        expect_equal<N>(a | b, ra | rb);
        expect_equal<N>(a ^ b, ra ^ rb);
        const size_t s = bit(rng);
        expect_equal<N>(a >> s, ra >> s);
        BOOST_TEST(a.count_and(b) == (ra & rb).count());
        BOOST_TEST((a == b) == (ra == rb));
        const auto fused = a.fused_xor(b);
        expect_equal<N>(fused.result, ra ^ rb);
        BOOST_TEST(fused.overlap == a.count_and(b));
        BOOST_TEST(fused.result_count == (ra ^ rb).count());
    }
}

// kInlineWords == 8 (512 bits): the trampoline's own boundary. 512 is the last inline width, 576 the
// first spilled one -- both must agree bit-for-bit with the oracle and with each other's operations.
BOOST_AUTO_TEST_CASE(bitset_trampoline_inline_spill_boundary) {
    std::mt19937_64 rng(0xB0DA51ULL);
    for (size_t n : {size_t{511}, size_t{512}, size_t{513}, size_t{576}, size_t{1024}, size_t{4096}}) {
        std::uniform_int_distribution<size_t> bit(0, n - 1);
        std::vector<size_t> pa;
        std::vector<size_t> pb;
        for (int k = 0; k < 20; ++k) {
            pa.push_back(bit(rng));
            pb.push_back(bit(rng));
        }
        Bitset a(n);
        Bitset b(n);
        std::vector<bool> ra(n, false);
        std::vector<bool> rb(n, false);
        for (size_t p : pa) {
            a.set(p);
            ra[p] = true;
        }
        for (size_t p : pb) {
            b.set(p);
            rb[p] = true;
        }
        BOOST_TEST(a.num_words() == (n + 63) / 64);
        BOOST_TEST(a.size() == n);

        size_t expected_and = 0;
        size_t expected_xor_count = 0;
        for (size_t i = 0; i < n; ++i) {
            expected_and += static_cast<size_t>(ra[i] && rb[i]);
            expected_xor_count += static_cast<size_t>(ra[i] != rb[i]);
        }
        BOOST_TEST(a.count_and(b) == expected_and);

        const auto x = a ^ b;
        BOOST_TEST(x.count() == expected_xor_count);
        for (size_t i = 0; i < n; ++i) {
            BOOST_TEST(x.test(i) == (ra[i] != rb[i]), "n=" << n << " bit " << i);
        }

        const auto fused = a.fused_xor(b);
        BOOST_TEST(fused.overlap == expected_and);
        BOOST_TEST(fused.result_count == expected_xor_count);
        BOOST_TEST((fused.result == x));
    }
}

// A spilled Bitset (n > 512 bits) must copy deeply -- mutating a copy must not alias the original's
// heap buffer.
BOOST_AUTO_TEST_CASE(bitset_spilled_copy_is_independent) {
    Bitset a(1024);
    a.set(1000);
    Bitset b = a; // copy
    b.set(5);
    BOOST_TEST(a.test(5) == false);
    BOOST_TEST(b.test(5) == true);
    BOOST_TEST(a.test(1000) == true);
    BOOST_TEST(b.test(1000) == true);
    BOOST_TEST(a.count() == 1U);
    BOOST_TEST(b.count() == 2U);
}

// find_first/find_next must walk past the inline/spill boundary and across many spilled words.
BOOST_AUTO_TEST_CASE(bitset_spilled_find_chain) {
    Bitset bs(2048);
    bs.set(0);
    bs.set(511);
    bs.set(512);
    bs.set(1000);
    bs.set(2047);
    BOOST_TEST(bs.find_first() == 0U);
    BOOST_TEST(bs.find_next(0) == 511U);
    BOOST_TEST(bs.find_next(511) == 512U);
    BOOST_TEST(bs.find_next(512) == 1000U);
    BOOST_TEST(bs.find_next(1000) == 2047U);
    BOOST_TEST(bs.find_next(2047) == 2048U);
}

// Equality must be symmetric even across widths. A width-0 bitset used to compare equal to everything
// while nothing compared equal to it, which in a hash map is silent corruption rather than a crash.
BOOST_AUTO_TEST_CASE(bitset_equality_is_symmetric_across_widths) {
    const Bitset zero;
    const Bitset narrow(64, 0xdeadbeefULL);
    const Bitset wide(256, 0xdeadbeefULL);

    BOOST_TEST(!(zero == narrow));
    BOOST_TEST(!(narrow == zero));
    // Same words, different widths: still distinct.
    BOOST_TEST(!(narrow == wide));
    BOOST_TEST(!(wide == narrow));
    // Same width, same words: equal both ways.
    BOOST_TEST((wide == Bitset(256, 0xdeadbeefULL)));
    BOOST_TEST((Bitset(256, 0xdeadbeefULL) == wide));
}
