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

// Bitset.h in isolation (single-word and multi-word) against a std::bitset oracle, so a regression
// in the hand-rolled shift / scan / mask surfaces here rather than as a distant energy drift.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bitset>
#include <cstdint>
#include <random>
#include <vector>

#include "monoprop/Bitset.h"

using monoprop::Bitset;

namespace {

template <size_t N>
auto make_pair(const std::vector<size_t> &positions) -> std::pair<Bitset<N>, std::bitset<N>> {
    Bitset<N> bs;
    std::bitset<N> ref;
    for (size_t p : positions) {
        bs.set(p);
        ref.set(p);
    }
    return {bs, ref};
}

template <size_t N>
auto expect_equal(const Bitset<N> &bs, const std::bitset<N> &ref) -> void {
    for (size_t i = 0; i < N; ++i) {
        INFO("bit " << i);
        CHECK(bs.test(i) == ref.test(i));
    }
    CHECK(bs.count() == ref.count());
}

} // namespace

// The ctor masks off bits beyond NumBits (kTopMask), so a partial top word never leaks stray high bits.
TEST_CASE("bitset_ctor_sanitizes_top") {
    const Bitset<10> b(0xFFFFULL);
    CHECK(b.count() == 10U);
    CHECK(b.word(0) == 0x3FFULL);
    const Bitset<64> full(~uint64_t{0});
    CHECK(full.count() == 64U);
}

TEST_CASE("bitset_set_test_word_boundaries") {
    auto [bs, ref] = make_pair<100>({0, 63, 64, 99});
    expect_equal<100>(bs, ref);
    CHECK(bs.test(63));
    CHECK(bs.test(64));
    CHECK(!bs.test(62));
    CHECK(!bs.test(65));
    CHECK(bs.count() == 4U);
}

TEST_CASE("bitset_count_and_parity_and_cross_word") {
    auto [a, ra] = make_pair<192>({1, 63, 64, 130, 191});
    auto [b, rb] = make_pair<192>({63, 64, 65, 130});
    const size_t expected = (ra & rb).count();
    CHECK(a.count_and(b) == expected);
    CHECK(a.parity_and(b) == ((expected & 1U) != 0U));
    auto [c, rc] = make_pair<192>({0, 2, 4});
    auto [d, rd] = make_pair<192>({1, 3, 5});
    CHECK(c.count_and(d) == 0U);
    CHECK(!c.parity_and(d));
}

TEST_CASE("bitset_not_respects_top_mask") {
    CHECK((~Bitset<100>{}).count() == 100U);
    CHECK((~Bitset<64>{}).count() == 64U);
    CHECK((~Bitset<10>{}).count() == 10U);
    auto [bs, ref] = make_pair<100>({3, 70, 99});
    CHECK((~~bs) == bs);
    (void)ref;
}

// Shift amounts cover exact word multiples, sub-word crossings, and >= NumBits (which must zero the set).
TEST_CASE("bitset_shift_right_cross_word") {
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
    // Single-word path (kNumWords == 1) takes a separate branch.
    auto [bs, ref] = make_pair<64>({0, 7, 31, 63});
    bs >>= 8;
    expect_equal<64>(bs, ref >> 8);
}

TEST_CASE("bitset_find_first_next_chain") {
    auto [bs, ref] = make_pair<192>({5, 63, 64, 130, 191});
    (void)ref;
    CHECK(bs.find_first() == 5U);
    CHECK(bs.find_next(5) == 63U);
    CHECK(bs.find_next(63) == 64U);
    CHECK(bs.find_next(64) == 130U);
    CHECK(bs.find_next(130) == 191U);
    CHECK(bs.find_next(191) == 192U); // past the last set bit -> NumBits
    CHECK(Bitset<192>{}.find_first() == 192U);
    // Single-word find_next branch.
    auto [sb, sref] = make_pair<64>({0, 40});
    (void)sref;
    CHECK(sb.find_first() == 0U);
    CHECK(sb.find_next(0) == 40U);
    CHECK(sb.find_next(40) == 64U);
}

// The multi-word hash must depend on which word carries a bit (the +i mix guard), and be deterministic.
TEST_CASE("bitset_splitmix_hash_position_sensitive") {
    Bitset<128> low;
    low.set(0);
    Bitset<128> high;
    high.set(64); // bit 0 of word 1 — same intra-word position as `low`'s bit
    const std::hash<Bitset<128>> h;
    CHECK(h(low) != h(high));
    CHECK(h(low) == h(low));
    Bitset<128> low_copy;
    low_copy.set(0);
    CHECK(h(low) == h(low_copy));
}

TEST_CASE("bitset_random_differential") {
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
        CHECK(a.count_and(b) == (ra & rb).count());
        CHECK((a == b) == (ra == rb));
    }
}
