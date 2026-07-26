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

#include <boost/test/unit_test.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;
using namespace monoprop::detail;

// Preserved from the former index_map_tests.cpp.
BOOST_AUTO_TEST_CASE(operator_index_term_index_width_matches_build) {
#if defined(monoprop_WIDE_TERM_INDEX)
    static_assert(sizeof(TermIndex) == 8, "wide build must use 64-bit TermIndex");
    BOOST_TEST(sizeof(TermIndex) == 8u);
#else
    static_assert(sizeof(TermIndex) == 4, "default build must use 32-bit TermIndex");
    BOOST_TEST(sizeof(TermIndex) == 4u);
#endif
}

namespace {
constexpr size_t N = 32;
using Store = OperatorIndex<N>;
using MSet = Monomial<N>;

// The store is non-copyable and non-movable: owners hold it by unique_ptr and share stable
// pointers to it, and clone() is the only deep copy. Lock this design invariant at compile time.
static_assert(!std::is_move_constructible_v<Store>, "OperatorIndex must remain non-movable");
static_assert(!std::is_copy_constructible_v<Store>, "OperatorIndex must remain non-copyable");

MSet bs(const VecZ &r) {
    return indices_to_bitset<N>(r);
}

// A bijection from [0, 21^3) to distinct 3-position monomials: three disjoint position bands inside
// the 2N = 64 valid indices, so every key is genuinely distinct and stays inline at any width >= 3.
MSet key3(size_t i) {
    return bs({i % 21, 21 + (i / 21) % 21, 42 + (i / 441) % 21});
}
} // namespace

BOOST_AUTO_TEST_CASE(rows_roundtrip_dense_popcount_positions) {
    Store s;
    s.push_back(bs({0, 3, 5}));
    s.push_back(bs({1, 2}));
    BOOST_TEST(s.size() == 2u);
    BOOST_TEST(s.popcount(0) == 3u);
    BOOST_TEST(s.popcount(1) == 2u);
    BOOST_TEST((s.row(0) == bs({0, 3, 5})));
    std::vector<size_t> pos;
    s.for_each_position(0, [&](size_t b) { pos.push_back(b); });
    BOOST_TEST(pos.size() == 3u);
    // for_each_position yields raw bit positions (ascending). indices_to_bitset<32>({0,3,5})
    // sets bits at 2*32-1-0=63, 2*32-1-3=60, 2*32-1-5=58, so find_first gives 58 first.
    BOOST_TEST(pos[0] == 58u);
    BOOST_TEST(pos[2] == 63u);
}

BOOST_AUTO_TEST_CASE(index_emplace_then_find_roundtrip) {
    Store s;
    s.push_back(bs({0, 3, 5}));
    s.emplace(bs({0, 3, 5}), 0);
    s.push_back(bs({1, 2}));
    s.emplace(bs({1, 2}), 1);
    auto f = s.find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);
    BOOST_TEST(!s.find(bs({7, 9})).has_value());
}

BOOST_AUTO_TEST_CASE(width_is_a_construction_invariant) {
    Store s(4);                    // stride = 1 + 4, fixed at construction
    s.push_back(bs({0, 2, 4, 6})); // a 4-position row fits inline at width 4
    s.reserve(20);                 // capacity only -- width/stride are never touched by reserve
    BOOST_TEST(s.popcount(0) == 4u);
    BOOST_TEST((s.row(0) == bs({0, 2, 4, 6}))); // round-trips inline after reserve
}

BOOST_AUTO_TEST_CASE(overflow_is_lossless_above_width) {
    Store s(2); // width 2; a 3-position row must overflow
    s.push_back(bs({0, 1, 2}));
    BOOST_TEST(s.popcount(0) == 3u);         // popcount recovered from the overflow map
    BOOST_TEST((s.row(0) == bs({0, 1, 2}))); // and the full row round-trips losslessly
}

// The store is intentionally non-movable (owners hold it by unique_ptr). The former
// `index_survives_store_move` case exercised a move that no longer exists by design; index
// integrity in its final, stable location is covered by the find/emplace round-trip below.
BOOST_AUTO_TEST_CASE(index_survives_rehash_in_place) {
    Store a;
    // Insert 64 distinct rows (varying both positions) to force >=1 rehash in the flat_set.
    // Using i and i+7 (mod 62) as positions; since 64 > 31, we vary the second axis too so
    // all 64 monomials are distinct.
    for (int i = 0; i < 64; ++i) {
        a.push_back(bs({static_cast<size_t>(i % 62), static_cast<size_t>((i + 7) % 62)}));
        a.emplace(a.row(static_cast<size_t>(i)), static_cast<size_t>(i));
    }
    auto f = a.find(a.row(50)); // find confirms against a's own rows after rehash
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 50u);
}

// clone() is the only deep-copy entry point: the store stays non-copyable/non-movable (the
// static_asserts above), so clone() must hand back a fresh, fully independent heap store whose
// index confirms against the CLONE's own rows, not the source's.
BOOST_AUTO_TEST_CASE(clone_is_deep_and_independent) {
    Store a(4); // non-default width must carry over
    a.push_back(bs({0, 3, 5}));
    a.emplace(bs({0, 3, 5}), 0);
    a.push_back(bs({1, 2}));
    a.emplace(bs({1, 2}), 1);

    auto b = a.clone(); // std::unique_ptr<Store>
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST((b->row(0) == bs({0, 3, 5})));
    auto f = b->find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);

    // Deep independence: growing the source must not touch the clone.
    a.push_back(bs({6, 7}));
    a.emplace(bs({6, 7}), 2);
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST(!b->find(bs({6, 7})).has_value());

    // Store locality: corrupting the SOURCE's row 0 must not perturb the clone's find, which
    // confirms against the CLONE's own rows. If the clone still referenced the source's rows,
    // this find would read a->row(0) (now {8,9}) and fail.
    a.set(0, bs({8, 9}));
    auto g = b->find(bs({0, 3, 5}));
    BOOST_TEST(g.has_value());
    BOOST_TEST(*g == 0u);
}

BOOST_AUTO_TEST_CASE(clone_preserves_overflow_rows) {
    Store a(2); // width 2; a 3-position row overflows losslessly
    a.push_back(bs({0, 1, 2}));
    a.emplace(bs({0, 1, 2}), 0);

    auto b = a.clone();
    BOOST_TEST(b->popcount(0) == 3u);
    BOOST_TEST((b->row(0) == bs({0, 1, 2})));
    BOOST_TEST(*b->find(bs({0, 1, 2})) == 0u);
}

// find_batch is the group-prefetch pipelined lookup used by the resolve phases. It must be
// semantically identical to n independent find() calls: out[i] = the row index of keys[i], or
// kNotFound. This drives a query mix that spans multiple G=16 groups plus a non-multiple tail, and
// interleaves present (2-position) and absent (3-position) keys so every branch runs — hit, empty
// slot (kNotFound), and the confirm step. The h32-collision fallback is not deterministically
// reachable in a unit test (it needs a 32-bit hash collision), but the equivalence assertion pins
// its observable behavior whichever path a given key takes.
BOOST_AUTO_TEST_CASE(find_batch_matches_scalar_find) {
    Store s;
    constexpr size_t kRows = 200; // > 12 groups of G=16
    // Distinct 2-position rows: (i/60, 4 + i%60) is a bijection for i < 240 with disjoint position
    // ranges {0..3} and {4..63}, so all rows differ and are genuine 2-position monomials.
    for (size_t i = 0; i < kRows; ++i) {
        const auto key = bs({i / 60, 4 + (i % 60)});
        s.push_back(key);
        s.emplace(key, i);
    }

    // Interleave each present key with an absent 3-position key (never inserted -> always missing),
    // then one trailing absent key so the total is not a multiple of G=16 and the final short group
    // (tail) path runs too.
    std::vector<MSet> queries;
    for (size_t i = 0; i < kRows; ++i) {
        queries.push_back(bs({i / 60, 4 + (i % 60)}));
        queries.push_back(bs({0, 1, 2 + (i % 20)}));
    }
    queries.push_back(bs({0, 1, 2}));       // 401 total
    BOOST_TEST(queries.size() % 16u != 0u); // a genuine short tail group

    std::vector<size_t> out(queries.size(), 424242);
    s.find_batch(queries.data(), queries.size(), out.data());

    bool all_match = true;
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto scalar = s.find(queries[i]);
        const size_t expected = scalar ? *scalar : Store::kNotFound;
        if (out[i] != expected) {
            all_match = false;
        }
    }
    BOOST_TEST(all_match);
    // Spot-check the two kinds explicitly.
    BOOST_TEST(out[0] == 0u);               // first present key -> row 0
    BOOST_TEST(out[1] == Store::kNotFound); // first absent key
}

// ---------------------------------------------------------------------------------------------
// Split (index + 1-byte tag) table. The table no longer stores a reconstructible 32-bit hash, so a
// rehash re-derives every surviving entry's hash from its row. These cases pin that path.
// ---------------------------------------------------------------------------------------------

// The slot is one TermIndex plus one tag byte, so the index arrays must measure exactly
// slot_count * (sizeof(TermIndex) + 1) bytes at a power-of-two slot count. Width-agnostic: this
// holds for both the default u32 and the -Dmonoprop_WIDE_TERM_INDEX u64 build.
BOOST_AUTO_TEST_CASE(index_slot_is_term_index_plus_one_tag_byte) {
    constexpr size_t kSlotBytes = sizeof(TermIndex) + 1;
    Store s;
    for (size_t n : {size_t{0}, size_t{100}, size_t{5000}}) {
        for (size_t i = s.size(); i < n; ++i) {
            s.push_back(key3(i));
            s.emplace(key3(i), i);
        }
        const size_t array_bytes = s.index_estimated_memory_bytes() - sizeof(Store);
        BOOST_TEST(array_bytes % kSlotBytes == 0u);
        const size_t slots = array_bytes / kSlotBytes;
        BOOST_TEST(std::has_single_bit(slots));
        BOOST_TEST(slots * 7u >= s.size() * 10u); // load factor stayed at or under 0.7
    }
}

// Many rehashes, each re-deriving hashes from rows: every key must still be findable, and find_batch
// must still agree with find (a 1/256 tag collision resolves through the exact-find fallback).
BOOST_AUTO_TEST_CASE(rehash_rederives_hashes_from_rows) {
    constexpr size_t kRows = 4000; // 16 -> 8192 slots: nine doublings
    Store s;
    std::vector<MSet> keys;
    keys.reserve(kRows);
    for (size_t i = 0; i < kRows; ++i) {
        keys.push_back(key3(i));
        s.push_back(keys.back());
        s.emplace(keys.back(), i); // emplace drives rehash_if_needed one entry at a time
    }
    bool all_found = true;
    for (size_t i = 0; i < kRows; ++i) {
        const auto f = s.find(keys[i]);
        if (!f || *f != i) {
            all_found = false;
        }
    }
    BOOST_TEST(all_found);

    std::vector<size_t> out(kRows, 424242);
    s.find_batch(keys.data(), kRows, out.data());
    bool batch_agrees = true;
    for (size_t i = 0; i < kRows; ++i) {
        if (out[i] != i) {
            batch_agrees = false;
        }
    }
    BOOST_TEST(batch_agrees);
}

// row_hash() of an overflow row must go through the side-map, not the (kOverflowMarker) header.
// Width 1 sends every 2-position row to overflow, so this rehashes a table of overflow-only rows.
BOOST_AUTO_TEST_CASE(rehash_rederives_hashes_of_overflow_rows) {
    constexpr size_t kRows = 300;
    Store s(1); // width 1: any 2-position row overflows
    for (size_t i = 0; i < kRows; ++i) {
        const auto key = bs({i % 30, 30 + (i / 30) % 30});
        s.push_back(key);
        s.emplace(key, i);
    }
    bool ok = true;
    for (size_t i = 0; i < kRows; ++i) {
        const auto key = bs({i % 30, 30 + (i / 30) % 30});
        const auto f = s.find(key);
        if (!f || *f != i || s.popcount(i) != 2u || !(s.row(i) == key)) {
            ok = false;
        }
    }
    BOOST_TEST(ok);
    // ... and the clone of an all-overflow store re-derives them against its OWN rows.
    const auto c = s.clone();
    BOOST_TEST(*c->find(bs({7, 37})) == 217u); // i=217 -> {217%30=7, 30+(217/30)%30=37}
}

// bulk_insert right-sizes the table up front, then inserts without rehashing. Driven in successive
// "layers" so the reserve path is exercised against a table that already holds entries.
BOOST_AUTO_TEST_CASE(bulk_insert_across_layers_finds_every_key) {
    Store s;
    const auto key_of = [](size_t i) { return key3(i); };
    size_t total = 0;
    for (size_t layer_n : {size_t{1}, size_t{17}, size_t{500}, size_t{2500}}) {
        const size_t base = s.grow_rows_geometric(layer_n);
        BOOST_TEST(base == total);
        for (size_t k = 0; k < layer_n; ++k) {
            s.set(base + k, key_of(base + k));
        }
        s.bulk_insert(layer_n, base, [&](size_t k) { return key_of(base + k); });
        total += layer_n;
    }
    BOOST_TEST(s.size() == total);
    bool ok = true;
    for (size_t i = 0; i < total; ++i) {
        const auto f = s.find(key_of(i));
        if (!f || *f != i) {
            ok = false;
        }
    }
    BOOST_TEST(ok);
    BOOST_TEST(!s.find(bs({0, 1, 2})).has_value()); // never inserted
}

// An empty store must report every key missing (find_batch's shard.count == 0 early-out).
BOOST_AUTO_TEST_CASE(find_batch_on_empty_store_is_all_missing) {
    Store s;
    const std::array<MSet, 3> keys{bs({0, 3}), bs({1, 2}), bs({4, 5, 6})};
    std::array<size_t, 3> out{0, 0, 0};
    s.find_batch(keys.data(), keys.size(), out.data());
    for (size_t i = 0; i < keys.size(); ++i) {
        BOOST_TEST(out[i] == Store::kNotFound);
    }
}
