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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h" // indices_to_bitset
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/OperatorIndex.h"
#include "monoprop/detail/parallel/Options.h"

using namespace monoprop;
using namespace monoprop::detail;

BOOST_AUTO_TEST_CASE(operator_index_term_index_is_32_bit) {
    static_assert(sizeof(TermIndex) == 4, "TermIndex is fixed at 32 bits; the packed row stores assume it");
    BOOST_TEST(sizeof(TermIndex) == 4u);
}

namespace {
constexpr size_t N = 32;
using Store = OperatorIndex<N>;
using MSet = Monomial<N>;

// Owners hold the store by unique_ptr and share stable pointers into it, so it must stay
// non-copyable and non-movable; clone() is the only deep copy.
static_assert(!std::is_move_constructible_v<Store>, "OperatorIndex must remain non-movable");
static_assert(!std::is_copy_constructible_v<Store>, "OperatorIndex must remain non-copyable");

MSet bs(const VecZ &r) {
    return indices_to_bitset<N>(r);
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

// The ceiling is declared, never materialised: 2^32 rows is hundreds of GiB, so every case here is
// arithmetic on the refusal path, which returns before a single row is allocated.
BOOST_AUTO_TEST_CASE(append_past_the_term_index_ceiling_is_refused_before_it_grows) {
    Store s;
    BOOST_CHECK_THROW(s.grow_rows_geometric(Store::kIndexCeiling + 1), TermIndexCeilingReached);
    BOOST_CHECK_EQUAL(s.size(), 0u);

    s.push_back(bs({0, 1}));
    // A store holding one term has room for kIndexCeiling - 1 more, so exactly kIndexCeiling is one too many.
    BOOST_CHECK_THROW(s.grow_rows_geometric(Store::kIndexCeiling), TermIndexCeilingReached);
    BOOST_CHECK_EQUAL(s.size(), 1u);
    BOOST_CHECK((s.row(0) == bs({0, 1})));

    // The count is checked as a subtraction, so a request that would wrap base + n is refused too.
    BOOST_CHECK_THROW(s.grow_rows_geometric(std::numeric_limits<size_t>::max()), TermIndexCeilingReached);
    BOOST_CHECK_EQUAL(s.size(), 1u);
}

// An empty append is legal at any size, including at the ceiling itself -- the last valid index is
// kIndexCeiling - 1, so a store of exactly kIndexCeiling terms is full, not over.
BOOST_AUTO_TEST_CASE(empty_append_never_throws_and_a_normal_append_still_grows) {
    Store s;
    BOOST_CHECK_NO_THROW(s.grow_rows_geometric(0));
    BOOST_CHECK_EQUAL(s.size(), 0u);
    s.push_back(bs({2}));
    BOOST_CHECK_NO_THROW(s.grow_rows_geometric(0));
    BOOST_CHECK_EQUAL(s.size(), 1u);
    const size_t base = s.grow_rows_geometric(2);
    BOOST_CHECK_EQUAL(base, 1u);
    BOOST_CHECK_EQUAL(s.size(), 3u);
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
    BOOST_TEST((s.row(0) == bs({0, 2, 4, 6})));
}

BOOST_AUTO_TEST_CASE(overflow_is_lossless_above_width) {
    Store s(2); // width 2; a 3-position row must overflow
    s.push_back(bs({0, 1, 2}));
    BOOST_TEST(s.popcount(0) == 3u); // popcount recovered from the overflow map
    BOOST_TEST((s.row(0) == bs({0, 1, 2})));
}

BOOST_AUTO_TEST_CASE(index_survives_rehash_in_place) {
    Store a;
    // 64 distinct rows (positions i and (i+7)%62) force at least one rehash of the in-place index.
    for (int i = 0; i < 64; ++i) {
        a.push_back(bs({static_cast<size_t>(i % 62), static_cast<size_t>((i + 7) % 62)}));
        a.emplace(a.row(static_cast<size_t>(i)), static_cast<size_t>(i));
    }
    auto f = a.find(a.row(50));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 50u);
}

BOOST_AUTO_TEST_CASE(clone_is_deep_and_independent) {
    Store a(4); // non-default width must carry over
    a.push_back(bs({0, 3, 5}));
    a.emplace(bs({0, 3, 5}), 0);
    a.push_back(bs({1, 2}));
    a.emplace(bs({1, 2}), 1);

    auto b = a.clone();
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST((b->row(0) == bs({0, 3, 5})));
    auto f = b->find(bs({1, 2}));
    BOOST_TEST(f.has_value());
    BOOST_TEST(*f == 1u);

    a.push_back(bs({6, 7}));
    a.emplace(bs({6, 7}), 2);
    BOOST_TEST(b->size() == 2u);
    BOOST_TEST(!b->find(bs({6, 7})).has_value());

    // If the clone still referenced the source's rows, this find would read a->row(0) (now {8,9})
    // and fail.
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

// find_batch (the group-prefetch pipelined lookup) must be semantically identical to n independent
// find() calls. The query mix below spans several G=16 groups plus a short tail and interleaves
// present and absent keys, so every branch but the h32-collision fallback runs; that one needs a
// real 32-bit hash collision, but the equivalence assertion pins it whichever path a key takes.
BOOST_AUTO_TEST_CASE(find_batch_matches_scalar_find) {
    Store s;
    constexpr size_t kRows = 200; // > 12 groups of G=16
    // (i/60, 4 + i%60) is a bijection for i < 240 over the disjoint ranges {0..3} and {4..63}.
    for (size_t i = 0; i < kRows; ++i) {
        const auto key = bs({i / 60, 4 + (i % 60)});
        s.push_back(key);
        s.emplace(key, i);
    }

    std::vector<MSet> queries;
    for (size_t i = 0; i < kRows; ++i) {
        queries.push_back(bs({i / 60, 4 + (i % 60)}));
        queries.push_back(bs({0, 1, 2 + (i % 20)}));
    }
    queries.push_back(bs({0, 1, 2})); // 401 total
    BOOST_TEST(queries.size() % 16u != 0u);

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
    BOOST_TEST(out[0] == 0u);               // first present key -> row 0
    BOOST_TEST(out[1] == Store::kNotFound); // first absent key
}

// Pins find_batch's partition.count == 0 early-out.
BOOST_AUTO_TEST_CASE(find_batch_on_empty_store_is_all_missing) {
    Store s;
    const std::array<MSet, 3> keys{bs({0, 3}), bs({1, 2}), bs({4, 5, 6})};
    std::array<size_t, 3> out{0, 0, 0};
    s.find_batch(keys.data(), keys.size(), out.data());
    for (size_t i = 0; i < keys.size(); ++i) {
        BOOST_TEST(out[i] == Store::kNotFound);
    }
}

// Shared invariant 14, pinned: bulk insertion does not deduplicate. Two rows holding the same key are both
// indexed, and a lookup returns the first one inserted (it sits earlier on the probe chain). Production never
// relies on this -- query keys are distinct by construction -- but a replacement index must either keep it or
// change it as an explicit, reviewed contract change. The options overload is the Task 11 publication seam;
// the packed index ignores it and publishes serially, identically.
BOOST_AUTO_TEST_CASE(bulk_insert_duplicate_keys_are_both_indexed) {
    const auto dup = bs({2, 7, 9});
    const auto other = bs({1, 4});
    const std::array<MSet, 3> keys{dup, other, dup};
    for (const int threads : {1, 4}) {
        for (const bool hashed : {false, true}) {
            Store s;
            const size_t base = s.grow_rows_geometric(keys.size());
            for (size_t k = 0; k < keys.size(); ++k) {
                s.set(base + k, keys[k]);
            }
            if (hashed) {
                s.bulk_insert_hashed(
                    keys.size(),
                    base,
                    [&](size_t k) {
                        std::vector<Store::PosT> pos;
                        s.for_each_position(base + k, [&](size_t b) { pos.push_back(static_cast<Store::PosT>(b)); });
                        return Store::fold_hash_positions(pos);
                    },
                    parallel::Options{.threads = threads});
            }
            else {
                s.bulk_insert(keys.size(), base, [&](size_t k) -> const MSet & { return keys[k]; });
            }
            BOOST_TEST_CONTEXT("threads=" << threads << " hashed=" << hashed) {
                BOOST_TEST(s.size() == 3u);
                size_t indexed = 0;
                std::vector<size_t> rows_for_dup;
                s.for_each([&](const MSet &key, size_t row) {
                    ++indexed;
                    if (key == dup) {
                        rows_for_dup.push_back(row);
                    }
                });
                BOOST_TEST(indexed == 3u);
                std::ranges::sort(rows_for_dup);
                BOOST_TEST((rows_for_dup == std::vector<size_t>{0, 2}));
                BOOST_REQUIRE(s.find(dup).has_value());
                BOOST_TEST(*s.find(dup) == 0u);
                BOOST_TEST(*s.find(other) == 1u);
            }
        }
    }
}

// The frozen probe slices one shared position buffer into blocks: absolute offsets, sliced counts and outputs.
// A sliced call must give exactly what the whole call gives for the same queries, hashes included.
BOOST_AUTO_TEST_CASE(find_batch_positions_on_sliced_spans_matches_the_whole_call) {
    Store s;
    for (size_t i = 0; i < 120; ++i) {
        const auto key = bs({i / 60, 4 + (i % 60)});
        s.push_back(key);
        s.emplace(key, i);
    }
    std::vector<Store::PosT> flat{63, 63, 63}; // unreferenced prefix: offsets are absolute
    std::vector<size_t> off;
    std::vector<uint32_t> k_of;
    for (size_t i = 0; i < 200; ++i) {
        const auto key = (i % 3 == 0) ? bs({0, 1, 2 + (i % 20)}) : bs({(i / 60) % 2, 4 + (i % 60)});
        off.push_back(flat.size());
        uint32_t k = 0;
        for (size_t b = key.find_first(); b < key.size(); b = key.find_next(b)) {
            flat.push_back(static_cast<Store::PosT>(b));
            ++k;
        }
        k_of.push_back(k);
    }
    std::vector<size_t> whole(off.size());
    std::vector<uint32_t> whole_hash(off.size());
    s.find_batch_positions(flat, off, k_of, whole, whole_hash);
    std::vector<size_t> sliced(off.size(), 7);
    std::vector<uint32_t> sliced_hash(off.size(), 7);
    for (size_t lo = 0; lo < off.size(); lo += 37) {
        const size_t m = std::min<size_t>(37, off.size() - lo);
        s.find_batch_positions(flat,
                               std::span<const size_t>(off).subspan(lo, m),
                               std::span<const uint32_t>(k_of).subspan(lo, m),
                               std::span<size_t>(sliced).subspan(lo, m),
                               std::span<uint32_t>(sliced_hash).subspan(lo, m));
    }
    BOOST_TEST(sliced == whole, boost::test_tools::per_element());
    BOOST_TEST(sliced_hash == whole_hash, boost::test_tools::per_element());
    BOOST_TEST(std::ranges::count(whole, Store::kNotFound) > 0);
    BOOST_TEST(std::ranges::count_if(whole, [](size_t v) { return v != Store::kNotFound; }) > 0);
}
