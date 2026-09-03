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

// SparseRowStore's own invariants. The three-way agreement with the dense and packed backends through
// the RowAccess.h accessors lives in row_accessor_tests.cpp; what is checked here is the part that has
// no counterpart in the other backends -- the codes word, and the row sizing that feeds it.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/operator/SparseRowStore.h"

#include "RandomMonomial.h"

using namespace monoprop;
using namespace monoprop::detail;

// Interchangeable with OperatorIndex means the same ownership rules, so the store cannot be silently
// copied out of MPOperator's unique_ptr.
static_assert(!std::is_move_constructible_v<SparseRowStore<32>>, "SparseRowStore must remain non-movable");
static_assert(!std::is_copy_constructible_v<SparseRowStore<32>>, "SparseRowStore must remain non-copyable");
// A mode lane addresses at most kMaxModes modes, and the widest store the suite builds must fit.
static_assert(SparseRowStore<SparseRowStore<32>::kMaxModes>::kMaxModes > 0, "kMaxModes must be instantiable");

namespace {

// The support-form identities, spelled out here against the dense cutoff_sums.
auto sums_from_codes(RowCodes codes) -> CutoffSums {
    const auto n = static_cast<size_t>(std::popcount(row_occupied_bits(codes)));
    const auto d = static_cast<size_t>(std::popcount(row_paired_bits(codes)));
    return {n - d, n + d, n};
}

// Owning lanes plus codes, so a test can hold a row the way the scan's scratch does. The lanes come out
// ascending and contiguous from slot 0, which is what the representation requires of every producer.
struct RowBuffer {
    std::vector<RowMode> lanes;
    RowCodes codes = 0;

    [[nodiscard]] auto view() const -> SparseRow { return SparseRow{lanes.data(), codes}; }
};

// Only for rows that fit one codes word (<= kRowMaxSlots slots) -- the shape a SparseRow can hold at all.
template <size_t NumBits>
auto row_of(const Bitset<NumBits> &mono) -> RowBuffer {
    RowBuffer out;
    for_each_mode_slot(mono, [&](size_t mode, unsigned int code) {
        out.codes |= static_cast<RowCodes>(code) << (2 * out.lanes.size());
        out.lanes.push_back(static_cast<RowMode>(mode));
    });
    return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_row_store_codes_encode_slot_pairs) {
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> store(8);

    // Modes 1 (both positions), 4 (upper only) and 9 (lower only), so the codes word must read
    // 0b11, 0b10, 0b01 from slot 0 up.
    Monomial<kNumModes> mono;
    mono.set(2);
    mono.set(3);
    mono.set(9);
    mono.set(18);
    store.push_back(mono);

    BOOST_TEST(!store.spilled(0));
    // Slot 0 = mode 1 (0b11) in bits 0-1, slot 1 = mode 4 (0b10) in bits 2-3, slot 2 = mode 9 (0b01)
    // in bits 4-5.
    BOOST_TEST(store.codes(0) == 0b01'10'11ULL);
    BOOST_TEST(store.slot_count(0) == 3U);
    BOOST_TEST(store.popcount(0) == 4U);

    std::vector<size_t> modes;
    std::vector<unsigned int> codes;
    store.for_each_slot(0, [&](size_t mode, unsigned int code) {
        modes.push_back(mode);
        codes.push_back(code);
    });
    BOOST_REQUIRE(modes.size() == 3U);
    BOOST_TEST(modes == (std::vector<size_t>{1U, 4U, 9U}), boost::test_tools::per_element());
    BOOST_TEST(codes == (std::vector<unsigned int>{0b11U, 0b10U, 0b01U}), boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(sparse_row_store_empty_row_has_empty_codes) {
    SparseRowStore<32> store(4);
    store.push_back(Monomial<32>{});
    BOOST_TEST(!store.spilled(0));
    BOOST_TEST(store.codes(0) == 0U);
    BOOST_TEST(store.slot_count(0) == 0U);
    BOOST_TEST(store.popcount(0) == 0U);
    BOOST_TEST(store.row(0) == Monomial<32>{});
}

namespace {

// The identities the whole support form rests on: or_sum = n, popcount_sum = n + d, xor_sum = n - d,
// against the dense cutoff_sums over the full storage window. Randomized rather than enumerated
// because what can break them is a particular occupancy pattern, not a particular width.
template <size_t NumModes>
auto check_codes_reproduce_cutoff_sums(std::mt19937_64 &rng) -> void {
    // Capacity above any row built below, so nothing spills and every row exercises the codes path.
    SparseRowStore<NumModes> store(SparseRowStore<NumModes>::kMaxSlots);
    for (size_t trial = 0; trial < 200; ++trial) {
        const auto mono = test_utils::random_monomial<NumModes>(rng, SparseRowStore<NumModes>::kMaxSlots);
        store.push_back(mono);
        const size_t i = store.size() - 1;
        BOOST_REQUIRE(!store.spilled(i));

        const auto dense = cutoff_sums<NumModes>(mono, NumModes);
        const auto sparse = sums_from_codes(store.codes(i));
        BOOST_TEST(sparse.or_sum == dense.or_sum);
        BOOST_TEST(sparse.popcount_sum == dense.popcount_sum);
        BOOST_TEST(sparse.xor_sum == dense.xor_sum);
        BOOST_TEST(store.slot_count(i) == dense.or_sum);
        BOOST_TEST(store.popcount(i) == dense.popcount_sum);
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_row_store_codes_reproduce_cutoff_sums) {
    std::mt19937_64 rng(20260812U);
    check_codes_reproduce_cutoff_sums<32>(rng);
    check_codes_reproduce_cutoff_sums<64>(rng);
    check_codes_reproduce_cutoff_sums<128>(rng);
    check_codes_reproduce_cutoff_sums<512>(rng);
}

// Spilled rows have no codes word, so the two measures must still come off the dense monomial.
BOOST_AUTO_TEST_CASE(sparse_row_store_spilled_rows_report_the_same_measures) {
    constexpr size_t kNumModes = 64;
    SparseRowStore<kNumModes> store(2);
    Monomial<kNumModes> mono;
    for (const size_t b : {0U, 1U, 4U, 20U, 21U, 99U}) { // modes 0 (paired), 2, 10 (paired), 49
        mono.set(b);
    }
    store.push_back(mono);

    BOOST_TEST(store.spilled(0));
    BOOST_TEST(store.row(0) == mono);
    const auto dense = cutoff_sums<kNumModes>(mono, kNumModes);
    BOOST_TEST(store.slot_count(0) == dense.or_sum);
    BOOST_TEST(store.popcount(0) == dense.popcount_sum);
}

BOOST_AUTO_TEST_CASE(sparse_row_store_clone_preserves_rows_and_spills) {
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> store(2);
    Monomial<kNumModes> inline_row;
    inline_row.set(4);
    inline_row.set(5);
    Monomial<kNumModes> spilled_row;
    for (const size_t b : {0U, 6U, 10U, 30U}) {
        spilled_row.set(b);
    }
    store.push_back(inline_row);
    store.push_back(spilled_row);

    const auto copy = store.clone();
    BOOST_REQUIRE(copy->size() == 2U);
    BOOST_TEST(copy->num_bits() == 2 * kNumModes);
    BOOST_TEST(copy->slots_per_row() == 2U);
    BOOST_TEST(!copy->spilled(0));
    BOOST_TEST(copy->codes(0) == store.codes(0));
    BOOST_TEST(copy->row(0) == inline_row);
    BOOST_TEST(copy->spilled(1));
    BOOST_TEST(copy->row(1) == spilled_row);
}

// resized() is the row-width migration: every row keeps its index and its content, whichever way the
// width moved -- including a row that crosses the overflow boundary, since set() re-decides that per
// row rather than trusting the old classification.
BOOST_AUTO_TEST_CASE(sparse_row_store_resized_preserves_rows_when_widening) {
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> store(2); // width 2: the 3-slot row below starts spilled
    Monomial<kNumModes> inline_row;
    inline_row.set(4);
    inline_row.set(5);
    Monomial<kNumModes> spilled_row;
    for (const size_t b : {0U, 6U, 10U, 30U}) {
        spilled_row.set(b);
    }
    store.push_back(inline_row);
    store.push_back(spilled_row);

    const auto wide = store.resized(4); // now fits inline
    BOOST_REQUIRE(wide->size() == 2U);
    BOOST_TEST(wide->slots_per_row() == 4U);
    BOOST_TEST(!wide->spilled(0));
    BOOST_TEST(wide->row(0) == inline_row);
    BOOST_TEST(!wide->spilled(1));
    BOOST_TEST(wide->row(1) == spilled_row);

    // Independent of the source: mutating store after the fact must not reach wide.
    store.set(0, Monomial<kNumModes>{});
    BOOST_TEST(wide->row(0) == inline_row);
}

BOOST_AUTO_TEST_CASE(sparse_row_store_resized_preserves_rows_when_narrowing) {
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> store(4); // width 4: both rows below fit inline
    Monomial<kNumModes> a;
    a.set(4);
    a.set(5);
    Monomial<kNumModes> b;
    for (const size_t bit : {0U, 6U, 10U, 30U}) {
        b.set(bit);
    }
    store.push_back(a);
    store.push_back(b);
    BOOST_REQUIRE(!store.spilled(1));

    const auto narrow = store.resized(2); // row 1 must now spill
    BOOST_REQUIRE(narrow->size() == 2U);
    BOOST_TEST(narrow->slots_per_row() == 2U);
    BOOST_TEST(!narrow->spilled(0));
    BOOST_TEST(narrow->row(0) == a);
    BOOST_TEST(narrow->spilled(1));
    BOOST_TEST(narrow->row(1) == b);
}

// K comes from the cutoff in modes, and is the same number for both cutoff kinds -- halving the slot
// bound for a support cutoff would truncate rows a length cutoff of the same size admits.
BOOST_AUTO_TEST_CASE(sparse_row_store_slots_come_from_the_cutoff_in_modes) {
    constexpr size_t kNumModes = 32;
    using Store = SparseRowStore<kNumModes>;

    CutoffFn<kNumModes> length = LengthCutoff<kNumModes>{.cutoff = 6U};
    CutoffFn<kNumModes> support = SupportCutoff<kNumModes>{.cutoff = 6U};
    BOOST_TEST(CutoffEvaluator<kNumModes>(length).max_slot_bound().value() == 6U);
    BOOST_TEST(CutoffEvaluator<kNumModes>(support).max_slot_bound().value() == 12U);

    BOOST_TEST(Store::slots_for_bound(6U) == 6U);
    // A bound past one codes word clamps rather than throwing: the excess rows spill.
    BOOST_TEST(Store::slots_for_bound(100U) == Store::kMaxSlots);
    BOOST_TEST(Store::slots_for_bound(0U) == 1U);
}

// The switch rule itself: a build-time constant, because the crossover follows the target ISA rather
// than anything known at run time. The values are the measured crossovers, so what this pins is that
// the CMake default reached the compiler at all -- a missing definition would silently fall back.
BOOST_AUTO_TEST_CASE(sparse_row_store_preference_threshold_matches_the_build) {
    using Store = SparseRowStore<32>;
    static_assert(Store::kMinModes > 0, "the sparse crossover must be a positive mode count");
    BOOST_TEST(!Store::preferred_for_modes(Store::kMinModes - 1));
    BOOST_TEST(Store::preferred_for_modes(Store::kMinModes));
    BOOST_TEST(Store::preferred_for_modes(Store::kMinModes + 1));
    // 32 modes is where sparse measured 1.9x behind dense even on baseline x86-64; no build should be
    // switching there.
    BOOST_TEST(!Store::preferred_for_modes(32U));
}

namespace {

// The row form of set() against the dense one, which is the only definition of what it must produce.
// Both fill the same row of two stores built alike; every observable of the two must agree, including
// the hash the table keys on -- a row written one way has to be findable by a key written the other.
template <size_t NumModes>
auto check_row_form_set_matches_dense(std::mt19937_64 &rng) -> void {
    using Store = SparseRowStore<NumModes>;
    using Key = SparseRowKey<2 * NumModes>;
    Store dense_written(Store::kMaxSlots);
    Store row_written(Store::kMaxSlots);
    for (size_t trial = 0; trial < 200; ++trial) {
        const auto mono = test_utils::random_monomial<NumModes>(rng, Store::kMaxSlots);
        const RowBuffer row = row_of(mono);

        dense_written.push_back(mono);
        const size_t i = row_written.grow_rows_geometric(1);
        row_written.set(i, row.view());

        BOOST_REQUIRE(i == dense_written.size() - 1);
        BOOST_TEST(row_written.spilled(i) == dense_written.spilled(i));
        BOOST_TEST(row_written.codes(i) == dense_written.codes(i));
        BOOST_TEST(row_written.slot_count(i) == dense_written.slot_count(i));
        BOOST_TEST(row_written.popcount(i) == dense_written.popcount(i));
        BOOST_TEST(row_written.row(i) == mono);
        // The three key forms are interchangeable only if they hash alike; the table's probe order
        // (and so MPI owner routing) is downstream of this.
        BOOST_TEST(sparse_row_hash(row.view()) == sparse_row_hash(mono));
        BOOST_TEST(sparse_row_hash(Key{.row = row.view()}) == sparse_row_hash(mono));
        BOOST_TEST(sparse_row_hash(Key{.spilled = &mono}) == sparse_row_hash(mono));
    }
    // Written rows are findable by either form, whichever way they went in.
    for (size_t i = 0; i < row_written.size(); ++i) {
        const auto mono = row_written.row(i);
        row_written.emplace(row_of(mono).view(), i);
    }
    for (size_t i = 0; i < row_written.size(); ++i) {
        const auto mono = row_written.row(i);
        const RowBuffer row = row_of(mono);
        const auto by_mono = row_written.find(mono);
        const auto by_row = row_written.find(row.view());
        const auto by_key = row_written.find(Key{.row = row.view()});
        BOOST_REQUIRE(by_mono.has_value());
        BOOST_TEST((by_row == by_mono));
        BOOST_TEST((by_key == by_mono));
        // Distinct monomials may repeat across trials, so the found row must equal this one rather
        // than be this index.
        BOOST_TEST(row_written.row(*by_mono) == mono);
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_row_store_row_form_set_matches_the_dense_one) {
    std::mt19937_64 rng(20260813U);
    check_row_form_set_matches_dense<32>(rng);
    check_row_form_set_matches_dense<96>(rng);
    check_row_form_set_matches_dense<512>(rng);
}

// A row wider than the store's capacity has to spill, exactly as the dense set() spills it. The capacity
// is sized from the cutoff and a fully paired term escapes the cutoff, so this arm is reachable by
// construction and cannot be sized away.
BOOST_AUTO_TEST_CASE(sparse_row_store_row_form_set_spills_a_row_past_the_capacity) {
    constexpr size_t kNumModes = 64;
    using Key = SparseRowKey<2 * kNumModes>;
    SparseRowStore<kNumModes> store(2);
    Monomial<kNumModes> mono;
    for (const size_t b : {0U, 1U, 4U, 20U, 21U, 99U}) { // modes 0 (paired), 2, 10 (paired), 49
        mono.set(b);
    }
    const RowBuffer row = row_of(mono);
    BOOST_REQUIRE(row.lanes.size() == 4U);

    const size_t i = store.grow_rows_geometric(1);
    store.set(i, row.view());
    BOOST_TEST(store.spilled(i));
    BOOST_TEST(store.row(i) == mono);
    BOOST_TEST(store.slot_count(i) == 4U);
    BOOST_TEST(store.popcount(i) == 6U);

    // A spilled row is found by the dense key; the row key finds it through either shape it carries.
    store.emplace(mono, i);
    BOOST_TEST((store.find(mono) == std::optional<size_t>(i)));
    BOOST_TEST((store.find(Key{.spilled = &mono}) == std::optional<size_t>(i)));
    BOOST_TEST((store.find(Key{.row = row.view()}) == std::optional<size_t>(i)));
}

// Rows are overwritten in place by the miss inserts, so a row must not inherit the previous occupant's
// spill -- in either direction.
BOOST_AUTO_TEST_CASE(sparse_row_store_row_form_set_clears_a_stale_spill) {
    constexpr size_t kNumModes = 32;
    SparseRowStore<kNumModes> store(2);
    Monomial<kNumModes> wide;
    for (const size_t b : {0U, 6U, 10U, 30U}) {
        wide.set(b);
    }
    Monomial<kNumModes> narrow;
    narrow.set(4);
    narrow.set(5);

    const size_t i = store.grow_rows_geometric(1);
    store.set(i, row_of(wide).view());
    BOOST_REQUIRE(store.spilled(i));

    store.set(i, row_of(narrow).view());
    BOOST_TEST(!store.spilled(i));
    BOOST_TEST(store.row(i) == narrow);
    BOOST_TEST(store.codes(i) == 0b11ULL); // one slot, mode 2, both positions

    store.set(i, row_of(wide).view());
    BOOST_TEST(store.spilled(i));
    BOOST_TEST(store.row(i) == wide);

    // The empty row is the case the lane padding exists for: it writes no lane of its own, so without the
    // pad the previous occupant's overflow marker survives in lane 0 and the row reads as spilled while
    // its side-map entry is gone.
    store.set(i, row_of(Monomial<kNumModes>{}).view());
    BOOST_TEST(!store.spilled(i));
    BOOST_TEST(store.codes(i) == 0U);
    BOOST_TEST(store.slot_count(i) == 0U);
    BOOST_TEST(store.row(i) == Monomial<kNumModes>{});
}

// A batch of keys is homogeneous by type but not by shape: the spilled ones are what a query record
// escapes to. find_batch must resolve both, in one pass, against a store holding both kinds of row.
BOOST_AUTO_TEST_CASE(sparse_row_key_batch_resolves_both_shapes) {
    constexpr size_t kNumModes = 64;
    using Store = SparseRowStore<kNumModes>;
    using Key = SparseRowKey<2 * kNumModes>;
    Store store(3);
    std::mt19937_64 rng(20260814U);

    std::vector<Monomial<kNumModes>> monos;
    for (size_t k = 0; k < 64; ++k) {
        // Up to 6 slots against a capacity of 3, so roughly half the rows spill.
        auto mono = test_utils::random_monomial<kNumModes>(rng, 6);
        if (std::find(monos.begin(), monos.end(), mono) != monos.end()) {
            continue;
        }
        store.push_back(mono);
        store.emplace(mono, store.size() - 1);
        monos.push_back(mono);
    }
    BOOST_REQUIRE(monos.size() > 8U);

    std::vector<RowBuffer> rows;
    std::vector<Key> keys;
    rows.reserve(monos.size());
    keys.reserve(monos.size());
    for (const auto &mono : monos) {
        rows.push_back(row_of(mono));
    }
    size_t spilled_keys = 0;
    for (size_t k = 0; k < monos.size(); ++k) {
        // The shape a record would have carried: within the capacity it stays a row, past it the record
        // escapes to the dense monomial.
        if (rows[k].lanes.size() > store.slots_per_row()) {
            keys.push_back(Key{.spilled = &monos[k]});
            ++spilled_keys;
        }
        else {
            keys.push_back(Key{.row = rows[k].view()});
        }
    }
    BOOST_REQUIRE(spilled_keys > 0U);
    BOOST_REQUIRE(spilled_keys < monos.size());

    std::vector<size_t> found(keys.size(), Store::kNotFound);
    store.find_batch(keys.data(), keys.size(), found.data());
    for (size_t k = 0; k < keys.size(); ++k) {
        BOOST_REQUIRE(found[k] < store.size());
        BOOST_TEST(store.row(found[k]) == monos[k]);
    }

    // An absent key must miss through either shape.
    Monomial<kNumModes> absent;
    absent.set(6);
    absent.set(7);
    absent.set(120);
    while (std::find(monos.begin(), monos.end(), absent) != monos.end()) {
        absent.set(9);
    }
    const RowBuffer absent_row = row_of(absent);
    BOOST_TEST(!store.find(Key{.row = absent_row.view()}).has_value());
    BOOST_TEST(!store.find(Key{.spilled = &absent}).has_value());
}

// The codes array's element width follows slots_per_row_, since a codes word only ever sets bits below
// 2 * slots_per_row_. The row array is the operator's largest, and rows are payload -- never a hash
// input, never serialized -- so a narrowing here changes no term and no energy and a baseline diff
// cannot see it. This is the footprint gate.
// memory_bytes() - slack_bytes() is the *used* part of the arrays, which makes the figure exact rather
// than allocator-dependent.
BOOST_AUTO_TEST_CASE(codes_width_follows_the_slot_count) {
    constexpr size_t kNumModes = 128;
    constexpr size_t kRows = 500;
    // 2 bytes of codes at up to 8 slots, 4 up to 16, 8 above -- with the mode lanes constant per slot.
    const std::array<std::pair<size_t, size_t>, 3> kCases{{{8, 2}, {16, 4}, {17, 8}}};

    for (const auto &[slots, codes_bytes] : kCases) {
        SparseRowStore<kNumModes> store(slots);
        for (size_t i = 0; i < kRows; ++i) {
            // One occupied mode per row: any slot count <= slots works, but staying at one keeps every
            // row off the overflow side-map, whose bytes are counted separately and would blur this.
            Monomial<kNumModes> mono;
            mono.set(2 * (i % kNumModes));
            store.push_back(mono);
        }
        const size_t expected = kRows * ((slots * sizeof(RowMode)) + codes_bytes);
        BOOST_TEST(store.memory_bytes() - store.slack_bytes() == expected);
    }
}

// The narrowed storage must be invisible above the seam: a store at each codes width has to hold and
// return the same rows, hash them the same way and find them the same way.
BOOST_AUTO_TEST_CASE(a_narrowed_codes_word_reads_back_unchanged) {
    constexpr size_t kNumModes = 128;
    std::mt19937_64 rng(20260828);

    std::vector<Monomial<kNumModes>> monos;
    for (size_t i = 0; i < 200; ++i) {
        // At most 8 occupied modes, so every row fits the narrowest store's slots and none spills.
        Monomial<kNumModes> mono;
        for (size_t k = 0; k < 8; ++k) {
            mono.set(rng() % (2 * kNumModes));
        }
        if (std::find(monos.begin(), monos.end(), mono) == monos.end()) {
            monos.push_back(mono);
        }
    }

    SparseRowStore<kNumModes> narrow(8);
    SparseRowStore<kNumModes> wide(SparseRowStore<kNumModes>::kMaxSlots);
    for (const auto &mono : monos) {
        narrow.push_back(mono);
        wide.push_back(mono);
        narrow.emplace(mono, narrow.size() - 1);
        wide.emplace(mono, wide.size() - 1);
    }
    BOOST_REQUIRE(narrow.memory_bytes() < wide.memory_bytes());

    for (size_t i = 0; i < monos.size(); ++i) {
        BOOST_REQUIRE(!narrow.spilled(i));
        BOOST_TEST(narrow.codes(i) == wide.codes(i));
        BOOST_TEST(narrow.row(i) == monos[i]);
        BOOST_TEST(narrow.popcount(i) == wide.popcount(i));
        BOOST_TEST(narrow.slot_count(i) == wide.slot_count(i));
        BOOST_TEST(sparse_row_hash(narrow.view(i)) == sparse_row_hash(wide.view(i)));
        BOOST_TEST(narrow.find(monos[i]).value() == i);
        BOOST_TEST(narrow.find(narrow.view(i)).value() == i);
    }
}
