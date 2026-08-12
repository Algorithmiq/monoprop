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
// the TypeAliases.h accessors lives in row_accessor_tests.cpp; what is checked here is the part that
// has no counterpart in the other backends -- the codes word, and the row sizing that feeds it.

#include <boost/test/unit_test.hpp>

#include <bit>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/operator/SparseRowStore.h"

using namespace monoprop;
using namespace monoprop::detail;

// Interchangeable with OperatorIndex means the same ownership rules, so the store cannot be silently
// copied out of MPOperator's unique_ptr.
static_assert(!std::is_move_constructible_v<SparseRowStore>, "SparseRowStore must remain non-movable");
static_assert(!std::is_copy_constructible_v<SparseRowStore>, "SparseRowStore must remain non-copyable");

namespace {

// The Stage 5 identities, spelled out here against the dense cutoff_sums so the port has something to
// be differentially tested against before any of it is written.
auto sums_from_codes(SparseRowStore::CodesT codes) -> CutoffSums {
    const auto n = static_cast<size_t>(std::popcount(SparseRowStore::occupied_bits(codes)));
    const auto d = static_cast<size_t>(std::popcount(SparseRowStore::paired_bits(codes)));
    return {n - d, n + d, n};
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_row_store_codes_encode_slot_pairs) {
    constexpr size_t kNumBits = 64;
    SparseRowStore store(kNumBits, 8);

    // Modes 1 (both positions), 4 (upper only) and 9 (lower only), so the codes word must read
    // 0b11, 0b10, 0b01 from slot 0 up.
    Bitset mono(kNumBits);
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
    SparseRowStore store(64, 4);
    store.push_back(Bitset(64));
    BOOST_TEST(!store.spilled(0));
    BOOST_TEST(store.codes(0) == 0U);
    BOOST_TEST(store.slot_count(0) == 0U);
    BOOST_TEST(store.popcount(0) == 0U);
    BOOST_TEST(store.row(0) == Bitset(64));
}

// The identities the whole support form rests on: or_sum = n, popcount_sum = n + d, xor_sum = n - d,
// against the dense cutoff_sums over the full storage window. Randomized rather than enumerated
// because what can break them is a particular occupancy pattern, not a particular width.
BOOST_AUTO_TEST_CASE(sparse_row_store_codes_reproduce_cutoff_sums) {
    std::mt19937_64 rng(20260812U);
    for (const size_t num_modes : {32U, 64U, 128U, 512U}) {
        const size_t num_bits = 2 * num_modes;
        // Capacity above any row built below, so nothing spills and every row exercises the codes path.
        SparseRowStore store(num_bits, SparseRowStore::kMaxSlots);
        const auto masks = CutoffMasks::make(num_bits, num_modes);
        for (size_t trial = 0; trial < 200; ++trial) {
            Bitset mono(num_bits);
            const size_t occupied = rng() % (SparseRowStore::kMaxSlots + 1);
            for (size_t k = 0; k < occupied; ++k) {
                const size_t mode = rng() % num_modes;
                // 1..3 so the mode is genuinely occupied, and all three codes appear.
                const unsigned int code = 1U + static_cast<unsigned int>(rng() % 3U);
                if ((code & 1U) != 0U) {
                    mono.set(2 * mode);
                }
                if ((code & 2U) != 0U) {
                    mono.set((2 * mode) + 1);
                }
            }
            store.push_back(mono);
            const size_t i = store.size() - 1;
            BOOST_REQUIRE(!store.spilled(i));

            const auto dense = cutoff_sums(mono, masks);
            const auto sparse = sums_from_codes(store.codes(i));
            BOOST_TEST(sparse.or_sum == dense.or_sum);
            BOOST_TEST(sparse.popcount_sum == dense.popcount_sum);
            BOOST_TEST(sparse.xor_sum == dense.xor_sum);
            BOOST_TEST(store.slot_count(i) == dense.or_sum);
            BOOST_TEST(store.popcount(i) == dense.popcount_sum);
        }
    }
}

// Spilled rows have no codes word, so the two measures must still come off the dense monomial.
BOOST_AUTO_TEST_CASE(sparse_row_store_spilled_rows_report_the_same_measures) {
    constexpr size_t kNumBits = 128;
    SparseRowStore store(kNumBits, 2);
    Bitset mono(kNumBits);
    for (const size_t b : {0U, 1U, 4U, 20U, 21U, 99U}) { // modes 0 (paired), 2, 10 (paired), 49
        mono.set(b);
    }
    store.push_back(mono);

    BOOST_TEST(store.spilled(0));
    BOOST_TEST(store.row(0) == mono);
    const auto dense = cutoff_sums(mono, CutoffMasks::make(kNumBits, kNumBits / 2));
    BOOST_TEST(store.slot_count(0) == dense.or_sum);
    BOOST_TEST(store.popcount(0) == dense.popcount_sum);
}

BOOST_AUTO_TEST_CASE(sparse_row_store_clone_preserves_rows_and_spills) {
    constexpr size_t kNumBits = 64;
    SparseRowStore store(kNumBits, 2);
    Bitset inline_row(kNumBits);
    inline_row.set(4);
    inline_row.set(5);
    Bitset spilled_row(kNumBits);
    for (const size_t b : {0U, 6U, 10U, 30U}) {
        spilled_row.set(b);
    }
    store.push_back(inline_row);
    store.push_back(spilled_row);

    const auto copy = store.clone();
    BOOST_REQUIRE(copy->size() == 2U);
    BOOST_TEST(copy->num_bits() == kNumBits);
    BOOST_TEST(copy->slots_per_row() == 2U);
    BOOST_TEST(!copy->spilled(0));
    BOOST_TEST(copy->codes(0) == store.codes(0));
    BOOST_TEST(copy->row(0) == inline_row);
    BOOST_TEST(copy->spilled(1));
    BOOST_TEST(copy->row(1) == spilled_row);
}

// K comes from the cutoff in modes, and is the same number for both cutoff kinds -- halving the slot
// bound for a support cutoff would truncate rows a length cutoff of the same size admits.
BOOST_AUTO_TEST_CASE(sparse_row_store_slots_come_from_the_cutoff_in_modes) {
    constexpr size_t kNumModes = 32;
    constexpr size_t kNumBits = 2 * kNumModes;

    const CutoffFn length = LengthCutoff(6U, kNumModes, kNumBits);
    const CutoffFn support = SupportCutoff(6U, kNumModes, kNumBits);
    BOOST_TEST(CutoffEvaluator(length).max_slot_bound().value() == 6U);
    BOOST_TEST(CutoffEvaluator(support).max_slot_bound().value() == 12U);
    BOOST_TEST(CutoffEvaluator(length).max_mode_bound().value() == 6U);
    BOOST_TEST(CutoffEvaluator(support).max_mode_bound().value() == 6U);

    BOOST_TEST(SparseRowStore::slots_for_bound(6U) == 6U);
    // A bound past one codes word clamps rather than throwing: the excess rows spill.
    BOOST_TEST(SparseRowStore::slots_for_bound(100U) == SparseRowStore::kMaxSlots);
    BOOST_TEST(SparseRowStore::slots_for_bound(0U) == 1U);
}

BOOST_AUTO_TEST_CASE(sparse_row_store_rejects_widths_past_the_lane_markers) {
    BOOST_CHECK_NO_THROW((SparseRowStore(2 * SparseRowStore::kMaxModes, 4)));
    BOOST_CHECK_THROW((SparseRowStore(2 * (SparseRowStore::kMaxModes + 1), 4)), SparseRowStoreUnsupported);
}

// The switch rule itself: a build-time constant, because the crossover follows the target ISA rather
// than anything known at run time. The values are the measured crossovers, so what this pins is that
// the CMake default reached the compiler at all -- a missing definition would silently fall back.
BOOST_AUTO_TEST_CASE(sparse_row_store_preference_threshold_matches_the_build) {
    static_assert(SparseRowStore::kMinModes > 0, "the sparse crossover must be a positive mode count");
    BOOST_TEST(!SparseRowStore::preferred_for_modes(SparseRowStore::kMinModes - 1));
    BOOST_TEST(SparseRowStore::preferred_for_modes(SparseRowStore::kMinModes));
    BOOST_TEST(SparseRowStore::preferred_for_modes(SparseRowStore::kMinModes + 1));
    // 32 modes is where the Stage 3 gate had sparse 1.9x behind dense even on baseline x86-64; no
    // build should be switching there.
    BOOST_TEST(!SparseRowStore::preferred_for_modes(32U));
}
