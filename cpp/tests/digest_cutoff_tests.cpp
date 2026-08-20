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

// paired_mode_count's d, differentially against the bitset cutoff_sums: the IDENTITY
// (d == popcount_sum - or_sum) and the PREDICATE built on it are separable, so they are separate cases.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/operator/MPOperator.h"

using namespace monoprop;

namespace {

// indices_to_bitset places every bit at or above the active offset: the precondition inherited here.
template <size_t N>
auto draw_well_formed(std::mt19937_64 &rng, size_t logical, size_t weight) -> Monomial<N> {
    VecZ idx;
    std::uniform_int_distribution<size_t> dist(0, (2 * logical) - 1);
    while (idx.size() < weight) {
        const size_t v = dist(rng);
        bool dup = false;
        for (const auto x : idx) {
            dup = dup || (x == v);
        }
        if (!dup) {
            idx.push_back(v);
        }
    }
    return indices_to_bitset<N>(idx);
}

// Weight spans both extremes: d == 0 and k == 2d are the cases the cutoffs branch on.
template <size_t N>
auto check_identity(std::mt19937_64 &rng, size_t logical, size_t &checked, size_t &paired_seen) -> void {
    for (size_t weight = 1; weight <= 2 * logical && weight <= 24; ++weight) {
        for (int rep = 0; rep < 40; ++rep) {
            const auto mono = draw_well_formed<N>(rng, logical, weight);
            const auto sums = cutoff_sums<N>(mono, logical);
            const size_t d = paired_mode_count<N>(mono);
            BOOST_REQUIRE_EQUAL(d, sums.popcount_sum - sums.or_sum);
            const auto rebuilt = cutoff_sums(sums.popcount_sum, d);
            BOOST_REQUIRE_EQUAL(rebuilt.xor_sum, sums.xor_sum);
            BOOST_REQUIRE_EQUAL(rebuilt.or_sum, sums.or_sum);
            BOOST_REQUIRE_EQUAL(rebuilt.popcount_sum, sums.popcount_sum);
            paired_seen += static_cast<size_t>(sums.xor_sum == 0);
            ++checked;
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(paired_mode_count_matches_cutoff_sums_across_widths) {
    std::mt19937_64 rng(0xD16E57U);
    size_t checked = 0;
    size_t paired_seen = 0;

    check_identity<32>(rng, 32, checked, paired_seen); // W = 64, one word, no active offset
    check_identity<32>(rng, 30, checked, paired_seen); // W = 64, active_bit_offset = 4
    check_identity<48>(rng, 45, checked, paired_seen); // W = 96 -- not a multiple of 64
    check_identity<64>(rng, 64, checked, paired_seen); // W = 128, exactly two words
    check_identity<128>(rng, 120, checked, paired_seen);
    check_identity<256>(rng, 250, checked, paired_seen); // the production shape

    BOOST_TEST(checked > 3000U);
    // The identity is only interesting on the fully-paired branch, so the draw must reach it.
    BOOST_TEST(paired_seen > 0U);
}

BOOST_AUTO_TEST_CASE(paired_mode_count_exhaustive_at_small_width) {
    constexpr size_t kN = 5; // W = 10
    size_t paired = 0;
    for (uint64_t bits = 0; bits < (uint64_t{1} << (2 * kN)); ++bits) {
        Monomial<kN> mono;
        for (size_t b = 0; b < 2 * kN; ++b) {
            if ((bits >> b) & 1U) {
                mono.set(b);
            }
        }
        const auto sums = cutoff_sums<kN>(mono, kN);
        BOOST_REQUIRE_EQUAL(paired_mode_count<kN>(mono), sums.popcount_sum - sums.or_sum);
        paired += static_cast<size_t>(sums.xor_sum == 0);
    }
    BOOST_TEST(paired == 32U); // 2^5: each mode independently empty or doubly occupied
}

// At the PREDICATE level, not the scan level: cutoff_sums is the independent form to compare against.
BOOST_AUTO_TEST_CASE(digest_predicate_matches_cutoff_sums_predicate) {
    std::mt19937_64 rng(0xC0FFEEU);
    size_t checked = 0;
    size_t kept = 0;
    size_t rejected = 0;

    // The popcount <= cutoff early-out is the asymmetry between the two, so cutoffs straddle it.
    for (const unsigned int cutoff : {1U, 2U, 4U, 6U, 10U, 20U}) {
        for (const bool support : {false, true}) {
            constexpr size_t kN = 32;
            constexpr size_t kLogical = 30;
            const CutoffFn<kN> fn = support ? CutoffFn<kN>{detail::SupportCutoff<kN>{cutoff, kLogical}}
                                            : CutoffFn<kN>{detail::LengthCutoff<kN>{cutoff, kLogical}};
            const detail::CutoffEvaluator<kN> eval(fn);
            for (size_t w = 1; w <= 12; ++w) {
                for (int rep = 0; rep < 40; ++rep) {
                    const auto mono = draw_well_formed<kN>(rng, kLogical, w);
                    const size_t k = mono.count();
                    const auto digest = eval.passes_from_dense(mono, k);
                    BOOST_REQUIRE(digest.has_value()); // a concrete cutoff must always decide
                    const bool reference = eval.passes_with_popcount(mono, k);
                    BOOST_REQUIRE_EQUAL(*digest, reference);
                    ++checked;
                    kept += static_cast<size_t>(*digest);
                    rejected += static_cast<size_t>(!*digest);
                }
            }
        }
    }
    BOOST_TEST(checked > 5000U);
    // A sweep that only ever kept would agree with any predicate that returns true.
    BOOST_TEST(kept > 0U);
    BOOST_TEST(rejected > 0U);
}
