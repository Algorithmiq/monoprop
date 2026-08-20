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

// The (k, d) integer predicates differentially against the dense bitset forms they displace; the emit
// path calls only these, so a disagreement is a silently wrong keep/reject, never a crash. Both
// populations are drawn on purpose: uniform draws land on the fully-paired branch 11 times in 28500.

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/core/SparseMonomial.h"

using namespace monoprop;

namespace {

// indices_to_bitset is the only constructor user input reaches; any other draw is unreachable state.
template <size_t N>
auto draw(std::mt19937_64 &rng, size_t logical, size_t weight) -> Monomial<N> {
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

template <size_t N>
auto draw_paired(std::mt19937_64 &rng, size_t logical, size_t modes) -> Monomial<N> {
    VecZ idx;
    std::uniform_int_distribution<size_t> dist(0, logical - 1);
    std::vector<size_t> chosen;
    while (chosen.size() < modes) {
        const size_t q = dist(rng);
        bool dup = false;
        for (const auto x : chosen) {
            dup = dup || (x == q);
        }
        if (!dup) {
            chosen.push_back(q);
            idx.push_back(2 * q);
            idx.push_back((2 * q) + 1);
        }
    }
    return indices_to_bitset<N>(idx);
}

struct Tally {
    size_t comparisons = 0;
    size_t mismatches = 0;
    size_t paired_out = 0; // samples that are fully paired (the unconditionally-kept branch)
    size_t kept = 0;
    size_t rejected = 0;
};

enum class Population : uint8_t { Uniform, Paired };

template <size_t N>
auto check_width(std::mt19937_64 &rng, size_t logical, Population pop, Tally &t) -> void {
    const bool paired_pop = pop == Population::Paired;
    for (int rep = 0; rep < 400; ++rep) {
        const size_t kw = 1 + (rng() % 12);
        const auto x = paired_pop ? draw_paired<N>(rng, logical, 1 + (kw % 5)) : draw<N>(rng, logical, kw);

        const size_t k = x.count();
        const size_t d = paired_mode_count<N>(x);

        const auto ref = cutoff_sums<N>(x, logical);
        const auto got = cutoff_sums(k, d);
        bool ok = got.xor_sum == ref.xor_sum && got.popcount_sum == ref.popcount_sum && got.or_sum == ref.or_sum
                  && is_paired(k, d) == is_paired<N>(x);
        t.comparisons += 4;
        if (is_paired(k, d)) {
            ++t.paired_out;
        }

        // 0 rejects all but the paired branch, 12 keeps everything, the rest straddle the weights.
        for (const unsigned int c : {0U, 1U, 4U, 6U, 12U}) {
            const bool len = length_cutoff(k, d, c);
            const bool sup = support_cutoff(k, d, c);
            ok = ok && len == length_cutoff<N>(x, c, logical) && sup == support_cutoff<N>(x, c, logical);
            // Assert the forwarding too, or a wrapper that dropped a term would hide behind itself.
            ok = ok && len == length_keeps(k, d, c) && sup == support_keeps(k, d, c);
            t.comparisons += 4;
            t.kept += static_cast<size_t>(len);
            t.rejected += static_cast<size_t>(!len);
        }

        if (!ok) {
            ++t.mismatches;
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_predicates_match_bitset_forms_across_widths) {
    std::mt19937_64 rng(0x5A5E0DDULL);
    Tally t;

    for (const auto pop : {Population::Uniform, Population::Paired}) {
        check_width<32>(rng, 32, pop, t); // W = 64, one word, no active offset
        check_width<32>(rng, 30, pop, t); // W = 64, active_bit_offset = 4
        check_width<48>(rng, 45, pop, t); // W = 96 -- not a multiple of 64
        check_width<64>(rng, 64, pop, t); // W = 128, exactly two words
        check_width<128>(rng, 120, pop, t);
        check_width<256>(rng, 250, pop, t); // the production shape
    }

    BOOST_TEST(t.mismatches == 0U);
    BOOST_TEST(t.comparisons > 40000U); // a loop that never ran would report zero mismatches too
    BOOST_TEST(t.paired_out > 0U);      // the unconditionally-kept branch must be reached
    BOOST_TEST(t.kept > 0U);
    BOOST_TEST(t.rejected > 0U);
}

// The boundaries as literals: length compares k, support compares k - d (a paired mode spans two).
BOOST_AUTO_TEST_CASE(sparse_predicates_pin_their_boundaries) {
    BOOST_TEST(is_paired(0U, 0U)); // the identity is fully paired by this definition
    BOOST_TEST(is_paired(4U, 2U));
    BOOST_TEST(!is_paired(3U, 1U));

    // Fully paired: kept at cutoff 0, which rejects everything else.
    BOOST_TEST(length_keeps(4U, 2U, 0U));
    BOOST_TEST(support_keeps(4U, 2U, 0U));
    BOOST_TEST(!length_keeps(1U, 0U, 0U));
    BOOST_TEST(!support_keeps(1U, 0U, 0U));

    // Unpaired k=5, d=1: length sees 5, support sees k - d = 4.
    BOOST_TEST(!length_keeps(5U, 1U, 4U));
    BOOST_TEST(length_keeps(5U, 1U, 5U));
    BOOST_TEST(support_keeps(5U, 1U, 4U));
    BOOST_TEST(!support_keeps(5U, 1U, 3U));
}
