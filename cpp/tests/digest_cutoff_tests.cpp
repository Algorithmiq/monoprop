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

// paired_mode_count computes d -- the number of modes carrying both Majoranas -- from the dense words,
// so that the emit path can reach the structural cutoff's verdict without calling cutoff_sums. Two
// things have to hold, and they are separable, so they are tested separately:
//
//   1. the IDENTITY: d == popcount_sum - or_sum for every well-formed monomial, at every width. The
//      bitset cutoff_sums is the oracle, per its own comment.
//   2. the WIRING: the scan reaches the same verdict, on the same terms, in the same record order.
//
// The identity is the interesting one, because paired_mode_count exploits two facts that are easy to
// get wrong: that a mode's two bits are word-internal (so `w >> 1` needs no cross-word carry), and
// that cutoff_sums' active_bit_offset masking is inert on well-formed monomials. Widths are chosen so
// that both the one-word specialisation and the multiword path are covered, including a width where
// 2*NumModes is not a multiple of 64.

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

// A well-formed monomial: indices_to_bitset places every set bit at physical position >= the active
// offset, which is exactly the precondition paired_mode_count inherits from cutoff_sums(k, d).
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

// Drive weight across the fully-unpaired and fully-paired extremes, not just the middle: d == 0 and
// k == 2d are the two cases the cutoffs branch on, and a bug in the even-mask would only show at one.
template <size_t N>
auto check_identity(std::mt19937_64 &rng, size_t logical, size_t &checked, size_t &paired_seen) -> void {
    for (size_t weight = 1; weight <= 2 * logical && weight <= 24; ++weight) {
        for (int rep = 0; rep < 40; ++rep) {
            const auto mono = draw_well_formed<N>(rng, logical, weight);
            const auto sums = cutoff_sums<N>(mono, logical);
            const size_t d = paired_mode_count<N>(mono);
            BOOST_REQUIRE_EQUAL(d, sums.popcount_sum - sums.or_sum);
            // The reconstruction the emit path actually performs, through the (k, d) overload.
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
    // A fully-paired monomial (xor_sum == 0) is the case both cutoffs keep unconditionally; if the
    // draw never produced one, the identity was only ever checked on the branch that does not matter.
    BOOST_TEST(paired_seen > 0U);
}

// Exhaustive at a tiny width: every monomial over 5 modes, so no draw can miss a case.
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

namespace {

auto build_op(const std::vector<Monomial<32>> &terms) -> detail::MPOperator<32> {
    detail::MPOperator<32> op;
    op.basis = Basis::Majorana;
    detail::insert_absent_terms<32>(
        op,
        terms.size(),
        [&](size_t k) -> const Monomial<32> & { return terms[k]; },
        [&](size_t k, size_t base) { assign_row<32>(*op.store, base + k, terms[k]); });
    return op;
}

auto same(const detail::FusedScanResult &a, const detail::FusedScanResult &b) -> bool {
    return a.leader_queries == b.leader_queries && a.follower_queries == b.follower_queries
           && a.leader_src == b.leader_src && a.follower_src == b.follower_src && a.leader_val == b.leader_val
           && a.follower_val == b.follower_val && a.cos_blocks.size() == b.cos_blocks.size();
}

} // namespace

// The wiring: same verdict, same terms, same record order. Order matters for the same reason it does
// in the emit-path tests -- Resolve.h mints each miss's term index in record order, and at the
// measured hit rate essentially every record is a miss, so record order is the accumulation order.
BOOST_AUTO_TEST_CASE(digest_cutoff_matches_cutoff_sums_in_the_scan) {
    constexpr size_t kN = 32;
    constexpr size_t kLogical = 30;
    std::mt19937_64 rng(0xC0FFEEU);
    size_t scans = 0;
    size_t records = 0;

    for (const unsigned int cutoff : {2U, 4U, 6U, 10U}) {
        for (const size_t gw : {1U, 2U, 3U, 4U}) {
            for (int rep = 0; rep < 8; ++rep) {
                std::vector<Monomial<kN>> terms;
                for (size_t i = 0; i < 200; ++i) {
                    terms.push_back(draw_well_formed<kN>(rng, kLogical, 1 + (rng() % 6)));
                }
                auto op = build_op(terms);
                const Monomial<kN> gen = draw_well_formed<kN>(rng, kLogical, gw);

                VecD coeffs(op.store->size());
                for (size_t i = 0; i < coeffs.size(); ++i) {
                    coeffs[i] = 0.25 + (0.5 * static_cast<double>(i % 7));
                }
                const CutoffFn<kN> fn = detail::LengthCutoff<kN>{cutoff, kLogical};
                const detail::CutoffEvaluator<kN> eval(fn);
                // upper_atol live so the is_above_upper rescue is exercised on both arms, not dead.
                const auto cut = detail::build_majorana_evolution_cutoff_state(std::optional<double>{1e-12},
                                                                               std::cref(coeffs),
                                                                               std::optional<double>{0.9},
                                                                               std::optional<double>{0.3});

                const auto base = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>>(op,
                                                                                          gen,
                                                                                          eval,
                                                                                          cut,
                                                                                          coeffs,
                                                                                          std::nullopt,
                                                                                          1,
                                                                                          0,
                                                                                          false,
                                                                                          nullptr,
                                                                                          1.0,
                                                                                          false);
                const auto digest = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>>(op,
                                                                                            gen,
                                                                                            eval,
                                                                                            cut,
                                                                                            coeffs,
                                                                                            std::nullopt,
                                                                                            1,
                                                                                            0,
                                                                                            false,
                                                                                            nullptr,
                                                                                            1.0,
                                                                                            true);

                BOOST_REQUIRE(same(base, digest));
                ++scans;
                for (const auto &q : digest.leader_queries) {
                    records += q.size();
                }
            }
        }
    }
    BOOST_TEST(scans == 128U);
    BOOST_TEST(records > 5000U); // a run that emitted nothing must not read as agreement
}

// Multi-rank routing, for the same reason the emit-path tests check it: the owner hash runs on the
// surviving partner, so a verdict that differed would scatter records into different per-rank buckets.
BOOST_AUTO_TEST_CASE(digest_cutoff_routes_identically_across_ranks) {
    constexpr size_t kN = 32;
    constexpr size_t kLogical = 30;
    std::mt19937_64 rng(0xBEEF01U);

    std::vector<Monomial<kN>> terms;
    for (size_t i = 0; i < 400; ++i) {
        terms.push_back(draw_well_formed<kN>(rng, kLogical, 1 + (rng() % 6)));
    }
    auto op = build_op(terms);
    const Monomial<kN> gen = draw_well_formed<kN>(rng, kLogical, 4);
    VecD coeffs(op.store->size(), 1.0);

    const CutoffFn<kN> fn = detail::LengthCutoff<kN>{6, kLogical};
    const detail::CutoffEvaluator<kN> eval(fn);
    const auto cut = detail::build_majorana_evolution_cutoff_state(std::nullopt,
                                                                   std::cref(coeffs),
                                                                   std::nullopt,
                                                                   std::optional<double>{0.3});

    for (const size_t ranks : {2U, 4U, 8U}) {
        const auto base = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>>(op,
                                                                                  gen,
                                                                                  eval,
                                                                                  cut,
                                                                                  coeffs,
                                                                                  std::nullopt,
                                                                                  ranks,
                                                                                  0,
                                                                                  false,
                                                                                  nullptr,
                                                                                  1.0,
                                                                                  false);
        const auto digest = detail::fused_find_and_collect<kN, MajoranaAlgebra<kN>>(op,
                                                                                    gen,
                                                                                    eval,
                                                                                    cut,
                                                                                    coeffs,
                                                                                    std::nullopt,
                                                                                    ranks,
                                                                                    0,
                                                                                    false,
                                                                                    nullptr,
                                                                                    1.0,
                                                                                    true);
        BOOST_REQUIRE_EQUAL(digest.leader_queries.size(), ranks);
        BOOST_TEST(same(base, digest));
        size_t total = 0;
        for (const auto &q : digest.leader_queries) {
            total += q.size();
        }
        BOOST_TEST(total > 0U);
    }
}
