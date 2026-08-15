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

// xor_gen against the composition of the primitives it replaces: Bitset ^, count_and, cutoff_sums,
// interleave_phase and both structural cutoffs. The comparison is the point -- xor_gen exists to
// return in one pass what those five compute separately, so anything it gets wrong is a silent
// numerical error rather than a crash.
//
// The differential loop counts its own comparisons and asserts the count, because a fuzz that
// reports zero failures proves nothing if the loop never ran. Mutation coverage is recorded next to
// each check: "always +1" sign, "always paired" d, zeroed overlap and a swapped output pair were all
// verified to fail this suite before it was committed.

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

template <size_t N>
auto positions_of(const Monomial<N> &m) -> std::vector<uint32_t> {
    std::vector<uint32_t> out;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        out.push_back(static_cast<uint32_t>(b));
    }
    return out;
}

// indices_to_bitset is the only constructor user input reaches, and it places every bit at physical
// position >= 2*(NumModes - logical) (AlgebraCommon.h:49). Drawing any other way would exercise a
// state the engine cannot produce and make the cutoff_sums comparison meaningless.
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

// Fully paired: both Majoranas of each chosen mode. Drawn explicitly because uniform monomials
// almost never land on this branch (11 samples in 28500 measured), yet it is the branch both cutoffs
// keep unconditionally and therefore the one that decides how wide a row must be.
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
    size_t odd_gen = 0;    // samples with an odd-weight generator (the row-parity correction's regime)
    size_t paired_out = 0; // samples whose product is fully paired (the unconditionally-kept branch)
};

enum class Population : uint8_t { Uniform, Paired };

template <size_t N>
auto differential(size_t logical, size_t iters, Tally &t, Population pop = Population::Uniform) -> void {
    std::mt19937_64 rng(0xC0FFEEULL + (N * 7919) + logical + (pop == Population::Paired ? 1U : 0U));
    std::vector<uint32_t> out((4 * N) + 64);
    constexpr unsigned kCut = 6;

    for (size_t it = 0; it < iters; ++it) {
        const size_t kw = 1 + (rng() % 12);
        const size_t gw = 1 + (rng() % 4); // reaches |G| = 1 and 3, not only the bench's 4
        // The paired arm draws both operands fully paired, so their product is paired too: that is the
        // only way to reach the unconditional-keep branch in bulk (uniform draws hit it ~4 times in 10^4).
        const auto m =
            (pop == Population::Paired) ? draw_paired<N>(rng, logical, 1 + (kw % 5)) : draw<N>(rng, logical, kw);
        const auto g =
            (pop == Population::Paired) ? draw_paired<N>(rng, logical, 1 + (gw % 2)) : draw<N>(rng, logical, gw);
        if ((gw % 2) != 0 && pop == Population::Uniform) {
            ++t.odd_gen;
        }

        const auto mp = positions_of<N>(m);
        const auto gp = positions_of<N>(g);
        const auto r = xor_gen<uint32_t>(mp.data(), mp.size(), gp.data(), gp.size(), out.data());

        const Monomial<N> x = m ^ g;
        const auto expect = positions_of<N>(x);
        const auto sums = cutoff_sums<N>(x, logical);
        const auto ctx = MajoranaAlgebra<N>::make_gen_context(g);
        if (is_paired(r.k, r.d)) {
            ++t.paired_out;
        }

        bool positions_match = r.k == expect.size();
        for (size_t j = 0; positions_match && j < r.k; ++j) {
            positions_match = out[j] == expect[j];
        }

        const bool ok = positions_match                                           //
                        && r.overlap == m.count_and(g)                            //
                        && sums.popcount_sum == r.k                               //
                        && sums.xor_sum == r.k - (2 * r.d)                        //
                        && sums.or_sum == r.k - r.d                               //
                        && r.sign == interleave_phase<N>(m, g)                    //
                        && r.sign == MajoranaAlgebra<N>::rotation_sign(ctx, m, x) //
                        && length_keeps(r.k, r.d, kCut) == length_cutoff<N>(x, kCut, logical)
                        && support_keeps(r.k, r.d, kCut) == support_cutoff<N>(x, kCut, logical)
                        && pair_digest<uint32_t>(out.data(), r.k).d == r.d;
        t.comparisons += 10;
        if (!ok) {
            ++t.mismatches;
        }
    }
}

struct DigestTally {
    size_t comparisons = 0;
    size_t mismatches = 0;
    size_t offset_checks = 0;     // monomials whose lowest set bit was measured against the offset
    size_t offset_violations = 0; // one below 2*(NumModes - logical) would void every comparison here
    size_t paired_out = 0;        // products on the unconditionally-kept branch
    size_t rejected_out = 0;      // products the length cutoff drops -- the emit path's rejected majority
};

// The (k, d) overloads in AlgebraCommon.h against the bitset forms they shadow, plus the precondition
// that makes the two equivalent: indices_to_bitset() places every bit at physical position
// >= 2*(NumModes - logical), and XOR keeps it there, so the bitset form's active-window masking has
// nothing to mask. Measured here rather than asserted in a comment, because the digest carries no
// logical_num_modes and silently answers for the whole register.
template <size_t N>
auto digest_overloads(size_t logical, size_t iters, DigestTally &t) -> void {
    std::mt19937_64 rng(0xD16E57ULL + (N * 104729) + logical);
    std::vector<uint32_t> out((4 * N) + 64);
    const size_t offset = 2 * (N - logical);

    CutoffFn<N> length_fn = detail::LengthCutoff<N>{.cutoff = 6, .logical_num_modes = logical};
    CutoffFn<N> support_fn = detail::SupportCutoff<N>{.cutoff = 4, .logical_num_modes = logical};
    CutoffFn<N> opaque_fn = [](const Monomial<N> &) { return true; };
    const detail::CutoffEvaluator<N> length_ev(length_fn);
    const detail::CutoffEvaluator<N> support_ev(support_fn);
    const detail::CutoffEvaluator<N> opaque_ev(opaque_fn);

    for (size_t it = 0; it < iters; ++it) {
        // Both populations, since a uniform draw essentially never lands on the fully paired branch
        // and that branch is the one the digest decides differently from a bare popcount compare.
        const bool paired_pop = (it % 4) == 0;
        const size_t kw = 1 + (rng() % 12);
        const size_t gw = 1 + (rng() % 4);
        const auto m = paired_pop ? draw_paired<N>(rng, logical, 1 + (kw % 5)) : draw<N>(rng, logical, kw);
        const auto g = paired_pop ? draw_paired<N>(rng, logical, 1 + (gw % 2)) : draw<N>(rng, logical, gw);
        const Monomial<N> x = m ^ g;

        // Bitset::find_first() returns size() when nothing is set, which clears every offset.
        for (const auto &probe : {m, g, x}) {
            ++t.offset_checks;
            if (probe.find_first() < offset) {
                ++t.offset_violations;
            }
        }

        const auto mp = positions_of<N>(m);
        const auto gp = positions_of<N>(g);
        const auto r = xor_gen<uint32_t>(mp.data(), mp.size(), gp.data(), gp.size(), out.data());
        const auto d = pair_digest<uint32_t>(out.data(), r.k); // independent of xor_gen's own d

        const auto ref = cutoff_sums<N>(x, logical);
        const auto got = cutoff_sums(d.k, d.d);
        bool ok = got.xor_sum == ref.xor_sum && got.popcount_sum == ref.popcount_sum && got.or_sum == ref.or_sum
                  && is_paired(d.k, d.d) == is_paired<N>(x);
        t.comparisons += 4;
        if (is_paired(d.k, d.d)) {
            ++t.paired_out;
        }

        for (const unsigned int c : {0U, 1U, 4U, 6U, 12U}) {
            ok = ok && length_cutoff(d.k, d.d, c) == length_cutoff<N>(x, c, logical)
                 && support_cutoff(d.k, d.d, c) == support_cutoff<N>(x, c, logical);
            t.comparisons += 2;
        }

        // The evaluator's digest path must match its bitset path, and must decline (nullopt) exactly
        // when the cutoff is opaque -- an early `true` there would keep terms the predicate rejects.
        const auto len_digest = length_ev.passes_with_popcount(d.k, d.d);
        const auto sup_digest = support_ev.passes_with_popcount(d.k, d.d);
        ok = ok && len_digest.has_value() && *len_digest == length_ev.passes_with_popcount(x, d.k)
             && sup_digest.has_value() && *sup_digest == support_ev.passes_with_popcount(x, d.k)
             && !opaque_ev.passes_with_popcount(d.k, d.d).has_value();
        t.comparisons += 3;
        if (len_digest.has_value() && !*len_digest) {
            ++t.rejected_out;
        }

        if (!ok) {
            ++t.mismatches;
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(sparse_xor_gen_matches_worked_example) {
    // Majorana indices 0 and 1 are one mode, so the product is fully paired: k = 2, d = 1.
    constexpr size_t N = 8;
    const auto m = indices_to_bitset<N>(VecZ{0, 1});
    const auto mp = positions_of<N>(m);
    std::vector<uint32_t> out(8);

    const auto self = xor_gen<uint32_t>(mp.data(), mp.size(), mp.data(), mp.size(), out.data());
    BOOST_TEST(self.k == 0U);       // M ^ M is the identity
    BOOST_TEST(self.overlap == 2U); // and every bit overlapped

    const auto pd = pair_digest<uint32_t>(mp.data(), mp.size());
    BOOST_TEST(pd.k == 2U);
    BOOST_TEST(pd.d == 1U);
    BOOST_TEST(is_paired(pd.k, pd.d));
    // A fully paired term is kept by both cutoffs whatever its length -- that is why a length cutoff
    // cannot bound the popcount a surviving row carries.
    BOOST_TEST(length_keeps(pd.k, pd.d, 0U));
    BOOST_TEST(support_keeps(pd.k, pd.d, 0U));
    // An unpaired single Majorana is not.
    BOOST_TEST(!length_keeps(1U, 0U, 0U));
}

BOOST_AUTO_TEST_CASE(sparse_xor_gen_differential_across_widths) {
    Tally t;
    // Offsets 2*(NumModes - logical): 0, 2, 12, 16 -- the two production shapes plus the degenerate
    // one, and two widths where 2*NumModes is not a multiple of 64 so kTopBits is exercised.
    differential<32>(32, 4000, t);    // W = 64, offset 0 (single word)
    differential<32>(31, 4000, t);    // W = 64, offset 2
    differential<128>(128, 4000, t);  // W = 256, offset 0
    differential<128>(120, 4000, t);  // W = 256, offset 16 (Hubbard)
    differential<256>(250, 4000, t);  // W = 512, offset 12 (the HPC configuration)
    differential<1024>(1020, 500, t); // W = 2048, offset 8
    differential<50>(50, 4000, t);    // W = 100, kTopBits != 0
    differential<50>(44, 4000, t);    // W = 100, offset 12
    // Fully paired operands, at the two production shapes: the branch both cutoffs keep regardless of
    // length, and the one a uniform draw essentially never reaches.
    differential<128>(120, 4000, t, Population::Paired);
    differential<256>(250, 4000, t, Population::Paired);

    BOOST_TEST(t.mismatches == 0U);
    // Assert the loop ran and reached both regimes it is supposed to cover, so a vacuous pass is not
    // mistaken for a clean one.
    BOOST_TEST(t.comparisons > 250000U);
    BOOST_TEST(t.odd_gen > 1000U);
    BOOST_TEST(t.paired_out > 5000U);
}

BOOST_AUTO_TEST_CASE(sparse_digest_overloads_match_bitset_forms) {
    DigestTally t;
    // active_bit_offset = 2*(NumModes - logical): 0, 2, 12 and 16, at the three production widths.
    digest_overloads<32>(32, 3000, t);   // offset 0, single word
    digest_overloads<32>(31, 3000, t);   // offset 2, single word
    digest_overloads<128>(128, 3000, t); // offset 0, multi word
    digest_overloads<128>(120, 3000, t); // offset 16 (Hubbard)
    digest_overloads<256>(256, 3000, t); // offset 0, widest
    digest_overloads<256>(250, 3000, t); // offset 12 (the HPC configuration)

    BOOST_TEST(t.mismatches == 0U);
    // The precondition the digest overloads rest on: no constructed monomial reached below the offset.
    BOOST_TEST(t.offset_violations == 0U);
    // Assert the loop ran and reached both decisions, so a vacuous pass is not mistaken for a clean one.
    BOOST_TEST(t.comparisons > 300000U);
    BOOST_TEST(t.offset_checks == 54000U);
    BOOST_TEST(t.paired_out > 3000U);
    BOOST_TEST(t.rejected_out > 3000U);
}
