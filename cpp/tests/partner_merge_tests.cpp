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

// The partner merge against the dense XOR it replaces. Random pairs alone do not reach the cases that
// decide it: the merge's whole surface is how many of G's slots the source already holds, so every
// overlap in [0, gen_pop] is drawn on purpose, and the paired case is drawn separately because a
// fully-paired monomial is 94 in 20.9M on production models.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"

using namespace monoprop;

namespace {

constexpr size_t kN = 64;
constexpr size_t kBits = 2 * kN;

// The dense reference: positions of M^G, ascending, straight off the bitset.
auto dense_partner(const Monomial<kN> &mono, const Monomial<kN> &gen) -> std::vector<size_t> {
    const Monomial<kN> nm = mono ^ gen;
    std::vector<size_t> out;
    for (size_t b = nm.find_first(); b < nm.size(); b = nm.find_next(b)) {
        out.push_back(b);
    }
    return out;
}

auto positions_of(const Monomial<kN> &m) -> std::vector<uint16_t> {
    std::vector<uint16_t> out;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        out.push_back(static_cast<uint16_t>(b));
    }
    return out;
}

// Checks the merge against the dense form on one pair, and returns the merged count so a caller can
// assert it saw work. Every field the emit site consumes is compared, not just the positions.
auto check_pair(const Monomial<kN> &mono, const Monomial<kN> &gen) -> size_t {
    const auto src = positions_of(mono);
    const auto gpos = positions_of(gen);
    std::vector<uint16_t> out(kBits);
    size_t overlap = 0;
    const size_t k =
        detail::merge_partner_positions(src.data(), src.size(), gpos.data(), gpos.size(), out.data(), overlap);

    const auto expect = dense_partner(mono, gen);
    BOOST_REQUIRE_EQUAL(k, expect.size());
    for (size_t j = 0; j < k; ++j) {
        BOOST_REQUIRE_EQUAL(static_cast<size_t>(out[j]), expect[j]);
    }
    BOOST_REQUIRE_EQUAL(overlap, mono.count_and(gen));
    // The popcount identity the emit site asserts on.
    BOOST_REQUIRE_EQUAL(k, mono.count() + gen.count() - (2 * overlap));
    return k;
}

// A generator of `gen_pop` slots, and a source holding exactly `overlap` of them plus `extra` others.
auto build_case(std::mt19937_64 &rng, size_t gen_pop, size_t overlap, size_t extra)
    -> std::pair<Monomial<kN>, Monomial<kN>> {
    std::vector<size_t> all(kBits);
    for (size_t i = 0; i < kBits; ++i) {
        all[i] = i;
    }
    std::shuffle(all.begin(), all.end(), rng);
    Monomial<kN> gen;
    for (size_t i = 0; i < gen_pop; ++i) {
        gen.set(all[i]);
    }
    Monomial<kN> mono;
    for (size_t i = 0; i < overlap; ++i) {
        mono.set(all[i]); // a slot G also holds: it cancels
    }
    for (size_t i = 0; i < extra; ++i) {
        mono.set(all[gen_pop + i]); // disjoint from G: it survives
    }
    return {mono, gen};
}

} // namespace

// Every overlap between the source and the generator, which is the branch the merge exists to take.
BOOST_AUTO_TEST_CASE(partner_merge_matches_dense_at_every_overlap) {
    std::mt19937_64 rng(0xA11CEU);
    size_t cases = 0;
    size_t nonempty = 0;
    for (size_t gen_pop = 1; gen_pop <= 6; ++gen_pop) {
        for (size_t overlap = 0; overlap <= gen_pop; ++overlap) {
            for (const size_t extra : {size_t{0}, size_t{1}, size_t{5}, size_t{20}}) {
                for (size_t rep = 0; rep < 8; ++rep) {
                    const auto [mono, gen] = build_case(rng, gen_pop, overlap, extra);
                    BOOST_REQUIRE_EQUAL(mono.count_and(gen), overlap); // the case is the one intended
                    nonempty += (check_pair(mono, gen) != 0) ? 1 : 0;
                    ++cases;
                }
            }
        }
    }
    BOOST_REQUIRE_EQUAL(cases, 6U * 4U * 8U + (1U + 2U + 3U + 4U + 5U + 6U) * 4U * 8U);
    // Total cancellation (overlap == gen_pop, extra == 0) is the only empty partner, so most must not be.
    BOOST_TEST(nonempty > 600U);
}

// The paired population separately: a uniform draw almost never produces a fully paired monomial, so
// this case is drawn on purpose rather than left to chance in the sweep above.
BOOST_AUTO_TEST_CASE(partner_merge_matches_dense_on_paired_monomials) {
    std::mt19937_64 rng(0xBEEFU);
    size_t paired_seen = 0;
    for (size_t rep = 0; rep < 400; ++rep) {
        // Whole modes only, so both slots of each are set and the result is fully paired.
        std::vector<size_t> modes(kN);
        for (size_t i = 0; i < kN; ++i) {
            modes[i] = i;
        }
        std::shuffle(modes.begin(), modes.end(), rng);
        Monomial<kN> mono;
        const size_t n_modes = 1 + (rng() % 8);
        for (size_t i = 0; i < n_modes; ++i) {
            mono.set(2 * modes[i]);
            mono.set((2 * modes[i]) + 1);
        }
        Monomial<kN> gen;
        const size_t g_modes = 1 + (rng() % 3);
        for (size_t i = 0; i < g_modes; ++i) {
            gen.set(2 * modes[kN - 1 - i]);
            gen.set((2 * modes[kN - 1 - i]) + 1);
        }
        check_pair(mono, gen);
        paired_seen += is_paired<kN>(mono ^ gen) ? 1 : 0;
    }
    // The draw is meant to land on the paired branch every time; a zero here means it stopped doing so.
    BOOST_REQUIRE_EQUAL(paired_seen, 400U);
}

// An empty generator and an empty source are both reachable (a truncated gate, a fresh row).
BOOST_AUTO_TEST_CASE(partner_merge_handles_empty_inputs) {
    Monomial<kN> mono;
    mono.set(4);
    mono.set(5);
    mono.set(70);
    const Monomial<kN> empty;
    BOOST_REQUIRE_EQUAL(check_pair(mono, empty), 3U);
    BOOST_REQUIRE_EQUAL(check_pair(empty, mono), 3U);
    BOOST_REQUIRE_EQUAL(check_pair(empty, empty), 0U);
}
