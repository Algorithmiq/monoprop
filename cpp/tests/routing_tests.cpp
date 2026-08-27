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

// routing::Router -- the term -> flat-slot map. Two properties carry the whole design:
//   * d = 0 is bit-for-bit today's `monomial_hash % P`, so the refactor cannot move a single term;
//   * at d = log2(R) the destination RANK of M^G is rank(M) ^ shift(G), which is what turns the dense
//     all-to-all into a pairwise exchange. A break here is silent -- wrong owner, not a crash -- so the
//     shift identity and the Scan/find_rank agreement are both asserted explicitly.
//
// Flat cases with a shared prefix, no suite nesting (suites break Boost's ctest discovery here).

#include <boost/test/unit_test.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/Routing.h"

using namespace monoprop;
using monoprop::routing::Router;

namespace {

constexpr size_t kN = 64; // 2N = 128 bits -> 2 words, so the multi-word hash path is exercised

// Exactly `weight` distinct bits, so a case can pin the popcount the old bit-walk paid for.
auto monomials_of_weight(size_t count, size_t weight, uint64_t seed) -> std::vector<Monomial<kN>> {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> slot(0, 2 * kN - 1);
    std::vector<Monomial<kN>> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        Monomial<kN> m;
        while (m.count() < weight) {
            m.set(slot(rng));
        }
        out.push_back(m);
    }
    return out;
}

// The map Router::dest computed before the basis was transposed: load one 64-bit vector per SET bit and
// XOR. Rebuilt from mix64 and the seed rather than read from linear_basis(), so the plane build and this
// share nothing but the specification.
template <size_t NumBits>
auto linear_hash_reference(const monoprop::Bitset<NumBits> &bits) -> uint64_t {
    const uint64_t seed = routing::seed_from_env();
    uint64_t h = 0;
    for (size_t i = bits.find_first(); i < NumBits; i = bits.find_next(i)) {
        h ^= routing::mix64(routing::mix64(seed) + (static_cast<uint64_t>(i) * 0x9E37'79B9'7F4A'7C15ULL));
    }
    return h;
}

auto random_monomials(size_t count, size_t weight, uint64_t seed) -> std::vector<Monomial<kN>> {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> slot(0, 2 * kN - 1);
    std::vector<Monomial<kN>> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        VecZ inds;
        for (size_t k = 0; k < weight; ++k) {
            inds.push_back(slot(rng));
        }
        out.push_back(indices_to_bitset<kN>(inds));
    }
    return out;
}

} // namespace

// d = 0 must not move a single term relative to `monomial_hash % P`: this is the regression gate that
// licenses everything else.
BOOST_AUTO_TEST_CASE(routing_zero_bits_is_bit_identical_to_splitmix) {
    const auto monos = random_monomials(500, 5, 0xC0FFEEULL);
    for (const size_t flat : {size_t{1}, size_t{2}, size_t{7}, size_t{112}, size_t{1792}}) {
        const auto router = Router::splitmix(flat);
        BOOST_TEST(router.linear_bits() == 0U);
        for (const auto &m : monos) {
            const size_t expected = monomial_hash<kN>(m) % flat;
            BOOST_TEST(router.dest<kN>(m) == expected);
            BOOST_TEST(find_rank<kN>(m, router) == expected);
        }
    }
}

// A two-level router with d = 0 is still today's routing, even though it knows about partitions:
// ((q/S) % R)*S + q%S == q % (R*S).
BOOST_AUTO_TEST_CASE(routing_zero_bits_two_level_collapses_to_flat_modulo) {
    const auto monos = random_monomials(300, 6, 0xBEEF01ULL);
    for (const auto [r, s] : {std::pair<size_t, size_t>{8, 14}, {4, 28}, {128, 14}, {64, 28}}) {
        const auto router = Router::for_modes<kN>(r, s, 0);
        for (const auto &m : monos) {
            BOOST_TEST(router.dest<kN>(m) == monomial_hash<kN>(m) % (r * s));
        }
    }
}

BOOST_AUTO_TEST_CASE(routing_linear_hash_is_gf2_linear) {
    const auto a = random_monomials(200, 5, 0x1234ULL);
    const auto b = random_monomials(200, 3, 0x5678ULL);
    for (size_t i = 0; i < a.size(); ++i) {
        const uint64_t ha = routing::linear_hash<2 * kN>(a[i]);
        const uint64_t hb = routing::linear_hash<2 * kN>(b[i]);
        BOOST_TEST(routing::linear_hash<2 * kN>(a[i] ^ b[i]) == (ha ^ hb));
    }
    // The empty monomial is the identity of the group, so its hash must be the identity of the codomain.
    BOOST_TEST(routing::linear_hash<2 * kN>(Monomial<kN>{}) == 0ULL);
}

// The load-bearing identity: at full linear bits the destination RANK of M^G is rank(M) ^ shift(G),
// so one rank's queries for one generator all land on one peer.
BOOST_AUTO_TEST_CASE(routing_shift_identity_holds_at_full_bits) {
    constexpr size_t kRanks = 16;
    constexpr size_t kParts = 14;
    const auto router = Router::for_modes<kN>(kRanks, kParts, 64); // clamped to log2(16) == 4
    BOOST_REQUIRE(router.linear_bits() == 4U);
    BOOST_REQUIRE(router.fanout() == 1U);

    const auto terms = random_monomials(400, 6, 0xAAAA01ULL);
    const auto gens = random_monomials(40, 4, 0xBBBB02ULL);
    for (const auto &g : gens) {
        const size_t shift = router.rank_shift<kN>(g);
        for (const auto &m : terms) {
            const size_t rank_m = router.dest<kN>(m) / kParts;
            const size_t rank_mg = router.dest<kN>(m ^ g) / kParts;
            BOOST_TEST(rank_mg == (rank_m ^ shift));
        }
    }
}

// The consequence that the transport will rely on: every term a rank owns sends its query for one
// generator to exactly ONE peer rank -- and the partition index within that peer still spreads.
BOOST_AUTO_TEST_CASE(routing_fanout_is_one_at_full_bits) {
    constexpr size_t kRanks = 8;
    constexpr size_t kParts = 14;
    const auto router = Router::for_modes<kN>(kRanks, kParts, 3);
    const auto terms = random_monomials(4000, 6, 0xCCCC03ULL);
    const auto gens = random_monomials(12, 4, 0xDDDD04ULL);

    std::vector<std::vector<Monomial<kN>>> owned(kRanks);
    for (const auto &m : terms) {
        owned[router.dest<kN>(m) / kParts].push_back(m);
    }
    for (size_t src = 0; src < kRanks; ++src) {
        BOOST_REQUIRE(!owned[src].empty());
        for (const auto &g : gens) {
            std::set<size_t> dest_ranks;
            std::set<size_t> dest_parts;
            for (const auto &m : owned[src]) {
                const size_t flat = router.dest<kN>(m ^ g);
                dest_ranks.insert(flat / kParts);
                dest_parts.insert(flat % kParts);
            }
            BOOST_TEST(dest_ranks.size() == 1U);
            BOOST_TEST(dest_parts.size() > 1U); // partitions keep full avalanche
        }
    }
}

// d in between: the low d bits shift deterministically and the high log2(R)-d are splitmix, so the
// realised fanout is R >> d -- the dial the balance/fanout trade is made on.
BOOST_AUTO_TEST_CASE(routing_partial_bits_give_fanout_ranks_over_two_to_the_d) {
    constexpr size_t kRanks = 32;
    constexpr size_t kParts = 14;
    const auto terms = random_monomials(20000, 6, 0xEEEE05ULL);
    const auto gen = random_monomials(1, 4, 0xFFFF06ULL).front();
    for (size_t d = 0; d <= 5; ++d) {
        const auto router = Router::for_modes<kN>(kRanks, kParts, d);
        BOOST_TEST(router.fanout() == (kRanks >> d));
        std::vector<std::set<size_t>> dest_of(kRanks);
        for (const auto &m : terms) {
            const size_t src = router.dest<kN>(m) / kParts;
            dest_of[src].insert(router.dest<kN>(m ^ gen) / kParts);
        }
        for (size_t src = 0; src < kRanks; ++src) {
            BOOST_TEST(dest_of[src].size() <= router.fanout());
        }
    }
}

// Without a power-of-two rank count there is no XOR structure to exploit, so the router must fall back
// to today's routing rather than silently produce a lopsided or out-of-range slot.
BOOST_AUTO_TEST_CASE(routing_non_power_of_two_ranks_falls_back_to_zero_bits) {
    for (const size_t r : {size_t{3}, size_t{7}, size_t{12}, size_t{112}}) {
        const auto router = Router::for_modes<kN>(r, 14, 8);
        BOOST_TEST(router.linear_bits() == 0U);
        BOOST_TEST(router.fanout() == r);
    }
    const auto pow2 = Router::for_modes<kN>(64, 14, 8);
    BOOST_TEST(pow2.linear_bits() == 6U); // clamped to log2(64), not 8
}

BOOST_AUTO_TEST_CASE(routing_dest_is_in_range_and_deterministic) {
    const auto monos = random_monomials(1000, 7, 0x9999ULL);
    for (const auto [r, s] : {std::pair<size_t, size_t>{1, 1}, {1, 112}, {8, 14}, {128, 14}, {64, 28}}) {
        for (size_t d = 0; d <= 7; ++d) {
            const auto router = Router::for_modes<kN>(r, s, d);
            for (const auto &m : monos) {
                const size_t slot = router.dest<kN>(m);
                BOOST_TEST(slot < r * s);
                BOOST_TEST(slot == router.dest<kN>(m)); // stateless
            }
        }
    }
}

// The transposed basis must be the SAME map, not merely a faster one: a divergence is a silently wrong
// owner. Pin dest() and rank_shift() against the old bit-walk over the geometries the dial spans --
// d = 0, d < log2(R), d == log2(R) -- and over popcounts from empty to full support.
BOOST_AUTO_TEST_CASE(routing_transposed_basis_is_bit_identical_to_the_bit_walk) {
    std::vector<Monomial<kN>> monos;
    for (const size_t w : {size_t{0},
                           size_t{1},
                           size_t{2},
                           size_t{3},
                           size_t{5},
                           size_t{8},
                           size_t{13},
                           size_t{21},
                           size_t{34},
                           size_t{2 * kN}}) {
        const auto batch = monomials_of_weight(500, w, 0xD15EA5E0ULL + w);
        monos.insert(monos.end(), batch.begin(), batch.end());
    }
    BOOST_REQUIRE_EQUAL(monos.size(), 5000U);

    // (R, S, d). log2(R) is 7, 4, 6, 3, 10, 1, 12, 5 respectively, so both d < log2(R) and d == log2(R)
    // appear, as does d = 0.
    const std::vector<std::array<size_t, 3>> geometries{
        {128, 14, 0},  {128, 14, 1}, {128, 14, 3}, {128, 14, 6},   {128, 14, 7}, {16, 1, 0}, {16, 1, 2}, {16, 1, 4},
        {64, 28, 1},   {64, 28, 5},  {64, 28, 6},  {8, 14, 0},     {8, 14, 1},   {8, 14, 2}, {8, 14, 3}, {1024, 1, 5},
        {1024, 1, 10}, {2, 112, 0},  {2, 112, 1},  {4096, 16, 12}, {32, 3, 4},   {32, 3, 5}};

    size_t checked = 0;
    for (const auto &[r, s, d] : geometries) {
        const auto router = Router::for_modes<kN>(r, s, d);
        BOOST_REQUIRE_EQUAL(router.linear_bits(), d); // no clamping in this table
        const uint64_t lin_mask = d == 0 ? 0ULL : (uint64_t{1} << d) - 1;
        for (const auto &m : monos) {
            const uint64_t q = monomial_hash<kN>(m);
            size_t expected = 0;
            if (d == 0) {
                expected = static_cast<size_t>(q % (r * s));
            }
            else {
                const uint64_t part = q % s;
                const uint64_t hi = (q / s) % (r >> d);
                const uint64_t lin = linear_hash_reference<2 * kN>(m) & lin_mask;
                expected = static_cast<size_t>(((lin | (hi << d)) * s) + part);
            }
            BOOST_REQUIRE_EQUAL(router.dest<kN>(m), expected);
            BOOST_REQUIRE_EQUAL(router.rank_shift<kN>(m),
                                static_cast<size_t>(linear_hash_reference<2 * kN>(m) & lin_mask));
            ++checked;
        }
    }
    BOOST_TEST_MESSAGE("bit-identity checks: " << checked);
    BOOST_TEST(checked >= 100000U);
}

// gf2_rank is the coverage diagnostic: shifts that span fewer than log2(R) dimensions leave ranks empty.
BOOST_AUTO_TEST_CASE(routing_gf2_rank_detects_a_degenerate_shift_set) {
    BOOST_TEST(routing::gf2_rank({}) == 0U);
    BOOST_TEST(routing::gf2_rank({0ULL, 0ULL}) == 0U);
    BOOST_TEST(routing::gf2_rank({0b001ULL, 0b010ULL, 0b011ULL}) == 2U); // third is the XOR of the first two
    BOOST_TEST(routing::gf2_rank({0b001ULL, 0b010ULL, 0b100ULL}) == 3U);

    // The real generator shifts must span at least log2(R) dimensions or the reachable ranks are a
    // strict subspace of the rank space.
    constexpr size_t kRanks = 128;
    const auto router = Router::for_modes<kN>(kRanks, 14, 7);
    std::vector<uint64_t> shifts;
    for (const auto &g : random_monomials(200, 4, 0x7777ULL)) {
        shifts.push_back(static_cast<uint64_t>(router.rank_shift<kN>(g)));
    }
    // Seed-independent, so this is NOT skipped when monoprop_ROUTE_SEED is overridden: 200 vectors fail
    // to span F_2^7 with probability ~2^-194, whatever basis the seed picks.
    BOOST_TEST(routing::gf2_rank(shifts) == 7U); // == log2(128): every rank is reachable
}

// The SHIPPED default. Flipping this is the whole point of the change, so it is pinned by a test
// rather than left to a comment: with no environment override, a power-of-two rank count routes at
// fanout 1, and a geometry with no XOR structure keeps the dense path instead of silently losing
// ranks. Skipped when the environment does override it, because then the default is not what is
// under test.
BOOST_AUTO_TEST_CASE(routing_default_is_linear_where_the_geometry_allows_it) {
    const char *mode = std::getenv("monoprop_ROUTING");
    const char *bits = std::getenv("monoprop_ROUTE_LINEAR_BITS");
    if ((mode != nullptr && *mode != '\0') || (bits != nullptr && *bits != '\0')) {
        BOOST_TEST_MESSAGE("routing overridden in the environment; default not under test");
        return;
    }
    BOOST_TEST(routing::make_router<kN>(8, 14).fanout() == 1U);
    BOOST_TEST(routing::make_router<kN>(128, 14).fanout() == 1U);
    BOOST_TEST(routing::make_router<kN>(1, 112).fanout() == 1U); // single rank: nothing to route between

    // 6 and 12 are not powers of two: no XOR structure, so Router clamps to d = 0 and every rank
    // stays reachable through splitmix rather than a subspace of them.
    BOOST_TEST(routing::make_router<kN>(6, 14).fanout() == 6U);
    BOOST_TEST(routing::make_router<kN>(12, 28).fanout() == 12U);
}
