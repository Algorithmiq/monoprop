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

// routing::Router -- the term -> flat-slot map. Two states, two properties that carry the whole design:
//   * splitmix is bit-for-bit today's `monomial_hash % P`, so the refactor cannot move a single term;
//   * under linear routing the destination of M^G is read off fp(M) ^ fp(G), and with a power-of-two
//     partition count the whole slot obeys flat(M^G) == flat(M) ^ flat_shift(G), which is what turns
//     the dense all-to-all into a pairwise exchange. A break here is silent -- wrong owner, not a crash
//     -- so the shift identity and the Scan/find_rank agreement are both asserted explicitly.
//
// Flat cases with a shared prefix, no suite nesting (suites break Boost's ctest discovery here).

#include <boost/test/unit_test.hpp>

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/mpi/Comm.h"
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

// The specified map: load one 64-bit label per SET bit and XOR. Rebuilt from mix64 and the seed rather
// than read from linear_basis(), so the router and this share nothing but the specification.
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

// The splitmix router must not move a single term relative to `monomial_hash % P`: this is the
// regression gate that licenses everything else.
BOOST_AUTO_TEST_CASE(routing_splitmix_is_bit_identical_to_hash_mod_p) {
    const auto monos = random_monomials(500, 5, 0xC0FFEEULL);
    for (const size_t flat : {size_t{1}, size_t{2}, size_t{7}, size_t{112}, size_t{1792}}) {
        const auto router = Router::splitmix(flat);
        BOOST_TEST(!router.is_linear());
        for (const auto &m : monos) {
            const size_t expected = monomial_hash<kN>(m) % flat;
            BOOST_TEST(router.dest<kN>(m) == expected);
            BOOST_TEST(find_rank<kN>(m, router) == expected);
        }
    }
}

// A two-level splitmix router is still today's routing, even though it knows about partitions:
// ((q/S) % R)*S + q%S == q % (R*S). 12 ranks is here too: splitmix carries no power-of-two condition.
BOOST_AUTO_TEST_CASE(routing_splitmix_two_level_collapses_to_flat_modulo) {
    const auto monos = random_monomials(300, 6, 0xBEEF01ULL);
    for (const auto [r, s] : {std::pair<size_t, size_t>{8, 14}, {4, 28}, {128, 14}, {64, 28}, {12, 5}}) {
        const auto router = Router::for_modes<kN>(r, s, false);
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

// The load-bearing identity: under linear routing the destination RANK of M^G is rank(M) ^ shift(G),
// so one rank's queries for one generator all land on one peer.
BOOST_AUTO_TEST_CASE(routing_shift_identity_holds_under_linear_routing) {
    constexpr size_t kRanks = 16;
    constexpr size_t kParts = 14;
    const auto router = Router::for_modes<kN>(kRanks, kParts, true);
    BOOST_REQUIRE(router.is_linear());
    BOOST_REQUIRE(router.linear_bits() == 4U); // every rank bit, log2(16)

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

// What the emit path actually calls: the partner's slot off its fingerprint fp(M) ^ fp(G). The check is
// exact equality against dest() over every linear geometry -- S == 1 (no partition index at all), S a
// power of two (the linear partition bits) and S = 3, 14 (the mixed modulo) are all here -- and, where
// the whole slot is linear, the shift identity flat(M^G) == flat(M) ^ flat_shift(G).
BOOST_AUTO_TEST_CASE(routing_dest_from_fingerprint_agrees_with_dest) {
    const auto terms = random_monomials(400, 6, 0x5E1F7A11ULL);
    const auto gens = random_monomials(25, 4, 0x9110F7E5ULL);
    const std::vector<std::pair<size_t, size_t>>
        geometries{{8, 1}, {16, 1}, {2, 2}, {32, 16}, {64, 8}, {4, 3}, {8, 14}};
    size_t checked = 0;
    size_t shifted = 0;
    for (const auto &[r, s] : geometries) {
        const auto router = Router::for_modes<kN>(r, s, true);
        BOOST_REQUIRE(router.is_linear());
        BOOST_REQUIRE_EQUAL(router.is_flat_linear(), std::has_single_bit(s));
        for (const auto &g : gens) {
            const uint64_t fp_g = routing::linear_hash<2 * kN>(g);
            const auto flat_shift = router.flat_shift<kN>(g);
            BOOST_REQUIRE_EQUAL(flat_shift.has_value(), router.is_flat_linear());
            for (const auto &m : terms) {
                const auto partner = m ^ g;
                const uint64_t fp_m = routing::linear_hash<2 * kN>(m);
                BOOST_REQUIRE_EQUAL(router.dest_from_fingerprint(fp_m ^ fp_g), router.dest<kN>(partner));
                BOOST_REQUIRE_EQUAL(router.dest_from_fingerprint(fp_m), router.dest<kN>(m));
                if (flat_shift) {
                    BOOST_REQUIRE_EQUAL(router.dest<kN>(partner), router.dest<kN>(m) ^ *flat_shift);
                    ++shifted;
                }
                ++checked;
            }
        }
    }
    BOOST_TEST_MESSAGE("dest_from_fingerprint checks: " << checked << " flat-shift checks: " << shifted);
    BOOST_TEST(checked >= 50000U);
    BOOST_TEST(shifted >= 20000U);
}

// The rank shift is the low log2(R) bits of the generator's image, zero under splitmix.
BOOST_AUTO_TEST_CASE(routing_rank_shift_is_the_low_rank_bits) {
    const auto gens = random_monomials(50, 4, 0x1234ULL);
    for (const auto &g : gens) {
        BOOST_TEST(Router::for_modes<kN>(16, 3, true).rank_shift<kN>(g) == (routing::linear_hash<2 * kN>(g) & 15U));
        BOOST_TEST(Router::for_modes<kN>(16, 3, false).rank_shift<kN>(g) == 0U);
        BOOST_TEST(!Router::for_modes<kN>(16, 3, false).flat_shift<kN>(g).has_value());
        BOOST_TEST(!Router::for_modes<kN>(1, 8, true).flat_shift<kN>(g).has_value()); // R == 1 is dense
    }
}

// The S == 1 skip: the partition index is 0 for every term, so under linear routing the destination is
// the rank index alone and monomial_hash is not on the path at all.
BOOST_AUTO_TEST_CASE(routing_single_partition_destination_is_the_rank_index) {
    constexpr size_t kRanks = 64;
    const auto router = Router::for_modes<kN>(kRanks, 1, true);
    for (const auto &m : random_monomials(500, 5, 0x0FAE7101ULL)) {
        BOOST_REQUIRE_EQUAL(router.dest<kN>(m), routing::linear_hash<2 * kN>(m) & (kRanks - 1));
    }
}

// With S a power of two the whole slot is linear: every term a SLOT owns sends its query for one
// generator to exactly one peer slot, inside the rank as well as across ranks.
BOOST_AUTO_TEST_CASE(routing_fanout_is_one_slot_when_partitions_are_a_power_of_two) {
    constexpr size_t kRanks = 8;
    constexpr size_t kParts = 16;
    const auto router = Router::for_modes<kN>(kRanks, kParts, true);
    BOOST_REQUIRE(router.is_flat_linear());
    BOOST_REQUIRE_EQUAL(router.flat_linear_bits(), 7U);
    const auto terms = random_monomials(6000, 6, 0xCCCC03ULL);
    const auto gens = random_monomials(12, 4, 0xDDDD04ULL);
    std::vector<std::vector<Monomial<kN>>> owned(kRanks * kParts);
    for (const auto &m : terms) {
        owned[router.dest<kN>(m)].push_back(m);
    }
    size_t populated = 0;
    for (size_t slot = 0; slot < owned.size(); ++slot) {
        if (owned[slot].empty()) {
            continue;
        }
        ++populated;
        for (const auto &g : gens) {
            std::set<size_t> dests;
            for (const auto &m : owned[slot]) {
                dests.insert(router.dest<kN>(m ^ g));
            }
            BOOST_REQUIRE_EQUAL(dests.size(), 1U);
            BOOST_REQUIRE_EQUAL(*dests.begin(), slot ^ *router.flat_shift<kN>(g));
        }
    }
    BOOST_TEST(populated == kRanks * kParts); // 6000 terms over 128 cosets: every slot populated
}

// With S not a power of two only the rank level is linear: one peer rank, and the partition index within
// that peer still spreads.
BOOST_AUTO_TEST_CASE(routing_fanout_is_one_rank_when_partitions_are_not_a_power_of_two) {
    constexpr size_t kRanks = 8;
    constexpr size_t kParts = 14;
    const auto router = Router::for_modes<kN>(kRanks, kParts, true);
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

// mpi::PeerPlan::window -- the slots a plan can reach, which is what lets the per-generator structures
// be S long instead of P. Dense is the count == P value of the same two expressions, so it is checked
// against the same formula rather than against a second one.
BOOST_AUTO_TEST_CASE(routing_slot_window_is_the_peer_ranks_partition_run) {
    for (const auto [r, s] : {std::pair<size_t, size_t>{8, 16}, {128, 16}, {16, 1}, {1, 1}, {4, 3}}) {
        const size_t p = r * s;
        for (size_t me = 0; me < p; ++me) {
            const mpi::SlotWindow dense = mpi::PeerPlan{}.window(me, r, s);
            BOOST_REQUIRE_EQUAL(dense.base, 0U);
            BOOST_REQUIRE_EQUAL(dense.count, p);
            BOOST_REQUIRE(dense.contains(me));

            for (size_t shift = 0; shift < r; ++shift) {
                const mpi::PeerPlan plan{.sparse = true, .shift = static_cast<int>(shift)};
                const mpi::SlotWindow w = plan.window(me, r, s);
                BOOST_REQUIRE_EQUAL(w.count, s);
                BOOST_REQUIRE_EQUAL(w.base, ((me / s) ^ shift) * s);
                // Every slot in the run belongs to the one peer rank the plan names, and no other.
                for (size_t k = 0; k < w.count; ++k) {
                    const size_t slot = w.slot(mpi::WindowIndex{k});
                    BOOST_REQUIRE(w.contains(slot));
                    BOOST_REQUIRE_EQUAL(w.index(slot).value, k);
                    BOOST_REQUIRE_EQUAL(slot / s, ((me / s) ^ shift));
                    BOOST_REQUIRE(plan.contains(static_cast<int>(me / s), static_cast<int>(slot / s)));
                }
                BOOST_REQUIRE(!w.contains(w.base + w.count));
                // Self is reachable only at shift 0; the engine's self-resolve leg turns on that.
                BOOST_REQUIRE_EQUAL(w.contains(me), shift == 0);
            }
        }
    }
}

// The window must be symmetric, or the two ends of one exchange size different arrays: XOR is an
// involution, so the peer's own window points back at this rank's run.
BOOST_AUTO_TEST_CASE(routing_slot_window_pairing_is_symmetric) {
    constexpr size_t kRanks = 32;
    constexpr size_t kParts = 8;
    for (size_t me = 0; me < kRanks * kParts; me += 3) {
        for (size_t shift = 0; shift < kRanks; ++shift) {
            const mpi::PeerPlan plan{.sparse = true, .shift = static_cast<int>(shift)};
            const mpi::SlotWindow mine = plan.window(me, kRanks, kParts);
            const mpi::SlotWindow theirs = plan.window(mine.base, kRanks, kParts);
            BOOST_REQUIRE_EQUAL(theirs.base, (me / kParts) * kParts);
            BOOST_REQUIRE_EQUAL(theirs.count, kParts);
        }
    }
}

// The property the re-basing rests on: every destination the emit path can produce for one generator
// lies inside that generator's window, so `slot - base` is always a legal index.
BOOST_AUTO_TEST_CASE(routing_every_dest_lands_inside_the_generators_window) {
    const auto terms = random_monomials(600, 6, 0x5107500DULL);
    const auto gens = random_monomials(20, 4, 0x1CE0FF1CEULL);
    const std::vector<std::pair<size_t, size_t>> geometries{{8, 16}, {16, 1}, {32, 4}, {1, 14}, {4, 3}};
    size_t checked = 0;
    for (const auto &[r, s] : geometries) {
        for (const bool linear : {false, true}) {
            const auto router = Router::for_modes<kN>(r, s, linear);
            for (const auto &g : gens) {
                const size_t shift = router.rank_shift<kN>(g);
                const mpi::PeerPlan plan{.sparse = router.is_linear(), .shift = static_cast<int>(shift)};
                for (const auto &m : terms) {
                    const size_t me = router.dest<kN>(m);
                    const mpi::SlotWindow w = plan.window(me, r, s);
                    BOOST_REQUIRE(w.contains(router.dest<kN>(m ^ g)));
                    ++checked;
                }
            }
        }
    }
    BOOST_TEST_MESSAGE("window containment checks: " << checked);
    BOOST_TEST(checked >= 100000U);
}

// WindowVec re-bases in exactly one place, so the flat slot the writer used is the flat slot the reader
// gets back -- including when the run does not start at 0.
BOOST_AUTO_TEST_CASE(routing_window_vec_round_trips_flat_slots) {
    const mpi::SlotWindow w{.base = 48, .count = 16};
    mpi::WindowVec<size_t> v(w);
    BOOST_REQUIRE_EQUAL(v.size(), w.count);
    for (size_t slot = w.base; slot < w.stop(); ++slot) {
        v.at_slot(slot) = slot * 7;
    }
    for (size_t k = 0; k < w.count; ++k) {
        BOOST_REQUIRE_EQUAL(v[mpi::WindowIndex{k}], (w.base + k) * 7);
        BOOST_REQUIRE_EQUAL(v.at_slot(w.slot(mpi::WindowIndex{k})), (w.base + k) * 7);
    }
    v.reset(mpi::SlotWindow{.base = 0, .count = 4});
    BOOST_REQUIRE_EQUAL(v.size(), 4U);
    BOOST_REQUIRE_EQUAL(v.at_slot(3), 0U);
}

// Without a power-of-two rank count there is no XOR structure to exploit, and there is no partial dial
// to fall back to, so the geometry is rejected at construction rather than routed on a subspace.
BOOST_AUTO_TEST_CASE(routing_non_power_of_two_ranks_throw_under_linear_routing) {
    for (const size_t r : {size_t{3}, size_t{7}, size_t{12}, size_t{112}}) {
        BOOST_CHECK_THROW(static_cast<void>(Router::for_modes<kN>(r, 14, true)), routing::UnroutableGeometry);
        BOOST_CHECK_NO_THROW(static_cast<void>(Router::for_modes<kN>(r, 14, false))); // splitmix has no condition
    }
    BOOST_CHECK_NO_THROW(static_cast<void>(Router::for_modes<kN>(64, 14, true)));
}

// R = 1 is a power of two, so it must NOT throw: it has no rank bit to take, which makes it the dense
// router and keeps the collective transport that serves every single-rank run.
BOOST_AUTO_TEST_CASE(routing_single_rank_is_dense_and_not_an_error) {
    const auto monos = random_monomials(200, 5, 0x51A61EULL);
    for (const size_t s : {size_t{1}, size_t{14}, size_t{112}}) {
        const auto router = Router::for_modes<kN>(1, s, true);
        BOOST_TEST(!router.is_linear());
        BOOST_TEST(router.linear_bits() == 0U);
        for (const auto &m : monos) {
            BOOST_TEST(router.dest<kN>(m) == monomial_hash<kN>(m) % s);
            BOOST_TEST(router.rank_shift<kN>(m) == 0U); // no bit to shift, so every generator is on-rank
        }
    }
}

BOOST_AUTO_TEST_CASE(routing_dest_is_in_range_and_deterministic) {
    const auto monos = random_monomials(1000, 7, 0x9999ULL);
    for (const auto [r, s] : {std::pair<size_t, size_t>{1, 1}, {1, 112}, {8, 14}, {128, 14}, {64, 28}}) {
        for (const bool linear : {false, true}) {
            const auto router = Router::for_modes<kN>(r, s, linear);
            for (const auto &m : monos) {
                const size_t slot = router.dest<kN>(m);
                BOOST_TEST(slot < r * s);
                BOOST_TEST(slot == router.dest<kN>(m)); // stateless
            }
        }
    }
}

// dest() must be the specified map, bit for bit, on both routers and every geometry: a divergence is a
// silently wrong owner. The reference rebuilds the label table from mix64 and the seed, so it shares
// nothing with linear_basis() but the specification; fingerprint_positions is pinned to it as well.
BOOST_AUTO_TEST_CASE(routing_dest_is_bit_identical_to_the_specified_map) {
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

    const uint64_t *labels = routing::linear_basis<2 * kN>().data();
    const std::vector<std::pair<size_t, size_t>> geometries{{128, 14},
                                                            {16, 1},
                                                            {64, 28},
                                                            {8, 14},
                                                            {1024, 1},
                                                            {2, 112},
                                                            {4096, 16},
                                                            {32, 3},
                                                            {1, 112},
                                                            {256, 2},
                                                            {2048, 1}};

    size_t checked = 0;
    for (const auto &[r, s] : geometries) {
        for (const bool linear : {false, true}) {
            const auto router = Router::for_modes<kN>(r, s, linear);
            const size_t d = router.linear_bits();
            BOOST_REQUIRE_EQUAL(d, linear ? static_cast<size_t>(std::countr_zero(r)) : 0U);
            const uint64_t lin_mask = d == 0 ? 0ULL : (uint64_t{1} << d) - 1;
            for (const auto &m : monos) {
                const uint64_t q = monomial_hash<kN>(m);
                const uint64_t fp = linear_hash_reference<2 * kN>(m);
                std::vector<uint16_t> pos;
                for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
                    pos.push_back(static_cast<uint16_t>(b));
                }
                BOOST_REQUIRE_EQUAL(routing::fingerprint_positions(labels, pos.data(), pos.size()), fp);
                BOOST_REQUIRE_EQUAL(routing::linear_hash<2 * kN>(m), fp);
                size_t expected = 0;
                if (d == 0) {
                    expected = static_cast<size_t>(q % (r * s)); // splitmix, and R = 1 under linear routing
                }
                else {
                    const size_t rank = static_cast<size_t>(fp & lin_mask);
                    const size_t part = std::has_single_bit(s) ? static_cast<size_t>((fp >> d) & (s - 1))
                                                               : static_cast<size_t>(routing::mix64(fp) % s);
                    expected = (rank * s) + part;
                }
                BOOST_REQUIRE_EQUAL(router.dest<kN>(m), expected);
                BOOST_REQUIRE_EQUAL(router.rank_shift<kN>(m), static_cast<size_t>(fp & lin_mask));
                ++checked;
            }
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
    const auto router = Router::for_modes<kN>(kRanks, 14, true);
    std::vector<uint64_t> shifts;
    for (const auto &g : random_monomials(200, 4, 0x7777ULL)) {
        shifts.push_back(static_cast<uint64_t>(router.rank_shift<kN>(g)));
    }
    // Seed-independent, so this is NOT skipped when monoprop_ROUTE_SEED is overridden: 200 vectors fail
    // to span F_2^7 with probability ~2^-194, whatever basis the seed picks.
    BOOST_TEST(routing::gf2_rank(shifts) == 7U); // == log2(128): every rank is reachable
}

// The SHIPPED default, pinned by a test rather than left to a comment: with no environment override a
// power-of-two rank count routes linearly at fanout 1, a single rank routes densely, and a geometry
// with no XOR structure is refused. Skipped when the environment does override it, because then the
// default is not what is under test.
BOOST_AUTO_TEST_CASE(routing_default_is_linear_where_the_geometry_allows_it) {
    const char *mode = std::getenv("monoprop_ROUTING");
    if (mode != nullptr && *mode != '\0') {
        BOOST_TEST_MESSAGE("routing overridden in the environment; default not under test");
        return;
    }
    BOOST_TEST(routing::make_router<kN>(8, 14).is_linear());
    BOOST_TEST(routing::make_router<kN>(128, 14).linear_bits() == 7U);
    BOOST_TEST(!routing::make_router<kN>(1, 112).is_linear()); // single rank: nothing to route between

    // 6 and 12 are not powers of two: no XOR structure and no partial dial to retreat to, so the
    // geometry is rejected instead of silently routing onto a subspace of the ranks.
    BOOST_CHECK_THROW(routing::make_router<kN>(6, 14), routing::UnroutableGeometry);
    BOOST_CHECK_THROW(routing::make_router<kN>(12, 28), routing::UnroutableGeometry);
}
