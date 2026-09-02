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

// The position-only emit kernels against their dense oracles: the rotation sign of both algebras from a
// term's positions, the (k, d) digest of the partner merge against the bitset cutoff, and the partner
// product as a whole (emit_partner) against the dense M ^ G.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/operator/OperatorIndex.h"

using namespace monoprop;

namespace {

template <size_t NumModes>
auto random_monomial(std::mt19937_64 &rng, size_t k) -> Monomial<NumModes> {
    Monomial<NumModes> m;
    std::uniform_int_distribution<size_t> bit(0, Monomial<NumModes>::size() - 1);
    k = std::min(k, Monomial<NumModes>::size()); // a narrow system cannot hold more positions than it has
    while (m.count() < k) {
        m.set(bit(rng));
    }
    return m;
}

template <size_t NumModes>
auto positions_of(const Monomial<NumModes> &m) -> std::vector<typename detail::OperatorIndex<NumModes>::PosT> {
    std::vector<typename detail::OperatorIndex<NumModes>::PosT> pos;
    for (size_t b = m.find_first(); b < m.size(); b = m.find_next(b)) {
        pos.push_back(static_cast<typename detail::OperatorIndex<NumModes>::PosT>(b));
    }
    return pos;
}

template <size_t NumModes, Algebra A>
auto check_sign_from_positions(uint64_t seed, size_t max_gen_weight) -> void {
    std::mt19937_64 rng(seed);
    size_t checked = 0;
    for (size_t trial = 0; trial < 3000; ++trial) {
        const auto gen = random_monomial<NumModes>(rng, 1 + (rng() % max_gen_weight));
        const auto ctx = A::make_gen_context(gen);
        BOOST_REQUIRE(A::sign_from_positions_ok(ctx));
        for (size_t t = 0; t < 4; ++t) {
            const auto mono = random_monomial<NumModes>(rng, rng() % 12);
            const auto pos = positions_of<NumModes>(mono);
            const int dense = A::rotation_sign(ctx, mono, mono ^ gen);
            const int sparse = A::rotation_sign_positions(ctx, pos.data(), pos.size());
            BOOST_REQUIRE_EQUAL(dense, sparse);
            ++checked;
        }
    }
    BOOST_TEST(checked == 12000U);
}

} // namespace

BOOST_AUTO_TEST_CASE(position_sign_matches_dense_majorana) {
    check_sign_from_positions<4, MajoranaAlgebra<4>>(1, 4);
    check_sign_from_positions<32, MajoranaAlgebra<32>>(2, 8);
    check_sign_from_positions<64, MajoranaAlgebra<64>>(3, 10);
    check_sign_from_positions<128, MajoranaAlgebra<128>>(4, 12);
    check_sign_from_positions<250, MajoranaAlgebra<250>>(5, 16);
}

BOOST_AUTO_TEST_CASE(position_sign_matches_dense_pauli) {
    check_sign_from_positions<4, PauliAlgebra<4>>(11, 4);
    check_sign_from_positions<32, PauliAlgebra<32>>(12, 8);
    check_sign_from_positions<64, PauliAlgebra<64>>(13, 12);
    check_sign_from_positions<128, PauliAlgebra<128>>(14, 16);
    check_sign_from_positions<250, PauliAlgebra<250>>(15, 20);
}

// A Pauli generator on more than 32 qubits has no compact word; the kernel says so instead of guessing.
BOOST_AUTO_TEST_CASE(position_sign_pauli_declines_wide_generators) {
    constexpr size_t kN = 64;
    Monomial<kN> wide;
    for (size_t q = 0; q < 33; ++q) {
        wide.set(2 * q);
    }
    Monomial<kN> narrow;
    for (size_t q = 0; q < 32; ++q) {
        narrow.set(2 * q);
    }
    BOOST_TEST(!PauliAlgebra<kN>::sign_from_positions_ok(PauliAlgebra<kN>::make_gen_context(wide)));
    BOOST_TEST(PauliAlgebra<kN>::sign_from_positions_ok(PauliAlgebra<kN>::make_gen_context(narrow)));
}

// The merge's (k, d) digest against the bitset predicates, for both structural cutoffs and the paired
// exception, on well-formed monomials over a logical width below the storage width.
BOOST_AUTO_TEST_CASE(position_digest_matches_the_bitset_cutoffs) {
    constexpr size_t kN = 32;
    constexpr size_t kLogical = 30;
    std::mt19937_64 rng(77);
    std::uniform_int_distribution<size_t> pos_dist(2 * (kN - kLogical), (2 * kN) - 1);
    const auto draw = [&](size_t k) {
        Monomial<kN> m;
        while (m.count() < k) {
            m.set(pos_dist(rng));
        }
        return m;
    };
    size_t paired_seen = 0;
    for (size_t trial = 0; trial < 5000; ++trial) {
        const auto a = draw(rng() % 9);
        const auto g = draw(1 + (rng() % 4));
        const auto pa = positions_of<kN>(a);
        std::vector<uint16_t> pg;
        for (size_t b = g.find_first(); b < g.size(); b = g.find_next(b)) {
            pg.push_back(static_cast<uint16_t>(b));
        }
        std::vector<detail::OperatorIndex<kN>::PosT> out(2 * kN);
        const auto merged = detail::merge_partner_positions(pa, pg, out);
        const auto partner = a ^ g;
        BOOST_REQUIRE_EQUAL(merged.count, partner.count());
        BOOST_REQUIRE_EQUAL(merged.overlap, a.count_and(g));
        const auto sums = cutoff_sums<kN>(partner, kLogical);
        BOOST_REQUIRE_EQUAL(merged.count - (2 * merged.paired), sums.xor_sum);
        BOOST_REQUIRE_EQUAL(merged.count - merged.paired, sums.or_sum);
        BOOST_REQUIRE_EQUAL(digest_is_paired(merged.count, merged.paired), is_paired<kN>(partner));
        paired_seen += static_cast<size_t>(digest_is_paired(merged.count, merged.paired) && merged.count > 0);
        for (const unsigned int cutoff : {0U, 2U, 4U, 6U}) {
            BOOST_REQUIRE_EQUAL(length_keeps(merged.count, merged.paired, cutoff),
                                length_cutoff<kN>(partner, cutoff, kLogical));
            BOOST_REQUIRE_EQUAL(support_keeps(merged.count, merged.paired, cutoff),
                                support_cutoff<kN>(partner, cutoff, kLogical));
            const CutoffFn<kN> lfn = detail::LengthCutoff<kN>{cutoff, kLogical};
            const detail::CutoffEvaluator<kN> leval(lfn);
            BOOST_REQUIRE(leval.has_digest_form());
            BOOST_REQUIRE_EQUAL(leval.passes_from_digest(merged.count, merged.paired),
                                leval.passes_with_popcount(partner, merged.count));
            const CutoffFn<kN> sfn = detail::SupportCutoff<kN>{cutoff, kLogical};
            const detail::CutoffEvaluator<kN> seval(sfn);
            BOOST_REQUIRE_EQUAL(seval.passes_from_digest(merged.count, merged.paired),
                                seval.passes_with_popcount(partner, merged.count));
        }
    }
    BOOST_TEST(paired_seen > 0U);
    const CutoffFn<kN> opaque = [](const Monomial<kN> &m) { return m.count() < 3; };
    BOOST_TEST(!detail::CutoffEvaluator<kN>(opaque).has_digest_form());
}

// emit_partner as a whole, inline and spilled rows, against the dense product.
BOOST_AUTO_TEST_CASE(emit_partner_matches_the_dense_product) {
    constexpr size_t kN = 48;
    std::mt19937_64 rng(4242);
    const size_t n = 300;
    std::vector<Monomial<kN>> terms;
    detail::OperatorIndex<kN> store(5); // rows above 5 positions spill
    store.grow_rows_geometric(n);
    for (size_t i = 0; i < n; ++i) {
        terms.push_back(random_monomial<kN>(rng, rng() % 9));
        store.set(i, terms[i]);
    }
    BOOST_REQUIRE(store.overflow_size() > 0U);
    using A = MajoranaAlgebra<kN>;
    std::vector<detail::OperatorIndex<kN>::PosT> out(2 * kN);
    for (size_t trial = 0; trial < 20; ++trial) {
        const auto gen = random_monomial<kN>(rng, 1 + (rng() % 5));
        const auto ctx = A::make_gen_context(gen);
        std::vector<uint16_t> gen_pos;
        for (size_t b = gen.find_first(); b < gen.size(); b = gen.find_next(b)) {
            gen_pos.push_back(static_cast<uint16_t>(b));
        }
        for (const bool need_dense : {false, true}) {
            for (size_t i = 0; i < n; ++i) {
                const auto p = detail::emit_partner<kN, A>(store,
                                                           i,
                                                           store.row_positions(i),
                                                           ctx,
                                                           std::span<const uint16_t>(gen_pos),
                                                           std::span<detail::OperatorIndex<kN>::PosT>(out),
                                                           need_dense);
                const auto partner = terms[i] ^ gen;
                BOOST_REQUIRE_EQUAL(p.k, partner.count());
                BOOST_REQUIRE_EQUAL(p.overlap, terms[i].count_and(gen));
                BOOST_REQUIRE_EQUAL(p.paired,
                                    cutoff_sums<kN>(partner, kN).or_sum - cutoff_sums<kN>(partner, kN).xor_sum);
                BOOST_REQUIRE_EQUAL(p.phase_factor, A::rotation_sign(ctx, terms[i], partner));
                Monomial<kN> from_pos;
                for (size_t j = 0; j < p.k; ++j) {
                    from_pos.set(static_cast<size_t>(out[j]));
                }
                BOOST_REQUIRE(from_pos == partner);
                if (need_dense || !store.row_positions(i).inlined()) {
                    BOOST_REQUIRE(p.has_dense);
                    BOOST_REQUIRE(p.dense == partner);
                }
            }
        }
    }
}
