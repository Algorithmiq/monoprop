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
// term's positions, and the (k, d) digest predicates against the bitset cutoffs they replace.

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "monoprop/algebra/Algebra.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/core/Monomial.h"
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

//! The paired-mode count of an ascending position list, written against the definition.
auto reference_paired(const std::vector<uint16_t> &pos) -> size_t {
    size_t d = 0;
    for (size_t j = 0; j + 1 < pos.size(); ++j) {
        if ((pos[j] % 2 == 0) && (pos[j + 1] == pos[j] + 1)) {
            ++d;
        }
    }
    return d;
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

// The (k, d) digest predicates against the bitset ones, for both structural cutoffs and the paired
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
        const auto term = draw(rng() % 9);
        std::vector<uint16_t> pos;
        for (size_t b = term.find_first(); b < term.size(); b = term.find_next(b)) {
            pos.push_back(static_cast<uint16_t>(b));
        }
        const size_t k = term.count();
        const size_t d = reference_paired(pos);
        // The digest is exactly what the two bitset sums carry: k − 2d is the xor sum, k − d the or sum.
        const auto sums = cutoff_sums<kN>(term, kLogical);
        BOOST_REQUIRE_EQUAL(k - (2 * d), sums.xor_sum);
        BOOST_REQUIRE_EQUAL(k - d, sums.or_sum);
        BOOST_REQUIRE_EQUAL(digest_is_paired(k, d), is_paired<kN>(term));
        paired_seen += static_cast<size_t>(digest_is_paired(k, d) && k > 0);
        for (const unsigned int cutoff : {0U, 2U, 4U, 6U}) {
            BOOST_REQUIRE_EQUAL(length_keeps(k, d, cutoff), length_cutoff<kN>(term, cutoff, kLogical));
            BOOST_REQUIRE_EQUAL(support_keeps(k, d, cutoff), support_cutoff<kN>(term, cutoff, kLogical));
            const CutoffFn<kN> lfn = detail::LengthCutoff<kN>{cutoff, kLogical};
            const detail::CutoffEvaluator<kN> leval(lfn);
            BOOST_REQUIRE(leval.has_digest_form());
            BOOST_REQUIRE_EQUAL(leval.passes_from_digest(k, d), leval.passes_with_popcount(term, k));
            const CutoffFn<kN> sfn = detail::SupportCutoff<kN>{cutoff, kLogical};
            const detail::CutoffEvaluator<kN> seval(sfn);
            BOOST_REQUIRE(seval.has_digest_form());
            BOOST_REQUIRE_EQUAL(seval.passes_from_digest(k, d), seval.passes_with_popcount(term, k));
        }
    }
    BOOST_TEST(paired_seen > 0U);
    // An opaque cutoff has no digest form, so nothing may answer it from (k, d).
    const CutoffFn<kN> opaque = [](const Monomial<kN> &m) { return m.count() < 3; };
    BOOST_TEST(!detail::CutoffEvaluator<kN>(opaque).has_digest_form());
}
