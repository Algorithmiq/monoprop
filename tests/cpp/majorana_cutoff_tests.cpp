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

// Sets are built directly in raw-bit space (Monomial::set) so the "fully paired" condition is
// unambiguous: a pair is raw bits (2k, 2k+1).

#include <boost/test/unit_test.hpp>

#include <complex>
#include <cstdint>
#include <random>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"

using namespace monoprop;
using cd = std::complex<double>;

// Raw bits {0,1} and {4,5} are two complete pairs.
BOOST_AUTO_TEST_CASE(majorana_cutoff_paired_kept_unconditionally) {
    constexpr size_t N = 32;
    Monomial<N> paired;
    paired.set(0);
    paired.set(1);
    paired.set(4);
    paired.set(5);
    BOOST_TEST(is_paired<N>(paired));
    BOOST_TEST(length_cutoff<N>(paired, 0));
    BOOST_TEST(length_cutoff<N>(paired, 2));
    BOOST_TEST(support_cutoff<N>(paired, 0));
}

// An unpaired set of length 3 (raw bits {0,2,4}: each even bit lacks its odd partner).
BOOST_AUTO_TEST_CASE(majorana_cutoff_length_and_support_thresholds) {
    constexpr size_t N = 32;
    Monomial<N> unpaired;
    unpaired.set(0);
    unpaired.set(2);
    unpaired.set(4);
    BOOST_TEST(!is_paired<N>(unpaired));

    // length = popcount = 3; support (distinct orbitals) = 3 here.
    BOOST_TEST(length_cutoff<N>(unpaired, 3));
    BOOST_TEST(!length_cutoff<N>(unpaired, 2));
    BOOST_TEST(support_cutoff<N>(unpaired, 3));
    BOOST_TEST(!support_cutoff<N>(unpaired, 2));

    // support <= length always, so passing length implies passing support at the same cutoff.
    std::mt19937_64 rng(0x50FA11ULL);
    std::uniform_int_distribution<size_t> bit(0, 2 * N - 1);
    for (int trial = 0; trial < 400; ++trial) {
        Monomial<N> m;
        for (int k = 0; k < 5; ++k) {
            m.set(bit(rng));
        }
        for (unsigned int c : {0U, 1U, 2U, 3U}) {
            if (length_cutoff<N>(m, c)) {
                BOOST_TEST(support_cutoff<N>(m, c));
            }
        }
        BOOST_TEST(length_cutoff<N>(m, 2 * N));
        BOOST_TEST(length_cutoff<N>(m, 0) == is_paired<N>(m));
    }
}

// logical_num_modes masks off the inactive low-mode prefix, so bits there must not count against the
// active window.
BOOST_AUTO_TEST_CASE(majorana_cutoff_logical_num_modes_masks_prefix_single_word) {
    constexpr size_t N = 32;
    constexpr size_t logical = 6; // active window = raw bits [2*(32-6), 64) = [52, 64)
    Monomial<N> prefix_only;
    prefix_only.set(0); // lone unpaired bit, inside the inactive prefix

    // Active window is empty -> treated as fully paired -> kept even at cutoff 0.
    BOOST_TEST(length_cutoff<N>(prefix_only, 0, logical));
    BOOST_TEST(support_cutoff<N>(prefix_only, 0, logical));
    // Over the whole register the lone bit is unpaired and exceeds cutoff 0 -> dropped.
    BOOST_TEST(!length_cutoff<N>(prefix_only, 0, N));
    BOOST_TEST(!length_cutoff<N>(prefix_only, 0)); // whole-register overload

    Monomial<N> active_bit;
    active_bit.set(52);
    BOOST_TEST(!length_cutoff<N>(active_bit, 0, logical));
    Monomial<N> active_pair;
    active_pair.set(52);
    active_pair.set(53);
    BOOST_TEST(length_cutoff<N>(active_pair, 0, logical));
}

BOOST_AUTO_TEST_CASE(majorana_cutoff_logical_num_modes_masks_prefix_multi_word) {
    constexpr size_t N = 96;
    constexpr size_t logical = 90; // active window = raw bits [2*(96-90), 192) = [12, 192)
    Monomial<N> prefix_only;
    prefix_only.set(4); // lone unpaired bit in the inactive prefix

    BOOST_TEST(length_cutoff<N>(prefix_only, 0, logical)); // active window empty -> kept
    BOOST_TEST(!length_cutoff<N>(prefix_only, 0, N));      // whole register -> dropped
}

BOOST_AUTO_TEST_CASE(majorana_cutoff_evaluator_dispatch_and_popcount) {
    constexpr size_t N = 32;

    CutoffFn<N> length_fn = detail::LengthCutoff<N>{.cutoff = 3};
    detail::CutoffEvaluator<N> length_ev(length_fn);
    BOOST_TEST((length_ev.length_cutoff() != nullptr));
    BOOST_TEST((length_ev.support_cutoff() == nullptr));
    BOOST_REQUIRE(length_ev.max_slot_bound().has_value());
    // A length cutoff counts set bits directly, so the slot bound IS the cutoff.
    BOOST_TEST(length_ev.max_slot_bound().value() == 3U);

    CutoffFn<N> support_fn = detail::SupportCutoff<N>{.cutoff = 2};
    detail::CutoffEvaluator<N> support_ev(support_fn);
    BOOST_TEST((support_ev.length_cutoff() == nullptr));
    BOOST_TEST((support_ev.support_cutoff() != nullptr));
    // A support cutoff counts modes/qubits and each spans two slots, so the slot bound doubles.
    BOOST_TEST(support_ev.max_slot_bound().value() == 4U);

    CutoffFn<N> opaque_fn = [](const Monomial<N> &) { return true; };
    detail::CutoffEvaluator<N> opaque_ev(opaque_fn);
    BOOST_TEST((opaque_ev.length_cutoff() == nullptr));
    BOOST_TEST((opaque_ev.support_cutoff() == nullptr));
    BOOST_TEST(!opaque_ev.max_slot_bound().has_value());

    // passes_with_popcount: pc <= cutoff short-circuits to true; otherwise it equals a direct eval.
    Monomial<N> unpaired; // length 4, not paired
    unpaired.set(0);
    unpaired.set(2);
    unpaired.set(4);
    unpaired.set(6);
    BOOST_TEST(length_ev.passes_with_popcount(unpaired, 3));
    BOOST_TEST(!length_ev.passes_with_popcount(unpaired, 4));
    BOOST_TEST(length_ev.passes_with_popcount(unpaired, 4) == length_ev(unpaired));

    Monomial<N> paired; // pc>cutoff but paired -> direct eval keeps it
    paired.set(0);
    paired.set(1);
    paired.set(2);
    paired.set(3);
    BOOST_TEST(length_ev.passes_with_popcount(paired, 10));
}

// interleave_phase (reference prefix-XOR scan) must equal the masked-parity form used on the hot path.
BOOST_AUTO_TEST_CASE(majorana_cutoff_interleave_phase_mask_cross_check) {
    auto check = [](auto tag) {
        constexpr size_t N = decltype(tag)::value;
        std::mt19937_64 rng(0xABCDEF01ULL + N);
        std::uniform_int_distribution<size_t> bit(0, 2 * N - 1);
        for (int trial = 0; trial < 500; ++trial) {
            Monomial<N> m;
            Monomial<N> g;
            for (int k = 0; k < 6; ++k) {
                m.set(bit(rng));
                g.set(bit(rng));
            }
            const int reference = interleave_phase<N>(m, g);
            const auto w = interleave_phase_mask<N>(g);
            const int masked = m.parity_and(w) ? -1 : 1;
            BOOST_TEST(reference == masked);
        }
    };
    check(std::integral_constant<size_t, 32>{}); // single word
    check(std::integral_constant<size_t, 96>{}); // multi word
}

BOOST_AUTO_TEST_CASE(majorana_cutoff_encode_decode_coeff) {
    constexpr size_t N = 32;
    Monomial<N> mono;
    mono.set(0);
    mono.set(3);
    mono.set(6);

    for (double r : {1.0, -2.5, 0.0, 7.25}) {
        const cd hermitian = decode_coeff<N>(cd(r, 0.0), mono); // r * hermitian_coefficient(mono)
        BOOST_TEST(encode_coeff<N>(hermitian, mono) == r);
    }

    // Multiply by i to break Hermiticity: the encoded value then has a nonzero imaginary part.
    const cd non_hermitian = decode_coeff<N>(cd(1.0, 0.0), mono) * cd(0.0, 1.0);
    BOOST_CHECK_THROW(encode_coeff<N>(non_hermitian, mono), std::runtime_error);
}
