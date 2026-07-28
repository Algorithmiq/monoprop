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

#include <boost/test/unit_test.hpp>

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "PauliTestOracle.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/algebra/PauliAlgebra.h"

using namespace monoprop;
using namespace pauli_oracle;

namespace {

// --- Reference oracles (test-only) -----------------------------------------------------------
// Readable closed forms for quantities the hot path derives inline (pauli_rotation_sign), built on
// the header's primitives (detail::pauli_uv, detail::mod4, pauli_y_count).

// Qubit Pauli weight = number of non-identity single-qubit letters = or_sum = |x | z|.
template <size_t NumModes>
[[nodiscard]] auto pauli_weight(const Monomial<NumModes> &p) -> size_t {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    size_t weight = 0;
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        const auto [v, u] = detail::pauli_uv(p.word(w), e_mask.word(w));
        weight += static_cast<size_t>(std::popcount(v | u));
    }
    return weight;
}

// The mod-4 exponent of the product-phase i^e for A*B, with A the left operand.
//   e = yA + yB - yR + 2*(zA . xB)  (mod 4),  R = A ^ B.
// e is odd iff A,B anticommute (phase = +/- i); even iff they commute (phase = +/- 1).
template <size_t NumModes>
[[nodiscard]] auto product_phase_exponent(const Monomial<NumModes> &a, const Monomial<NumModes> &b) -> int {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    const auto r = a ^ b;
    const long y_a = static_cast<long>(pauli_y_count<NumModes>(a));
    const long y_b = static_cast<long>(pauli_y_count<NumModes>(b));
    const long y_r = static_cast<long>(pauli_y_count<NumModes>(r));
    long cross = 0; // zA . xB = popcount(v-plane(A) & x-plane(B))
    for (size_t w = 0; w < Monomial<NumModes>::num_words(); ++w) {
        const uint64_t e = e_mask.word(w);
        const uint64_t z_a = a.word(w) & e; // v-plane of A
        const auto [v_b, u_b] = detail::pauli_uv(b.word(w), e);
        const uint64_t x_b = u_b ^ v_b; // x-plane of B
        cross += std::popcount(z_a & x_b);
    }
    return detail::mod4(y_a + y_b - y_r + 2 * cross);
}

// Product phase phi (unit modulus) such that A*B = phi * (A ^ B), A the left operand.
template <size_t NumModes>
[[nodiscard]] auto pauli_product_phase(const Monomial<NumModes> &a, const Monomial<NumModes> &b)
    -> std::complex<double> {
    return POWERS_OF_I[product_phase_exponent<NumModes>(a, b)];
}

// Emit sign +/-1 such that A*B = sign * i * (A ^ B), valid when A,B anticommute (exponent e odd).
// The raw product sign; pauli_rotation_sign returns exactly -pauli_emit_sign_antic.
template <size_t NumModes>
[[nodiscard]] auto pauli_emit_sign_antic(const Monomial<NumModes> &a, const Monomial<NumModes> &b) -> int {
    return product_phase_exponent<NumModes>(a, b) == 1 ? 1 : -1;
}

} // namespace

// pair_swap involution + anticommutation, against string-level and dense-matrix oracles.
BOOST_AUTO_TEST_CASE(pauli_algebra_pair_swap_and_anticommutation) {
    constexpr size_t N = 8;

    // Exhaustive 1- and 2-qubit checks (also cross-checked against dense matrices).
    for (size_t n : {size_t{1}, size_t{2}}) {
        for (const auto &pa : all_strings(n)) {
            const auto a = native_bitset<N>(pa);
            BOOST_TEST((pair_swap<N>(pair_swap<N>(a)) == a));
            for (const auto &pb : all_strings(n)) {
                const auto b = native_bitset<N>(pb);
                const bool antic = pauli_anticommutes<N>(a, b);
                BOOST_TEST(antic == string_anticommutes(pa, pb));
                // Independent dense-matrix check: anticommute iff AB == -BA.
                const auto ma = matrix_from_string(pa);
                const auto mb = matrix_from_string(pb);
                const size_t d = ma.size() == 4 ? 2 : 4;
                const auto ab = matmul(ma, mb, d);
                const auto ba = matmul(mb, ma, d);
                const bool mat_antic = approx_equal(ab, scalar_mul(cd(-1, 0), ba));
                BOOST_TEST(antic == mat_antic);
            }
        }
    }

    // Randomized up to 6 qubits (single-word) + multiword (N=40) coverage.
    std::mt19937 rng(0xC0FFEEU);
    for (size_t trial = 0; trial < 4000; ++trial) {
        const size_t n = 1 + (rng() % 6);
        const auto pa = random_string(rng, n);
        const auto pb = random_string(rng, n);
        const auto a = native_bitset<N>(pa);
        const auto b = native_bitset<N>(pb);
        BOOST_TEST((pair_swap<N>(pair_swap<N>(a)) == a));
        BOOST_TEST(pauli_anticommutes<N>(a, b) == string_anticommutes(pa, pb));
    }

    constexpr size_t NW = 40; // 2N = 80 bits -> 2 words: exercises multiword kernels.
    for (size_t trial = 0; trial < 2000; ++trial) {
        const auto pa = random_string(rng, NW);
        const auto pb = random_string(rng, NW);
        const auto a = native_bitset<NW>(pa);
        const auto b = native_bitset<NW>(pb);
        BOOST_TEST((pair_swap<NW>(pair_swap<NW>(a)) == a));
        BOOST_TEST(pauli_anticommutes<NW>(a, b) == string_anticommutes(pa, pb));
    }
}

// The encoding is exactly the Jordan-Wigner image: native == change_basis(jw(P), jw_basis).
BOOST_AUTO_TEST_CASE(pauli_algebra_encoding_is_jw_image) {
    constexpr size_t N = 8;
    for (size_t n : {size_t{1}, size_t{2}}) {
        const auto basis = jw_basis<N>(n);
        for (const auto &p : all_strings(n)) {
            BOOST_TEST((native_bitset<N>(p) == change_basis<N>(jw_bitset<N>(p), basis)));
        }
    }
    std::mt19937 rng(0x1234ABCDU);
    for (size_t trial = 0; trial < 4000; ++trial) {
        const size_t n = 1 + (rng() % 6);
        const auto basis = jw_basis<N>(n);
        const auto p = random_string(rng, n);
        BOOST_TEST((native_bitset<N>(p) == change_basis<N>(jw_bitset<N>(p), basis)));
    }
}

// Product phase pinned by dense-matrix brute force; emit sign for anticommuting pairs.
BOOST_AUTO_TEST_CASE(pauli_algebra_product_phase_vs_brute_force) {
    constexpr size_t N = 4;
    for (size_t n : {size_t{1}, size_t{2}, size_t{3}}) {
        const size_t d = size_t{1} << n;
        const auto strs = all_strings(n);
        for (const auto &pa : strs) {
            const auto a = native_bitset<N>(pa);
            const auto ma = matrix_from_string(pa);
            for (const auto &pb : strs) {
                const auto b = native_bitset<N>(pb);
                const auto r = a ^ b;

                std::string pr(n, 'I');
                for (size_t q = 0; q < n; ++q) {
                    pr[q] = letter_from_bitset<N>(r, q);
                }
                const auto mr = matrix_from_string(pr);
                const auto mb = matrix_from_string(pb);
                const auto ab = matmul(ma, mb, d);

                const cd phi = pauli_product_phase<N>(a, b);
                BOOST_TEST(std::abs(std::abs(phi) - 1.0) < 1e-12);
                BOOST_TEST(approx_equal(ab, scalar_mul(phi, mr)));

                if (pauli_anticommutes<N>(a, b)) {
                    const int sign = pauli_emit_sign_antic<N>(a, b);
                    BOOST_TEST((sign == 1 || sign == -1));
                    // A*B = sign * i * R for anticommuting Hermitian Paulis.
                    BOOST_TEST(approx_equal(ab, scalar_mul(cd(0, static_cast<double>(sign)), mr)));
                    // Hot kernel returns the rotation sign = negated raw emit sign.
                    const auto ctx = make_pauli_gen_context<N>(b);
                    BOOST_TEST(pauli_rotation_sign<N>(ctx, a, r) == -sign);
                }
            }
        }
    }

    // pauli_rotation_sign == -pauli_emit_sign_antic for all pairs, including multiword (N=40).
    constexpr size_t NW = 40;
    std::mt19937 rng(0xBEEF01U);
    for (size_t trial = 0; trial < 3000; ++trial) {
        const auto a = native_bitset<NW>(random_string(rng, NW));
        const auto b = native_bitset<NW>(random_string(rng, NW));
        const auto ctx = make_pauli_gen_context<NW>(b);
        BOOST_TEST(pauli_rotation_sign<NW>(ctx, a, a ^ b) == -pauli_emit_sign_antic<NW>(a, b));
    }
}

// Cutoff / weight / Z-only equivalence under the native encoding, incl. logical < NumModes.
BOOST_AUTO_TEST_CASE(pauli_algebra_cutoff_and_weight_equivalence) {
    constexpr size_t N = 32; // single word (2N = 64)
    constexpr size_t logical = 6;
    const auto basis = jw_basis<N>(logical);

    std::mt19937 rng(0x0DDBALLU);
    for (size_t trial = 0; trial < 3000; ++trial) {
        const auto p = random_string(rng, logical); // P on the low qubits 0..logical-1
        const auto native = native_bitset<N>(p);
        const auto via_jw = change_basis<N>(jw_bitset<N>(p), basis);
        BOOST_TEST((native == via_jw));

        for (unsigned int c : {0U, 1U, 2U, 3U, 6U}) {
            BOOST_TEST(support_cutoff<N>(native, c, logical) == support_cutoff<N>(via_jw, c, logical));
            // Also exercise the whole-register (logical == NumModes) code path.
            BOOST_TEST(support_cutoff<N>(native, c) == support_cutoff<N>(via_jw, c));
        }

        size_t true_weight = 0;
        for (char ch : p) {
            true_weight += (ch != 'I') ? 1 : 0;
        }
        BOOST_TEST(pauli_weight<N>(native) == true_weight);

        // is_paired (support_cutoff's xor_sum == 0) detects exactly the Z-only strings.
        BOOST_TEST(is_paired<N>(native) == is_z_only(p));
    }
}

// Initial-state phase vs brute-force <b|P|b>.
BOOST_AUTO_TEST_CASE(pauli_algebra_state_phase) {
    constexpr size_t N = 8;
    constexpr size_t n = 5;
    std::mt19937 rng(0xFACE42U);
    std::bernoulli_distribution occ(0.5);
    std::bernoulli_distribution use_z(0.5);

    for (size_t trial = 0; trial < 3000; ++trial) {
        // Random computational basis state b and state_mask (even/z-plane bits of occupied qubits).
        std::vector<int> b(n);
        VecZ occupied_slots;
        for (size_t q = 0; q < n; ++q) {
            b[q] = occ(rng) ? 1 : 0;
            if (b[q] != 0) {
                occupied_slots.push_back(2 * q + 1); // z-plane bit of qubit q (even physical bit)
            }
        }
        const auto state_mask = indices_to_bitset<N>(occupied_slots);

        // Z-only Pauli: pauli_state_phase must match (-1)^{|Z ∩ occupied|} and dense <b|P|b>.
        std::string pz(n, 'I');
        for (size_t q = 0; q < n; ++q) {
            pz[q] = use_z(rng) ? 'Z' : 'I';
        }
        const auto z_mono = native_bitset<N>(pz);
        int expected = 1;
        for (size_t q = 0; q < n; ++q) {
            if (pz[q] == 'Z' && b[q] != 0) {
                expected = -expected;
            }
        }
        const double phase = pauli_state_phase<N>(z_mono, state_mask);
        BOOST_TEST(phase == static_cast<double>(expected));

        const size_t d = size_t{1} << n;
        size_t idx = 0;
        for (size_t q = 0; q < n; ++q) {
            if (b[q] != 0) {
                idx |= (size_t{1} << (n - 1 - q));
            }
        }
        const auto mz = matrix_from_string(pz);
        BOOST_TEST(std::abs(mz[idx * d + idx] - cd(static_cast<double>(expected), 0)) < 1e-9);

        // Non-diagonal Pauli: <b|P|b> == 0 (documents why the state-phase guard is Z-only).
        std::string pnd = random_string(rng, n);
        if (is_z_only(pnd)) {
            pnd[rng() % n] = 'X'; // force at least one off-diagonal letter
        }
        const auto mnd = matrix_from_string(pnd);
        BOOST_TEST(std::abs(mnd[idx * d + idx]) < 1e-9);
    }
}
