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

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/PauliAlgebra.h"

using namespace monoprop;

namespace {

using cd = std::complex<double>;

constexpr char LETTERS[4] = {'I', 'X', 'Y', 'Z'};

// --- Encoding helpers (independent of the library under test) --------------------------------

// native_bitset: set gamma slots per the X/Y/Z rule then map to physical bits via
// indices_to_bitset. X_q -> slot 2q, Y_q -> slot 2q+1, Z_q -> slots {2q, 2q+1}.
template <size_t NumModes>
auto native_bitset(const std::string &p) -> MajoranaSet<NumModes> {
    VecZ slots;
    for (size_t q = 0; q < p.size(); ++q) {
        switch (p[q]) {
        case 'X':
            slots.push_back(2 * q);
            break;
        case 'Y':
            slots.push_back(2 * q + 1);
            break;
        case 'Z':
            slots.push_back(2 * q);
            slots.push_back(2 * q + 1);
            break;
        default:
            break; // 'I'
        }
    }
    return indices_to_bitset<NumModes>(slots);
}

// --- Reference oracles (test-only) -----------------------------------------------------------
// Closed-form phase/weight computations kept here rather than in the shipped PauliAlgebra.h: the
// library's hot path derives the same quantities inline (pauli_rotation_sign). These readable
// forms exist only to pin that inline kernel against an independent reference in the cases below.
// They reuse the header's still-shipped primitives (detail::pauli_uv, detail::mod4, pauli_y_count).

// Qubit Pauli weight = number of non-identity single-qubit letters = or_sum = |x | z|.
template <size_t NumModes>
[[nodiscard]] auto pauli_weight(const MajoranaSet<NumModes> &p) -> size_t {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    size_t weight = 0;
    for (size_t w = 0; w < MajoranaSet<NumModes>::num_words(); ++w) {
        const auto [v, u] = detail::pauli_uv(p.word(w), e_mask.word(w));
        weight += static_cast<size_t>(std::popcount(v | u));
    }
    return weight;
}

// The mod-4 exponent of the product-phase i^e for A*B, with A the LEFT operand.
//   e = yA + yB - yR + 2*(zA . xB)  (mod 4),  R = A ^ B.
// e is odd iff A,B anticommute (phase = +/- i); even iff they commute (phase = +/- 1).
template <size_t NumModes>
[[nodiscard]] auto product_phase_exponent(const MajoranaSet<NumModes> &a, const MajoranaSet<NumModes> &b) -> int {
    constexpr auto e_mask = pauli_even_mask<NumModes>();
    const auto r = a ^ b;
    const long y_a = static_cast<long>(pauli_y_count<NumModes>(a));
    const long y_b = static_cast<long>(pauli_y_count<NumModes>(b));
    const long y_r = static_cast<long>(pauli_y_count<NumModes>(r));
    long cross = 0; // zA . xB = popcount(v-plane(A) & x-plane(B))
    for (size_t w = 0; w < MajoranaSet<NumModes>::num_words(); ++w) {
        const uint64_t e = e_mask.word(w);
        const uint64_t z_a = a.word(w) & e; // v-plane of A
        const auto [v_b, u_b] = detail::pauli_uv(b.word(w), e);
        const uint64_t x_b = u_b ^ v_b; // x-plane of B
        cross += std::popcount(z_a & x_b);
    }
    return detail::mod4(y_a + y_b - y_r + 2 * cross);
}

// Product phase phi (unit modulus) such that A*B = phi * (A ^ B), A the LEFT operand.
template <size_t NumModes>
[[nodiscard]] auto pauli_product_phase(const MajoranaSet<NumModes> &a, const MajoranaSet<NumModes> &b)
    -> std::complex<double> {
    return POWERS_OF_I[product_phase_exponent<NumModes>(a, b)];
}

// Emit sign +/-1 such that A*B = sign * i * (A ^ B), valid when A,B ANTICOMMUTE (exponent e odd).
// The RAW product sign; pauli_rotation_sign returns exactly -pauli_emit_sign_antic.
template <size_t NumModes>
[[nodiscard]] auto pauli_emit_sign_antic(const MajoranaSet<NumModes> &a, const MajoranaSet<NumModes> &b) -> int {
    return product_phase_exponent<NumModes>(a, b) == 1 ? 1 : -1;
}

// Faithful C++ port of _pauli_to_fermi (conversion_utils.py) -- indices only (coeff dropped;
// the bitset only cares about which Majorana modes are present, not their order/phase).
auto pauli_to_fermi_indices(const std::string &pauli) -> VecZ {
    std::vector<size_t> acc;
    bool flag_z = false;
    for (int i = static_cast<int>(pauli.size()) - 1; i >= 0; --i) {
        const char p = pauli[static_cast<size_t>(i)];
        const auto ii = static_cast<size_t>(i);
        if ((p == 'Z' && !flag_z) || (p == 'I' && flag_z)) {
            acc.push_back(2 * ii + 1);
            acc.push_back(2 * ii);
        }
        else if (p == 'X' && !flag_z) {
            acc.push_back(2 * ii);
            flag_z = true;
        }
        else if (p == 'X' && flag_z) {
            acc.push_back(2 * ii + 1);
            flag_z = false;
        }
        else if (p == 'Y' && !flag_z) {
            acc.push_back(2 * ii + 1);
            flag_z = true;
        }
        else if (p == 'Y' && flag_z) {
            acc.push_back(2 * ii);
            flag_z = false;
        }
        // (Z, flag_z) and (I, !flag_z): no-op
    }
    return VecZ(acc.rbegin(), acc.rend());
}

template <size_t NumModes>
auto jw_bitset(const std::string &p) -> MajoranaSet<NumModes> {
    return indices_to_bitset<NumModes>(pauli_to_fermi_indices(p));
}

// Port of jordan_wigner_basis_change(n): basis[2i] = [0..2i-1, 2i], basis[2i+1] = [0..2i-1, 2i+1].
// Returned as a full-width (2*NumModes) basis so change_basis can index it by gamma slot.
template <size_t NumModes>
auto jw_basis(size_t n) -> MajoranaVector<NumModes> {
    MajoranaVector<NumModes> basis(2 * NumModes);
    for (size_t i = 0; i < n; ++i) {
        VecZ z_str;
        for (size_t z = 0; z < 2 * i; ++z) {
            z_str.push_back(z);
        }
        VecZ even_vec = z_str;
        even_vec.push_back(2 * i);
        VecZ odd_vec = z_str;
        odd_vec.push_back(2 * i + 1);
        basis[2 * i] = indices_to_bitset<NumModes>(even_vec);
        basis[2 * i + 1] = indices_to_bitset<NumModes>(odd_vec);
    }
    return basis;
}

// Decode the single-qubit letter of qubit q from a native-encoded bitset.
template <size_t NumModes>
auto letter_from_bitset(const MajoranaSet<NumModes> &maj, size_t q) -> char {
    const bool u = maj.test(2 * NumModes - 1 - 2 * q); // slot 2q   (odd physical bit)
    const bool v = maj.test(2 * NumModes - 2 - 2 * q); // slot 2q+1 (even physical bit)
    if (!u && !v) {
        return 'I';
    }
    if (u && !v) {
        return 'X';
    }
    if (!u && v) {
        return 'Y';
    }
    return 'Z';
}

// --- Dense Pauli-matrix brute force ----------------------------------------------------------

auto single_letter(char c) -> std::vector<cd> {
    switch (c) {
    case 'X':
        return {cd(0, 0), cd(1, 0), cd(1, 0), cd(0, 0)};
    case 'Y':
        return {cd(0, 0), cd(0, -1), cd(0, 1), cd(0, 0)};
    case 'Z':
        return {cd(1, 0), cd(0, 0), cd(0, 0), cd(-1, 0)};
    default:
        return {cd(1, 0), cd(0, 0), cd(0, 0), cd(1, 0)}; // I
    }
}

// Kronecker product of A (da x da) and B (db x db); A is the more-significant factor.
auto kron(const std::vector<cd> &a, size_t da, const std::vector<cd> &b, size_t db) -> std::vector<cd> {
    const size_t d = da * db;
    std::vector<cd> r(d * d, cd(0, 0));
    for (size_t i = 0; i < da; ++i) {
        for (size_t j = 0; j < da; ++j) {
            const cd aij = a[i * da + j];
            for (size_t k = 0; k < db; ++k) {
                for (size_t l = 0; l < db; ++l) {
                    r[(i * db + k) * d + (j * db + l)] = aij * b[k * db + l];
                }
            }
        }
    }
    return r;
}

auto matmul(const std::vector<cd> &a, const std::vector<cd> &b, size_t d) -> std::vector<cd> {
    std::vector<cd> r(d * d, cd(0, 0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t k = 0; k < d; ++k) {
            const cd aik = a[i * d + k];
            if (aik == cd(0, 0)) {
                continue;
            }
            for (size_t j = 0; j < d; ++j) {
                r[i * d + j] += aik * b[k * d + j];
            }
        }
    }
    return r;
}

// Dense matrix of a Pauli string (qubit 0 = most-significant tensor factor).
auto matrix_from_string(const std::string &p) -> std::vector<cd> {
    std::vector<cd> m = single_letter(p[0]);
    size_t d = 2;
    for (size_t q = 1; q < p.size(); ++q) {
        m = kron(m, d, single_letter(p[q]), 2);
        d *= 2;
    }
    return m;
}

auto approx_equal(const std::vector<cd> &a, const std::vector<cd> &b) -> bool {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-9) {
            return false;
        }
    }
    return true;
}

auto scalar_mul(cd s, const std::vector<cd> &a) -> std::vector<cd> {
    std::vector<cd> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = s * a[i];
    }
    return r;
}

// Local anticommutation from the strings alone: strings anticommute iff an odd number of qubits
// carry two distinct non-identity letters.
auto string_anticommutes(const std::string &a, const std::string &b) -> bool {
    size_t local = 0;
    for (size_t q = 0; q < a.size(); ++q) {
        if (a[q] != 'I' && b[q] != 'I' && a[q] != b[q]) {
            ++local;
        }
    }
    return (local & 1U) != 0U;
}

auto is_z_only(const std::string &p) -> bool {
    for (char c : p) {
        if (c == 'X' || c == 'Y') {
            return false;
        }
    }
    return true;
}

// Enumerate all 4^n Pauli strings on n qubits.
auto all_strings(size_t n) -> std::vector<std::string> {
    std::vector<std::string> out;
    size_t total = 1;
    for (size_t i = 0; i < n; ++i) {
        total *= 4;
    }
    out.reserve(total);
    for (size_t idx = 0; idx < total; ++idx) {
        std::string s(n, 'I');
        size_t v = idx;
        for (size_t q = 0; q < n; ++q) {
            s[q] = LETTERS[v & 3U];
            v >>= 2U;
        }
        out.push_back(s);
    }
    return out;
}

auto random_string(std::mt19937 &rng, size_t n) -> std::string {
    std::uniform_int_distribution<int> d(0, 3);
    std::string s(n, 'I');
    for (size_t q = 0; q < n; ++q) {
        s[q] = LETTERS[d(rng)];
    }
    return s;
}

} // namespace

// The repo's ctest discovery (boostAddTests.cmake) treats every --list_content line as a
// top-level test name and cannot address suite-nested cases, so tests use flat cases with a
// shared name prefix (as coeff_frame_*, inverted_index_*, etc. do) rather than a
// BOOST_AUTO_TEST_SUITE. Run just this group with --run_test=pauli_algebra_*.

// T1: pair_swap involution + anticommutation vs an independent second computation.
BOOST_AUTO_TEST_CASE(pauli_algebra_pair_swap_and_anticommutation) {
    constexpr size_t N = 8;

    // Exhaustive 1- and 2-qubit checks (also cross-checked against dense matrices).
    for (size_t n : {size_t{1}, size_t{2}}) {
        for (const auto &pa : all_strings(n)) {
            const auto a = native_bitset<N>(pa);
            // Involution.
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

// T2: the encoding is EXACTLY the Jordan-Wigner image: native == change_basis(jw(P), jw_basis).
BOOST_AUTO_TEST_CASE(pauli_algebra_encoding_is_jw_image) {
    constexpr size_t N = 8;
    // Exhaustive for n = 1, 2.
    for (size_t n : {size_t{1}, size_t{2}}) {
        const auto basis = jw_basis<N>(n);
        for (const auto &p : all_strings(n)) {
            BOOST_TEST((native_bitset<N>(p) == change_basis<N>(jw_bitset<N>(p), basis)));
        }
    }
    // Randomized for n up to 6.
    std::mt19937 rng(0x1234ABCDU);
    for (size_t trial = 0; trial < 4000; ++trial) {
        const size_t n = 1 + (rng() % 6);
        const auto basis = jw_basis<N>(n);
        const auto p = random_string(rng, n);
        BOOST_TEST((native_bitset<N>(p) == change_basis<N>(jw_bitset<N>(p), basis)));
    }
}

// T3: product phase pinned by dense-matrix brute force; emit sign for anticommuting pairs.
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

                // Reconstruct R's string from the bitset and build its dense matrix.
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
                    // Hot kernel returns the ROTATION sign = negated raw emit sign.
                    const auto ctx = make_pauli_gen_context<N>(b);
                    BOOST_TEST(pauli_rotation_sign<N>(ctx, a, r) == -sign);
                }
            }
        }
    }

    // pauli_rotation_sign == -pauli_emit_sign_antic for ALL pairs, including multiword (N=40).
    constexpr size_t NW = 40;
    std::mt19937 rng(0xBEEF01U);
    for (size_t trial = 0; trial < 3000; ++trial) {
        const auto a = native_bitset<NW>(random_string(rng, NW));
        const auto b = native_bitset<NW>(random_string(rng, NW));
        const auto ctx = make_pauli_gen_context<NW>(b);
        BOOST_TEST(pauli_rotation_sign<NW>(ctx, a, a ^ b) == -pauli_emit_sign_antic<NW>(a, b));
    }
}

// T4: cutoff / weight / Z-only equivalence under the native encoding, incl. logical < NumModes.
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

        // pauli_weight == number of non-identity letters.
        size_t true_weight = 0;
        for (char ch : p) {
            true_weight += (ch != 'I') ? 1 : 0;
        }
        BOOST_TEST(pauli_weight<N>(native) == true_weight);

        // is_paired (support_cutoff's xor_sum == 0) detects exactly the Z-only strings.
        BOOST_TEST(is_paired<N>(native) == is_z_only(p));
    }
}

// T5: Hartree-Fock phase vs brute-force <b|P|b>.
BOOST_AUTO_TEST_CASE(pauli_algebra_hf_phase) {
    constexpr size_t N = 8;
    constexpr size_t n = 5;
    std::mt19937 rng(0xFACE42U);
    std::bernoulli_distribution occ(0.5);
    std::bernoulli_distribution use_z(0.5);

    for (size_t trial = 0; trial < 3000; ++trial) {
        // Random computational basis state b and hf_mask (even/z-plane bits of occupied qubits).
        std::vector<int> b(n);
        VecZ hf_slots;
        for (size_t q = 0; q < n; ++q) {
            b[q] = occ(rng) ? 1 : 0;
            if (b[q] != 0) {
                hf_slots.push_back(2 * q + 1); // z-plane bit of qubit q (even physical bit)
            }
        }
        const auto hf_mask = indices_to_bitset<N>(hf_slots);

        // Z-only Pauli: pauli_hf_phase must match (-1)^{|Z ∩ occupied|} and dense <b|P|b>.
        std::string pz(n, 'I');
        for (size_t q = 0; q < n; ++q) {
            pz[q] = use_z(rng) ? 'Z' : 'I';
        }
        const auto zmaj = native_bitset<N>(pz);
        int expected = 1;
        for (size_t q = 0; q < n; ++q) {
            if (pz[q] == 'Z' && b[q] != 0) {
                expected = -expected;
            }
        }
        const double hf = pauli_hf_phase<N>(zmaj, hf_mask);
        BOOST_TEST(hf == static_cast<double>(expected));

        const size_t d = size_t{1} << n;
        size_t idx = 0;
        for (size_t q = 0; q < n; ++q) {
            if (b[q] != 0) {
                idx |= (size_t{1} << (n - 1 - q));
            }
        }
        const auto mz = matrix_from_string(pz);
        BOOST_TEST(std::abs(mz[idx * d + idx] - cd(static_cast<double>(expected), 0)) < 1e-9);

        // Non-diagonal Pauli: <b|P|b> == 0 (documents why the hf-phase guard is Z-only).
        std::string pnd = random_string(rng, n);
        if (is_z_only(pnd)) {
            pnd[rng() % n] = 'X'; // force at least one off-diagonal letter
        }
        const auto mnd = matrix_from_string(pnd);
        BOOST_TEST(std::abs(mnd[idx * d + idx]) < 1e-9);
    }
}
