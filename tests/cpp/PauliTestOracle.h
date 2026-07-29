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

#pragma once

// Independent Pauli reference oracle shared by the Pauli test files. Nothing here touches the
// library under test beyond indices_to_bitset: the dense-matrix brute force and the JW image are
// computed from first principles so the engine's inline kernels can be pinned against them.

#include <bit>
#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"

namespace pauli_oracle {

using namespace monoprop;
using cd = std::complex<double>;

inline constexpr char LETTERS[4] = {'I', 'X', 'Y', 'Z'};

// Native symplectic-slot list for a Pauli string: X_q -> slot 2q, Y_q -> slot 2q+1,
// Z_q -> {2q, 2q+1}. This is the format the propagator's initial_operator and
// generators expect.
inline auto slots_of_string(const std::string &p) -> VecZ {
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
    return slots;
}

template <size_t NumModes>
auto native_bitset(const std::string &p) -> Monomial<NumModes> {
    return indices_to_bitset<NumModes>(slots_of_string(p));
}

// Decode the single-qubit letter of qubit q from a native-encoded bitset
// (MSb0 physical mapping): slot 2q is the x-plane bit, slot 2q+1 the z-plane bit.
template <size_t NumModes>
auto letter_from_bitset(const Monomial<NumModes> &mono, size_t q) -> char {
    const bool u = mono.test(2 * NumModes - 1 - 2 * q); // slot 2q
    const bool v = mono.test(2 * NumModes - 2 - 2 * q); // slot 2q+1
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

// Faithful C++ port of _pauli_to_fermi (conversion_utils.py) -- indices only
// (coeff dropped; the bitset only cares which Majorana modes are present).
inline auto pauli_to_fermi_indices(const std::string &pauli) -> VecZ {
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
auto jw_bitset(const std::string &p) -> Monomial<NumModes> {
    return indices_to_bitset<NumModes>(pauli_to_fermi_indices(p));
}

// jordan_wigner_basis_change(n) as a full-width (2*NumModes) basis so
// change_basis can index it by slot.
template <size_t NumModes>
auto jw_basis(size_t n) -> MonomialList<NumModes> {
    MonomialList<NumModes> basis(2 * NumModes);
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

inline auto single_letter(char c) -> std::vector<cd> {
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
inline auto kron(const std::vector<cd> &a, size_t da, const std::vector<cd> &b, size_t db) -> std::vector<cd> {
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

inline auto matmul(const std::vector<cd> &a, const std::vector<cd> &b, size_t d) -> std::vector<cd> {
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
inline auto matrix_from_string(const std::string &p) -> std::vector<cd> {
    std::vector<cd> m = single_letter(p[0]);
    size_t d = 2;
    for (size_t q = 1; q < p.size(); ++q) {
        m = kron(m, d, single_letter(p[q]), 2);
        d *= 2;
    }
    return m;
}

inline auto approx_equal(const std::vector<cd> &a, const std::vector<cd> &b, double tol = 1e-9) -> bool {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

inline auto scalar_mul(cd s, const std::vector<cd> &a) -> std::vector<cd> {
    std::vector<cd> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = s * a[i];
    }
    return r;
}

// Local anticommutation from the strings alone: anticommute iff an odd number of
// qubits carry two distinct non-identity letters.
inline auto string_anticommutes(const std::string &a, const std::string &b) -> bool {
    size_t local = 0;
    for (size_t q = 0; q < a.size(); ++q) {
        if (a[q] != 'I' && b[q] != 'I' && a[q] != b[q]) {
            ++local;
        }
    }
    return (local & 1U) != 0U;
}

inline auto is_z_only(const std::string &p) -> bool {
    for (char c : p) {
        if (c == 'X' || c == 'Y') {
            return false;
        }
    }
    return true;
}

inline auto all_strings(size_t n) -> std::vector<std::string> {
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

inline auto random_string(std::mt19937 &rng, size_t n) -> std::string {
    std::uniform_int_distribution<int> d(0, 3);
    std::string s(n, 'I');
    for (size_t q = 0; q < n; ++q) {
        s[q] = LETTERS[d(rng)];
    }
    return s;
}

} // namespace pauli_oracle
