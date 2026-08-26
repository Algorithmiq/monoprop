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

#include "PauliTestOracle.h"

namespace pauli_oracle {

namespace {
constexpr char LETTERS[4] = {'I', 'X', 'Y', 'Z'};
} // namespace

auto slots_of_string(const std::string &p) -> VecZ {
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

auto native_bitset(size_t num_modes, const std::string &p) -> Bitset {
    return indices_to_bitset(slots_of_string(p), 2 * num_modes);
}

auto letter_from_bitset(size_t num_modes, const Bitset &mono, size_t q) -> char {
    const bool u = mono.test(2 * num_modes - 1 - 2 * q); // slot 2q
    const bool v = mono.test(2 * num_modes - 2 - 2 * q); // slot 2q+1
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

auto jw_bitset(size_t num_modes, const std::string &p) -> Bitset {
    return indices_to_bitset(pauli_to_fermi_indices(p), 2 * num_modes);
}

auto jw_basis(size_t num_modes, size_t n) -> MonomialList {
    // The fill value carries the width: a sized MonomialList would otherwise hold width-0 bitsets, and
    // the slots past 2*n are never assigned below yet still reach change_basis's XOR.
    MonomialList basis(2 * num_modes, Bitset(2 * num_modes));
    for (size_t i = 0; i < n; ++i) {
        VecZ z_str;
        for (size_t z = 0; z < 2 * i; ++z) {
            z_str.push_back(z);
        }
        VecZ even_vec = z_str;
        even_vec.push_back(2 * i);
        VecZ odd_vec = z_str;
        odd_vec.push_back(2 * i + 1);
        basis[2 * i] = indices_to_bitset(even_vec, 2 * num_modes);
        basis[2 * i + 1] = indices_to_bitset(odd_vec, 2 * num_modes);
    }
    return basis;
}

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

auto matrix_from_string(const std::string &p) -> std::vector<cd> {
    std::vector<cd> m = single_letter(p[0]);
    size_t d = 2;
    for (size_t q = 1; q < p.size(); ++q) {
        m = kron(m, d, single_letter(p[q]), 2);
        d *= 2;
    }
    return m;
}

auto approx_equal(const std::vector<cd> &a, const std::vector<cd> &b, double tol) -> bool {
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

auto scalar_mul(cd s, const std::vector<cd> &a) -> std::vector<cd> {
    std::vector<cd> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = s * a[i];
    }
    return r;
}

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

} // namespace pauli_oracle
