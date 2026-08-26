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
//
// Definitions live in PauliTestOracle.cpp: five translation units include this header, so
// header-inline definitions would be five copies of an oracle that is never on a hot path.

#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/core/Monomial.h"

namespace pauli_oracle {

using namespace monoprop;
using cd = std::complex<double>;

// Native symplectic-slot list for a Pauli string: X_q -> slot 2q, Y_q -> slot 2q+1,
// Z_q -> {2q, 2q+1}. This is the format the propagator's initial_operator and
// generators expect.
auto slots_of_string(const std::string &p) -> VecZ;

auto native_bitset(size_t num_modes, const std::string &p) -> Bitset;

// Decode the single-qubit letter of qubit q from a native-encoded bitset
// (MSb0 physical mapping): slot 2q is the x-plane bit, slot 2q+1 the z-plane bit.
auto letter_from_bitset(size_t num_modes, const Bitset &mono, size_t q) -> char;

// Faithful C++ port of _pauli_to_fermi (conversion_utils.py) -- indices only
// (coeff dropped; the bitset only cares which Majorana modes are present).
auto pauli_to_fermi_indices(const std::string &pauli) -> VecZ;

auto jw_bitset(size_t num_modes, const std::string &p) -> Bitset;

// jordan_wigner_basis_change(n) as a full-width (2*num_modes) basis so
// change_basis can index it by slot.
auto jw_basis(size_t num_modes, size_t n) -> MonomialList;

auto single_letter(char c) -> std::vector<cd>;

// Kronecker product of A (da x da) and B (db x db); A is the more-significant factor.
auto kron(const std::vector<cd> &a, size_t da, const std::vector<cd> &b, size_t db) -> std::vector<cd>;

auto matmul(const std::vector<cd> &a, const std::vector<cd> &b, size_t d) -> std::vector<cd>;

// Dense matrix of a Pauli string (qubit 0 = most-significant tensor factor).
auto matrix_from_string(const std::string &p) -> std::vector<cd>;

auto approx_equal(const std::vector<cd> &a, const std::vector<cd> &b, double tol = 1e-9) -> bool;

auto scalar_mul(cd s, const std::vector<cd> &a) -> std::vector<cd>;

// Local anticommutation from the strings alone: anticommute iff an odd number of
// qubits carry two distinct non-identity letters.
auto string_anticommutes(const std::string &a, const std::string &b) -> bool;

auto is_z_only(const std::string &p) -> bool;

auto all_strings(size_t n) -> std::vector<std::string>;

auto random_string(std::mt19937 &rng, size_t n) -> std::string;

} // namespace pauli_oracle
