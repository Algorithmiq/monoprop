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

#include <filesystem>
#include <vector>

#include "monoprop/TypeAliases.h"

namespace test_utils {

/// Minimal test-case payload loaded from a `tests/data/*.msgpack` fixture.
///
/// Mirrors the flat msgpack schema documented in `tests/data/README.md`; only
/// the fields exercised by the C++ test suite are kept.
struct CaseData {
    double actual_expval{0.0};
    monoprop::VecZ initial_state;
    monoprop::VecZ param_inds;
    monoprop::VecD gen_coeffs;
    monoprop::VecD parameters;
    std::vector<monoprop::VecZ> majoranas;
    monoprop::OperatorDict hamiltonian;
    size_t num_modes{0};
};

/// Load a test case from a minimal-schema msgpack fixture.
///
/// @param filename Path to the `.msgpack` fixture.
/// @return The parsed @ref CaseData.
/// @throws std::runtime_error if the file cannot be read or parsed.
auto load_case(const std::filesystem::path& filename) -> CaseData;

} // namespace test_utils
