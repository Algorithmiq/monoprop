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

// The subset of the tests/data/README.md fixture schema that the C++ suite uses.
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

// Throws std::runtime_error if the fixture cannot be read or parsed.
auto load_case(const std::filesystem::path& filename) -> CaseData;

// A monotone injection of a case's modes into the modes of a wider system -- the C++ twin of
// tests/cases.py's ModeEmbedding, carrying the same map.
//
// Relabelling modes monotonically is a canonical transformation: the map is strictly increasing, so a
// sorted Majorana index tuple stays sorted and no anticommutation sign appears, and the physics -- the
// reference expectation value included -- is the source problem's. That is how the suite reaches a
// storage width no checked-in fixture has, with no second reference calculation and no second fixture
// whose only difference from an existing one is a permutation.
struct ModeEmbedding {
    size_t num_modes{0};  ///< Width of the embedding system.
    monoprop::VecZ modes; ///< Where source mode m lands; strictly increasing and below num_modes.

    /// One Majorana index of the source system, in the embedding system.
    [[nodiscard]] auto majorana(size_t index) const -> size_t { return (2 * modes[index / 2]) + (index % 2); }
};

/// `data` relabelled through `embedding`. Parameters, coefficients and actual_expval carry over as they
/// are; only mode-indexed data moves.
inline auto embed_case(const CaseData& data, const ModeEmbedding& embedding) -> CaseData {
    const auto map_indices = [&](const monoprop::VecZ& indices) {
        monoprop::VecZ out;
        out.reserve(indices.size());
        for (const auto index : indices) {
            out.push_back(embedding.majorana(index));
        }
        return out;
    };

    CaseData out = data;
    out.num_modes = embedding.num_modes;
    for (auto& mode : out.initial_state) {
        mode = embedding.modes.at(mode);
    }
    for (auto& mono : out.majoranas) {
        mono = map_indices(mono);
    }
    out.hamiltonian.clear();
    for (const auto& [indices, coeff] : data.hamiltonian) {
        out.hamiltonian.emplace(map_indices(indices), coeff);
    }
    return out;
}

} // namespace test_utils
