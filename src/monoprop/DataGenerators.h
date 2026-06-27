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

#include <complex>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/monopropExport.h"

namespace monoprop {
struct monoprop_EXPORT MPData {
    double actual_expval{0.0};
    double circuit_expval{0.0};
    VecZ slater_determinant;
    VecZ param_inds;
    VecD gen_coeffs;
    VecD parameters;
    std::vector<VecZ> majoranas;
    std::map<VecZ, std::complex<double>> fermionic_operator;
    size_t num_modes{0};
};

monoprop_EXPORT auto load_from_msgpack(const std::filesystem::path& filename) -> MPData;
} // namespace monoprop
