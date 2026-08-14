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

#include "Monomial.h"

#include <format>
#include <stdexcept>
#include <string>

namespace monoprop {
auto cutoff_type_str_2_enum(const std::string &cutoff_type) -> CutoffType {
    if (cutoff_type == "length") {
        return monoprop::CutoffType::Length;
    }

    if (cutoff_type == "support") {
        return monoprop::CutoffType::Support;
    }

    throw std::invalid_argument(
        std::format("Unknown CutoffType string: '{}'. Valid options are: 'length', 'support'.", cutoff_type));
}

auto cutoff_type_enum_2_str(CutoffType cutoff_type) -> std::string {
    switch (cutoff_type) {
        case monoprop::CutoffType::Length:
            return "length";
        case monoprop::CutoffType::Support:
            return "support";
        default:
            throw std::invalid_argument("Unknown CutoffType enum value");
    }
}

auto basis_str_2_enum(const std::string &basis) -> Basis {
    if (basis == "majorana") {
        return monoprop::Basis::Majorana;
    }

    if (basis == "pauli") {
        return monoprop::Basis::Pauli;
    }

    throw std::invalid_argument(
        std::format("Unknown Basis string: '{}'. Valid options are: 'majorana', 'pauli'.", basis));
}

auto basis_enum_2_str(Basis basis) -> std::string {
    switch (basis) {
        case monoprop::Basis::Majorana:
            return "majorana";
        case monoprop::Basis::Pauli:
            return "pauli";
        default:
            throw std::invalid_argument("Unknown Basis enum value");
    }
}
} // namespace monoprop
