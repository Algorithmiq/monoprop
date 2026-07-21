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

#include <algorithm>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/evolution/LayerBuilder.h"
#include "monoprop/detail/evolution/EvolutionHelpers.h"

namespace monoprop {

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expected_num_params(const VecZ &parameter_mapping) -> size_t {
    return parameter_mapping.empty() ? 0 : *std::max_element(parameter_mapping.begin(), parameter_mapping.end()) + 1;
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_parameter_validated_functional(size_t expected_num_params, Fn func)
    -> std::function<R(const VecD &)> {
    return [expected_num_params, func = std::move(func)](const VecD &params) -> R {
        validate_functional_call(params, expected_num_params);
        return func(params);
    };
}

} // namespace monoprop
