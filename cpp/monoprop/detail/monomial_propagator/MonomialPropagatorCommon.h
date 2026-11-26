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

#include <stdexcept>
#include <utility>

#include "monoprop/MPFunctions.h"
#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/AlgebraCommon.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/core/Monomial.h"

namespace monoprop::detail {
class UnknownCutoffTypeError : public std::runtime_error {
public:
    UnknownCutoffTypeError() : std::runtime_error("Unknown cutoff type") {}
};

template <size_t NumModes>
auto cutoff_function(CutoffType cutoff_type, unsigned int cutoff, size_t logical_num_modes = NumModes)
    -> CutoffFn<NumModes> {
    switch (cutoff_type) {
        case CutoffType::Length:
            return detail::LengthCutoff<NumModes>{cutoff, logical_num_modes};
        case CutoffType::Support:
            return detail::SupportCutoff<NumModes>{cutoff, logical_num_modes};
        default:
            throw UnknownCutoffTypeError();
    }
}

template <size_t NumModes>
auto cutoff_function_basis_change(CutoffType cutoff_type,
                                  unsigned int cutoff,
                                  const MonomialList<NumModes> &basis,
                                  size_t logical_num_modes = NumModes) -> CutoffFn<NumModes> {
    switch (cutoff_type) {
        case CutoffType::Length:
            return [cutoff, logical_num_modes, basis_copy = basis](const Monomial<NumModes> &mono) {
                const auto mapped_mono = change_basis<NumModes>(mono, basis_copy);
                return length_cutoff<NumModes>(mapped_mono, cutoff, logical_num_modes);
            };
        case CutoffType::Support:
            return [cutoff, logical_num_modes, basis_copy = basis](const Monomial<NumModes> &mono) {
                const auto mapped_mono = change_basis<NumModes>(mono, basis_copy);
                return support_cutoff<NumModes>(mapped_mono, cutoff, logical_num_modes);
            };
        default:
            throw UnknownCutoffTypeError();
    }
}

/*!
 * \brief The graph's (parameter_mapping, gen_coeffs) arrays, one entry per layer, in optimizer order.
 *
 * Layers store gate info in simulation order; the evaluation machinery expects optimizer order (the reverse).
 */
inline auto gate_arrays(const MPGraph &graph) -> std::pair<VecZ, VecD> {
    const size_t count = graph.layers();
    VecZ parameter_mapping(count);
    VecD gen_coeffs(count);
    for (size_t layer = 0; layer < count; ++layer) {
        const auto traversal = graph.get_layer_traversal(layer);
        const size_t optimizer_index = count - 1 - layer;
        parameter_mapping[optimizer_index] = traversal.param_index();
        gen_coeffs[optimizer_index] = traversal.gen_coeff();
    }
    return {std::move(parameter_mapping), std::move(gen_coeffs)};
}

/*!
 * \brief The angles that replay a picture through a graph's layers, as contract_partially() does.
 *
 * Heisenberg replays the oldest layers first in reverse optimizer order; Schrödinger replays the reversed window at
 * the negated angles, starting from the dense state.
 */
inline auto replay_angles(const VecD &parameters,
                          const VecZ &parameter_mapping,
                          const VecD &gen_coeffs,
                          bool schrodinger) -> VecD {
    return schrodinger ? map_params(parameters, parameter_mapping, gen_coeffs, -1.0)
                       : map_params(parameters, parameter_mapping, gen_coeffs, 1.0, true);
}
} // namespace monoprop::detail
