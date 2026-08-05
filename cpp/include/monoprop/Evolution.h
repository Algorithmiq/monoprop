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

#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/monopropExport.h"

namespace monoprop {
struct Layer;
class MPGraphView;

/// One layer's rotation angle in factored form: the layer rotates by 2·gen_coeff·param.
struct LayerAngle {
    double gen_coeff = 1.0; ///< the generator's coefficient for this layer
    double param = 0.0;     ///< the optimizer parameter driving this layer
};

/// Forward-evolve `op` through one layer; each rank owns its local coefficients, cross-rank cycles are communicated.
monoprop_EXPORT auto evolve_step(VecD &op,
                                 const Layer &layer,
                                 double param,
                                 mpi::Comm comm,
                                 const detail::LayerCosScale &cos_scale) -> void;

/// Forward-evolve `coeffs` through every layer of `graph`, returning this rank's evolved coefficients.
monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                                     const MPGraphView &graph,
                                     const VecD &params,
                                     mpi::Comm comm,
                                     const detail::LayerCosScale &cos_scale) -> VecD;

/// Reverse-mode derivative of one layer: inverse-rotates (state, op) in place and returns the gradient term.
monoprop_EXPORT auto state_operator_derivative_local(VecD &state,
                                                     VecD &op,
                                                     const MPGraphView &graph,
                                                     size_t layer_idx,
                                                     LayerAngle angle,
                                                     mpi::Comm comm,
                                                     const detail::LayerCosAccumulate &cos_acc) -> double;
} // namespace monoprop
