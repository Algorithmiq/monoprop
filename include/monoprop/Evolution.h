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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "monoprop/MPFunctions.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/monopropExport.h"

namespace monoprop {
struct Layer;
class MPGraph;
class MPGraphView;

struct LayerCore;

/**
 * @brief Perform a single-monomial evolution step using MPI communication.
 *
 * Each rank owns its local operator coefficients; cross-rank cycles are communicated via
 * Used by the in-built contraction and replay.
 */
monoprop_EXPORT auto evolve_step(VecD &op,
                                 const Layer &layer,
                                 double param,
                                 const detail::LayerCosScale &cos_scale,
                                 mpi::Comm comm) -> void;

/**
 * @brief Evolves an operator through the graph using MPI communication.
 *
 * This function applies a series of evolutions to an operator based on the
 * provided MP graph and parameters. Each rank processes its local data
 * and communicates as needed.
 *
 * @param coeffs The local rank's initial coefficients (state or operator)
 * @param graph The local rank's MPGraph containing the evolution circuit structure
 * @param params The parameters to use for each evolution step
 * @param comm MPI communicator
 * @return The evolved operator coefficients for this rank
 */
// ── Recompute-routed forward evolution (cos scaling via the mandatory callback) ─
// gradient functional replay. Callers pass a view (MPGraph::replay_view() / slice_view()).
monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                                     const MPGraphView &graph,
                                     const VecD &params,
                                     const detail::LayerCosScale &cos_scale,
                                     mpi::Comm comm) -> VecD;

// ── Recompute-routed reverse derivative (cos accumulation via the mandatory callback) ──
monoprop_EXPORT auto state_operator_derivative_local(VecD &state,
                                                     VecD &op,
                                                     const MPGraphView &graph,
                                                     size_t layer_idx,
                                                     double gen_coeff,
                                                     double param,
                                                     const detail::LayerCosAccumulate &cos_acc,
                                                     mpi::Comm comm) -> double;
} // namespace monoprop

#include "monoprop/detail/evolution/EvolutionHelpers.h"
