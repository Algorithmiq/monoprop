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

/// @brief Perform a single-monomial evolution step (MPI: each rank owns its local coefficients,
/// cross-rank cycles are communicated). Used by the in-built contraction and replay.
monoprop_EXPORT auto evolve_step(VecD &op,
                                 const Layer &layer,
                                 double param,
                                 const detail::LayerCosScale &cos_scale,
                                 mpi::Comm comm) -> void;

/// @brief Evolve an operator through the graph (per-rank local data + MPI as needed). Returns this
/// rank's evolved coefficients.
// Recompute-routed forward evolution (cos scaling via the mandatory callback). Callers pass a view
// (MPGraph::replay_view() / slice_view()).
monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                                     const MPGraphView &graph,
                                     const VecD &params,
                                     const detail::LayerCosScale &cos_scale,
                                     mpi::Comm comm) -> VecD;

// Recompute-routed reverse derivative (cos accumulation via the mandatory callback).
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
