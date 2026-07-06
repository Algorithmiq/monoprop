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

#include <functional>
#include <utility>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/MajoranaAlgebra.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

monoprop_EXPORT auto inner_product(const VecD &v, const VecD &w) -> double;

monoprop_EXPORT auto map_params(const VecD &parameters,
                                const VecZ &parameter_mapping,
                                const VecD &gen_coeffs,
                                double phase,
                                bool reverse = false) -> VecD;

// The forward (cos_scale) and reverse (cos_acc) callbacks recompute each layer's cosine set from the
// prepared fold; passing default-constructed (empty) std::function selects the stored-cos path.
monoprop_EXPORT auto ev(double e_core,
                        const VecD &state,
                        const VecD &op,
                        const VecZ &parameter_mapping,
                        const VecD &gen_coeffs,
                        const MPGraph &graph,
                        const VecD &params,
                        MPI_Comm comm = MPI_COMM_WORLD,
                        const detail::LayerCosScale &cos_scale = {}) -> double;

monoprop_EXPORT auto ev_and_grad(double e_core,
                                 const VecD &state,
                                 const VecD &op,
                                 const VecZ &parameter_mapping,
                                 const VecD &gen_coeffs,
                                 const MPGraph &graph,
                                 const VecD &params,
                                 MPI_Comm comm = MPI_COMM_WORLD,
                                 const detail::LayerCosScale &cos_scale = {},
                                 const detail::LayerCosAccumulate &cos_acc = {})
    -> std::pair<double, VecD>;

// Streaming backward keep-set sweep producing a typed-layer MPGraph (FoldLayer / PrunedLayer) that
// reuses `graph`'s layer cores (shared_ptr — no cross-rank copy) and stores a pruned CosMask
// only on layers whose cos was trimmed. `full_cos_of_layer(layer_idx)` is invoked at most once per
// layer, in sweep order, so the caller materializes one layer's cos at a time. The cross-rank D/B
// reachability exchange runs (propagating the keep-set across ranks) but stores NO positions —
// cross-rank replays unmasked, exactly as today. Definitions live in src/monoprop/detail/pare/PareGraph.cpp.
monoprop_EXPORT auto pare_graph(const MPGraph &graph,
                const VecZ &nonzero_inds,
                size_t local_index_count,
                bool schrodinger,
                MPI_Comm comm,
                const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph;

monoprop_EXPORT auto get_pared_graph(const VecD &state,
                     const VecD &op,
                     double threshold,
                     const MPGraph &graph,
                     bool schrodinger,
                     MPI_Comm comm,
                     const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph;
} // namespace monoprop
