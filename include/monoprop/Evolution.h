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
 * @brief Perform a single-Majorana evolution step using MPI communication.
 *
 * Each rank owns its local operator coefficients; cross-rank cycles are communicated via
 * MPI_Alltoallv. Recompute-routed (cos via callback): the layer stores no cos bitmap, so
 * cos_scale (recompute fold / transient or filtered word list) performs the cosine scaling.
 * Used by the in-build contraction and replay.
 */
monoprop_EXPORT auto evolve_step(VecD &op, const Layer &layer, double param, const detail::LayerCosScale &cos_scale, mpi::Comm comm)
    -> void;

/**
 * @brief Evolves an operator through the graph using MPI communication.
 *
 * This function applies a series of evolutions to an operator based on the
 * provided MBS graph and parameters. Each rank processes its local data
 * and communicates as needed.
 *
 * @param coeffs The local rank's initial coefficients (state or operator)
 * @param graph The local rank's MPGraph containing the evolution circuit structure
 * @param params The parameters to use for each evolution step
 * @param comm MPI communicator
 * @return The evolved operator coefficients for this rank
 */
// ── Recompute-routed forward evolution (cos scaling via the mandatory callback) ─
// Each layer's cosine scaling is performed by `cos_scale(layer_index, …)` (the prepared-fold
// recompute / transient or filtered word list) — no layer stores its cos bitmap. Used by the energy/
// gradient functional replay. Callers pass a view (MPGraph::replay_view() / slice_view()).
monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                     const MPGraphView &graph,
                     const VecD &params,
                     const detail::LayerCosScale &cos_scale,
                     mpi::Comm comm) -> VecD;

// ── Recompute-routed reverse derivative (cos accumulation via the mandatory callback) ──
// The cosine accumulate pass is performed by `cos_acc(layer_index, …)` (the prepared-fold recompute /
// transient or filtered word list) — no layer stores its cos bitmap.
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
