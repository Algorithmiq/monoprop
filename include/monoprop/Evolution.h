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
#include <numeric>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <tbb/combinable.h>

#include "monoprop/MPFunctions.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/Utilities.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/monopropExport.h"

namespace monoprop {
struct Layer;
class MPGraph;
class MPGraphView;
class MPExecutionPlan;

template <size_t NumModes>
struct EvolveMajResult;

/**
 * @brief Perform a single-Majorana evolution step using MPI communication
 *
 * Each rank owns its local operator coefficients. Cross-rank cycles are
 * communicated via MPI_Alltoallv.
 *
 * @param op The local rank's coefficients to be evolved
 * @param graph The local rank's MPGraph
 * @param param The parameter to evolve by
 * @param layer_idx The layer index to process
 * @param comm MPI communicator
 */
monoprop_EXPORT auto evolve_step(VecD &op,
                                 const MPGraph &graph,
                                 double param,
                                 size_t layer_idx,
                                 MPI_Comm comm = MPI_COMM_WORLD) -> void;

monoprop_EXPORT auto evolve_step(VecD &op,
                                 const MPGraphView &graph,
                                 double param,
                                 size_t layer_idx,
                                 MPI_Comm comm = MPI_COMM_WORLD) -> void;

monoprop_EXPORT auto evolve_step(VecD &op,
                                 const MPExecutionPlan &graph,
                                 double param,
                                 size_t layer_idx,
                                 MPI_Comm comm = MPI_COMM_WORLD) -> void;

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
monoprop_EXPORT auto evolve_operator(const VecD &coeffs,
                                     const MPGraph &graph,
                                     const VecD &params,
                                     MPI_Comm comm = MPI_COMM_WORLD) -> VecD;

monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                                     const MPGraph &graph,
                                     const VecD &params,
                                     MPI_Comm comm = MPI_COMM_WORLD) -> VecD;

monoprop_EXPORT auto evolve_operator(const VecD &coeffs,
                                     const MPGraphView &graph,
                                     const VecD &params,
                                     MPI_Comm comm = MPI_COMM_WORLD) -> VecD;

monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                                     const MPGraphView &graph,
                                     const VecD &params,
                                     MPI_Comm comm = MPI_COMM_WORLD) -> VecD;

monoprop_EXPORT auto evolve_operator(const VecD &coeffs,
                                     const MPExecutionPlan &graph,
                                     const VecD &params,
                                     MPI_Comm comm = MPI_COMM_WORLD) -> VecD;

monoprop_EXPORT auto evolve_operator(VecD &&coeffs,
                                     const MPExecutionPlan &graph,
                                     const VecD &params,
                                     MPI_Comm comm = MPI_COMM_WORLD) -> VecD;

monoprop_EXPORT auto state_operator_derivative(VecD &state,
                                               VecD &op,
                                               const MPGraph &graph,
                                               size_t layer_idx,
                                               double gen_coeff,
                                               double param,
                                               MPI_Comm comm = MPI_COMM_WORLD) -> double;

monoprop_EXPORT auto state_operator_derivative_local(VecD &state,
                                                     VecD &op,
                                                     const MPGraph &graph,
                                                     size_t layer_idx,
                                                     double gen_coeff,
                                                     double param,
                                                     MPI_Comm comm = MPI_COMM_WORLD) -> double;
monoprop_EXPORT auto state_operator_derivative_local(VecD &state,
                                                     VecD &op,
                                                     const MPGraphView &graph,
                                                     size_t layer_idx,
                                                     double gen_coeff,
                                                     double param,
                                                     MPI_Comm comm = MPI_COMM_WORLD) -> double;

monoprop_EXPORT auto state_operator_derivative_local(VecD &state,
                                                     VecD &op,
                                                     const MPExecutionPlan &graph,
                                                     size_t layer_idx,
                                                     double gen_coeff,
                                                     double param,
                                                     MPI_Comm comm = MPI_COMM_WORLD) -> double;

// Provides optional per-layer forward cache for cosine-only derivative recovery.
// The cache layout is [layer][coefficient], flattened with row stride = layer_stride.
monoprop_EXPORT auto set_derivative_cosine_cache(const double *cache_data, size_t num_layers, size_t layer_stride)
    -> void;

// Clears any previously configured derivative cosine cache.
monoprop_EXPORT auto clear_derivative_cosine_cache() -> void;
/*!
 * @brief Evolves a single rank's operators against a Majorana generator
 *
 * This is the core building block for MPI-ready evolution. It processes only
 * the local rank's data and produces outputs organized by target rank for
 * easy communication. In an MPI setting, each rank calls this function with
 * its local data.
 *
 * @tparam NumModes Number of fermionic modes in the system
 * @param local_mp_op The local rank's MPOperator
 * @param gen_maj Majorana generator operator in bitset representation
 * @param cutoff_fn Cutoff function to filter new terms
 * @param atol Lower absolute tolerance for coefficient truncation
 * @param local_coeffs Optional coefficients for the local rank's operators
 * @param upper_atol Upper absolute tolerance for coefficient truncation
 * @param param Optional evolution parameter (for sin/cos computation)
 * @param only_rotate_len_k If > 0, apply gates to monomials of length <= k in the evolved operator
 *                           even if they anticommute. This is useful for when you apply many free
 *                           fermionic gates (ie: gates generated by length 2 majorana monomials)
 *                           before expectation value estimation in schrodinger picture simulations.
 *                           0 disables this filter.
 * @param comm MPI communicator used for ownership queries
 * @param num_ranks Total number of ranks in the distributed system
 * @return EvolveMajResult containing evolution data organized by target rank
 */
template <size_t NumModes>
auto evolve_maj_single_rank(const MPOperator<NumModes> &local_mp_op,
                            const MajoranaSet<NumModes> &gen_maj,
                            const CutoffFn<NumModes> &cutoff_fn,
                            const std::optional<double> &atol,
                            std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                            const std::optional<double> &upper_atol,
                            const std::optional<double> &param,
                            int only_rotate_len_k,
                            MPI_Comm comm,
                            size_t num_ranks) -> EvolveMajResult<NumModes>;

/*!
 * @brief Identifies operators that anticommute with a generator and builds MP evolution data
 *
 * This function analyzes a set of Majorana operators to identify those that anticommute
 * with a specified generator. It produces the necessary data structures for time evolution,
 * including cosine indices, cycles, and phases. Uses MPI for inter-rank communication.
 *
 * @tparam NumModes Number of fermionic modes in the system
 * @param local_mp_op The local rank's MPOperator
 * @param gen_maj Majorana generator operator in bitset representation
 * @param cutoff_fn Cutoff function to filter new terms
 * @param atol Lower absolute tolerance for coefficient truncation
 * @param local_coeffs Optional coefficients for the local rank's operators
 * @param upper_atol Upper absolute tolerance for coefficient truncation
 * @param param Optional evolution parameter (for sin/cos computation)
 * @param only_rotate_len_k If > 0, apply gates to monomials of length <= k in the evolved operator
 *                           even if they anticommute. 0 disables this filter.
 * @param comm MPI communicator
 * @return Evolution data for this rank, including cycles, cosine indices, and any new terms
 */
template <size_t NumModes>
auto evolve_maj(const MPOperator<NumModes> &local_mp_op,
                const MajoranaSet<NumModes> &gen_maj,
                const CutoffFn<NumModes> &cutoff_fn,
                const std::optional<double> &atol,
                std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                const std::optional<double> &upper_atol,
                const std::optional<double> &param,
                int only_rotate_len_k,
                MPI_Comm comm = MPI_COMM_WORLD) -> EvolveMajResult<NumModes>;
} // namespace monoprop

#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/EvolutionMajorana.h"
