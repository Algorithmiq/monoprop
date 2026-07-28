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
#include <span>
#include <utility>
#include <vector>

#include "monoprop/MPGraph.h"
#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/MajoranaAlgebra.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"
#include "monoprop/detail/mpi/MPICompat.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/monopropExport.h"

namespace monoprop {

monoprop_EXPORT auto inner_product(const VecD &v, const VecD &w) -> double;

/// The indices of @p v whose magnitude exceeds @p threshold, ascending. A negative threshold clears
/// even the exact zeros, so every index qualifies -- see EvalState::indices_above, which must agree.
monoprop_EXPORT auto indices_above(const VecD &v, double threshold) -> VecZ;

/**
 * @brief The evolved operator's contraction partner: the reference state, sparse or dense.
 *
 * Heisenberg stores the reference state SPARSELY (ascending rows carrying a unit +-1 phase; on
 * production models ~0.07% of rows are nonzero), and the energy path only ever dots it against the
 * evolved operator -- so nothing densifies it. Schrödinger's state is the LIVE evolved coefficient
 * vector, which is dense in general and must be snapshotted whole.
 *
 * The three operations below are everything any consumer needs: dot it (energy), scatter it into a
 * mutable dense buffer (the gradient's in-place back-evolution), or ask which rows clear a paring
 * threshold. Values are OWNED, and `length` is snapshotted: the operator's sparse rows grow by
 * push_back as terms are appended, so a view would both dangle and outrun the captured operator.
 */
class monoprop_EXPORT EvalState {
public:
    EvalState() = default;

    /// The sparse form: @p rows must be strictly ascending and < @p length, @p values parallel to it.
    static auto sparse(size_t length, std::span<const TermIndex> rows, std::span<const double> values) -> EvalState;

    /// The dense form: @p values is the whole vector, one entry per operator term.
    static auto dense(VecD values) -> EvalState;

    /// The number of operator terms this state spans (its logical dense length).
    auto length() const -> size_t { return length_; }

    /// @brief The inner product with @p op, which must be at least length() long.
    /// Bit-identical across both forms for finite @p op: the skipped rows contribute an exact 0.0 and
    /// ascending rows preserve the dense summation order.
    auto dot(const VecD &op) const -> double;

    /// @brief Overwrite @p out with this state, resized to length().
    /// Assigns every entry rather than resizing: the sole caller's buffer is thread-local scratch
    /// reused across calls and across propagators, and it arrives holding a previous back-evolution.
    auto scatter_into(VecD &out) const -> void;

    /// The rows whose magnitude exceeds @p threshold, ascending; the paring keep-set. Agrees exactly
    /// with indices_above() over the equivalent dense vector, for every threshold.
    auto indices_above(double threshold) const -> VecZ;

private:
    size_t length_ = 0;
    bool is_dense_ = false;
    /// Sparse form only: ascending nonzero rows. Widened from TermIndex, whose width is a build knob
    /// (monoprop_WIDE_TERM_INDEX) that has no business in an exported signature.
    VecZ rows_ = {};
    VecD values_ = {}; ///< sparse: parallel to rows_; dense: the whole vector
};

monoprop_EXPORT auto map_params(const VecD &parameters,
                                const VecZ &parameter_mapping,
                                const VecD &gen_coeffs,
                                double phase,
                                bool reverse = false) -> VecD;

/// Expectation value of the evolved operator against `state` plus `e_core`, summed over all ranks.
monoprop_EXPORT auto ev(double e_core,
                        const EvalState &state,
                        const VecD &op,
                        const VecZ &parameter_mapping,
                        const VecD &gen_coeffs,
                        const MPGraph &graph,
                        const VecD &params,
                        mpi::Comm comm = MPI_COMM_WORLD,
                        const detail::LayerCosScale &cos_scale = {}) -> double;

/// As ev(), plus the gradient with respect to each parameter; both callbacks are required if `params` is non-empty.
monoprop_EXPORT auto ev_and_grad(double e_core,
                                 const EvalState &state,
                                 const VecD &op,
                                 const VecZ &parameter_mapping,
                                 const VecD &gen_coeffs,
                                 const MPGraph &graph,
                                 const VecD &params,
                                 mpi::Comm comm = MPI_COMM_WORLD,
                                 const detail::LayerCosScale &cos_scale = {},
                                 const detail::LayerCosAccumulate &cos_acc = {}) -> std::pair<double, VecD>;

/// Prune `graph` to the subgraph reaching `nonzero_inds`; `full_cos_of_layer(i)` supplies layer i's full cosine set.
monoprop_EXPORT auto pare_graph(const MPGraph &graph,
                                const VecZ &nonzero_inds,
                                size_t local_index_count,
                                bool schrodinger,
                                mpi::Comm comm,
                                const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph;
} // namespace monoprop
