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

/// The indices of `v` whose magnitude exceeds `threshold`, ascending. A negative threshold admits even
/// the exact zeros, so every index qualifies -- EvalState::indices_above must agree.
monoprop_EXPORT auto indices_above(const VecD &v, double threshold) -> VecZ;

/// The evolved operator's contraction partner: the reference state, sparse (Heisenberg) or dense
/// (Schrödinger, whose state is the live evolved coefficient vector). Values are OWNED and `length` is
/// snapshotted: the operator's sparse rows grow by push_back as terms are appended, so a view would
/// both dangle and outrun the captured operator.
class monoprop_EXPORT EvalState {
public:
    EvalState() = default;

    /// `rows` must be strictly ascending and < `length`, `values` parallel to it.
    static auto sparse(size_t length, std::span<const TermIndex> rows, std::span<const double> values) -> EvalState;

    /// `values` is the whole vector, one entry per operator term.
    static auto dense(VecD values) -> EvalState;

    /// The logical dense length: the number of operator terms spanned, in either form.
    auto length() const -> size_t { return length_; }

    /// The inner product with `op`, which must be at least length() long. Bit-identical across both
    /// forms for finite `op`: the skipped rows contribute an exact 0.0 and ascending rows preserve the
    /// dense summation order.
    auto dot(const VecD &op) const -> double;

    /// Overwrite `out` with this state, resized to length(); entries this state does not name read as
    /// an exact zero.
    auto scatter_into(VecD &out) const -> void;

    /// The rows whose magnitude exceeds `threshold`, ascending; the paring keep-set. Agrees with
    /// indices_above() over the equivalent dense vector, for every threshold.
    auto indices_above(double threshold) const -> VecZ;

private:
    size_t length_ = 0;
    bool is_dense_ = false;
    /// Widened from TermIndex, whose width is a build knob (monoprop_WIDE_TERM_INDEX) that has no
    /// business in an exported signature.
    VecZ rows_ = {};
    /// sparse: parallel to rows_; dense: the whole vector
    VecD values_ = {};
};

monoprop_EXPORT auto map_params(const VecD &parameters,
                                const VecZ &parameter_mapping,
                                const VecD &gen_coeffs,
                                double phase,
                                bool reverse = false) -> VecD;

/// Everything one expectation-value evaluation reads, in one object: the graph plus the arrays that index
/// into it. Built at the call site and consumed there -- every member but `graph` is a non-owning
/// reference, and `graph` is itself a view over layers the caller keeps alive.
struct EvalRequest {
    double e_core;                 ///< the identity term, added to the summed expectation value
    const EvalState &state;        ///< the contraction partner; see EvalState
    const VecD &op;                ///< un-evolved operator coefficients, one per term on this rank
    const VecZ &parameter_mapping; ///< optimizer order: which parameter drives graph layer i
    const VecD &gen_coeffs;        ///< optimizer order, parallel to parameter_mapping
    MPGraphView graph;             ///< replay window; MPGraph callers pass graph.replay_view()
    const VecD &params;            ///< the optimizer's parameter vector, indexed by parameter_mapping
};

/// Expectation value of the evolved operator against `request.state` plus `request.e_core`, summed over all
/// ranks. `cos.scale` is required if `request.params` is non-empty; `cos.accumulate` is ignored here.
monoprop_EXPORT auto ev(const EvalRequest &request,
                        mpi::Comm comm = MPI_COMM_WORLD,
                        const detail::CosCallbacks &cos = {}) -> double;

/// As ev(), plus the gradient with respect to each parameter; both callbacks are required if
/// `request.params` is non-empty.
monoprop_EXPORT auto ev_and_grad(const EvalRequest &request,
                                 mpi::Comm comm = MPI_COMM_WORLD,
                                 const detail::CosCallbacks &cos = {}) -> std::pair<double, VecD>;

/// Prune `graph` to the subgraph reaching `nonzero_inds`; `full_cos_of_layer(i)` supplies layer i's full cosine set.
monoprop_EXPORT auto pare_graph(const MPGraph &graph,
                                const VecZ &nonzero_inds,
                                size_t local_index_count,
                                mpi::Comm comm,
                                const std::function<CosMask(size_t)> &full_cos_of_layer) -> MPGraph;
} // namespace monoprop
