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

#include "monoprop/MPFunctions.h"

#include "monoprop/Evolution.h"
#include "monoprop/Threading.h"

namespace monoprop {

namespace {

// Reuse these buffers per thread so repeated ev/grad calls stay allocation-light
// after the plan-building logic moved out of this translation unit.
struct EvalScratch {
    VecD state;
    VecD op;
    VecD mapped_params;
    VecD gradient;
};

auto eval_scratch() -> EvalScratch & {
    static thread_local EvalScratch scratch;
    return scratch;
}

// The graph is traversed in simulation order, while parameter_mapping is stored
// in optimizer order. This helper writes either forward or reversed mapped
// coefficients without rebuilding any metadata.
void fill_mapped_params(VecD &result,
                        const VecD &parameters,
                        const VecZ &parameter_mapping,
                        const VecD &gen_coeffs,
                        double phase,
                        bool reverse) {
    const size_t count = parameter_mapping.size();
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t dst = reverse ? (count - 1 - i) : i;
        result[dst] = phase * parameters[parameter_mapping[i]] * gen_coeffs[i];
    }
}

// Forward-evolve a copy of the operator into scratch.op: map the raw params to per-layer angles, then
// apply every layer's rotation. Shared setup for ev_impl and ev_and_grad_impl.
auto prepare_evolved_operator(EvalScratch &scratch,
                              const VecD &op,
                              const VecD &params,
                              const VecZ &parameter_mapping,
                              const VecD &gen_coeffs,
                              const MPGraphView &graph,
                              mpi::Comm comm,
                              const detail::LayerCosScale &cos_scale) -> void {
    fill_mapped_params(scratch.mapped_params, params, parameter_mapping, gen_coeffs, 1.0, true);
    scratch.op = op;
    // Every functional supplies a non-empty cos_scale callback (the layer cosine set is always
    // recomputed/transient, never read from a stored bitmap), so the cos pass routes through it.
    scratch.op = evolve_operator(std::move(scratch.op), graph, scratch.mapped_params, cos_scale, comm);
}

// Expectation value ⟨state|evolved op⟩ + e_core, summed across ranks. Empty params ⇒ evaluate the
// unevolved operator directly.
auto ev_impl(double e_core,
             const VecD &state,
             const VecD &op,
             const VecZ &parameter_mapping,
             const VecD &gen_coeffs,
             const MPGraphView &graph,
             const VecD &params,
             mpi::Comm comm,
             const detail::LayerCosScale &cos_scale) -> double {
    if (params.empty()) {
        return e_core + mpi::allreduce_sum(inner_product(state, op), comm);
    }

    auto &scratch = eval_scratch();
    prepare_evolved_operator(scratch, op, params, parameter_mapping, gen_coeffs, graph, comm, cos_scale);
    return e_core + mpi::allreduce_sum(inner_product(state, scratch.op), comm);
}

// Expectation value and its gradient w.r.t. the parameters. Forward-evolves the operator, then walks
// the layers in reverse accumulating each parameter's derivative (allreduced across ranks). Empty
// params ⇒ value only, with an empty gradient.
auto ev_and_grad_impl(double e_core,
                      const VecD &state,
                      const VecD &op,
                      const VecZ &parameter_mapping,
                      const VecD &gen_coeffs,
                      const MPGraphView &graph,
                      const VecD &params,
                      mpi::Comm comm,
                      const detail::LayerCosScale &cos_scale,
                      const detail::LayerCosAccumulate &cos_acc) -> std::pair<double, VecD> {
    if (params.empty()) {
        return {e_core + mpi::allreduce_sum(inner_product(state, op), comm), VecD(0)};
    }

    // The stored-cos fallback was removed: both callbacks are consumed on the with-parameters path
    // (cos_scale in the forward prepare, cos_acc in the reverse sweep). Fail loudly rather than with a
    // cryptic std::bad_function_call if a caller relied on the old empty-callback default.
    if (!cos_scale || !cos_acc) {
        throw std::invalid_argument("ev_and_grad requires both cos_scale (forward) and cos_acc (reverse) callbacks; "
                                    "the stored-cos fallback no longer exists.");
    }

    auto &scratch = eval_scratch();
    scratch.state = state;
    prepare_evolved_operator(scratch, op, params, parameter_mapping, gen_coeffs, graph, comm, cos_scale);

    auto &state_ = scratch.state;
    auto &op_ = scratch.op;
    const auto expectation_value = mpi::allreduce_sum(inner_product(state_, op_), comm);

    scratch.gradient.assign(params.size(), 0.0);
    for (size_t i = 0; i < parameter_mapping.size(); ++i) {
        const auto idx = parameter_mapping.size() - 1 - i;
        const auto param_ind = parameter_mapping[i];
        // cos_acc is always non-empty (see prepare_evolved_operator): the reverse-derivative
        // cosine accumulation always routes through the recompute/transient callback.
        scratch.gradient[param_ind] +=
            state_operator_derivative_local(state_, op_, graph, idx, gen_coeffs[i], params[param_ind], cos_acc, comm);
    }

    mpi::allreduce_sum_inplace(scratch.gradient, comm);
    return {e_core + expectation_value, scratch.gradient};
}

} // namespace

auto inner_product(const VecD &v, const VecD &w) -> double {
    const auto *v_data = v.data();
    const auto *w_data = w.data();
    double result = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        result += v_data[i] * w_data[i];
    }
    return result;
}

auto map_params(const VecD &parameters,
                const VecZ &parameter_mapping,
                const VecD &gen_coeffs,
                double phase,
                bool reverse) -> VecD {
    VecD result;
    fill_mapped_params(result, parameters, parameter_mapping, gen_coeffs, phase, reverse);
    return result;
}

auto ev(double e_core,
        const VecD &state,
        const VecD &op,
        const VecZ &parameter_mapping,
        const VecD &gen_coeffs,
        const MPGraph &graph,
        const VecD &params,
        mpi::Comm comm,
        const detail::LayerCosScale &cos_scale) -> double {
    return ev_impl(e_core, state, op, parameter_mapping, gen_coeffs, graph.replay_view(), params, comm, cos_scale);
}

auto ev_and_grad(double e_core,
                 const VecD &state,
                 const VecD &op,
                 const VecZ &parameter_mapping,
                 const VecD &gen_coeffs,
                 const MPGraph &graph,
                 const VecD &params,
                 mpi::Comm comm,
                 const detail::LayerCosScale &cos_scale,
                 const detail::LayerCosAccumulate &cos_acc) -> std::pair<double, VecD> {
    return ev_and_grad_impl(e_core,
                            state,
                            op,
                            parameter_mapping,
                            gen_coeffs,
                            graph.replay_view(),
                            params,
                            comm,
                            cos_scale,
                            cos_acc);
}

} // namespace monoprop
