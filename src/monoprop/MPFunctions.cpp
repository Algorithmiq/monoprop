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
    VecD cosine_cache;
    size_t cosine_cache_stride = 0;
    size_t cosine_cache_layers = 0;
};

auto eval_scratch() -> EvalScratch & {
    static thread_local EvalScratch scratch;
    return scratch;
}

// The graph is traversed in simulation order, while parameter_mapping is stored
// in optimizer order. This helper writes either forward or reversed mapped
// coefficients without rebuilding any metadata.
auto fill_mapped_params(VecD &result,
                        const VecD &parameters,
                        const VecZ &parameter_mapping,
                        const VecD &gen_coeffs,
                        double phase,
                        bool reverse) -> void {
    const size_t count = parameter_mapping.size();
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t dst = reverse ? (count - 1 - i) : i;
        result[dst] = phase * parameters[parameter_mapping[i]] * gen_coeffs[i];
    }
}

template <typename GraphType>
auto prepare_evolved_operator(EvalScratch &scratch,
                              const VecD &op,
                              const VecD &params,
                              const VecZ &parameter_mapping,
                              const VecD &gen_coeffs,
                              const GraphType &graph,
                              MPI_Comm comm) -> void {
    fill_mapped_params(scratch.mapped_params, params, parameter_mapping, gen_coeffs, 1.0, true);
    scratch.op = op;

    scratch.cosine_cache_layers = graph.layers();
    scratch.cosine_cache_stride = scratch.op.size();
    scratch.cosine_cache.resize(scratch.cosine_cache_layers * scratch.cosine_cache_stride);

    for (size_t layer_idx = 0; layer_idx < graph.layers(); ++layer_idx) {
        std::copy_n(scratch.op.data(),
                    scratch.cosine_cache_stride,
                    scratch.cosine_cache.data() + (layer_idx * scratch.cosine_cache_stride));
        evolve_step(scratch.op, graph, scratch.mapped_params[layer_idx], layer_idx, comm);
    }
}

template <typename GraphType>
auto ev_impl(double e_core,
             const VecD &state,
             const VecD &op,
             const VecZ &parameter_mapping,
             const VecD &gen_coeffs,
             const GraphType &graph,
             const VecD &params,
             MPI_Comm comm) -> double {
    if (params.empty()) {
        return e_core + mpi::allreduce_sum(inner_product(state, op), comm);
    }

    auto &scratch = eval_scratch();
    prepare_evolved_operator(scratch, op, params, parameter_mapping, gen_coeffs, graph, comm);
    return e_core + mpi::allreduce_sum(inner_product(state, scratch.op), comm);
}

template <typename GraphType>
auto ev_and_grad_impl(double e_core,
                      const VecD &state,
                      const VecD &op,
                      const VecZ &parameter_mapping,
                      const VecD &gen_coeffs,
                      const GraphType &graph,
                      const VecD &params,
                      MPI_Comm comm) -> std::pair<double, VecD> {
    if (params.empty()) {
        return {e_core + mpi::allreduce_sum(inner_product(state, op), comm), VecD(0)};
    }

    auto &scratch = eval_scratch();
    scratch.state = state;
    prepare_evolved_operator(scratch, op, params, parameter_mapping, gen_coeffs, graph, comm);

    auto &state_ = scratch.state;
    auto &op_ = scratch.op;
    const auto expectation_value = mpi::allreduce_sum(inner_product(state_, op_), comm);

    scratch.gradient.assign(params.size(), 0.0);
    set_derivative_cosine_cache(scratch.cosine_cache.data(), scratch.cosine_cache_layers, scratch.cosine_cache_stride);
    for (size_t i = 0; i < parameter_mapping.size(); ++i) {
        const auto idx = parameter_mapping.size() - 1 - i;
        const auto param_ind = parameter_mapping[i];
        scratch.gradient[param_ind] +=
            state_operator_derivative_local(state_, op_, graph, idx, gen_coeffs[i], params[param_ind], comm);
    }
    clear_derivative_cosine_cache();

    mpi::allreduce_sum_inplace(scratch.gradient, comm);
    return {e_core + expectation_value, scratch.gradient};
}

} // namespace

auto inner_product(const VecD &v, const VecD &w) -> double {
    const auto *v_data = v.data();
    const auto *w_data = w.data();
    return threading::parallel_reduce_indices(
        v.size(),
        0.0,
        [&v_data, &w_data](size_t i, double &local) { local += v_data[i] * w_data[i]; },
        std::plus<>{});
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
        MPI_Comm comm) -> double {
    return ev_impl(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
}

auto ev(double e_core,
        const VecD &state,
        const VecD &op,
        const VecZ &parameter_mapping,
        const VecD &gen_coeffs,
        const MPExecutionPlan &graph,
        const VecD &params,
        MPI_Comm comm) -> double {
    return ev_impl(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
}

auto ev_and_grad(double e_core,
                 const VecD &state,
                 const VecD &op,
                 const VecZ &parameter_mapping,
                 const VecD &gen_coeffs,
                 const MPGraph &graph,
                 const VecD &params,
                 MPI_Comm comm) -> std::pair<double, VecD> {
    return ev_and_grad_impl(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
}

auto ev_and_grad(double e_core,
                 const VecD &state,
                 const VecD &op,
                 const VecZ &parameter_mapping,
                 const VecD &gen_coeffs,
                 const MPExecutionPlan &graph,
                 const VecD &params,
                 MPI_Comm comm) -> std::pair<double, VecD> {
    return ev_and_grad_impl(e_core, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
}

} // namespace monoprop
