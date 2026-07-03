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

#include "monoprop/MonomialPropagator.h"

namespace monoprop {

template <size_t NumModes>
auto MonomialPropagator<NumModes>::regenerate_cutoff_fn_() -> void {
    if (basis_change_.has_value()) {
        MajoranaVector<NumModes> basis(2 * logical_num_modes_);
        for (size_t i = 0; i < 2 * logical_num_modes_; ++i) {
            basis[i] = indices_to_bitset<NumModes>(basis_change_.value()[i]);
        }
        cutoff_fn_ = detail::cutoff_function_basis_change<NumModes>(cutoff_type_, cutoff_, basis, logical_num_modes_);
    }
    else {
        cutoff_fn_ = detail::cutoff_function<NumModes>(cutoff_type_, cutoff_, logical_num_modes_);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::initialize_operator_caches_() -> void {
    (void)mp_op_.get_operator();
    (void)mp_op_.get_state();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::current_picture_coeffs_() -> const VecD & {
    return schrodinger_ ? mp_op_.get_state() : mp_op_.get_operator();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::extend_coeffs_from_current_picture_if_needed_(VecD &coeffs) -> void {
    if (coeffs.size() >= mp_op_.size()) {
        return;
    }

    const auto &current = current_picture_coeffs_();
    if (&coeffs == &current) {
        return;
    }

    coeffs.insert(coeffs.end(), current.begin() + static_cast<std::ptrdiff_t>(coeffs.size()), current.end());
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_build_graph_(const std::vector<VecZ> &majoranas,
                                                            const VecZ &parameter_mapping,
                                                            const VecD &gen_coeffs,
                                                            const VecZ &gate_indices,
                                                            int only_rotate_len_k) -> void {
    const auto majoranas_size = majoranas.size();
    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &parameter_mapping, &gen_coeffs, &gate_indices, majoranas_size](const VecZ &maj, int rot_len, size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            propagate_one_(maj,
                           rot_len,
                           std::nullopt,
                           std::nullopt,
                           parameter_mapping[idx],
                           gen_coeffs[idx],
                           gate_indices[idx]);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_graph_with_coeffs_(const std::vector<VecZ> &majoranas,
                                                                  const VecZ &parameter_mapping,
                                                                  const VecD &gen_coeffs,
                                                                  const VecZ &gate_indices,
                                                                  const VecD &parameters,
                                                                  const VecD &operator_coeffs,
                                                                  int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    auto coeffs = operator_coeffs;
    const auto majoranas_size = majoranas.size();

    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &parameter_mapping, &gen_coeffs, &gate_indices, &mapped_params, &coeffs, majoranas_size](
            const VecZ &maj,
            int only_rotate_len_k,
            size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            propagate_one_(maj,
                           only_rotate_len_k,
                           std::cref(coeffs),
                           mapped_params[idx],
                           parameter_mapping[idx],
                           gen_coeffs[idx],
                           gate_indices[idx]);
            const auto param = schrodinger_ ? -mapped_params[idx] : mapped_params[idx];
            const auto graph_idx = schrodinger_ ? 0 : i;
            extend_coeffs_from_current_picture_if_needed_(coeffs);

            evolve_step(coeffs, graph_, param, graph_idx, comm_);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_contract_immediately_(const std::vector<VecZ> &majoranas,
                                                                     const VecZ &parameter_mapping,
                                                                     const VecD &gen_coeffs,
                                                                     const VecD &parameters,
                                                                     int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    VecD *op_coeffs = schrodinger_ ? &mp_op_.state_coeffs : &mp_op_.op_coeffs;
    *op_coeffs = current_picture_coeffs_();
    const auto majoranas_size = majoranas.size();

    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &mapped_params, op_coeffs, majoranas_size](const VecZ &maj, int only_rotate_len_k, size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            propagate_one_(maj, only_rotate_len_k, std::cref(*op_coeffs), mapped_params[idx]);
            extend_coeffs_from_current_picture_if_needed_(*op_coeffs);

            const auto param = schrodinger_ ? -mapped_params[idx] : mapped_params[idx];
            evolve_step(*op_coeffs, graph_.slice_view(1), param, 0, comm_);
            graph_.consume_prefix(1);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_graph(const std::vector<VecZ> &majoranas,
                                               const VecZ &parameter_mapping,
                                               const VecD &gen_coeffs,
                                               std::optional<VecZ> gate_indices,
                                               std::optional<VecD> parameters,
                                               int only_rotate_len_k) -> void {
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);

    // Resolve gate indices: default to one gate per generator (iota). These are local and
    // 0-based per call; offset by the gate count already in the graph so they are absolute.
    VecZ local_gates;
    if (gate_indices.has_value()) {
        local_gates = std::move(*gate_indices);
    }
    else {
        local_gates.resize(majoranas.size());
        std::iota(local_gates.begin(), local_gates.end(), size_t{0});
    }
    validate_gate_indices(local_gates, majoranas.size());
    const size_t gate_offset = n_gates();
    for (auto &g : local_gates) {
        g += gate_offset;
    }

    if (!parameters.has_value()) {
        // Pure structural build: append layers recording their gate information.
        evolve_mode_build_graph_(majoranas, parameter_mapping, gen_coeffs, local_gates, only_rotate_len_k);
    }
    else {
        // Coefficient-informed build: regenerate the seed by contracting the existing graph
        // (replacing the former operator_coeffs input), then build+contract the new layers into it
        // so atol truncation sees realistic coefficients. The existing graph references the
        // parameter prefix [0, m); slice `parameters` to that so the exact-length check passes.
        VecD seed;
        if (graph_layers() > 0) {
            const auto existing = graph_gate_arrays_();
            const size_t m = expected_num_params(existing.first);
            const VecD existing_params(
                parameters->begin(),
                parameters->begin() + static_cast<std::ptrdiff_t>(std::min(m, parameters->size())));
            seed = contract_partially(existing_params, false);
        }
        else {
            seed = current_picture_coeffs_();
        }
        evolve_mode_graph_with_coeffs_(majoranas,
                                       parameter_mapping,
                                       gen_coeffs,
                                       local_gates,
                                       *parameters,
                                       seed,
                                       only_rotate_len_k);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate(const std::vector<VecZ> &majoranas,
                                             const VecZ &parameter_mapping,
                                             const VecD &gen_coeffs,
                                             const VecD &parameters,
                                             int only_rotate_len_k) -> void {
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);
    validate_parameters_length(parameters, parameter_mapping);
    evolve_mode_contract_immediately_(majoranas, parameter_mapping, gen_coeffs, parameters, only_rotate_len_k);
}

template <size_t NumModes>
template <typename EvolutionFunc>
auto MonomialPropagator<NumModes>::propagate_with_timing_(const std::vector<VecZ> &majoranas,
                                                          int only_rotate_len_k,
                                                          EvolutionFunc evolution_func) -> void {
    for (size_t i = 0; i < majoranas.size(); ++i) {
        const auto idx = !schrodinger_ ? majoranas.size() - 1 - i : i;
        const auto &maj = majoranas[idx];

        evolution_func(maj, only_rotate_len_k, i);
    }

    initialize_operator_caches_();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate_one_(const VecZ &gen_vec,
                                                  int only_rotate_len_k,
                                                  std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                  std::optional<double> param,
                                                  size_t param_index,
                                                  double gen_coeff,
                                                  size_t gate_index) -> void {
    const auto gen_maj = indices_to_bitset<NumModes>(gen_vec);

    auto evolve_result = evolve_maj<NumModes>(mp_op_,
                                              gen_maj,
                                              cutoff_fn_,
                                              lower_atol_,
                                              coeffs,
                                              upper_atol_,
                                              param,
                                              only_rotate_len_k,
                                              comm_);

    SplitCycleResult split;
    update_mp<NumModes>(mp_op_,
                        evolve_result.half_op,
                        evolve_result.half_cycles,
                        evolve_result.half_phases,
                        evolve_result.cycles,
                        evolve_result.phases,
                        comm_);
    split = split_and_exchange_cycles(evolve_result.cycles, evolve_result.phases, comm_);

    append_to_graph(graph_,
                    evolve_result.cos_inds,
                    evolve_result.compressed_cos_data,
                    split,
                    comm_,
                    param_index,
                    gen_coeff,
                    gate_index);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::n_gates() const -> size_t {
    const size_t count = graph_.layers();
    size_t max_gate = 0;
    bool any = false;
    for (size_t layer = 0; layer < count; ++layer) {
        const size_t g = graph_.get_layer_traversal(layer).gate_index();
        max_gate = any ? std::max(max_gate, g) : g;
        any = true;
    }
    return any ? max_gate + 1 : 0;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::set_parameter_mapping(const VecZ &parameter_mapping) -> void {
    const size_t count = graph_.layers();
    const size_t gates = n_gates();

    if (parameter_mapping.size() == count) {
        // Per-layer mapping in optimizer order; layer `layer` holds optimizer index
        // count-1-layer (see graph_gate_arrays_). Relabel in place, preserving each layer's
        // generator coeff and gate index.
        for (size_t layer = 0; layer < count; ++layer) {
            const size_t optimizer_index = count - 1 - layer;
            auto &target = graph_.get_layer(layer);
            target.set_gate_info(parameter_mapping[optimizer_index], target.gen_coeff(), target.gate_index());
        }
    }
    else if (parameter_mapping.size() == gates) {
        // Per-gate mapping indexed by absolute gate index: relabel each layer via its own
        // stored gate index (order-agnostic, correct in both pictures and across builds).
        for (size_t layer = 0; layer < count; ++layer) {
            auto &target = graph_.get_layer(layer);
            target.set_gate_info(parameter_mapping[target.gate_index()], target.gen_coeff(), target.gate_index());
        }
    }
    else {
        throw std::runtime_error(std::format("parameter_mapping has {} entries; expected {} (per graph "
                                             "layer) or {} (per gate).",
                                             parameter_mapping.size(),
                                             count,
                                             gates));
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_gate_arrays_() const -> std::pair<VecZ, VecD> {
    const size_t count = graph_.layers();
    VecZ parameter_mapping(count);
    VecD gen_coeffs(count);
    // Layers store gate info in simulation order; the evaluation machinery expects it in
    // optimizer order (the reverse), so place layer `layer` at optimizer index count-1-layer.
    for (size_t layer = 0; layer < count; ++layer) {
        const auto traversal = graph_.get_layer_traversal(layer);
        const size_t optimizer_index = count - 1 - layer;
        parameter_mapping[optimizer_index] = traversal.param_index();
        gen_coeffs[optimizer_index] = traversal.gen_coeff();
    }
    return {std::move(parameter_mapping), std::move(gen_coeffs)};
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_functional_(Fn &&func, std::optional<double> pare_threshold)
    -> std::function<R(const VecD &)> {
    auto gate_arrays = graph_gate_arrays_();
    auto parameter_mapping = std::move(gate_arrays.first);
    auto gen_coeffs = std::move(gate_arrays.second);
    const auto num_params = expected_num_params(parameter_mapping);

    VecD state = mp_op_.get_state();
    VecD op = mp_op_.get_operator();
    const auto core_term = this->core_term();
    const auto comm = comm_;

    if (pare_threshold.has_value()) {
        auto masked_graph = get_masked_execution_plan(state, op, *pare_threshold, graph_, schrodinger_, comm_);
        const auto expected_layers = graph_layers();
        return make_parameter_validated_functional(
            num_params,
            [func = std::forward<Fn>(func),
             core_term,
             state = std::move(state),
             op = std::move(op),
             graph = std::move(masked_graph),
             parameter_mapping = std::move(parameter_mapping),
             gen_coeffs = std::move(gen_coeffs),
             expected_layers,
             comm](const VecD &params) -> R {
                validate_expected_graph_layers(graph.layers(), expected_layers);
                return func(core_term, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
            });
    }

    const auto expected_layers = graph_layers();
    return make_parameter_validated_functional(
        num_params,
        [func = std::forward<Fn>(func),
         core_term,
         state = std::move(state),
         op = std::move(op),
         &graph = graph_,
         parameter_mapping = std::move(parameter_mapping),
         gen_coeffs = std::move(gen_coeffs),
         expected_layers,
         comm](const VecD &params) -> R {
            validate_expected_graph_layers(graph.layers(), expected_layers);
            return func(core_term, state, op, parameter_mapping, gen_coeffs, graph, params, comm);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_functional(std::optional<double> pare_threshold)
    -> std::function<double(const VecD &)> {
    return make_functional_(ev_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient_functional(std::optional<double> pare_threshold)
    -> std::function<std::pair<double, VecD>(const VecD &)> {
    return make_functional_(ev_and_grad_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value(const VecD &parameters) -> double {
    return expectation_value_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD> {
    return expectation_value_and_gradient_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::contract_partially(const VecD &parameters, bool inplace) -> VecD {
    const auto gate_arrays = graph_gate_arrays_();
    const auto &parameter_mapping = gate_arrays.first;
    const auto &gen_coeffs = gate_arrays.second;
    validate_parameters_length(parameters, parameter_mapping);

    if (parameters.empty()) {
        return current_picture_coeffs_();
    }

    const size_t num_majoranas = parameter_mapping.size();
    if (schrodinger_) {
        const auto &state = mp_op_.get_state();
        const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, -1.0);
        const auto evolved_state =
            inplace ? evolve_operator(state, graph_.slice_graph(num_majoranas, true), mapped_params, comm_)
                    : evolve_operator(state, graph_.slice_view(num_majoranas), mapped_params, comm_);
        if (inplace) {
            mp_op_.state_coeffs = evolved_state;
        }
        return evolved_state;
    }

    const auto &op = mp_op_.get_operator();
    const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0, true);
    const auto evolved_op = inplace ? evolve_operator(op, graph_.slice_graph(num_majoranas, true), mapped_params, comm_)
                                    : evolve_operator(op, graph_.slice_view(num_majoranas), mapped_params, comm_);
    if (inplace) {
        mp_op_.op_coeffs = evolved_op;
    }
    return evolved_op;
}

} // namespace monoprop
