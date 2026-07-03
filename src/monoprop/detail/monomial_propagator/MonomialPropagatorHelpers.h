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
auto MonomialPropagator<NumModes>::append_to_graph(MPGraph &graph,
                                                   VecZ &cos_inds,
                                                   std::optional<CompressedCosineData> &compressed_cos_data,
                                                   SplitCycleResult &split,
                                                   MPI_Comm comm,
                                                   size_t param_index,
                                                   double gen_coeff,
                                                   size_t gate_index) -> void {
    if (mpi::size(comm) > 1) {
        compressed_cos_data = detail::remove_incoming_cycle_targets_compressed(cos_inds, split);
        cos_inds.clear();
    }
    if (compressed_cos_data.has_value()) {
        graph.append(std::move(*compressed_cos_data),
                     std::move(split.local_cycles),
                     std::move(split.cross_rank),
                     param_index,
                     gen_coeff,
                     gate_index);
    }
    else {
        graph.append(std::move(cos_inds),
                     std::move(split.local_cycles),
                     std::move(split.cross_rank),
                     param_index,
                     gen_coeff,
                     gate_index);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expected_num_params(const VecZ &parameter_mapping) -> size_t {
    return parameter_mapping.empty() ? 0 : *std::max_element(parameter_mapping.begin(), parameter_mapping.end()) + 1;
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_parameter_validated_functional(size_t expected_num_params, Fn func)
    -> std::function<R(const VecD &)> {
    return [expected_num_params, func = std::move(func)](const VecD &params) -> R {
        validate_functional_call(params, expected_num_params);
        return func(params);
    };
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::format_bytes_(size_t bytes) -> std::string {
    static constexpr std::array<std::string_view, 5> units = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    size_t unit_idx = 0;
    while (value >= 1024.0 && unit_idx + 1 < units.size()) {
        value /= 1024.0;
        ++unit_idx;
    }
    return std::format("{:.3f} {}", value, units[unit_idx]);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::print_memory_row_(std::string_view name, size_t local_bytes) const -> void {
    const auto total_bytes = mpi::allreduce_sum(local_bytes, comm_);
    if (mpi::rank(comm_) != 0) {
        return;
    }
    if (name != "total" && total_bytes == 0) {
        return;
    }

    std::print("  {:<20} {}\n", name, format_bytes_(total_bytes));
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::print_object_memory_report_(std::string_view label) const -> void {
    if (mpi::rank(comm_) == 0) {
        std::print("Object memory report: {}\n", label);
        std::print("--------------------------------\n");
    }

    const auto graph_breakdown = graph_memory_usage();
    const auto operator_breakdown = operator_memory_usage();

    if (mpi::rank(comm_) == 0) {
        std::print("Graph\n");
    }
    print_memory_row_("total", graph_breakdown.total_bytes());
    print_memory_row_("layer descriptors", graph_breakdown.layer_descriptor_bytes);
    print_memory_row_("layer storage", graph_breakdown.layer_storage_object_bytes);
    print_memory_row_("cos data", graph_breakdown.cos_data_bytes);
    print_memory_row_("local cycles", graph_breakdown.local_cycle_bytes);
    print_memory_row_("cross rank", graph_breakdown.cross_rank_bytes);
    print_memory_row_("exchange layouts", graph_breakdown.exchange_layout_bytes);
    print_memory_row_("execution plan", graph_breakdown.execution_plan_bytes);

    if (mpi::rank(comm_) == 0) {
        std::print("\n");
    }

    if (mpi::rank(comm_) == 0) {
        std::print("Operator\n");
    }
    print_memory_row_("total", operator_breakdown.total_bytes());
    print_memory_row_("terms", operator_breakdown.operator_terms_bytes);
    print_memory_row_("op coeffs", operator_breakdown.op_coeffs_bytes);
    print_memory_row_("state coeffs", operator_breakdown.state_coeffs_bytes);
    print_memory_row_("indexing", operator_breakdown.indexing_bytes);
    print_memory_row_("initial operator", operator_breakdown.init_operator_bytes);
    print_memory_row_("slater determinant", operator_breakdown.slater_determinant_bytes);

    if (mpi::rank(comm_) == 0) {
        std::print("--------------------------------\n");
    }
}

} // namespace monoprop
