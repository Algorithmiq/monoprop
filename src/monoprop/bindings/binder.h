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
#include <cctype>
#include <complex>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace monoprop::bindings::detail {
// mpi4py comm object -> MPI_Comm.
auto get_mpi_comm(nb::object obj) -> MPI_Comm;

auto cutoff_type_str_2_enum(const std::string &cutoff_type) -> CutoffType;
auto cutoff_type_enum_2_str(CutoffType cutoff_type) -> std::string;

auto basis_str_2_enum(const std::string &basis) -> Basis;
auto basis_enum_2_str(Basis basis) -> std::string;

template <size_t NumModes>
auto bind_monomial_propagator(nb::module_ &mod) -> void {
    using namespace monoprop;

    auto name = std::format("MonomialPropagator{:03d}", NumModes);
    auto cls = nb::class_<MonomialPropagator<NumModes>>(mod, name.c_str());

    cls.def(
        "__init__",
        [](MonomialPropagator<NumModes> *t,
           const std::map<std::vector<size_t>, std::complex<double>> &initial_operator,
           unsigned int cutoff,
           const std::vector<size_t> &initial_state,
           nb::object py_comm,
           std::optional<unsigned int> schrodinger_cutoff,
           std::optional<double> lower_atol,
           std::optional<double> upper_atol,
           const std::string &cutoff_type,
           std::optional<std::vector<std::vector<size_t>>> basis_change,
           size_t logical_num_modes,
           const std::string &basis,
           size_t partitions) {
            new (t) MonomialPropagator<NumModes>(initial_operator,
                                                 cutoff,
                                                 initial_state,
                                                 schrodinger_cutoff,
                                                 get_mpi_comm(py_comm),
                                                 lower_atol,
                                                 upper_atol,
                                                 cutoff_type_str_2_enum(cutoff_type),
                                                 basis_change,
                                                 logical_num_modes,
                                                 basis_str_2_enum(basis),
                                                 partitions);
        },
        "initial_operator"_a,
        "cutoff"_a,
        "initial_state"_a,
        "comm"_a = nb::none(),
        "schrodinger_cutoff"_a = std::nullopt,
        "lower_atol"_a = std::nullopt,
        "upper_atol"_a = std::nullopt,
        "cutoff_type"_a = "length",
        "basis_change"_a = std::nullopt,
        "logical_num_modes"_a = NumModes,
        "basis"_a = "majorana",
        "partitions"_a = 0,
        "Instantiate the simulator.");

    cls.def("build_graph",
            &MonomialPropagator<NumModes>::build_graph,
            "majoranas"_a,
            "parameter_mapping"_a,
            "gen_coeffs"_a,
            "gate_indices"_a = std::nullopt,
            "parameters"_a = std::nullopt,
            "only_rotate_len_k"_a = std::nullopt,
            "Build the propagation graph, recording per-layer gate information");

    // Deep copy: the operator store is cloned, the immutable graph layer cores are shared.
    cls.def(
        "__deepcopy__",
        [](const MonomialPropagator<NumModes> &self, nb::handle) { return MonomialPropagator<NumModes>(self); },
        "memo"_a = nb::none());

    cls.def("propagate",
            &MonomialPropagator<NumModes>::propagate,
            "majoranas"_a,
            "parameter_mapping"_a,
            "gen_coeffs"_a,
            "parameters"_a,
            "only_rotate_len_k"_a = std::nullopt,
            "Evolve and contract immediately without storing a graph");

    cls.def("expectation_value",
            &MonomialPropagator<NumModes>::expectation_value,
            "parameters"_a,
            "Expectation value at the given variational parameters");

    cls.def("expectation_value_and_gradient",
            &MonomialPropagator<NumModes>::expectation_value_and_gradient,
            "parameters"_a,
            "Expectation value and its gradient at the given variational parameters");

    cls.def("expectation_value_functional",
            &MonomialPropagator<NumModes>::expectation_value_functional,
            "pare_threshold"_a = std::nullopt,
            "Reusable callable giving the expectation value from parameters; None keeps the exact graph");

    cls.def("expectation_value_and_gradient_functional",
            &MonomialPropagator<NumModes>::expectation_value_and_gradient_functional,
            "pare_threshold"_a = std::nullopt,
            "Reusable callable giving (expectation value, gradient) from parameters; None keeps the exact graph");

    cls.def("contract_partially",
            &MonomialPropagator<NumModes>::contract_partially,
            "parameters"_a,
            "inplace"_a,
            "Contract the graph at parameters; inplace consumes the graph and updates internal state");

    cls.def("update_initial_operator",
            &MonomialPropagator<NumModes>::update_initial_operator,
            "op_dict"_a,
            "Rewrite the initial operator from an {indices: coefficient} dict");

    cls.def_prop_rw("lower_atol",
                    &MonomialPropagator<NumModes>::lower_atol,
                    &MonomialPropagator<NumModes>::update_lower_atol,
                    "lower_atol"_a = std::nullopt,
                    "Lower absolute tolerance of the cutoff function, or None");

    cls.def_prop_rw("upper_atol",
                    &MonomialPropagator<NumModes>::upper_atol,
                    &MonomialPropagator<NumModes>::update_upper_atol,
                    "upper_atol"_a = std::nullopt,
                    "Upper absolute tolerance of the cutoff function, or None");

    cls.def_prop_rw("cutoff",
                    &MonomialPropagator<NumModes>::cutoff,
                    &MonomialPropagator<NumModes>::update_cutoff,
                    "Cutoff value the cutoff function is built from");

    cls.def_prop_rw(
        "cutoff_type",
        [](const MonomialPropagator<NumModes> &self) -> std::string {
            return cutoff_type_enum_2_str(self.cutoff_type());
        },
        [](MonomialPropagator<NumModes> &self, const std::string &cutoff_type) {
            self.update_cutoff_type(cutoff_type_str_2_enum(cutoff_type));
        },
        "Cutoff scheme: 'length' or 'support'");

    cls.def_prop_rw("basis_change",
                    &MonomialPropagator<NumModes>::basis_change,
                    &MonomialPropagator<NumModes>::update_basis_change,
                    "basis_change"_a = std::nullopt,
                    "Optional per-Majorana basis change with 2 * num_modes entries, or None");

    cls.def_prop_ro("schrodinger",
                    &MonomialPropagator<NumModes>::schrodinger,
                    "Whether the propagator uses Schrodinger picture");

    cls.def_prop_ro(
        "basis",
        [](const MonomialPropagator<NumModes> &self) -> std::string { return basis_enum_2_str(self.basis()); },
        "The operator basis: 'majorana' (default) or 'pauli'");

    cls.def(
        "evolved_operator",
        [](MonomialPropagator<NumModes> &self, const VecD &parameters, double atol) -> nb::dict {
            nb::dict py_result;
            for (const auto &[indices, coeff] : self.evolved_operator_terms(parameters, atol)) {
                nb::tuple_builder key(indices.size());
                for (const auto &i : indices) {
                    key.put(i);
                }
                py_result[key.commit()] = coeff;
            }

            if (!self.schrodinger() && std::abs(self.core_term()) >= atol) {
                py_result[nb::tuple()] = std::complex{self.core_term(), 0.0};
            }

            return py_result;
        },
        "parameters"_a,
        "atol"_a,
        "The evolved operator as a {indices: coefficient} dict, keeping terms with |coeff| >= atol");

    cls.def_prop_ro("num_modes",
                    &MonomialPropagator<NumModes>::logical_num_modes,
                    "Number of modes the operator actually uses");

    cls.def_prop_ro_static(
        "storage_num_modes",
        [](nb::handle /*unused*/) { return MonomialPropagator<NumModes>::storage_num_modes; },
        "Mode width this compiled template instantiation stores");

    cls.def("size", &MonomialPropagator<NumModes>::size, "Number of monomial terms on this rank");

    cls.def("graph_size",
            &MonomialPropagator<NumModes>::graph_size,
            "(cosine-only indices, cycles) in the graph on this rank");

    cls.def("graph_layers", &MonomialPropagator<NumModes>::graph_layers, "Number of layers in the graph");
    cls.def("n_gates",
            &MonomialPropagator<NumModes>::n_gates,
            "Number of distinct gates in the graph; a multi-term gate spans several layers, so <= graph_layers()");

    cls.def_prop_rw("parameter_mapping",
                    &MonomialPropagator<NumModes>::parameter_mapping,
                    &MonomialPropagator<NumModes>::set_parameter_mapping,
                    "Variational-parameter index driving each graph layer; assigning re-wires it in place");

    cls.def(
        "operator_memory_bytes",
        [](const MonomialPropagator<NumModes> &self) { return self.operator_memory_usage().total_bytes(); },
        "Total bytes held by the operator on this rank");
    cls.def(
        "graph_memory_bytes",
        [](const MonomialPropagator<NumModes> &self) { return self.graph_memory_usage().total_bytes(); },
        "Total bytes held by the graph on this rank");

    // total_bytes() alone cannot say whether the row store or the transposed inverted index dominates.
    cls.def("operator_memory_breakdown", [](const MonomialPropagator<NumModes> &self) {
        const auto b = self.operator_memory_usage();
        return std::map<std::string, size_t>{{"operator_terms_bytes", b.operator_terms_bytes},
                                             {"op_coeffs_bytes", b.op_coeffs_bytes},
                                             {"state_coeffs_bytes", b.state_coeffs_bytes},
                                             {"indexing_bytes", b.indexing_bytes},
                                             {"init_operator_bytes", b.init_operator_bytes},
                                             {"initial_state_bytes", b.initial_state_bytes},
                                             {"inverted_index_bytes", b.inverted_index_bytes},
                                             {"total_bytes", b.total_bytes()},
                                             // Diagnostics (not part of total_bytes; see the struct).
                                             {"d_invidx_dense_bytes", b.inverted_index_dense_bytes},
                                             {"d_invidx_sparse_bytes", b.inverted_index_sparse_bytes},
                                             {"d_invidx_dense_columns", b.inverted_index_dense_columns},
                                             {"d_terms_slack_bytes", b.operator_terms_slack_bytes},
                                             {"d_state_coeffs_nonzero", b.state_coeffs_nonzero}};
    });

    // The operator partitions but the graph does not: its per-layer arrays are indexed by rank,
    // and on a partitioned run that index space is the FLAT world (ranks x partitions). Splitting
    // the total is what separates memory that grows with the problem from memory that grows with
    // the machine. d_occupied_slots / d_slot_records is the occupancy that says whether a sparse
    // layout would pay; d_slot_records / d_layer_cores recovers the world size P.
    cls.def("graph_memory_breakdown", [](const MonomialPropagator<NumModes> &self) {
        const auto b = self.graph_memory_usage();
        return std::map<std::string, size_t>{{"layer_descriptor_bytes", b.layer_descriptor_bytes},
                                             {"layer_storage_object_bytes", b.layer_storage_object_bytes},
                                             {"cos_data_bytes", b.cos_data_bytes},
                                             {"cross_rank_bytes", b.cross_rank_bytes},
                                             {"exchange_layout_bytes", b.exchange_layout_bytes},
                                             {"total_bytes", b.total_bytes()},
                                             // Diagnostics (not part of total_bytes; see the struct).
                                             {"d_slot_record_bytes", b.slot_record_bytes},
                                             {"d_recv_cache_bytes", b.recv_cache_bytes},
                                             {"d_derivative_layout_bytes", b.derivative_layout_bytes},
                                             {"d_layer_cores", b.layer_cores},
                                             {"d_slot_records", b.slot_records},
                                             {"d_occupied_slots", b.occupied_slots}};
    });
}
} // namespace monoprop::bindings::detail
