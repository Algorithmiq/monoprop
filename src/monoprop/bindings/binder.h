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
/*! Return a MPI communicator from mpi4py communicator object. */
auto get_mpi_comm(nb::object obj) -> MPI_Comm;

auto cutoff_type_str_2_enum(const std::string &cutoff_type) -> CutoffType;
auto cutoff_type_enum_2_str(CutoffType cutoff_type) -> std::string;

auto basis_str_2_enum(const std::string &basis) -> Basis;
auto basis_enum_2_str(Basis basis) -> std::string;

/// @brief Binds the MonomialPropagator class to Python.
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
           const std::vector<size_t> &slater_determinant,
           nb::object py_comm,
           std::optional<unsigned int> schrodinger_cutoff,
           std::optional<double> lower_atol,
           std::optional<double> upper_atol,
           const std::string &cutoff_type,
           std::optional<std::vector<std::vector<size_t>>> basis_change,
           size_t logical_num_modes,
           const std::string &basis,
           size_t shards) {
            new (t) MonomialPropagator<NumModes>(initial_operator,
                                                 cutoff,
                                                 slater_determinant,
                                                 schrodinger_cutoff,
                                                 get_mpi_comm(py_comm),
                                                 lower_atol,
                                                 upper_atol,
                                                 cutoff_type_str_2_enum(cutoff_type),
                                                 basis_change,
                                                 logical_num_modes,
                                                 basis_str_2_enum(basis),
                                                 shards);
        },
        "initial_operator"_a,
        "cutoff"_a,
        "slater_determinant"_a,
        "comm"_a = nb::none(),
        "schrodinger_cutoff"_a = std::nullopt,
        "lower_atol"_a = std::nullopt,
        "upper_atol"_a = std::nullopt,
        "cutoff_type"_a = "length",
        "basis_change"_a = std::nullopt,
        "logical_num_modes"_a = NumModes,
        "basis"_a = "majorana",
        "shards"_a = 0,
        "Instantiate the simulator.");

    cls.def("build_graph",
            &MonomialPropagator<NumModes>::build_graph,
            "majoranas"_a,
            "parameter_mapping"_a,
            "gen_coeffs"_a,
            "gate_indices"_a = std::nullopt,
            "parameters"_a = std::nullopt,
            "only_rotate_len_k"_a = 0,
            "Build the propagation graph, recording per-layer gate information");

    // Deep copy (operator store deep-cloned; immutable graph layer cores shared). Only __deepcopy__ is exposed.
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
            "only_rotate_len_k"_a = 0,
            "Evolve and contract immediately without storing a graph");

    cls.def("expectation_value", &MonomialPropagator<NumModes>::expectation_value, "parameters"_a);

    cls.def("expectation_value_and_gradient",
            &MonomialPropagator<NumModes>::expectation_value_and_gradient,
            "parameters"_a);

    cls.def("expectation_value_functional",
            &MonomialPropagator<NumModes>::expectation_value_functional,
            "pare_threshold"_a = std::nullopt);

    cls.def("expectation_value_and_gradient_functional",
            &MonomialPropagator<NumModes>::expectation_value_and_gradient_functional,
            "pare_threshold"_a = std::nullopt);

    cls.def("contract_partially", &MonomialPropagator<NumModes>::contract_partially, "parameters"_a, "inplace"_a);

    cls.def("update_initial_operator", &MonomialPropagator<NumModes>::update_initial_operator, "op_dict"_a);

    cls.def_prop_rw("lower_atol",
                    &MonomialPropagator<NumModes>::lower_atol,
                    &MonomialPropagator<NumModes>::update_lower_atol,
                    "lower_atol"_a = std::nullopt);

    cls.def_prop_rw("upper_atol",
                    &MonomialPropagator<NumModes>::upper_atol,
                    &MonomialPropagator<NumModes>::update_upper_atol,
                    "upper_atol"_a = std::nullopt);

    cls.def_prop_rw("cutoff", &MonomialPropagator<NumModes>::cutoff, &MonomialPropagator<NumModes>::update_cutoff);

    cls.def_prop_rw(
        "cutoff_type",
        [](const MonomialPropagator<NumModes> &self) -> std::string {
            return cutoff_type_enum_2_str(self.cutoff_type());
        },
        [](MonomialPropagator<NumModes> &self, const std::string &cutoff_type) {
            self.update_cutoff_type(cutoff_type_str_2_enum(cutoff_type));
        });

    cls.def_prop_rw("basis_change",
                    &MonomialPropagator<NumModes>::basis_change,
                    &MonomialPropagator<NumModes>::update_basis_change,
                    "basis_change"_a = std::nullopt);

    cls.def_prop_ro("schrodinger",
                    &MonomialPropagator<NumModes>::schrodinger,
                    "Whether the propagator uses Schrodinger picture");

    cls.def_prop_ro(
        "basis",
        [](const MonomialPropagator<NumModes> &self) -> std::string { return basis_enum_2_str(self.basis()); },
        "The operator basis: 'majorana' (default) or 'pauli'");

    // Contract the graph, then decode every above-atol term into a Python {indices: coeff} dict.
    // evolved_operator_terms is shard-transparent (merges each shard's disjoint hash partition).
    cls.def(
        "evolved_operator",
        [](MonomialPropagator<NumModes> &self, const VecD &parameters, double atol) -> nb::dict {
            nb::dict py_result;
            for (const auto &[indices, coeff] : self.evolved_operator_terms(parameters, atol)) {
                nb::list key;
                for (const auto &i : indices) {
                    key.append(i);
                }
                py_result[nb::tuple(key)] = coeff;
            }

            if (!self.schrodinger() && std::abs(self.core_term()) >= atol) {
                // Add the core term if in Heisenberg picture
                py_result[nb::tuple()] = std::complex{self.core_term(), 0.0};
            }

            return py_result;
        },
        "parameters"_a,
        "atol"_a);

    cls.def_prop_ro("num_modes", &MonomialPropagator<NumModes>::logical_num_modes);

    cls.def_prop_ro_static("storage_num_modes",
                           [](nb::handle /*unused*/) { return MonomialPropagator<NumModes>::storage_num_modes; });

    cls.def("size", &MonomialPropagator<NumModes>::size);

    cls.def("graph_size", &MonomialPropagator<NumModes>::graph_size);

    cls.def("graph_layers", &MonomialPropagator<NumModes>::graph_layers);
    cls.def("n_gates", &MonomialPropagator<NumModes>::n_gates);

    cls.def_prop_rw("parameter_mapping",
                    &MonomialPropagator<NumModes>::parameter_mapping,
                    &MonomialPropagator<NumModes>::set_parameter_mapping);

    cls.def("operator_memory_bytes",
            [](const MonomialPropagator<NumModes> &self) { return self.operator_memory_usage().total_bytes(); });
    cls.def("graph_memory_bytes",
            [](const MonomialPropagator<NumModes> &self) { return self.graph_memory_usage().total_bytes(); });

    // Per-field operator memory split (shard-aggregated). total_bytes() alone cannot say whether the
    // row store or the transposed inverted index dominates, which is what sizing decisions turn on.
    cls.def("operator_memory_breakdown", [](const MonomialPropagator<NumModes> &self) {
        const auto b = self.operator_memory_usage();
        return std::map<std::string, size_t>{{"operator_terms_bytes", b.operator_terms_bytes},
                                             {"op_coeffs_bytes", b.op_coeffs_bytes},
                                             {"state_coeffs_bytes", b.state_coeffs_bytes},
                                             {"indexing_bytes", b.indexing_bytes},
                                             {"init_operator_bytes", b.init_operator_bytes},
                                             {"slater_determinant_bytes", b.slater_determinant_bytes},
                                             {"inverted_index_bytes", b.inverted_index_bytes},
                                             {"total_bytes", b.total_bytes()},
                                             // Diagnostics (NOT part of total_bytes; see the struct).
                                             {"d_invidx_dense_bytes", b.inverted_index_dense_bytes},
                                             {"d_invidx_sparse_bytes", b.inverted_index_sparse_bytes},
                                             {"d_invidx_dense_columns", b.inverted_index_dense_columns},
                                             {"d_invidx_delta_bytes", b.inverted_index_delta_bytes},
                                             {"d_invidx_oracle_bytes", b.inverted_index_oracle_bytes},
                                             {"d_invidx_delta_wins", b.inverted_index_delta_wins},
                                             {"d_terms_slack_bytes", b.operator_terms_slack_bytes},
                                             {"d_state_coeffs_nonzero", b.state_coeffs_nonzero}};
    });
}
} // namespace monoprop::bindings::detail
