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

#include <algorithm>
#include <cctype>
#include <complex>
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

#ifdef monoprop_ENABLE_MPI
#include <mpi4py/mpi4py.h>
#endif

#include "monoprop/Info.h"
#include "monoprop/MPFunctions.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

namespace nb = nanobind;
using namespace nanobind::literals;
using namespace monoprop;

static auto get_mpi_comm(nb::object obj) -> MPI_Comm {
#ifdef monoprop_ENABLE_MPI
    if (!obj || obj.is_none()) {
        return MPI_COMM_WORLD;
    }
    auto *comm_ptr = PyMPIComm_Get(obj.ptr());

    if (!comm_ptr)
        throw nb::python_error();

    return *comm_ptr;
#else
    (void)obj;
    return MPI_COMM_WORLD; // dummy when MPI disabled
#endif
}

NB_MODULE(_core, mod) {
    mod.doc() = "Monomial Propagator";

#ifdef monoprop_ENABLE_MPI
    // initialize mpi4py's C-API
    if (import_mpi4py() < 0) {
        // mpi4py calls the Python C API
        // we let nanobind give us the detailed traceback
        throw nb::python_error();
    }
#endif

    mod.attr("__build_type__") = std::string(build_type());
    mod.attr("__compiler_flags__") = compiler_flags();
    // From the build system, since this TU is a real source and not a configured template: the version
    // has to be the one _core was compiled against, which the installed nanobind package need not be.
    mod.attr("__nanobind_version__") = std::string(monoprop_NANOBIND_VERSION);
    mod.attr("__variant__") = std::string(variant());

#ifdef monoprop_ENABLE_MPI
    mod.attr("has_mpi") = true;
#else
    mod.attr("has_mpi") = false;
#endif

    using namespace monoprop;

    mod.def("is_antihermitian",
            &is_antihermitian,
            "indices"_a,
            "Check if a Majorana operator (represented by indices) is antihermitian.");
    mod.def("antihermitian_generator_correction",
            &antihermitian_generator_correction,
            "indices"_a,
            "Get the generator correction for a Majorana operator (represented by indices).");

    auto cls = nb::class_<MonomialPropagator>(mod, "MonomialPropagator");

    cls.def(
        "__init__",
        [](MonomialPropagator *t,
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
           size_t partitions,
           std::optional<size_t> storage_num_modes) {
            new (t) MonomialPropagator(initial_operator,
                                       cutoff,
                                       initial_state,
                                       logical_num_modes,
                                       schrodinger_cutoff,
                                       get_mpi_comm(py_comm),
                                       lower_atol,
                                       upper_atol,
                                       cutoff_type_str_2_enum(cutoff_type),
                                       basis_change,
                                       basis_str_2_enum(basis),
                                       partitions,
                                       storage_num_modes);
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
        "logical_num_modes"_a,
        "basis"_a = "majorana",
        "partitions"_a = 0,
        "storage_num_modes"_a = std::nullopt,
        "Instantiate the simulator.");

    cls.def("build_graph",
            &MonomialPropagator::build_graph,
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
        [](const MonomialPropagator &self, nb::handle) { return MonomialPropagator(self); },
        "memo"_a = nb::none());

    cls.def("propagate",
            &MonomialPropagator::propagate,
            "majoranas"_a,
            "parameter_mapping"_a,
            "gen_coeffs"_a,
            "parameters"_a,
            "only_rotate_len_k"_a = std::nullopt,
            "Evolve and contract immediately without storing a graph");

    cls.def("expectation_value",
            &MonomialPropagator::expectation_value,
            "parameters"_a,
            "Expectation value at the given variational parameters");

    cls.def("expectation_value_and_gradient",
            &MonomialPropagator::expectation_value_and_gradient,
            "parameters"_a,
            "Expectation value and its gradient at the given variational parameters");

    cls.def("expectation_value_functional",
            &MonomialPropagator::expectation_value_functional,
            "pare_threshold"_a = std::nullopt,
            "Reusable callable giving the expectation value from parameters; None keeps the exact graph");

    cls.def("expectation_value_and_gradient_functional",
            &MonomialPropagator::expectation_value_and_gradient_functional,
            "pare_threshold"_a = std::nullopt,
            "Reusable callable giving (expectation value, gradient) from parameters; None keeps the exact graph");

    cls.def("contract_partially",
            &MonomialPropagator::contract_partially,
            "parameters"_a,
            "inplace"_a,
            "Contract the graph at parameters; inplace consumes the graph and updates internal state");

    cls.def("update_initial_operator",
            &MonomialPropagator::update_initial_operator,
            "op_dict"_a,
            "Rewrite the initial operator from an {indices: coefficient} dict");

    cls.def_prop_rw("lower_atol",
                    &MonomialPropagator::lower_atol,
                    &MonomialPropagator::update_lower_atol,
                    "lower_atol"_a = std::nullopt,
                    "Lower absolute tolerance of the cutoff function, or None");

    cls.def_prop_rw("upper_atol",
                    &MonomialPropagator::upper_atol,
                    &MonomialPropagator::update_upper_atol,
                    "upper_atol"_a = std::nullopt,
                    "Upper absolute tolerance of the cutoff function, or None");

    cls.def_prop_rw("cutoff",
                    &MonomialPropagator::cutoff,
                    &MonomialPropagator::update_cutoff,
                    "Cutoff value the cutoff function is built from");

    cls.def_prop_rw(
        "cutoff_type",
        [](const MonomialPropagator &self) { return cutoff_type_enum_2_str(self.cutoff_type()); },
        [](MonomialPropagator &self, const std::string &cutoff_type) {
            self.update_cutoff_type(cutoff_type_str_2_enum(cutoff_type));
        },
        "Cutoff scheme: 'length' or 'support'");

    cls.def_prop_rw("basis_change",
                    &MonomialPropagator::basis_change,
                    &MonomialPropagator::update_basis_change,
                    "basis_change"_a = std::nullopt,
                    "Optional per-Majorana basis change with 2 * num_modes entries, or None");

    cls.def_prop_ro("schrodinger", &MonomialPropagator::schrodinger, "Whether the propagator uses Schrodinger picture");

    cls.def_prop_ro(
        "basis",
        [](const MonomialPropagator &self) { return basis_enum_2_str(self.basis()); },
        "The operator basis: 'majorana' (default) or 'pauli'");

    cls.def(
        "evolved_operator",
        [](MonomialPropagator &self, const VecD &parameters, double atol) -> nb::dict {
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

    cls.def_prop_ro("num_modes", &MonomialPropagator::logical_num_modes, "Number of modes the operator actually uses");

    cls.def_prop_ro("storage_num_modes",
                    &MonomialPropagator::storage_num_modes,
                    "Mode width this propagator stores monomials at; >= num_modes");

    // Exposed for the test suite: which row backend a width selects is otherwise unobservable from
    // Python, so a run forced onto the support-form store could silently have used the dense one.
    cls.def_prop_ro("rows_are_sparse",
                    &MonomialPropagator::rows_are_sparse,
                    "Whether this propagator stores operator rows in the support form (see monoprop_ROW_STORE)");

    cls.def("size", &MonomialPropagator::size, "Number of monomial terms on this rank");

    cls.def("graph_size", &MonomialPropagator::graph_size, "(cosine-only indices, cycles) in the graph on this rank");

    cls.def("graph_layers", &MonomialPropagator::graph_layers, "Number of layers in the graph");
    cls.def("n_gates",
            &MonomialPropagator::n_gates,
            "Number of distinct gates in the graph; a multi-term gate spans several layers, so <= graph_layers()");

    cls.def_prop_rw("parameter_mapping",
                    &MonomialPropagator::parameter_mapping,
                    &MonomialPropagator::set_parameter_mapping,
                    "Variational-parameter index driving each graph layer; assigning re-wires it in place");

    cls.def(
        "operator_memory_bytes",
        [](const MonomialPropagator &self) { return self.operator_memory_usage().total_bytes(); },
        "Total bytes held by the operator on this rank");
    cls.def(
        "graph_memory_bytes",
        [](const MonomialPropagator &self) { return self.graph_memory_usage().total_bytes(); },
        "Total bytes held by the graph on this rank");

    // total_bytes() alone cannot say whether the row store or the transposed inverted index dominates.
    cls.def("operator_memory_breakdown", [](const MonomialPropagator &self) {
        const auto b = self.operator_memory_usage();
        return std::map<std::string, size_t>{{"operator_terms_bytes", b.operator_terms_bytes},
                                             {"op_coeffs_bytes", b.op_coeffs_bytes},
                                             {"state_coeffs_bytes", b.state_coeffs_bytes},
                                             {"indexing_bytes", b.indexing_bytes},
                                             {"init_operator_bytes", b.init_operator_bytes},
                                             {"initial_state_bytes", b.initial_state_bytes},
                                             {"inverted_index_bytes", b.inverted_index_bytes},
                                             {"matched_scratch_bytes", b.matched_scratch_bytes},
                                             {"total_bytes", b.total_bytes()},
                                             // Diagnostics (not part of total_bytes; see the struct).
                                             {"d_invidx_dense_bytes", b.inverted_index_dense_bytes},
                                             {"d_invidx_sparse_bytes", b.inverted_index_sparse_bytes},
                                             {"d_invidx_columns_bytes", b.inverted_index_columns_bytes},
                                             {"d_invidx_dense_columns", b.inverted_index_dense_columns},
                                             {"d_terms_slack_bytes", b.operator_terms_slack_bytes},
                                             {"d_state_coeffs_nonzero", b.state_coeffs_nonzero},
                                             {"d_init_operator_entries", b.init_operator_entries}};
    });
}
