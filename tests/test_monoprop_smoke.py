# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from monoprop import MajoranaPropagator, jordan_wigner_basis_change
from tests.cases import CasesFermionicProblem


def _bound_simulator_type(problem, serial_comm):
    wrapper = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        comm=serial_comm,
    )
    return type(wrapper._simulator)


def _make_bound_simulator(bound_type, problem, serial_comm, *, schrodinger: bool):
    return bound_type(
        initial_operator=problem.operator.terms,
        cutoff=2 * problem.n_modes,
        slater_determinant=problem.monomial_circuit.initial_state,
        comm=serial_comm,
        schrodinger_cutoff=2 * problem.n_modes if schrodinger else None,
        lower_atol=None,
        upper_atol=None,
        cutoff_type="length",
        basis_change=None,
    )


def _make_bound_core(problem, serial_comm, *, schrodinger: bool):
    bound_type = _bound_simulator_type(problem, serial_comm)
    return _make_bound_simulator(
        bound_type,
        problem,
        serial_comm,
        schrodinger=schrodinger,
    )


def _evolve_bound_core(core, monomial_circuit):
    parameter_mapping = monomial_circuit.param_inds.tolist()
    gen_coeffs = monomial_circuit.gen_coeffs.tolist()
    parameters = monomial_circuit.parameters.tolist()
    core.propagate_build_graph(
        monomial_circuit.majoranas,
        parameter_mapping,
        gen_coeffs,
    )
    coeffs = core.contract_partially(parameters, inplace=False)
    return parameters, coeffs


@parametrize_with_cases(
    "problem", cases=CasesFermionicProblem, has_tag="has_commutator_data"
)
@pytest.mark.parametrize(
    "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
)
def test_bound_constructor_accepts_declared_arguments(
    problem, serial_comm, schrodinger
):
    core = _make_bound_core(problem, serial_comm, schrodinger=schrodinger)

    assert core.num_modes == problem.n_modes
    assert core.schrodinger is schrodinger


@parametrize_with_cases(
    "problem", cases=CasesFermionicProblem, has_tag="has_commutator_data"
)
@pytest.mark.parametrize(
    "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
)
def test_bound_mutators_accept_declared_arguments(problem, serial_comm, schrodinger):
    core = _make_bound_core(problem, serial_comm, schrodinger=schrodinger)

    core.lower_atol = 1e-12
    core.lower_atol = None
    core.upper_atol = 1e-12
    core.upper_atol = None
    core.cutoff = 2 * problem.n_modes
    core.cutoff_type = "length"
    core.basis_change = jordan_wigner_basis_change(problem.n_modes)
    core.basis_change = None

    assert core.cutoff == 2 * problem.n_modes
    assert core.cutoff_type == "length"
    assert core.basis_change is None


@parametrize_with_cases(
    "problem", cases=CasesFermionicProblem, has_tag="has_commutator_data"
)
@pytest.mark.parametrize(
    "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
)
def test_bound_expectation_value_methods_accept_declared_arguments(
    problem, serial_comm, schrodinger
):
    monomial_circuit = problem.monomial_circuit
    core = _make_bound_core(problem, serial_comm, schrodinger=schrodinger)
    parameters, coeffs = _evolve_bound_core(core, monomial_circuit)

    expectation_value_fn = core.expectation_value_functional(pare_threshold=None)
    expectation_value = expectation_value_fn(parameters)
    assert isinstance(expectation_value, float)

    expectation_value_grad_fn = core.expectation_value_and_gradient_functional(
        pare_threshold=None
    )
    expectation_value_with_grad, gradient = expectation_value_grad_fn(parameters)
    assert isinstance(expectation_value_with_grad, float)
    assert len(gradient) == len(parameters)

    assert len(coeffs) > 0


@parametrize_with_cases(
    "problem", cases=CasesFermionicProblem, has_tag="has_commutator_data"
)
@pytest.mark.parametrize(
    "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
)
def test_bound_graph_methods_accept_declared_arguments(
    problem, serial_comm, schrodinger
):
    monomial_circuit = problem.monomial_circuit
    core = _make_bound_core(problem, serial_comm, schrodinger=schrodinger)
    parameters, _ = _evolve_bound_core(core, monomial_circuit)

    evolved_operator = core.evolved_operator_dict(parameters, 1e-12)
    assert isinstance(evolved_operator, dict)

    assert core.size() > 0
    graph_size = core.graph_size()
    assert len(graph_size) == 2
    assert core.graph_data() is not None
    assert core.graph_layers() is not None

    core.update_initial_operator(op_dict=problem.operator.terms)
