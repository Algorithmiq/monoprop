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

"""Coverage for ``term_expectation_values`` and ``update_initial_coefficients``.

Oracles: the expectation value the values must reassemble, a re-weight to a single term, and
the Schrodinger evolved state, which is the adjoint the Heisenberg replay computes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monoprop import Circuit, ExpGate, MajoranaPropagator, PauliPropagator
from monoprop.pauli import Pauli, PauliOperator
from tests.cases import load_problem

DATA = Path(__file__).parent / "data"

N_QUBITS = 6
OBSERVABLES = [
    Pauli("ZZ", (0, 1)),
    Pauli("XY", (2, 3)),
    Pauli("Z", (4,)),
    Pauli("YX", (1, 5)),
    Pauli("ZZ", (2, 4)),
]


def _circuit() -> Circuit:
    """Single-qubit and nearest-neighbour rotations, enough to spread every observable."""
    gates = []
    for layer in range(3):
        axis = "XYZ"[layer % 3]
        gates += [
            ExpGate(PauliOperator({Pauli(axis, (q,)): 1.0}, N_QUBITS))
            for q in range(N_QUBITS)
        ]
        gates += [
            ExpGate(PauliOperator({Pauli("ZZ", (q, q + 1)): 1.0}, N_QUBITS))
            for q in range(layer % 2, N_QUBITS - 1, 2)
        ]
    return Circuit(tuple(gates), N_QUBITS)


def _heisenberg(cutoff: int, serial_comm, coefficients=None) -> PauliPropagator:
    coefficients = (
        np.linspace(0.3, 1.2, len(OBSERVABLES))
        if coefficients is None
        else coefficients
    )
    prop = PauliPropagator(
        PauliOperator(dict(zip(OBSERVABLES, coefficients)), N_QUBITS),
        initial_state=[0, 3],
        cutoff=cutoff,
        comm=serial_comm,
    )
    prop.build_graph(_circuit())
    return prop


@pytest.fixture(scope="module")
def parameters() -> np.ndarray:
    return np.random.default_rng(7).uniform(0, 2 * np.pi, _circuit().n_parameters)


@pytest.mark.parametrize("cutoff", [2, 3, N_QUBITS])
def test_pauli_values_reassemble_the_expectation_value(
    cutoff, parameters, serial_comm
) -> None:
    """Initial coefficients times values sum to the expectation value, truncated or not."""
    prop = _heisenberg(cutoff, serial_comm)
    values = prop.term_expectation_values(parameters)

    assert values.dtype == np.float64
    assert values.shape == (len(OBSERVABLES),)
    coefficients = np.linspace(0.3, 1.2, len(OBSERVABLES))
    assert coefficients @ values == pytest.approx(
        prop.expectation_value(parameters), abs=1e-12
    )


@pytest.mark.parametrize("cutoff", [2, N_QUBITS])
def test_each_value_matches_a_single_term_reweight(
    cutoff, parameters, serial_comm
) -> None:
    """A value is the expectation value once the initial operator is that term alone."""
    prop = _heisenberg(cutoff, serial_comm)
    values = prop.term_expectation_values(parameters)

    for i in range(len(OBSERVABLES)):
        prop.update_initial_coefficients(np.eye(len(OBSERVABLES))[i])
        assert prop.expectation_value(parameters) == pytest.approx(values[i], abs=1e-12)


def test_values_match_a_schrodinger_read(parameters, serial_comm) -> None:
    """Untruncated, Heisenberg values equal the Schrodinger evolved-state coefficients."""
    heisenberg = _heisenberg(N_QUBITS, serial_comm).term_expectation_values(parameters)

    schrodinger = PauliPropagator(
        PauliOperator({OBSERVABLES[0]: 1.0}, N_QUBITS),
        initial_state=[0, 3],
        cutoff=N_QUBITS,
        schrodinger_cutoff=N_QUBITS,
        comm=serial_comm,
    )
    schrodinger.build_graph(_circuit())
    state = schrodinger.evolved_operator_coefficients(OBSERVABLES, parameters)
    np.testing.assert_allclose(heisenberg, state.real, atol=1e-11)
    # The Schrodinger picture reads its own state's adjoint, which is the same vector.
    np.testing.assert_allclose(
        schrodinger.term_expectation_values(parameters, terms=OBSERVABLES),
        state.real,
        atol=1e-11,
    )


def test_explicit_terms_follow_query_order(parameters, serial_comm) -> None:
    """``terms`` reorders and repeats exactly as given; the default is construction order."""
    prop = _heisenberg(3, serial_comm)
    default = prop.term_expectation_values(parameters)
    order = [3, 0, 3, 4]
    queried = prop.term_expectation_values(
        parameters, terms=[OBSERVABLES[i] for i in order]
    )
    np.testing.assert_array_equal(queried, default[order])


def test_identity_reads_one(parameters, serial_comm) -> None:
    """The identity is the core term, whose derivative is 1 in either picture."""
    prop = PauliPropagator(
        PauliOperator({Pauli("", ()): 0.5, OBSERVABLES[0]: 1.0}, N_QUBITS),
        initial_state=[],
        cutoff=3,
        comm=serial_comm,
    )
    prop.build_graph(_circuit())
    values = prop.term_expectation_values(parameters)
    assert 1.0 in values.tolist()
    assert np.array([0.5, 1.0]) @ values == pytest.approx(
        prop.expectation_value(parameters), abs=1e-12
    )


def test_heisenberg_absent_term_raises(parameters, serial_comm) -> None:
    """Heisenberg has no value for a term outside its operator."""
    prop = _heisenberg(2, serial_comm)
    with pytest.raises(RuntimeError, match="not found"):
        prop.term_expectation_values(
            parameters, terms=[Pauli("XXXXX", (0, 1, 2, 3, 4))]
        )


def test_schrodinger_absent_term_reads_zero(parameters, serial_comm) -> None:
    """The truncated Schrodinger state carries no weight past its cutoff."""
    prop = PauliPropagator(
        PauliOperator({OBSERVABLES[0]: 1.0}, N_QUBITS),
        initial_state=[],
        cutoff=2,
        schrodinger_cutoff=2,
        comm=serial_comm,
    )
    prop.build_graph(_circuit())
    values = prop.term_expectation_values(
        parameters, terms=[Pauli("XXXXX", (0, 1, 2, 3, 4))]
    )
    np.testing.assert_array_equal(values, [0.0])


def test_majorana_values_reassemble_the_expectation_value(serial_comm) -> None:
    """Complex Majorana values decompose the expectation value over the decoded coefficients."""
    problem = load_problem(DATA / "random_exact.msgpack")
    prop = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=problem.n_modes // 2,
        comm=serial_comm,
    )
    prop.build_graph(problem.monomial_circuit.to_circuit())
    parameters = problem.monomial_circuit.parameters

    values = prop.term_expectation_values(parameters)
    assert values.dtype == complex
    coefficients = np.array(list(problem.operator.terms.values()))
    total = coefficients @ values
    assert total.real == pytest.approx(prop.expectation_value(parameters), abs=1e-10)
    assert total.imag == pytest.approx(0.0, abs=1e-10)


def test_update_initial_coefficients_matches_update_initial_operator(
    parameters, serial_comm
) -> None:
    """The array re-weight is the operator re-weight, bit for bit."""
    weights = np.random.default_rng(3).normal(size=len(OBSERVABLES))
    by_operator = _heisenberg(3, serial_comm)
    by_operator.update_initial_operator(
        PauliOperator(dict(zip(OBSERVABLES, weights)), N_QUBITS)
    )
    by_array = _heisenberg(3, serial_comm)
    by_array.update_initial_coefficients(weights)

    value_operator, grad_operator = by_operator.expectation_value_and_gradient(
        parameters
    )
    value_array, grad_array = by_array.expectation_value_and_gradient(parameters)
    assert value_array == value_operator
    np.testing.assert_array_equal(grad_array, grad_operator)


def test_update_initial_coefficients_rejects_a_wrong_length(serial_comm) -> None:
    prop = _heisenberg(3, serial_comm)
    with pytest.raises(ValueError, match="one per initial-operator term"):
        prop.update_initial_coefficients(np.ones(len(OBSERVABLES) + 1))
