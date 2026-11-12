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

"""Tests for ``term_expectation_values``.

The values are checked against the expectation value they reassemble, the expectation value
with each term alone, and the Schrodinger evolved state, which holds the same values when
untruncated.

Values are allreduced, so the tests run on ``comm``; the Schrodinger read compares rank-local
evolved-state coefficients and stays serial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monoprop import Circuit, ExpGate, MajoranaPropagator, PauliPropagator
from monoprop.pauli import Pauli, PauliOperator
from tests.reference_data import load_problem

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


def _heisenberg(cutoff: int, comm, coefficients=None) -> PauliPropagator:
    """A Heisenberg propagator over ``OBSERVABLES`` with the circuit stored as a graph."""
    coefficients = (
        np.linspace(0.3, 1.2, len(OBSERVABLES))
        if coefficients is None
        else coefficients
    )
    prop = PauliPropagator(
        PauliOperator(dict(zip(OBSERVABLES, coefficients)), N_QUBITS),
        initial_state=[0, 3],
        cutoff=cutoff,
        comm=comm,
    )
    prop.build_graph(_circuit())
    return prop


@pytest.fixture(scope="module")
def parameters() -> np.ndarray:
    """Generic angles, so no gate is trivially the identity."""
    return np.random.default_rng(7).uniform(0, 2 * np.pi, _circuit().n_parameters)


@pytest.mark.parametrize("cutoff", [2, 3, N_QUBITS])
def test_pauli_values_reassemble_the_expectation_value(
    cutoff, parameters, comm
) -> None:
    """Initial coefficients times values sum to the expectation value, truncated or not."""
    prop = _heisenberg(cutoff, comm)
    values = prop.term_expectation_values(OBSERVABLES, parameters)

    assert values.dtype == np.float64
    assert values.shape == (len(OBSERVABLES),)
    coefficients = np.linspace(0.3, 1.2, len(OBSERVABLES))
    assert coefficients @ values == pytest.approx(
        prop.expectation_value(parameters), abs=1e-12
    )


@pytest.mark.parametrize("cutoff", [2, N_QUBITS])
def test_each_value_matches_a_single_term_reweight(cutoff, parameters, comm) -> None:
    """Each value matches the expectation value once the initial operator is that term alone."""
    prop = _heisenberg(cutoff, comm)
    values = prop.term_expectation_values(OBSERVABLES, parameters)

    for i in range(len(OBSERVABLES)):
        prop.update_initial_operator(PauliOperator({OBSERVABLES[i]: 1.0}, N_QUBITS))
        assert prop.expectation_value(parameters) == pytest.approx(values[i], abs=1e-12)


def test_values_match_a_schrodinger_read(parameters) -> None:
    """Untruncated, Heisenberg values equal the Schrodinger evolved-state coefficients."""
    heisenberg = _heisenberg(N_QUBITS, None).term_expectation_values(
        OBSERVABLES, parameters
    )

    schrodinger = PauliPropagator(
        PauliOperator({OBSERVABLES[0]: 1.0}, N_QUBITS),
        initial_state=[0, 3],
        cutoff=N_QUBITS,
        schrodinger_cutoff=N_QUBITS,
    )
    schrodinger.build_graph(_circuit())
    state = schrodinger.evolved_operator_coefficients(OBSERVABLES, parameters)
    np.testing.assert_allclose(heisenberg, state.real, atol=1e-11)
    # Schrodinger's own values are its evolved-state coefficients (real for Pauli strings).
    np.testing.assert_allclose(
        schrodinger.term_expectation_values(OBSERVABLES, parameters),
        state.real,
        atol=1e-11,
    )


def test_terms_follow_query_order(parameters, comm) -> None:
    """``terms`` reorders and repeats exactly as given."""
    prop = _heisenberg(3, comm)
    values = prop.term_expectation_values(OBSERVABLES, parameters)
    order = [3, 0, 3, 4]
    queried = prop.term_expectation_values([OBSERVABLES[i] for i in order], parameters)
    np.testing.assert_array_equal(queried, values[order])


def test_identity_returns_one(parameters, comm) -> None:
    """The identity returns 1, and its coefficient still counts when reassembling the expectation value."""
    prop = PauliPropagator(
        PauliOperator({Pauli("", ()): 0.5, OBSERVABLES[0]: 1.0}, N_QUBITS),
        initial_state=[],
        cutoff=3,
        comm=comm,
    )
    prop.build_graph(_circuit())
    values = prop.term_expectation_values([Pauli("", ()), OBSERVABLES[0]], parameters)
    assert values[0] == 1.0
    assert np.array([0.5, 1.0]) @ values == pytest.approx(
        prop.expectation_value(parameters), abs=1e-12
    )


def test_heisenberg_absent_term_raises(parameters, comm) -> None:
    """Heisenberg raises for a term absent from the evolved operator."""
    prop = _heisenberg(2, comm)
    terms = [Pauli("XXXXX", (0, 1, 2, 3, 4))]
    with pytest.raises(RuntimeError, match="not found"):
        prop.term_expectation_values(terms, parameters)


def test_heisenberg_rejects_absorbed_gates(parameters, comm) -> None:
    """Heisenberg raises after an in-place contract_partially or propagate.

    The values would miss the absorbed gates.
    """
    contracted = _heisenberg(2, comm)
    contracted.contract_partially(parameters, inplace=True)
    with pytest.raises(RuntimeError, match="absorbed gates"):
        contracted.term_expectation_values(OBSERVABLES)

    propagated = PauliPropagator(
        PauliOperator(dict.fromkeys(OBSERVABLES, 1.0), N_QUBITS),
        initial_state=[0, 3],
        cutoff=2,
        comm=comm,
    )
    circuit = _circuit()
    propagated.propagate(Circuit(circuit.gates, N_QUBITS, parameters=tuple(parameters)))
    with pytest.raises(RuntimeError, match="absorbed gates"):
        propagated.term_expectation_values(OBSERVABLES)


def test_schrodinger_absent_term_returns_zero(parameters, comm) -> None:
    """Schrodinger returns 0 for a term past the state's cutoff, which the truncated state lacks."""
    prop = PauliPropagator(
        PauliOperator({OBSERVABLES[0]: 1.0}, N_QUBITS),
        initial_state=[],
        cutoff=2,
        schrodinger_cutoff=2,
        comm=comm,
    )
    prop.build_graph(_circuit())
    values = prop.term_expectation_values([Pauli("XXXXX", (0, 1, 2, 3, 4))], parameters)
    np.testing.assert_array_equal(values, [0.0])


def test_majorana_values_reassemble_the_expectation_value(comm) -> None:
    """Majorana values are complex, and coefficients times values sum to the expectation value."""
    problem = load_problem(DATA / "random_exact.msgpack")
    prop = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=problem.n_modes // 2,
        comm=comm,
    )
    prop.build_graph(problem.monomial_circuit.to_circuit())
    parameters = problem.monomial_circuit.parameters

    values = prop.term_expectation_values(list(problem.operator.terms), parameters)
    assert values.dtype == complex
    coefficients = np.array(list(problem.operator.terms.values()))
    total = coefficients @ values
    assert total.real == pytest.approx(prop.expectation_value(parameters), abs=1e-10)
    assert total.imag == pytest.approx(0.0, abs=1e-10)
