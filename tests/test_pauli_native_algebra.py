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

"""Dense-matrix ground truth for the native Pauli engine (<=4 qubits).

Builds explicit Pauli matrices via Kronecker products of {I, X, Y, Z} and checks the native
``PauliPropagator`` (``engine_basis="pauli"``) against the exact Heisenberg evolution
``U^dag O U`` for a single Pauli-rotation gate. This pins the gamma-slot encoding and the emit
phase end-to-end from Python.

Convention note: the engine implements the gate ``U = exp(+i * theta * g * P_gen)`` where
``theta`` is the parameter value and ``g`` the generator coefficient (the pinned convention the
C++ T7 dense-matrix test validates: ``U = cos(g*theta) I + i sin(g*theta) G``). The dense
reference here therefore uses ``expm(+1j * theta * g * P_gen)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from monoprop import Circuit, Exp, Pauli, PauliOperator, PauliPropagator

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_SINGLE = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

_LETTERS = "IXYZ"


def _dense(string: str) -> np.ndarray:
    """Dense matrix of a full-width Pauli string (qubit 0 = most-significant factor)."""
    matrix = _SINGLE[string[0]]
    for letter in string[1:]:
        matrix = np.kron(matrix, _SINGLE[letter])
    return matrix


def _dense_from_operator(operator: PauliOperator, num_qubits: int) -> np.ndarray:
    """Dense matrix of a PauliOperator on ``num_qubits`` qubits."""
    dim = 1 << num_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for pauli, coeff in operator.terms.items():
        full = ["I"] * num_qubits
        for qubit, letter in zip(pauli.qubits, pauli.string):
            full[qubit] = letter
        matrix += coeff * _dense("".join(full))
    return matrix


def _reference_evolved(
    observable: PauliOperator, gen_string: str, g: float, theta: float, num_qubits: int
) -> np.ndarray:
    """Exact Heisenberg-evolved observable ``U^dag O U`` for ``U = exp(+i theta g P)``."""
    obs_dense = _dense_from_operator(observable, num_qubits)
    gen_dense = _dense(gen_string)
    unitary = expm(1j * theta * g * gen_dense)
    return unitary.conj().T @ obs_dense @ unitary


def _evolve_native(
    observable: PauliOperator,
    gen_string: str,
    g: float,
    theta: float,
    num_qubits: int,
    serial_comm,
) -> PauliPropagator:
    """Propagate one native Pauli gate ``exp(+i theta g P)`` in the Heisenberg picture."""
    propagator = PauliPropagator(
        observable,
        [],
        cutoff=num_qubits,
        comm=serial_comm,
        engine_basis="pauli",
    )
    generator = PauliOperator(
        {Pauli(gen_string, tuple(range(num_qubits))): g}, num_qubits=num_qubits
    )
    circuit = Circuit(gates=(Exp(generator),), parameters=(theta,), initial_state=[])
    propagator.propagate(circuit)
    return propagator


# -- all 16 two-qubit letter pairs (skipping the identity generator) ------------------------


@pytest.mark.parametrize(
    "gen_string",
    [a + b for a in _LETTERS for b in _LETTERS if a + b != "II"],
)
@pytest.mark.parametrize("theta", [0.37, -0.8, 1.3])
def test_two_qubit_all_letter_pairs(serial_comm, gen_string, theta):
    """Every non-identity two-qubit generator against a Y-heavy observable."""
    num_qubits = 2
    observable = PauliOperator(
        {"XY": 0.5, "ZZ": -0.3, "YX": 0.7, "IZ": 0.2, "YY": -0.15}, num_qubits=num_qubits
    )
    g = 0.9
    propagator = _evolve_native(
        observable, gen_string, g, theta, num_qubits, serial_comm
    )
    got = _dense_from_operator(propagator.evolved_pauli_operator(), num_qubits)
    want = _reference_evolved(observable, gen_string, g, theta, num_qubits)
    assert np.allclose(got, want, atol=1e-10), gen_string


# -- Y-heavy single-qubit generators (phase-sensitive: distinguishes +i from -i) ------------


@pytest.mark.parametrize(
    ("gen_string", "observable_string"),
    [
        ("X", "Y"),
        ("Y", "Z"),
        ("Z", "X"),
        ("Y", "X"),
    ],
)
@pytest.mark.parametrize("theta", [0.6, -1.1])
def test_single_qubit_phase_sensitive(
    serial_comm, gen_string, observable_string, theta
):
    """A [gen, obs] pair whose rotation direction distinguishes exp(+i..) from exp(-i..)."""
    num_qubits = 1
    observable = PauliOperator({observable_string: 1.0}, num_qubits=num_qubits)
    g = 1.0
    propagator = _evolve_native(
        observable, gen_string, g, theta, num_qubits, serial_comm
    )
    got = _dense_from_operator(propagator.evolved_pauli_operator(), num_qubits)
    want = _reference_evolved(observable, gen_string, g, theta, num_qubits)
    assert np.allclose(got, want, atol=1e-10)


# -- expectation value against a computational-basis reference ------------------------------


@pytest.mark.parametrize("initial_state", [[], [0], [1], [0, 2]])
def test_expectation_value_matches_dense(serial_comm, initial_state):
    """<b|U^dag O U|b> from the engine matches the dense computation, Y-heavy observable."""
    num_qubits = 3
    observable = PauliOperator(
        {"XYZ": 0.4, "YYY": -0.6, "ZIZ": 0.25, "IYX": 0.5}, num_qubits=num_qubits
    )
    gen_string = "YIY"
    g, theta = 0.8, 0.55
    propagator = PauliPropagator(
        observable,
        initial_state,
        cutoff=num_qubits,
        comm=serial_comm,
        engine_basis="pauli",
    )
    generator = PauliOperator(
        {Pauli(gen_string, (0, 1, 2)): g}, num_qubits=num_qubits
    )
    circuit = Circuit(
        gates=(Exp(generator),), parameters=(theta,), initial_state=initial_state
    )
    propagator.build_graph(circuit)
    got = propagator.expectation_value(circuit.parameters)

    evolved = _reference_evolved(observable, gen_string, g, theta, num_qubits)
    # Basis index of |b>: qubit 0 is the most-significant tensor factor.
    index = sum(1 << (num_qubits - 1 - q) for q in initial_state)
    want = evolved[index, index].real
    assert np.isclose(got, want, atol=1e-10)
