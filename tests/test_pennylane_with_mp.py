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

import json

import numpy as np
import pytest

from monoprop import PauliPropagator
from monoprop.pauli import PauliOperator

try:
    import pennylane as qml

    from monoprop.pennylane_conversion import (
        from_pennylane_circuit,
        from_pennylane_operator,
        to_pennylane_operator,
    )

    _pennylane_available = True
except ImportError:
    _pennylane_available = False

requires_pennylane = pytest.mark.skipif(
    not _pennylane_available, reason="pennylane not installed"
)

_NUM_QUBITS = 12
_OCCUPIED_QUBITS = list(range(1, _NUM_QUBITS, 2))
_ANGLES = (0.2, 0.1, 0.3, 0.1, 0.3, 0.1)


def _pennylane_evolution(t1, t2, t3, t4, t5, t6):
    """A 12-qubit evolution circuit using only single-Pauli-generator gates.

    Unlike the qiskit test's ``simple_ev_circuit``, every window here is a single Pauli word --
    a multi-term ``PauliEvolutionGate``-style window has no PennyLane equivalent
    ``from_pennylane_circuit`` accepts (see its docstring).
    """
    qml.PauliRot(t1, "XZ", wires=[0, 1])
    qml.IsingZZ(t2, wires=[2, 3])
    qml.RY(t3, wires=4)
    qml.RZ(t4, wires=6)
    qml.PauliRot(t5, "YI", wires=[8, 9])
    qml.IsingZZ(t6, wires=[10, 11])


@pytest.fixture
def hamiltonian_lih(lazy_shared_datadir) -> qml.ops.LinearCombination:
    path = lazy_shared_datadir / "hamiltonian_lih.json"
    with path.open() as f:
        hamiltonian = json.load(f)
    num_qubits = max((len(label) for label in hamiltonian), default=0)
    return to_pennylane_operator(PauliOperator(hamiltonian, num_qubits=num_qubits))


@pytest.fixture
def pennylane_result(hamiltonian_lih: qml.ops.LinearCombination) -> float:
    """Run the full circuit (state prep + evolution) on PennyLane's own simulator."""

    @qml.qnode(qml.device("default.qubit", wires=_NUM_QUBITS))
    def circuit():
        for wire in _OCCUPIED_QUBITS:
            qml.PauliX(wires=wire)
        _pennylane_evolution(*_ANGLES)
        return qml.expval(hamiltonian_lih)

    return circuit()


@requires_pennylane
@pytest.mark.pennylane
def test_pennylane_with_mp(
    hamiltonian_lih: qml.ops.LinearCombination,
    pennylane_result: float,
):
    """Integration test for circuits coming from PennyLane and running them with the PauliPropagator."""
    operator = from_pennylane_operator(hamiltonian_lih, wires=range(_NUM_QUBITS))
    circuit = from_pennylane_circuit(
        _pennylane_evolution,
        _OCCUPIED_QUBITS,
        *_ANGLES,
        wires=range(_NUM_QUBITS),
    )
    mp = PauliPropagator(
        operator,
        circuit.initial_state,
        cutoff=6,
    )
    mp.propagate(circuit)
    test_expval = mp.expval()
    assert np.isclose(test_expval, pennylane_result, atol=1e-6)


@requires_pennylane
@pytest.mark.pennylane
@pytest.mark.parametrize(
    ("gate", "angle", "qubit", "observable"),
    [
        ("RX", 0.7, 0, "YI"),
        ("RY", 0.4, 1, "IX"),
        ("RZ", 0.9, 0, "XI"),
        ("RX", -0.3, 1, "IZ"),
    ],
)
def test_rotation_sign_matches_pennylane(gate, angle, qubit, observable):
    """A converted rotation must reproduce PennyLane's own sign, not its conjugate.

    Each case pairs a single rotation with an observable that anticommutes with its generator on
    the same qubit the gate acts on, so the expectation value is an odd function of the angle and
    a spurious sign flip would show up exactly. Unlike qiskit, PennyLane's own convention needs no
    negation (see [monoprop.pennylane_conversion][]), so this pins that down against PennyLane's
    own simulator rather than relying on a round trip, which cannot catch a consistently-applied
    sign bug.
    """
    pauli_letters = {
        "I": qml.Identity,
        "X": qml.PauliX,
        "Y": qml.PauliY,
        "Z": qml.PauliZ,
    }
    hamiltonian = qml.ops.LinearCombination(
        [1.0], [pauli_letters[observable[0]](0) @ pauli_letters[observable[1]](1)]
    )

    @qml.qnode(qml.device("default.qubit", wires=2))
    def reference_circuit():
        getattr(qml, gate)(angle, wires=qubit)
        return qml.expval(hamiltonian)

    expected_expval = reference_circuit()

    def qfunc(theta):
        getattr(qml, gate)(theta, wires=qubit)

    mp = PauliPropagator(
        from_pennylane_operator(hamiltonian, wires=range(2)), [], cutoff=4
    )
    mp.propagate(from_pennylane_circuit(qfunc, [], angle, wires=range(2)))

    assert np.isclose(mp.expectation_value(), expected_expval, atol=1e-9)
