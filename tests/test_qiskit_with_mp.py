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

import json

import numpy as np
import pytest

from monoprop import QubitPropagator, gates_from_qubit_circuit
from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    from monoprop.qiskit_conversion import to_qiskit_operator

    _qiskit_available = True
except ImportError:
    _qiskit_available = False

requires_qiskit = pytest.mark.skipif(
    not _qiskit_available, reason="qiskit not installed"
)


@pytest.fixture
def hamiltonian_lih(lazy_shared_datadir) -> SparsePauliOp:
    path = lazy_shared_datadir / "hamiltonian_lih.json"
    with path.open() as f:
        hamiltonian = json.load(f)
    return to_qiskit_operator(hamiltonian)


@pytest.fixture
def simple_initial_state() -> QuantumCircuit:
    num_qubits = 12
    circuit = QuantumCircuit(num_qubits)

    for i in range(1, num_qubits, 2):
        circuit.x(i)
    return circuit


@pytest.fixture
def simple_ev_circuit() -> QuantumCircuit:
    num_qubits = 12
    circuit = QuantumCircuit(num_qubits)

    op1 = SparsePauliOp(["II", "XZ"], np.array([1, 1]))
    op2 = SparsePauliOp(["IZ", "ZZ"], np.array([0.1, 0.1]))
    op3 = SparsePauliOp(["YI", "IY"], np.array([0.7, 0.1]))
    op4 = SparsePauliOp(["ZI", "ZZ"], np.array([0.5, 0.6]))
    evo1 = PauliEvolutionGate(op1, time=0.2)
    evo2 = PauliEvolutionGate(op2, time=0.1)
    evo3 = PauliEvolutionGate(op3, time=0.3)
    evo4 = PauliEvolutionGate(op4, time=0.1)

    for i, evo_op in enumerate([evo1, evo2, evo3, evo4, evo3, evo2]):
        circuit.append(evo_op, range(2 * i, 2 + 2 * i))

    return circuit


@pytest.fixture
def qiskit_full_circuit(simple_initial_state, simple_ev_circuit) -> QuantumCircuit:
    return simple_initial_state.compose(simple_ev_circuit)


@pytest.fixture
def qiskit_result(qiskit_full_circuit, hamiltonian_lih) -> list[complex]:
    """Run the full circuit on a qiskit simulator to get the expected result."""
    estimator = StatevectorEstimator()
    job = estimator.run([(qiskit_full_circuit, hamiltonian_lih)])
    result = job.result()
    return result[0].data.evs


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_qiskit_with_mp(
    hamiltonian_lih: SparsePauliOp,
    simple_ev_circuit: QuantumCircuit,
    qiskit_result: list[complex],
):
    """Integration test for circuits comming from Qiskit and running them with the MonomialPropagator."""

    operator = from_qiskit_operator(hamiltonian_lih)
    quantum_circuit = from_qiskit_circuit(
        simple_ev_circuit, initial_state=list(range(1, 12, 2))
    )
    gates, _ = gates_from_qubit_circuit(quantum_circuit)
    parameters = [gate.parameter for gate in quantum_circuit.gates]
    mp = QubitPropagator(
        operator,
        quantum_circuit.initial_state,
        cutoff=6,
    )
    mp.propagate(gates, parameters)
    test_expval = mp.expectation_value()
    assert np.isclose(test_expval, qiskit_result, atol=1e-6)
