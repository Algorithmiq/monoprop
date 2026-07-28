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

from monoprop import PauliPropagator
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
    """Integration test for circuits coming from Qiskit and running them with the PauliPropagator."""

    operator = from_qiskit_operator(hamiltonian_lih)
    circuit = from_qiskit_circuit(
        simple_ev_circuit, initial_state=list(range(1, 12, 2))
    )
    mp = PauliPropagator(
        operator,
        circuit.initial_state,
        cutoff=6,
    )
    mp.propagate(circuit)
    test_expval = mp.expval()
    assert np.isclose(test_expval, qiskit_result, atol=1e-6)


@requires_qiskit
@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("gate", "angle", "qubit", "observable"),
    [
        ("rx", 0.7, 0, "IY"),
        ("ry", 0.4, 1, "XI"),
        ("rz", 0.9, 0, "IX"),
        ("rx", -0.3, 1, "YI"),
    ],
)
def test_rotation_sign_matches_qiskit(gate, angle, qubit, observable):
    """A converted rotation must reproduce qiskit's sign, not its conjugate.

    Qiskit rotates by ``exp(-i t P/2)`` while :class:`~monoprop.circuit.ExpGate` applies
    ``exp(+i theta H)``. Each case pairs a single rotation with an observable that anticommutes
    with its generator, so the expectation value is an odd function of the angle and a missing
    negation across the boundary shows up as an exact sign flip. The
    ``from_qiskit_circuit``/``to_qiskit_circuit`` roundtrip cannot catch this -- it negates both
    ways -- so this compares against qiskit's own simulator.
    """
    qiskit_circuit = QuantumCircuit(2)
    getattr(qiskit_circuit, gate)(angle, qubit)
    hamiltonian = SparsePauliOp.from_list([(observable, 1.0)])

    expected = StatevectorEstimator().run([(qiskit_circuit, hamiltonian)]).result()
    expected_expval = expected[0].data.evs

    mp = PauliPropagator(from_qiskit_operator(hamiltonian), [], cutoff=4)
    mp.propagate(from_qiskit_circuit(qiskit_circuit, []))

    assert np.isclose(mp.expectation_value(), expected_expval, atol=1e-9)
