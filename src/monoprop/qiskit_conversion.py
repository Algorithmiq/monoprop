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

"""Module to convert qiskit objects."""

from __future__ import annotations

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
except ImportError as e:
    raise ImportError(
        "qiskit is required to use monoprop.qiskit_conversion. Install it with: pip install qiskit"
    ) from e

from monoprop.pauli_data import PauliEvCircuit, PauliEvGate, PauliOperator

PAULI_EVOLUTION_EQUIVALENT = {
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "rxy",
    "rxz",
    "ryz",
}

VALID_PAULI_GATES = PAULI_EVOLUTION_EQUIVALENT.union({"PauliEvolution"})


def from_qiskit_operator(
    qiskit_op: SparsePauliOp, *, atol: float = 1e-8, force_real: bool = False
) -> PauliOperator:
    """Convert a Qiskit operator to a PauliOperator.

    Args:
        qiskit_op: A qiskit Pauli operator.
        atol: Absolute tolerance for cutoff with qiskit's simplify()
        force_real: Cast operator coefficients as real values. Default is `False`
    Returns:
        A PauliOperator instance representing the given operator.
    """
    # Make sure the operator is reduced
    qiskit_op = qiskit_op.simplify(atol=atol)
    pauli_strings: list[str] = qiskit_op.paulis.to_labels(array=True)  # type: ignore
    pauli_strings = [
        s[::-1] for s in pauli_strings
    ]  # reverse the strings to match monoprop convention
    coeffs = qiskit_op.coeffs
    if force_real:
        coeffs = np.real_if_close(coeffs)  # type: ignore
        if np.iscomplexobj(coeffs):
            raise ValueError("Operator has complex terms")
    return PauliOperator(strings=pauli_strings, coefficients=coeffs)  # type: ignore


def to_qiskit_operator(
    pauli_dict: dict[str, complex] | dict[str, float],
) -> SparsePauliOp:
    """Convert a dictionary of Pauli strings with their coefficients to a Qiskit operator.

    Args:
        pauli_dict: A dictionary of Pauli strings with their coefficients,
            in the right qubit order.

    Returns:
        the Qiskit operator.
    """
    return SparsePauliOp.from_list([(k[::-1], v) for k, v in pauli_dict.items()])


def from_qiskit_circuit(
    circuit: QuantumCircuit,
    initial_state: list[int],
) -> PauliEvCircuit:
    """Convert a Qiskit circuit to a PauliOperator.

    Note that the qiskit circuit must be composed only by PauliEvolutionGates with commuting
    operators.

    Args:
        circuit: A qiskit quantum circuit.
        initial_state: Initial quantum state as a list of integers.

    Returns:
        A PauliEvCircuit instance representing the given circuit.
    """
    gates = []
    qregs = circuit.qregs[0]
    for gate in circuit.data:
        g_op = gate.operation
        gate_name = g_op.name

        if gate_name == "barrier":
            continue

        qubits: list[int] = [qregs.index(qb) for qb in gate.qubits]  # type: ignore

        if gate_name == "PauliEvolution":
            parameter = g_op.time
            paulis = from_qiskit_operator(g_op.operator)
            converted_gate = PauliEvGate(
                qubits=qubits, paulis=paulis, parameter=parameter
            )
            gates.append(converted_gate)
        elif gate_name in PAULI_EVOLUTION_EQUIVALENT:
            parameter = g_op.params[0] * 0.5
            pauli_string = gate_name[1:].upper()  # Remove the leading 'R' and uppercase
            paulis = PauliOperator([pauli_string], [1.0])  # Remove the leading 'R'
            converted_gate = PauliEvGate(
                qubits=qubits,
                paulis=paulis,
                parameter=parameter,
            )
            gates.append(converted_gate)
        else:
            raise ValueError(
                f"Unsupported gate {gate_name}. Only PauliEvolutionGate or equivalent gates are supported."
            )

    return PauliEvCircuit(
        initial_state=initial_state, gates=gates, num_qubits=len(qregs)
    )  # type: ignore


def to_qiskit_circuit(circuit: PauliEvCircuit) -> QuantumCircuit:
    """Convert a PauliEvCircuit to a Qiskit circuit.

    Note that the resulting qiskit circuit will be composed only by PauliEvolutionGates with
    commuting operators.

    Args:
        circuit: A PauliEvCircuit instance representing the given circuit.

    Returns:
        A qiskit quantum circuit.
    """
    num_qubits = max(qb for gate in circuit.gates for qb in gate.qubits) + 1  # type: ignore
    qiskit_circuit = QuantumCircuit(num_qubits)
    for gate in circuit.gates:
        qiskit_op = to_qiskit_operator(gate.paulis.to_dict())
        qiskit_circuit.append(
            PauliEvolutionGate(qiskit_op, time=gate.parameter), gate.qubits
        )
    return qiskit_circuit
