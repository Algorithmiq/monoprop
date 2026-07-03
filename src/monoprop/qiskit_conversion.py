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

from monoprop.circuit import PauliCircuit, PauliGate
from monoprop.pauli_data import PauliOperator

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
) -> PauliCircuit:
    """Convert a Qiskit circuit to a :class:`~monoprop.circuit.PauliCircuit`.

    Note that the qiskit circuit must be composed only by PauliEvolutionGates with commuting
    operators. Each qiskit gate becomes one :class:`~monoprop.circuit.PauliGate` driven by
    its own angle (the identity parameter mapping).

    Args:
        circuit: A qiskit quantum circuit.
        initial_state: Initial quantum state as a list of integers.

    Returns:
        A :class:`~monoprop.circuit.PauliCircuit` representing the given circuit.
    """
    gates: list[PauliGate] = []
    parameters: list[float] = []
    qregs = circuit.qregs[0]
    for gate in circuit.data:
        g_op = gate.operation
        gate_name = g_op.name

        if gate_name == "barrier":
            continue

        qubits: tuple[int, ...] = tuple(qregs.index(qb) for qb in gate.qubits)  # type: ignore

        if gate_name == "PauliEvolution":
            parameter = g_op.time
            paulis = from_qiskit_operator(g_op.operator)
        elif gate_name in PAULI_EVOLUTION_EQUIVALENT:
            parameter = g_op.params[0] * 0.5
            pauli_string = gate_name[1:].upper()  # Remove the leading 'R' and uppercase
            paulis = PauliOperator([pauli_string], [1.0])
        else:
            raise ValueError(
                f"Unsupported gate {gate_name}. Only PauliEvolutionGate or equivalent gates are supported."
            )

        gates.append(PauliGate(qubits, paulis))
        parameters.append(float(parameter))

    return PauliCircuit(
        gates=tuple(gates),
        parameters=tuple(parameters),
        initial_state=tuple(initial_state),
        num_qubits=len(qregs),
    )


def to_qiskit_circuit(circuit: PauliCircuit) -> QuantumCircuit:
    """Convert a :class:`~monoprop.circuit.PauliCircuit` to a Qiskit circuit.

    Note that the resulting qiskit circuit will be composed only by PauliEvolutionGates with
    commuting operators. Each gate's evolution time is taken from the circuit's
    ``parameters`` via its parameter mapping.

    Args:
        circuit: A :class:`~monoprop.circuit.PauliCircuit` representing the given circuit.

    Returns:
        A qiskit quantum circuit.
    """
    qiskit_circuit = QuantumCircuit(circuit.num_qubits)
    mapping = circuit.resolved_mapping
    for gate, param_index in zip(circuit.gates, mapping, strict=True):
        qiskit_op = to_qiskit_operator(gate.paulis.to_dict())
        qiskit_circuit.append(
            PauliEvolutionGate(qiskit_op, time=circuit.parameters[param_index]),
            gate.qubits,
        )
    return qiskit_circuit
