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

from monoprop.conversion_utils import _extend_pauli_string

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
except ImportError as e:
    raise ImportError(
        "qiskit is required to use monoprop.qiskit_conversion. Install it with: pip install qiskit"
    ) from e

from monoprop.circuit import Circuit, ExpGate
from monoprop.pauli import Pauli, PauliOperator

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
    return PauliOperator._from_terms(
        pauli_strings,
        list(coeffs),  # type: ignore[arg-type]
        num_qubits=qiskit_op.num_qubits,
    )


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


def _place_operator(
    local_op: PauliOperator, qubits: tuple[int, ...], num_qubits: int
) -> PauliOperator:
    """Remap a local operator (on qubits ``0..len(qubits)-1``) onto global ``qubits``.

    The result is a gate generator carrying the full-circuit ``num_qubits`` (an operator
    carries its own qubit count).
    """
    return PauliOperator._from_terms(
        [
            Pauli(pauli.string, tuple(qubits[q] for q in pauli.qubits))
            for pauli in local_op.terms
        ],
        list(local_op.terms.values()),
        num_qubits=num_qubits,
    )


def from_qiskit_circuit(
    circuit: QuantumCircuit,
    initial_state: list[int],
) -> Circuit:
    """Convert a Qiskit circuit to a :class:`~monoprop.circuit.Circuit`.

    Note that the qiskit circuit must be composed only by PauliEvolutionGates with commuting
    operators. Each qiskit gate becomes one qubit :class:`~monoprop.circuit.ExpGate` driven by
    its own angle (the identity parameter mapping).

    Args:
        circuit: A qiskit quantum circuit.
        initial_state: Initial quantum state as a list of integers.

    Returns:
        A :class:`~monoprop.circuit.Circuit` representing the given circuit.
    """
    if len(circuit.qregs) != 1:
        raise ValueError(
            f"from_qiskit_circuit only supports a single quantum register; got {len(circuit.qregs)}."
        )

    gates: list[ExpGate] = []
    parameters: list[float] = []
    qregs = circuit.qregs[0]
    num_qubits = len(qregs)
    for gate in circuit.data:
        g_op = gate.operation
        gate_name = g_op.name

        if gate_name == "barrier":
            continue

        qubits: tuple[int, ...] = tuple(qregs.index(qb) for qb in gate.qubits)  # type: ignore

        if gate_name == "PauliEvolution":
            parameter = g_op.time
            generator = _place_operator(
                from_qiskit_operator(g_op.operator), qubits, num_qubits
            )
        elif gate_name in PAULI_EVOLUTION_EQUIVALENT:
            parameter = g_op.params[0] * 0.5
            pauli_string = gate_name[1:].upper()  # Remove the leading 'R' and uppercase
            generator = PauliOperator._from_terms(
                [Pauli(pauli_string, qubits)], [1.0], num_qubits=num_qubits
            )
        else:
            raise ValueError(
                f"Unsupported gate {gate_name}. Only PauliEvolutionGate or equivalent gates are supported."
            )

        gates.append(ExpGate(generator))
        parameters.append(float(parameter))

    return Circuit(
        gates=tuple(gates),
        parameters=tuple(parameters),
        initial_state=tuple(initial_state),
    )


def _extend_generator_minimally(
    generator: PauliOperator,
) -> tuple[dict[str, complex], list[int]]:
    qubits = sorted({q for p in generator.terms for q in p.qubits})
    localizing_qubit_map = {q: i for i, q in enumerate(qubits)}
    result = {
        _extend_pauli_string(
            "".join(pauli.string),
            [localizing_qubit_map[q] for q in pauli.qubits],
            len(qubits),
        ): coeff
        for pauli, coeff in generator.terms.items()
    }

    return result, qubits


def to_qiskit_circuit(circuit: Circuit, num_qubits: int) -> QuantumCircuit:
    """Convert a :class:`~monoprop.circuit.Circuit` to a Qiskit circuit.

    Note that the resulting qiskit circuit will be composed only by PauliEvolutionGates with
    commuting operators. Each gate's evolution time is taken from the circuit's
    ``parameters`` via its parameter mapping.

    Args:
        circuit: A :class:`~monoprop.circuit.Circuit` representing the given circuit.
        num_qubits: Total number of qubits (the circuit no longer carries it; supply the
            observable's ``num_qubits``).

    Returns:
        A qiskit quantum circuit.
    """
    if len(circuit.parameters) != circuit.n_parameters:
        raise ValueError(
            f"to_qiskit_circuit needs a bound circuit: it has {circuit.n_parameters} "
            f"parameter(s) but {len(circuit.parameters)} angle value(s). Supply the angles, "
            "e.g. Circuit(..., parameters=...)."
        )
    qiskit_circuit = QuantumCircuit(num_qubits)
    mapping = circuit.resolved_mapping
    for gate, param_index in zip(circuit.gates, mapping, strict=True):
        generator = gate.generator
        if not isinstance(generator, PauliOperator):
            raise TypeError(
                "to_qiskit_circuit requires a qubit (Pauli) circuit; got a "
                f"{circuit.family}-family gate."
            )
        pauli_dict, qubits = _extend_generator_minimally(generator)
        qiskit_circuit.append(
            PauliEvolutionGate(
                to_qiskit_operator(pauli_dict), time=circuit.parameters[param_index]
            ),
            qubits,
        )
    return qiskit_circuit
