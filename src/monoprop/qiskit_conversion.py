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
from monoprop.utils import _validate_system_size

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
    qiskit_op: SparsePauliOp, *, atol: float = 1e-8
) -> PauliOperator:
    """Convert a Qiskit operator to a PauliOperator.

    Requires the operator to be Hermitian

    Args:
        qiskit_op: A qiskit Pauli operator.
        atol: Absolute tolerance for the ``simplify()`` run first, which drops smaller terms.

    Returns:
        A PauliOperator instance representing the given operator.
    """
    qiskit_op = qiskit_op.simplify(atol=atol)
    pauli_strings: list[str] = qiskit_op.paulis.to_labels(array=True)  # type: ignore
    pauli_strings = [
        s[::-1] for s in pauli_strings
    ]  # reverse the strings to match monoprop convention
    return PauliOperator._from_terms(
        pauli_strings, list(qiskit_op.coeffs), num_qubits=qiskit_op.num_qubits
    )


def _to_qiskit_operator(pauli_dict: dict[str, float], num_qubits: int) -> SparsePauliOp:
    """Convert a dictionary of Pauli strings with their coefficients to a Qiskit operator.

    Args:
        pauli_dict: A dictionary mapping Pauli strings to their coefficients.
        num_qubits: Number of qubits in the system.

    Returns:
        The Qiskit operator.
    """
    return SparsePauliOp.from_list(
        [(s[::-1], c) for s, c in pauli_dict.items()], num_qubits=num_qubits
    )


def to_qiskit_operator(
    pauli_operator: PauliOperator, num_qubits: int | None = None
) -> SparsePauliOp:
    """Convert a PauliOperator to a Qiskit SparsePauliOp.

    Args:
        pauli_operator: A PauliOperator instance.
        num_qubits: Number of qubits in the system. If None, it will be inferred from the
            PauliOperator.

    Returns:
        The Qiskit operator.
    """
    num_qubits = num_qubits if num_qubits is not None else pauli_operator.num_qubits
    if num_qubits is None:
        raise ValueError(
            "Number of qubits must be specified either in the PauliOperator or as an argument."
        )

    operator = {
        _extend_pauli_string(p.string, p.qubits, num_qubits): coeff
        for p, coeff in pauli_operator.terms.items()
    }

    return _to_qiskit_operator(operator, num_qubits=num_qubits)


def _place_operator(
    local_op: PauliOperator, qubits: tuple[int, ...], num_qubits: int
) -> PauliOperator:
    """Remap a local operator on ``0..len(qubits)-1`` onto global ``qubits``, at full width."""
    return PauliOperator._from_terms(
        [
            Pauli(pauli.string, tuple(qubits[q] for q in pauli.qubits))
            for pauli in local_op.terms
        ],
        list(local_op.terms.values()),
        num_qubits=num_qubits,
    )


def _negated(operator: PauliOperator) -> PauliOperator:
    """Flip the sign of every coefficient of ``operator``.

    Qiskit evolves by ``exp(-i t H)`` while [ExpGate][monoprop.circuit.ExpGate] applies
    ``exp(+i theta H)``, so a generator changes sign when it crosses the boundary. The sign goes on
    the *generator*, not on the angle, which keeps a converted circuit's ``parameters`` (and hence
    gradients with respect to them) numerically equal to the qiskit evolution times.
    """
    return PauliOperator._from_terms(
        list(operator.terms),
        [-coeff for coeff in operator.terms.values()],
        num_qubits=operator.num_qubits,
    )


def from_qiskit_circuit(
    circuit: QuantumCircuit,
    initial_state: list[int],
) -> Circuit:
    """Convert a Qiskit circuit to a [Circuit][monoprop.circuit.Circuit].

    The qiskit circuit must hold only PauliEvolutionGates (or the equivalent rotations in
    ``PAULI_EVOLUTION_EQUIVALENT``) with commuting operators; barriers are ignored. Each gate
    becomes one [ExpGate][monoprop.circuit.ExpGate] driven by its own angle (the identity parameter
    mapping), with the generator's coefficients negated so the angles carry through unchanged (see
    ``_negated``).
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
            generator = _negated(
                _place_operator(from_qiskit_operator(g_op.operator), qubits, num_qubits)
            )
        elif gate_name in PAULI_EVOLUTION_EQUIVALENT:
            parameter = g_op.params[0]
            pauli_string = gate_name[1:].upper()
            # R<P>(t) == exp(-i t P/2), i.e. exp(+i t (-P/2)).
            generator = PauliOperator._from_terms(
                [Pauli(pauli_string, qubits)], [-0.5], num_qubits=num_qubits
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
        system_size=num_qubits,
    )


def _extend_generator_minimally(
    generator: PauliOperator,
) -> tuple[dict[str, float], list[int]]:
    """Relabel a generator onto the qubits it touches, returning the strings and those qubits."""
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
    """Convert a [Circuit][monoprop.circuit.Circuit] to a Qiskit circuit.

    The result holds only PauliEvolutionGates. Each gate's evolution time is taken from the
    circuit's ``parameters`` via its parameter mapping, and each generator's coefficients are
    negated to turn monoprop's ``exp(+i theta H)`` back into qiskit's ``exp(-i t H)`` (see
    ``_negated``). Pass the observable's ``num_qubits`` as the width of the result.

    Args:
        circuit: A [Circuit][monoprop.circuit.Circuit] representing the given circuit.
        num_qubits: Total number of qubits. Must match ``circuit.system_size``.

    Returns:
        A qiskit quantum circuit.
    """
    num_qubits = _validate_system_size(num_qubits, argument_name="num_qubits")
    if num_qubits != circuit.system_size:
        raise ValueError(
            f"num_qubits={num_qubits} does not match circuit.system_size={circuit.system_size}."
        )
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
        pauli_dict, qubits = _extend_generator_minimally(_negated(generator))
        qiskit_circuit.append(
            PauliEvolutionGate(
                _to_qiskit_operator(pauli_dict, num_qubits=len(qubits)),
                time=circuit.parameters[param_index],
            ),
            qubits,
        )
    return qiskit_circuit
