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

"""Module to convert Cirq objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import cirq
except ImportError as e:
    raise ImportError(
        "cirq is required to use monoprop_cirq. Install it with: pip install cirq-core"
    ) from e

from monoprop.circuit import Circuit, ExpGate
from monoprop.pauli import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_CIRQ_PAULIS = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
_PAULI_LETTERS = {pauli: letter for letter, pauli in _CIRQ_PAULIS.items()}

# Expanding a gate's generator matrix in the Pauli basis leaves floating-point noise on terms that
# are exactly zero; it is dropped before the commutation check so it cannot make that check fail.
_EXPANSION_ATOL = 1e-12

_UNSUPPORTED_HINT = (
    "Only cirq.EigenGate gates (rotations, X/Y/Z/XX/YY/ZZ powers, CZ, CNOT, ISWAP, SWAP, CCZ, "
    "CCX, ...), cirq.PauliStringPhasor and cirq.PauliString operations whose generator has "
    "pairwise-commuting Pauli terms are supported; decompose other gates first, e.g. "
    "cirq.Circuit(cirq.decompose(circuit))."
)


def _qubit_index(
    touched: Iterable[cirq.Qid], qubit_order: cirq.QubitOrderOrList | None
) -> dict[cirq.Qid, int]:
    """Map each qubit to its monoprop index; the map's length is the system width.

    An explicit ``qubit_order`` resolves the way Cirq resolves one: a list pins the order (and may
    include untouched qubits), a ``cirq.QubitOrder`` such as ``cirq.QubitOrder.DEFAULT`` orders the
    touched qubits. Without one, ``cirq.LineQubit(x)`` becomes qubit ``x`` in a ``max(x) + 1``-wide
    system, so an operator and a circuit converted separately agree on every qubit even when they
    touch different ones. Sorting each object's own qubits would not, which is why no other qubit
    type has a default.
    """
    touched = frozenset(touched)
    if qubit_order is not None:
        order = cirq.QubitOrder.as_qubit_order(qubit_order).order_for(touched)
    elif touched and all(isinstance(q, cirq.LineQubit) and q.x >= 0 for q in touched):
        order = tuple(cirq.LineQubit.range(max(q.x for q in touched) + 1))
    else:
        raise ValueError(
            "Cannot infer a qubit ordering: the default needs every qubit to be a non-negative "
            "cirq.LineQubit, with at least one touched; pass qubit_order=... explicitly."
        )
    if any(q.dimension != 2 for q in order):
        raise ValueError(
            "Only qubits are supported; got a qid of dimension other than 2."
        )
    return {q: i for i, q in enumerate(order)}


def _output_qubits(
    num_qubits: int, qubits: Sequence[cirq.Qid] | None
) -> tuple[cirq.Qid, ...]:
    """Return the Cirq qubit for each monoprop index, defaulting to ``cirq.LineQubit.range``."""
    if qubits is None:
        return tuple(cirq.LineQubit.range(num_qubits))
    qubit_tuple = tuple(qubits)
    if len(qubit_tuple) != num_qubits:
        raise ValueError(
            f"qubits has {len(qubit_tuple)} entries but the input has {num_qubits} qubits."
        )
    if len(set(qubit_tuple)) != num_qubits:
        raise ValueError("qubits has duplicate entries.")
    return qubit_tuple


def _to_pauli(pauli_string: cirq.PauliString, index: dict[cirq.Qid, int]) -> Pauli:
    """Translate a Cirq Pauli string's letters and qubits, ignoring its coefficient."""
    qubits = pauli_string.qubits
    # Valid by construction: X/Y/Z letters on the distinct, non-negative indices of `index`.
    return Pauli(
        "".join(_PAULI_LETTERS[pauli_string[q]] for q in qubits),
        tuple(index[q] for q in qubits),
        skip_validation=True,
    )


def _to_pauli_string(
    pauli: Pauli, coeff: float, qubits: Sequence[cirq.Qid]
) -> cirq.PauliString:
    """Translate a monoprop Pauli term onto the given Cirq qubits."""
    return cirq.PauliString(
        qubit_pauli_map={
            qubits[q]: _CIRQ_PAULIS[letter]
            for q, letter in zip(pauli.qubits, pauli.string, strict=True)
        },
        coefficient=coeff,
    )


def from_cirq_operator(
    cirq_op: cirq.PauliSumLike,
    *,
    qubit_order: cirq.QubitOrderOrList | None = None,
    atol: float = 1e-8,
    skip_validation: bool = False,
) -> PauliOperator:
    """Convert a Cirq Pauli operator to a [PauliOperator][monoprop.pauli.PauliOperator].

    Requires the operator to be Hermitian, i.e. every coefficient real.

    Args:
        cirq_op: Anything ``cirq.PauliSum.wrap`` accepts: a ``cirq.PauliSum``, a
            ``cirq.PauliString``, a single-qubit Pauli operation such as ``cirq.Z(q)``, or a scalar.
        qubit_order: The qubit ordering, with its ``i``-th qubit becoming qubit ``i``: a list of
            qubits (which may include untouched ones) or a ``cirq.QubitOrder``. Defaults to
            ``cirq.LineQubit(x)`` becoming qubit ``x``, so it is required for any other qubit type
            and for an operator that touches no qubit.
        atol: Absolute tolerance below which a term's coefficient is dropped.
        skip_validation: If ``True``, skip the check that every coefficient is real; the
            imaginary parts are then silently dropped. Only pass ``True`` for operators already
            known to be Hermitian.

    Returns:
        A PauliOperator instance representing the given operator.

    Raises:
        ValueError: If a coefficient is complex, unless ``skip_validation`` is ``True``, or if no
            qubit ordering can be inferred.
    """
    pauli_sum = cirq.PauliSum.wrap(cirq_op)
    index = _qubit_index(pauli_sum.qubits, qubit_order)
    terms = {
        _to_pauli(pauli_string, index): pauli_string.coefficient
        for pauli_string in pauli_sum
        if abs(pauli_string.coefficient) > atol
    }
    return PauliOperator(terms, num_qubits=len(index), skip_validation=skip_validation)


def to_cirq_operator(
    pauli_operator: PauliOperator, *, qubits: Sequence[cirq.Qid] | None = None
) -> cirq.PauliSum:
    """Convert a [PauliOperator][monoprop.pauli.PauliOperator] to a ``cirq.PauliSum``.

    Args:
        pauli_operator: A PauliOperator instance.
        qubits: The Cirq qubits to use, with qubit ``i`` placed on ``qubits[i]``. Defaults to
            ``cirq.LineQubit.range(pauli_operator.num_qubits)``.

    Returns:
        A ``cirq.PauliSum`` over the given qubits.

    Raises:
        ValueError: If ``qubits`` does not have ``pauli_operator.num_qubits`` distinct entries.
    """
    qubit_tuple = _output_qubits(pauli_operator.num_qubits, qubits)
    return cirq.PauliSum.from_pauli_strings(
        [
            _to_pauli_string(pauli, coeff, qubit_tuple)
            for pauli, coeff in pauli_operator.terms.items()
        ]
    )


def _eigen_gate_generator(gate: cirq.EigenGate) -> tuple[np.ndarray, float]:
    """Return ``(G, t)`` with ``gate`` equal to ``exp(i t G)`` up to a global phase.

    ``gate`` applies ``exp(i pi t (theta_k + s))`` on each eigenspace ``P_k``, for its exponent
    ``t`` and global shift ``s``, so ``G = pi * sum_k theta_k P_k`` and ``s`` is only a phase.
    ``cirq.Rx``/``Ry``/``Rz`` instead return their angle in radians as ``t``, with ``G`` scaled to
    match, so the parameter is the value they were built with.
    """
    # _eigen_components() is protected, but it is the one method every EigenGate subclass defines.
    generator = sum(
        np.pi * theta * projector for theta, projector in gate._eigen_components()
    )
    exponent = float(gate.exponent)
    if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
        return generator / np.pi, np.pi * exponent
    return generator, exponent


def _matrix_terms(matrix: np.ndarray, qubits: Sequence[int]) -> dict[Pauli, float]:
    """Expand a Hermitian matrix acting on ``qubits`` in the Pauli basis."""
    basis = cirq.kron_bases(cirq.PAULI_BASIS, repeat=len(qubits))
    # Valid by construction: kron_bases labels are I/X/Y/Z strings of length len(qubits).
    return {
        Pauli(label, qubits, skip_validation=True): float(np.real(coeff))
        for label, coeff in cirq.expand_matrix_in_orthogonal_basis(
            matrix, basis
        ).items()
        if abs(coeff) > _EXPANSION_ATOL
    }


def _operation_terms(
    op: cirq.Operation, index: dict[cirq.Qid, int]
) -> tuple[dict[Pauli, float], float] | None:
    """Return ``(terms, t)`` with ``op`` equal to ``exp(i t sum(terms))`` up to a global phase.

    Returns ``None`` for an identity or global-phase operation, which Heisenberg propagation
    cannot see. ``terms`` may still hold an identity term, also a global phase.
    """
    if isinstance(op.gate, (cirq.IdentityGate, cirq.GlobalPhaseGate)):
        return None
    if isinstance(op, cirq.PauliStringPhasor):
        # Cirq keeps the string's coefficient at +1, folding a sign into the exponents; the phasor
        # is then exp(-i pi/2 * exponent_relative * P) up to phase.
        return {_to_pauli(op.pauli_string, index): -np.pi / 2}, float(
            op.exponent_relative
        )
    if isinstance(op.gate, cirq.EigenGate):
        matrix, parameter = _eigen_gate_generator(op.gate)
        return _matrix_terms(matrix, tuple(index[q] for q in op.qubits)), parameter
    if isinstance(op, cirq.PauliString):
        # A Pauli product P is exp(-i pi/2 P) up to phase.
        return {_to_pauli(op, index): -np.pi / 2}, 1.0
    raise ValueError(f"Unsupported gate {op!r}. {_UNSUPPORTED_HINT}")


def from_cirq_circuit(
    circuit: cirq.AbstractCircuit,
    initial_state: Sequence[int],
    *,
    qubit_order: cirq.QubitOrderOrList | None = None,
    param_resolver: cirq.ParamResolverOrSimilarType = None,
) -> Circuit:
    """Convert a Cirq circuit to a [Circuit][monoprop.circuit.Circuit].

    Every operation must equal ``exp(i t G)`` up to a global phase, for a generator ``G`` whose
    Pauli terms pairwise commute. That covers every ``cirq.EigenGate`` with such a generator
    (``cirq.rx``/``ry``/``rz``, the X/Y/Z/XX/YY/ZZ powers, ``cirq.CZ``/``CNOT``/``ISWAP``/``SWAP``/
    ``CCZ``/``CCX`` and their powers, ...), ``cirq.PauliStringPhasor`` (what
    ``cirq.PauliSumExponential`` expands into) and ``cirq.PauliString`` operations. Identity and
    global-phase operations are skipped, and ``cirq.CircuitOperation`` subcircuits are unrolled.
    Each remaining operation becomes one [ExpGate][monoprop.circuit.ExpGate] driven by its own
    angle (the identity parameter mapping): the gate's ``exponent``, or its angle in radians for
    ``cirq.Rx``/``Ry``/``Rz``, with every constant factor on the generator. A gate's global shift
    is a phase and is dropped.

    Args:
        circuit: The Cirq circuit.
        initial_state: The reference state (occupied qubit indices).
        qubit_order: The qubit ordering, as for ``from_cirq_operator``. Defaults to
            ``cirq.LineQubit(x)`` becoming qubit ``x``; pass it explicitly for any other qubit
            type, or when the circuit leaves qubits idle past the highest one it touches.
        param_resolver: Values for the circuit's sympy symbols, applied with
            ``cirq.resolve_parameters`` before conversion.

    Returns:
        A Circuit instance representing the given circuit.

    Raises:
        ValueError: If the circuit has unresolved symbols or a measurement, if an operation is
            unsupported, or if no qubit ordering can be inferred.
    """
    if param_resolver is not None:
        circuit = cirq.resolve_parameters(circuit, param_resolver)
    if cirq.is_parameterized(circuit):
        raise ValueError(
            f"Circuit has unresolved parameters {sorted(cirq.parameter_names(circuit))}; "
            "pass param_resolver=... with their values."
        )
    circuit = cirq.unroll_circuit_op(circuit, deep=True, tags_to_check=None)
    index = _qubit_index(circuit.all_qubits(), qubit_order)
    num_qubits = len(index)

    gates: list[ExpGate] = []
    parameters: list[float] = []
    for tagged_op in circuit.all_operations():
        op = tagged_op.untagged
        if cirq.is_measurement(op):
            raise ValueError(
                f"Unsupported measurement {op!r}. Drop terminal measurements first with "
                "cirq.drop_terminal_measurements(circuit)."
            )
        converted = _operation_terms(op, index)
        if converted is None:
            continue
        terms, parameter = converted
        # Pauli() drops identity letters, so an identity term (a global phase) has no qubits. An
        # operation with nothing else, such as the qubit-less phasor cirq.PauliSumExponential
        # emits for a constant, is skipped rather than kept as a gate with its own parameter.
        terms = {pauli: coeff for pauli, coeff in terms.items() if pauli.qubits}
        if not terms:
            continue
        # Real coefficients on in-range qubits by construction, so validation is skipped.
        generator = PauliOperator(terms, num_qubits=num_qubits, skip_validation=True)
        if not generator.all_pairwise_commute():
            raise ValueError(
                f"Unsupported gate {op!r}: its generator's Pauli terms do not commute. "
                f"{_UNSUPPORTED_HINT}"
            )
        gates.append(ExpGate(generator))
        parameters.append(parameter)

    return Circuit(
        gates=tuple(gates),
        parameters=tuple(parameters),
        initial_state=tuple(initial_state),
        system_size=num_qubits,
    )
