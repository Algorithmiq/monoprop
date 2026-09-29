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

try:
    import cirq
except ImportError as e:
    raise ImportError(
        "cirq is required to use monoprop_cirq. Install it with: pip install cirq-core"
    ) from e

from monoprop.pauli import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_CIRQ_PAULIS = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
_PAULI_LETTERS = {pauli: letter for letter, pauli in _CIRQ_PAULIS.items()}


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
