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

"""Module to convert PennyLane objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import pennylane as qml
except ImportError as e:
    raise ImportError(
        "pennylane is required to use monoprop.pennylane_conversion. "
        "Install it with: pip install pennylane"
    ) from e

from monoprop.circuit import Circuit, ExpGate
from monoprop.pauli import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence


def _resolve_wires(
    touched: Iterable[Hashable], wires: Sequence[Hashable] | None
) -> tuple[Hashable, ...]:
    """Return an explicit, ordered wire tuple, inferring one from ``touched`` if not given.

    Without an explicit ``wires``, the wire label doubles as the qubit index: the default is
    ``range(max(touched) + 1)``, so a wire below the highest one touched becomes an implicit
    idle qubit rather than being silently renumbered away (e.g. touching only wire 1 defaults
    to a 2-qubit system with that term on qubit 1, not a 1-qubit system with it moved to qubit
    0). This only works when every touched wire is a non-negative integer with at least one
    touched; anything else needs an explicit ``wires=``.
    """
    if wires is not None:
        wire_tuple = tuple(wires)
        if len(set(wire_tuple)) != len(wire_tuple):
            raise ValueError("wires must contain unique labels.")
        return wire_tuple
    touched_set = set(touched)
    if not touched_set or not all(isinstance(w, int) and w >= 0 for w in touched_set):
        raise ValueError(
            "Cannot infer a qubit ordering from these wire labels (they must be non-negative "
            "integers, with at least one touched); pass wires=... explicitly."
        )
    return tuple(range(max(touched_set) + 1))


def _validate_output_wires(
    wires: Sequence[Hashable] | None, width: int, width_name: str
) -> tuple[Hashable, ...]:
    """Return output wires after validating their width and uniqueness."""
    wire_tuple = tuple(range(width)) if wires is None else tuple(wires)
    if len(wire_tuple) != width:
        raise ValueError(
            f"wires has {len(wire_tuple)} entries but {width_name}={width}."
        )
    if len(set(wire_tuple)) != len(wire_tuple):
        raise ValueError("wires must contain unique labels.")
    return wire_tuple


def _pauli_sentence_terms(
    pauli_sentence: qml.pauli.PauliSentence, wire_map: dict[Hashable, int]
) -> tuple[list[Pauli], list[complex]]:
    """Translate a ``PauliSentence`` into monoprop [Pauli][monoprop.pauli.Pauli] terms."""
    paulis: list[Pauli] = []
    coeffs: list[complex] = []
    for word, coeff in pauli_sentence.items():
        qubits = tuple(wire_map[w] for w in word)
        letters = "".join(word[w] for w in word)
        paulis.append(Pauli(letters, qubits))
        coeffs.append(coeff)
    return paulis, coeffs


def from_pennylane_operator(
    pennylane_op: qml.operation.Operator,
    *,
    wires: Sequence[Hashable] | None = None,
    atol: float = 1e-8,
) -> PauliOperator:
    """Convert a PennyLane operator to a [PauliOperator][monoprop.pauli.PauliOperator].

    Requires the operator to carry a Pauli decomposition (its ``pauli_rep``, e.g. a
    ``qml.ops.LinearCombination``/``qml.Hamiltonian``, a ``qml.ops.Sum``/``qml.ops.Prod`` of Pauli
    operators, or a bare single-qubit Pauli operator) and to be Hermitian.

    Args:
        pennylane_op: A PennyLane operator expressible as a linear combination of Pauli words.
        wires: The wire ordering to use for the resulting qubit indices, with ``wires[i]``
            becoming qubit ``i``. Defaults to ``range(max(touched wire) + 1)`` -- the wire label
            doubles as the qubit index -- so pass this explicitly when a touched wire is not a
            non-negative integer, or a specific ordering is wanted. Labels must be unique.
        atol: Absolute tolerance below which a term's coefficient is dropped.

    Returns:
        A PauliOperator instance representing the given operator.
    """
    # pauli_sentence() returns the operator's own cached pauli_rep, and prune() mutates in place
    # and returns None -- so a fresh copy is pruned instead, to avoid corrupting the caller's op.
    pauli_sentence = qml.pauli.PauliSentence(
        dict(qml.pauli.pauli_sentence(pennylane_op))
    )
    pauli_sentence.prune(atol)
    wire_tuple = _resolve_wires(pauli_sentence.wires, wires)
    wire_map = {w: i for i, w in enumerate(wire_tuple)}
    paulis, coeffs = _pauli_sentence_terms(pauli_sentence, wire_map)
    return PauliOperator._from_terms(paulis, coeffs, num_qubits=len(wire_tuple))


def to_pennylane_operator(
    pauli_operator: PauliOperator, *, wires: Sequence[Hashable] | None = None
) -> qml.ops.LinearCombination:
    """Convert a [PauliOperator][monoprop.pauli.PauliOperator] to a PennyLane operator.

    Args:
        pauli_operator: A PauliOperator instance.
        wires: The wire labels to use, with qubit ``i`` placed on ``wires[i]``. Defaults to plain
            integer wires ``0..num_qubits-1``.

    Returns:
        A ``qml.ops.LinearCombination`` over the given wires.

    Raises:
        ValueError: If ``wires`` does not contain exactly ``pauli_operator.num_qubits`` unique
            labels.
    """
    wire_tuple = _validate_output_wires(
        wires, pauli_operator.num_qubits, "pauli_operator.num_qubits"
    )
    coeffs: list[float] = []
    ops: list[qml.operation.Operator] = []
    for pauli, coeff in pauli_operator.terms.items():
        word = qml.pauli.PauliWord(
            {
                wire_tuple[q]: letter
                for q, letter in zip(pauli.qubits, pauli.string, strict=True)
            }
        )
        coeffs.append(coeff)
        ops.append(word.operation(wire_order=list(wire_tuple)))
    return qml.ops.LinearCombination(coeffs, ops)


def from_pennylane_circuit(
    qfunc: Callable[..., Any],
    initial_state: list[int],
    *args: Any,
    wires: Sequence[Hashable] | None = None,
    **kwargs: Any,
) -> Circuit:
    """Convert a PennyLane circuit to a [Circuit][monoprop.circuit.Circuit].

    ``qfunc`` (a plain quantum function or a ``QNode``) is traced with ``qml.tape.make_qscript``,
    without executing on any device. Only gates with a single trainable parameter and a
    Pauli-expressible generator are supported (e.g. ``qml.RX``/``qml.RY``/``qml.RZ``/
    ``qml.PauliRot``/``qml.MultiRZ``/``qml.IsingXX`` and similar); ``qml.Barrier`` is ignored, and a
    ``qml.exp``/``qml.evolve`` gate (a multi-term-generator exponential, the PennyLane analog of
    qiskit's ``PauliEvolutionGate``) is explicitly unsupported -- decomposing its ``coeff`` into a
    single real angle is not generally well-defined. Each gate becomes one
    [ExpGate][monoprop.circuit.ExpGate] driven by its own angle (the identity parameter mapping).
    Unlike [monoprop.qiskit_conversion][], no sign flip is needed: PennyLane's ``generator()`` is
    already defined by ``U(phi) = exp(+i phi G)``, the same convention
    [ExpGate][monoprop.circuit.ExpGate] uses.

    Args:
        qfunc: A quantum function or ``QNode`` to trace.
        initial_state: The reference state (occupied qubit indices).
        *args: Positional arguments to call ``qfunc`` with.
        wires: The wire ordering to use for the resulting qubit indices. Defaults to
            ``range(max(touched wire) + 1)`` -- the wire label doubles as the qubit index -- so
            pass this explicitly if the circuit has idle wires past the highest one touched, or
            a touched wire is not a non-negative integer. Labels must be unique.
        **kwargs: Keyword arguments to call ``qfunc`` with.

    Returns:
        A Circuit instance representing the given circuit.

    Raises:
        ValueError: If a gate is not a single-parameter, Pauli-generator gate.
    """
    qfunc = qfunc.func if isinstance(qfunc, qml.QNode) else qfunc
    tape = qml.tape.make_qscript(qfunc)(*args, **kwargs)
    all_wires = [w for op in tape.operations for w in op.wires]
    wire_tuple = _resolve_wires(all_wires, wires)
    wire_map = {w: i for i, w in enumerate(wire_tuple)}
    num_qubits = len(wire_tuple)

    gates: list[ExpGate] = []
    parameters: list[float] = []
    for op in tape.operations:
        if isinstance(op, qml.Barrier):
            continue
        # qml.ops.Exp is excluded explicitly: a single-term instance would otherwise slip past
        # the has_generator/num_params==1 check below (Exp.generator() returns its own base
        # unscaled, and Exp.parameters holds the raw, possibly-complex `coeff`), which is not the
        # single-real-angle form the rest of this loop assumes (see this function's docstring).
        if isinstance(op, qml.ops.Exp) or not (op.has_generator and op.num_params == 1):
            raise ValueError(
                f"Unsupported gate {type(op).__name__}. Only single-parameter Pauli-generator "
                "gates are supported."
            )
        pauli_sentence = op.generator().pauli_rep
        if pauli_sentence is None:
            raise ValueError(
                f"Unsupported gate {type(op).__name__}: its generator is not expressible as a "
                "sum of Pauli words."
            )
        paulis, coeffs = _pauli_sentence_terms(pauli_sentence, wire_map)
        gates.append(
            ExpGate(PauliOperator._from_terms(paulis, coeffs, num_qubits=num_qubits))
        )
        parameters.append(float(op.parameters[0]))

    return Circuit(
        gates=tuple(gates),
        parameters=tuple(parameters),
        initial_state=tuple(initial_state),
        system_size=num_qubits,
    )


def to_pennylane_circuit(
    circuit: Circuit, *, wires: Sequence[Hashable] | None = None
) -> Callable[..., None]:
    """Convert a [Circuit][monoprop.circuit.Circuit] to a PennyLane quantum function.

    Calling the returned ``qfunc`` inside a queuing context (e.g. a ``QNode`` or
    ``qml.tape.make_qscript``) queues one ``qml.exp(generator, 1j * theta)`` operation per gate,
    where ``generator`` is the gate's own [PauliOperator][monoprop.pauli.PauliOperator] rebuilt as
    a native PennyLane operator. Called with no arguments, ``qfunc()`` replays ``circuit``'s own
    bound angles (``circuit.parameters``); called with ``circuit.n_parameters`` positional angles,
    ``qfunc(*thetas)`` uses those instead -- this also works when ``circuit`` itself is unbound
    (``circuit.parameters == ()``), as long as angles are then supplied explicitly. Unlike
    [monoprop.qiskit_conversion][], no sign flip is applied (see [from_pennylane_circuit][]):
    ``exp(+i theta H) == qml.exp(H, 1j * theta)`` directly.

    Args:
        circuit: A [Circuit][monoprop.circuit.Circuit] representing the given circuit.
        wires: The wire labels to use for the circuit's qubits. Defaults to plain integer wires
            ``0..circuit.system_size-1``.

    Returns:
        A quantum function ``qfunc(*thetas)``, as described above.

    Raises:
        ValueError: If ``wires`` does not contain exactly ``circuit.system_size`` unique labels,
            or ``qfunc`` is called with a number of angles other than ``circuit.n_parameters``
            (and not zero, in which case ``circuit.parameters`` is used instead).
        TypeError: If ``circuit`` holds a Majorana-family gate rather than a Pauli one.
    """
    wire_tuple = _validate_output_wires(
        wires, circuit.system_size, "circuit.system_size"
    )
    generators: list[PauliOperator] = []
    for gate in circuit.gates:
        if not isinstance(gate.generator, PauliOperator):
            raise TypeError(
                "to_pennylane_circuit requires a qubit (Pauli) circuit; got a "
                f"{circuit.family}-family gate."
            )
        generators.append(gate.generator)
    mapping = circuit.resolved_mapping

    def qfunc(*thetas: float) -> None:
        angles = thetas or circuit.parameters
        if len(angles) != circuit.n_parameters:
            raise ValueError(
                f"qfunc expects {circuit.n_parameters} angle(s); got {len(angles)}."
            )
        for generator, param_index in zip(generators, mapping, strict=True):
            coeffs: list[float] = []
            term_ops: list[qml.operation.Operator] = []
            for pauli, coeff in generator.terms.items():
                word = qml.pauli.PauliWord(
                    {
                        wire_tuple[q]: letter
                        for q, letter in zip(pauli.qubits, pauli.string, strict=True)
                    }
                )
                coeffs.append(coeff)
                term_ops.append(word.operation())
            qml.exp(qml.dot(coeffs, term_ops), 1j * angles[param_index])

    return qfunc
