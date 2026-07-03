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

"""Authoring types for Majorana/qubit circuits.

A :class:`MajoranaGate` (or its qubit analogue :class:`PauliGate`) is a *pure generator*:
the bundle of Majorana monomials -- or Pauli operators -- driven by a single variational
angle. Gates carry no parameter index.

A circuit bundles a sequence of those gates with the angle *values* that drive them
(:attr:`Circuit.parameters`), the ``parameter_mapping`` wiring each gate to an angle, and
the reference :attr:`Circuit.initial_state`. Circuits come in a typed family sharing the
same shape and feeding directly into the propagator:

- :class:`MajoranaCircuit` -- native Majorana gates, consumed by
  :class:`~monoprop.majorana_propagator.MajoranaPropagator`.
- :class:`PauliCircuit` -- qubit (Pauli) gates, consumed by
  :class:`~monoprop.pauli_propagator.PauliPropagator`.
- :class:`~monoprop.fermi_data.FermiCircuit` -- fermionic gates converted to Majorana
  gates at construction; a :class:`MajoranaCircuit`, so also consumed by
  :class:`~monoprop.majorana_propagator.MajoranaPropagator`.

:class:`Circuit` itself is an abstract base: construct one of the typed subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .pauli_data import PauliOperator


@dataclass(frozen=True, slots=True)
class Term:
    """One generated Majorana monomial and its structural coefficient.

    Attributes:
        majorana: Majorana indices of the monomial.
        gen_coeff: Generator coefficient ``g`` (the layer angle is
            ``parameters[param] * gen_coeff``).
    """

    majorana: tuple[int, ...]
    gen_coeff: float


@dataclass(frozen=True, slots=True)
class MajoranaGate:
    """The bundle of Majorana monomials driven by a single variational angle.

    A pure generator: it carries only its generated monomials. The angle that drives it is
    assigned by position via the circuit's ``parameter_mapping``, not stored on the gate.

    Attributes:
        terms: The generated monomials.
    """

    terms: tuple[Term, ...]


@dataclass(frozen=True, slots=True)
class PauliGate:
    """A qubit (Pauli) evolution gate; the qubit analogue of :class:`MajoranaGate`.

    ``PauliPropagator`` maps it into a :class:`MajoranaGate` via the Jordan-Wigner
    transform. Like :class:`MajoranaGate` it is a pure generator with no parameter index.

    Attributes:
        qubits: Qubit indices the gate acts on.
        paulis: Commuting Pauli operators applied on ``qubits``.
    """

    qubits: tuple[int, ...]
    paulis: PauliOperator


@dataclass(frozen=True)
class Circuit:
    """Abstract base for a variational circuit: gates, angles, and a reference state.

    Bundles everything the propagator needs to build or evaluate an evolution:

    - ``gates``: the ordered generators.
    - ``parameter_mapping``: the angle index driving each gate (``parameter_mapping[i]``
      drives gate ``i``); ``None`` gives each gate its own distinct angle. Repeat an index
      to tie gates to a shared angle.
    - ``parameters``: the angle *values* (a point in parameter space). Empty means unbound
      -- author the structure now and supply values at evaluation time.
    - ``initial_state``: the reference Slater determinant / computational-basis state.

    Compose circuits with ``+`` (temporal concatenation within the same gate family; the
    right operand's angles are appended on a fresh axis). A *bound* circuit is
    self-consistent: when ``parameters`` is non-empty its length must equal
    :attr:`n_parameters`.

    :class:`Circuit` is abstract -- construct :class:`MajoranaCircuit`,
    :class:`PauliCircuit`, or :class:`~monoprop.fermi_data.FermiCircuit`.

    Attributes:
        gates: The ordered generators.
        parameters: The angle values, or empty for an unbound circuit.
        parameter_mapping: Per-gate angle index, or ``None`` for the identity mapping.
        initial_state: The reference state (occupied mode / qubit indices).
    """

    gates: tuple[MajoranaGate | PauliGate, ...] = ()
    parameters: tuple[float, ...] = ()
    parameter_mapping: tuple[int, ...] | None = None
    initial_state: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize fields to tuples and validate the mapping, params, and gate types."""
        if type(self) is Circuit:
            raise TypeError(
                "Circuit is abstract; construct MajoranaCircuit, PauliCircuit, or "
                "FermiCircuit."
            )
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "parameters", tuple(float(v) for v in self.parameters))
        object.__setattr__(
            self, "initial_state", tuple(int(i) for i in self.initial_state)
        )
        mapping = (
            None
            if self.parameter_mapping is None
            else tuple(int(p) for p in self.parameter_mapping)
        )
        object.__setattr__(self, "parameter_mapping", mapping)
        _resolve_mapping(len(self.gates), mapping)  # validates contiguity + length
        if self.parameters and len(self.parameters) != self.n_parameters:
            raise ValueError(
                f"parameters has {len(self.parameters)} values but the circuit has "
                f"{self.n_parameters} parameters."
            )
        self._validate_gates()

    def _validate_gates(self) -> None:
        """Subclass hook: check every gate is of the family's gate type."""

    @property
    def resolved_mapping(self) -> tuple[int, ...]:
        """Per-gate angle index, with the identity filled in when unset."""
        if self.parameter_mapping is None:
            return tuple(range(len(self.gates)))
        return self.parameter_mapping

    @property
    def n_parameters(self) -> int:
        """Number of distinct variational angles the circuit references."""
        mapping = self.resolved_mapping
        return max(mapping) + 1 if mapping else 0

    def __len__(self) -> int:
        """Number of gates."""
        return len(self.gates)

    def __iter__(self) -> Iterator[MajoranaGate | PauliGate]:
        """Iterate over the gates in application order."""
        return iter(self.gates)

    def __add__(self, other: Circuit) -> Circuit:
        """Concatenate two circuits of the same family, appending ``other``'s angles.

        The result applies ``self``'s gates then ``other``'s; ``other``'s angle indices are
        shifted up by ``self.n_parameters`` so the two halves keep independent angles. Build
        the whole thing in a single :meth:`~monoprop.MajoranaPropagator.build_graph` call to
        avoid the picture-dependent ordering of incremental multi-call building.

        A :class:`PauliCircuit` may only be concatenated with another :class:`PauliCircuit`;
        Majorana-family circuits (:class:`MajoranaCircuit` /
        :class:`~monoprop.fermi_data.FermiCircuit`) concatenate to a :class:`MajoranaCircuit`.
        """
        if not isinstance(other, Circuit):
            return NotImplemented
        if isinstance(self, PauliCircuit) != isinstance(other, PauliCircuit):
            raise TypeError(
                "Cannot concatenate a PauliCircuit with a non-Pauli circuit; the gate "
                "families differ."
            )
        if (
            self.initial_state
            and other.initial_state
            and self.initial_state != other.initial_state
        ):
            raise ValueError(
                "Cannot concatenate circuits with different initial states."
            )
        offset = self.n_parameters
        mapping = self.resolved_mapping + tuple(
            m + offset for m in other.resolved_mapping
        )
        gates = self.gates + other.gates
        parameters = self.parameters + other.parameters
        initial_state = self.initial_state or other.initial_state
        if isinstance(self, PauliCircuit):
            if self.num_qubits != other.num_qubits:
                raise ValueError(
                    "Cannot concatenate PauliCircuits with different qubit counts "
                    f"({self.num_qubits} != {other.num_qubits})."
                )
            return PauliCircuit(
                gates=gates,
                parameters=parameters,
                parameter_mapping=mapping,
                initial_state=initial_state,
                num_qubits=self.num_qubits,
            )
        return MajoranaCircuit(
            gates=gates,
            parameters=parameters,
            parameter_mapping=mapping,
            initial_state=initial_state,
        )


@dataclass(frozen=True)
class MajoranaCircuit(Circuit):
    """A variational circuit of native :class:`MajoranaGate` generators.

    Consumed directly by :class:`~monoprop.majorana_propagator.MajoranaPropagator`.
    """

    def _validate_gates(self) -> None:
        for gate in self.gates:
            if not isinstance(gate, MajoranaGate):
                raise TypeError(
                    f"MajoranaCircuit gates must be MajoranaGate; got "
                    f"{type(gate).__name__}. Use PauliCircuit for PauliGate circuits."
                )

    @classmethod
    def from_dense_arrays(
        cls,
        majoranas: Sequence[Sequence[int]],
        gen_coeffs: Sequence[float],
        param_inds: Sequence[int],
        parameters: Sequence[float] = (),
        initial_state: Sequence[int] = (),
    ) -> MajoranaCircuit:
        """Build a circuit from flat, per-monomial dense arrays.

        This is the native dense/wire format (also the on-disk msgpack-fixture layout):
        consecutive monomials sharing a ``param_ind`` become one :class:`MajoranaGate`, and
        the per-gate ``param_ind`` values become the circuit's parameter mapping, so
        weight-tying is preserved and the expanded engine arrays stay identical to the
        original.

        Args:
            majoranas: One Majorana-index sequence per monomial.
            gen_coeffs: Generator coefficient per monomial.
            param_inds: Variational-angle index per monomial (contiguous runs group into
                gates).
            parameters: Optional angle values.
            initial_state: Optional reference state (occupied mode indices).

        Returns:
            A :class:`MajoranaCircuit` carrying the grouped gates, angle values, mapping,
            and initial state.
        """
        indices = [int(p) for p in param_inds]
        gates: list[MajoranaGate] = []
        mapping: list[int] = []
        current_index: int | None = None
        current_terms: list[Term] = []
        for maj, coeff, pidx in zip(majoranas, gen_coeffs, indices, strict=True):
            if current_terms and pidx != current_index:
                gates.append(MajoranaGate(tuple(current_terms)))
                mapping.append(current_index)  # type: ignore[arg-type]
                current_terms = []
            current_index = pidx
            current_terms.append(Term(tuple(int(i) for i in maj), float(coeff)))
        if current_terms:
            gates.append(MajoranaGate(tuple(current_terms)))
            mapping.append(current_index)  # type: ignore[arg-type]

        return cls(
            gates=tuple(gates),
            parameters=tuple(float(p) for p in parameters),
            parameter_mapping=tuple(mapping),
            initial_state=tuple(int(i) for i in initial_state),
        )


@dataclass(frozen=True)
class PauliCircuit(Circuit):
    """A variational circuit of qubit :class:`PauliGate` generators.

    Consumed directly by :class:`~monoprop.pauli_propagator.PauliPropagator`, which maps the
    gates into the Majorana basis via Jordan-Wigner.

    Attributes:
        num_qubits: Total number of qubits the circuit acts on (not derivable from the
            gates, which touch only the qubits they act on).
    """

    num_qubits: int = field(kw_only=True, default=0)

    def __post_init__(self) -> None:
        """Normalize ``num_qubits`` then run the shared circuit validation."""
        object.__setattr__(self, "num_qubits", int(self.num_qubits))
        super().__post_init__()

    def _validate_gates(self) -> None:
        for gate in self.gates:
            if not isinstance(gate, PauliGate):
                raise TypeError(
                    f"PauliCircuit gates must be PauliGate; got {type(gate).__name__}. "
                    "Use MajoranaCircuit for MajoranaGate circuits."
                )


def validate_parameter_mapping(
    mapping: Sequence[int], expected_len: int, unit: str = "gates"
) -> None:
    """Validate that a parameter mapping has the right length and contiguous indices.

    Args:
        mapping: Per-unit angle indices to validate.
        expected_len: Number of entries the mapping must have.
        unit: Noun naming what each entry covers (e.g. ``"gates"`` or ``"graph layers"``),
            used only in the length-mismatch error message.

    Raises:
        ValueError: If the mapping length does not match ``expected_len`` or its indices are
            not contiguous ``0..max`` (an index gap would silently invent a phantom
            parameter).
    """
    if len(mapping) != expected_len:
        raise ValueError(
            f"parameter_mapping has {len(mapping)} entries but there are "
            f"{expected_len} {unit}."
        )
    if mapping and set(mapping) != set(range(max(mapping) + 1)):
        raise ValueError(
            "parameter_mapping indices must be contiguous 0..n-1 with no gaps; "
            f"got {sorted(set(mapping))}."
        )


def _resolve_mapping(
    n_gates: int, parameter_mapping: Sequence[int] | None
) -> list[int]:
    """Resolve and validate a per-gate parameter mapping.

    Args:
        n_gates: Number of gates the mapping must cover.
        parameter_mapping: Per-gate angle index, or ``None`` for the identity mapping
            (one distinct angle per gate).

    Returns:
        The mapping as a list of ints, one per gate.

    Raises:
        ValueError: If the mapping length does not match ``n_gates`` or its indices are not
            contiguous ``0..max`` (an index gap would silently invent a phantom parameter).
    """
    if parameter_mapping is None:
        return list(range(n_gates))
    mapping = [int(p) for p in parameter_mapping]
    validate_parameter_mapping(mapping, n_gates, "gates")
    return mapping


def expand_monomials(
    gates: Sequence[MajoranaGate],
    mapping: Sequence[int],
) -> tuple[list[tuple[int, ...]], list[float], list[int], list[int]]:
    """Flatten gates + an already-resolved per-gate mapping into per-monomial arrays.

    Args:
        gates: Majorana gates, in application order.
        mapping: The angle index driving each gate (one entry per gate).

    Returns:
        A tuple ``(majoranas, gen_coeffs, parameter_mapping, gate_indices)`` for the C++
        engine, expanded per monomial. ``gate_indices[i]`` is the (local, 0-based) index of
        the authoring gate monomial ``i`` came from, so the engine can recover gate
        boundaries; monomials from a multi-term gate share one gate index.
    """
    majoranas: list[tuple[int, ...]] = []
    gen_coeffs: list[float] = []
    per_monomial: list[int] = []
    gate_indices: list[int] = []
    for gate_index, (gate, param) in enumerate(zip(gates, mapping, strict=True)):
        for term in gate.terms:
            majoranas.append(tuple(term.majorana))
            gen_coeffs.append(float(term.gen_coeff))
            per_monomial.append(param)
            gate_indices.append(gate_index)
    return majoranas, gen_coeffs, per_monomial, gate_indices
