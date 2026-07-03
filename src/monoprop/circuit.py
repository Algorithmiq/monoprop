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

A :class:`Circuit` bundles a sequence of those gates with the angle *values* that drive
them (:attr:`Circuit.parameters`), the ``parameter_mapping`` wiring each gate to an angle,
and the reference :attr:`Circuit.initial_state`. The propagator consumes a ``Circuit``;
domain types (Majorana/Pauli/Fermi) build one via ``to_circuit()``.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    assigned by position via the propagator's ``parameter_mapping`` argument, not stored on
    the gate.

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


@dataclass(frozen=True, slots=True)
class Circuit:
    """A variational circuit: gates, the angles that drive them, and a reference state.

    Bundles everything the propagator needs to build or evaluate an evolution:

    - ``gates``: the ordered generators (:class:`MajoranaGate` or :class:`PauliGate`).
    - ``parameter_mapping``: the angle index driving each gate (``parameter_mapping[i]``
      drives gate ``i``); ``None`` gives each gate its own distinct angle. Repeat an index
      to tie gates to a shared angle.
    - ``parameters``: the angle *values* (a point in parameter space). Empty means unbound
      -- author the structure now and supply values at evaluation time.
    - ``initial_state``: the reference Slater determinant / computational-basis state.

    Compose circuits with ``+`` (temporal concatenation; the right operand's angles are
    appended on a fresh axis). A *bound* circuit is self-consistent: when ``parameters`` is
    non-empty its length must equal :attr:`n_parameters`.

    Attributes:
        gates: The ordered generators.
        parameters: The angle values, or empty for an unbound circuit.
        parameter_mapping: Per-gate angle index, or ``None`` for the identity mapping.
        initial_state: The reference state (occupied mode / qubit indices).
    """

    gates: tuple[MajoranaGate | PauliGate, ...]
    parameters: tuple[float, ...] = ()
    parameter_mapping: tuple[int, ...] | None = None
    initial_state: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize fields to tuples and validate the mapping and parameter length."""
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
        """Concatenate two circuits, appending ``other``'s angles on a fresh axis.

        The result applies ``self``'s gates then ``other``'s; ``other``'s angle indices are
        shifted up by ``self.n_parameters`` so the two halves keep independent angles. Build
        the whole thing in a single :meth:`~monoprop.MajoranaPropagator.build_graph`
        call to avoid the picture-dependent ordering of incremental multi-call building.
        """
        if not isinstance(other, Circuit):
            return NotImplemented
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
        return Circuit(
            gates=self.gates + other.gates,
            parameters=self.parameters + other.parameters,
            parameter_mapping=mapping,
            initial_state=self.initial_state or other.initial_state,
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
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Flatten gates + an already-resolved per-gate mapping into per-monomial arrays.

    Args:
        gates: Majorana gates, in application order.
        mapping: The angle index driving each gate (one entry per gate).

    Returns:
        A tuple ``(majoranas, gen_coeffs, parameter_mapping)`` for the C++ engine, expanded
        per monomial.
    """
    majoranas: list[tuple[int, ...]] = []
    gen_coeffs: list[float] = []
    per_monomial: list[int] = []
    for gate, param in zip(gates, mapping, strict=True):
        for term in gate.terms:
            majoranas.append(tuple(term.majorana))
            gen_coeffs.append(float(term.gen_coeff))
            per_monomial.append(param)
    return majoranas, gen_coeffs, per_monomial
