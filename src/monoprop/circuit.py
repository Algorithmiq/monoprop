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

"""Authoring types for Majorana/qubit gate sequences.

A :class:`MajoranaGate` (or its qubit analogue :class:`PauliGate`) is a *pure generator*:
the bundle of Majorana monomials -- or Pauli operators -- driven by a single variational
angle. Gates carry no parameter index. Which variational angle drives each gate is a
property of the *ansatz*, expressed separately as a ``parameter_mapping`` argument to the
propagator (``parameter_mapping[i]`` is the angle index driving gate ``i``; ``None`` means
one distinct angle per gate). Sharing an angle across gates is expressed by repeating an
index in that mapping. Angle *values* are a plain sequence of floats indexed the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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


def combine_parameters(
    first: Iterable[float],
    second: Iterable[float],
    *,
    schrodinger: bool = False,
) -> list[float]:
    """Stitch two independently-indexed circuit halves into one parameter vector.

    Use this when a circuit is split into two chunks that are each authored with their own
    ``0``-based parameter indices and rebuilt across two
    :meth:`~monoprop.MajoranaPropagator.propagate_build_graph` calls. ``first`` holds the
    angles of the temporally-first chunk, ``second`` the second. The result is ordered to
    match the propagator's parameter axis when the chunks are fed in the order that
    reproduces a single-call evolution:

    - Schrodinger picture (``schrodinger=True``): gates apply front-to-back, so the chunks
      are fed in order (first, then second) and the axis is ``[*first, *second]``.
    - Heisenberg picture (``schrodinger=False``, the default): each call applies its chunk
      back-to-front, so reproducing a single call feeds the chunks *reversed* (second, then
      first) and the axis is ``[*second, *first]``.

    Args:
        first: The temporally-first chunk's angle values.
        second: The second chunk's angle values.
        schrodinger: Whether the propagator is in the Schrodinger picture.

    Returns:
        The combined angle values in parameter-axis order.
    """
    first_vals = [float(v) for v in first]
    second_vals = [float(v) for v in second]
    lead, tail = (first_vals, second_vals) if schrodinger else (second_vals, first_vals)
    return [*lead, *tail]


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
    if len(mapping) != n_gates:
        raise ValueError(
            f"parameter_mapping has {len(mapping)} entries but there are {n_gates} gates."
        )
    if mapping and set(mapping) != set(range(max(mapping) + 1)):
        raise ValueError(
            "parameter_mapping indices must be contiguous 0..n-1 with no gaps; "
            f"got {sorted(set(mapping))}."
        )
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


def to_engine_arrays(
    gates: Sequence[MajoranaGate],
    parameter_mapping: Sequence[int] | None = None,
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Compile Majorana gates into the dense engine arrays (0-based, one-shot).

    Resolves ``parameter_mapping`` in the *local* 0-based sense (the mapping must be
    contiguous ``0..n-1``) — as used by :meth:`~monoprop.MajoranaPropagator.propagate`. For
    graph accumulation, where indices extend a global axis, see
    :func:`expand_monomials`.

    Args:
        gates: Majorana gates to compile, in application order.
        parameter_mapping: Per-gate angle index (``parameter_mapping[i]`` drives gate ``i``),
            or ``None`` for the identity mapping (one distinct angle per gate). Must be
            contiguous ``0..n-1``.

    Returns:
        A tuple ``(majoranas, gen_coeffs, parameter_mapping)`` for the C++ engine, where the
        returned ``parameter_mapping`` is expanded per monomial.
    """
    mapping = _resolve_mapping(len(gates), parameter_mapping)
    return expand_monomials(gates, mapping)
