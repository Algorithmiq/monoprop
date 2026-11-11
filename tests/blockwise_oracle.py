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

"""Dict-based Heisenberg reference for standard and blockwise Majorana truncation.

Independent of the engine: it works on the Hermitian operator algebra (complex coefficients on
sorted Majorana strings) rather than the engine's encoded real coefficients, and implements the
blockwise rule literally -- no structural cutoff inside a block, and at each block end every
monomial created during the block that is unpaired and over the cutoff is dropped. Standard
truncation is the same rule with every layer its own block.

Conventions, as monoprop documents them: a gate applies ``exp(+i theta H)`` for a Hermitian
generator ``H`` (one layer per generator term, in ``generator.terms`` order), the Heisenberg sweep
conjugates the observable by the last gate first, ``a_j = (gamma_{2j} + i gamma_{2j+1}) / 2`` so
``gamma_{2j} gamma_{2j+1} = i (1 - 2 n_j)``, and ``initial_state`` lists the occupied modes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from monoprop import Circuit, MajoranaOperator

Term = tuple[int, ...]


def _product(left: Term, right: Term) -> tuple[Term, int]:
    """Return ``(string, sign)`` with ``gamma_left gamma_right = sign * gamma_string``."""
    swaps = sum(1 for a in left for b in right if a > b)
    return tuple(sorted(set(left) ^ set(right))), -1 if swaps % 2 else 1


def _anticommute(left: Term, right: Term) -> bool:
    overlap = len(set(left) & set(right))
    return (len(left) * len(right) - overlap) % 2 == 1


def is_paired(term: Term) -> bool:
    """True if every mode holds both of its Majoranas or neither."""
    modes = [i // 2 for i in term]
    return all(modes.count(m) == 2 for m in modes)


def complexity(term: Term, cutoff_type: str) -> int:
    """Length (Majorana count) or support (distinct modes) of a monomial."""
    if cutoff_type == "length":
        return len(term)
    return len({i // 2 for i in term})


def passes(term: Term, cutoff: int, cutoff_type: str) -> bool:
    """Eq. 4: kept iff fully paired or within the cutoff."""
    return is_paired(term) or complexity(term, cutoff_type) <= cutoff


def _layers(
    circuit: Circuit, parameters: Sequence[float]
) -> list[tuple[int, Term, complex, float]]:
    """``(gate position, string, Hermitian coefficient, angle)`` per layer, in circuit order."""
    out = []
    for position, (gate, index) in enumerate(
        zip(circuit.gates, circuit.resolved_mapping, strict=True)
    ):
        for term, coeff in gate.generator.terms.items():
            if term:
                string: Term = tuple(term)  # type: ignore[arg-type]
                out.append((position, string, complex(coeff), float(parameters[index])))
    return out


def _rotate(
    op: dict[Term, complex], string: Term, coeff: complex, theta: float
) -> dict[Term, complex]:
    """Conjugate ``op`` by ``exp(i theta h)`` with ``h = coeff * gamma_string`` Hermitian.

    For a term anticommuting with ``h``, ``U^dag O U = O U^2`` with
    ``U^2 = cos(2 theta |c|) + i sin(2 theta |c|) h / |c|``. Every partner is emitted; the caller
    filters what it keeps.
    """
    magnitude = abs(coeff)
    cos_v = math.cos(2.0 * theta * magnitude)
    sin_unit = 1j * math.sin(2.0 * theta * magnitude) * coeff / magnitude
    out: dict[Term, complex] = {}
    for term, value in op.items():
        if not _anticommute(term, string):
            out[term] = out.get(term, 0.0) + value
            continue
        out[term] = out.get(term, 0.0) + value * cos_v
        partner, sign = _product(term, string)
        out[partner] = out.get(partner, 0.0) + value * sin_unit * sign
    return out


def evolve(
    operator: MajoranaOperator,
    circuit: Circuit,
    parameters: Sequence[float],
    *,
    cutoff: int,
    cutoff_type: str = "length",
    blocks: Sequence[Hashable] | str | None = None,
    per_layer_blocks: Sequence[Hashable] | None = None,
    upper_atol: float | None = None,
    naive_regeneration: bool = False,
    stats: dict[str, int] | None = None,
) -> dict[Term, complex]:
    """Heisenberg-evolve ``operator`` through ``circuit`` with standard or blockwise truncation.

    Args:
        operator: Initial observable; its terms are never truncated.
        circuit: Majorana-family circuit.
        parameters: Angle values, indexed by the circuit's mapping.
        cutoff: Structural cutoff w*.
        cutoff_type: ``"length"`` or ``"support"``.
        blocks: One label per gate (contiguous runs), ``"gate"`` for one block per gate, or
            ``None`` for standard truncation (every layer its own block).
        per_layer_blocks: One label per non-identity layer instead of per gate.
        upper_atol: At a block end, also keep a new term with ``|coeff| >= upper_atol``.
        naive_regeneration: Model the bug blockwise truncation must avoid: a dropped monomial
            stays dead, so any later contribution to it is lost instead of starting a fresh term.
        stats: If given, receives ``dropped`` (terms dropped at block ends) and ``regenerated``
            (dropped monomials that a later layer produced again).

    Returns:
        The evolved operator as ``{sorted Majorana tuple: complex coefficient}``.
    """
    layers = _layers(circuit, parameters)
    if per_layer_blocks is not None:
        labels = list(per_layer_blocks)
    elif blocks is None:
        labels = list(range(len(layers)))
    elif blocks == "gate":
        labels = [position for position, *_ in layers]
    else:
        labels = [blocks[position] for position, *_ in layers]

    def kept(term: Term, value: complex) -> bool:
        if passes(term, cutoff, cutoff_type):
            return True
        return upper_atol is not None and abs(value) >= upper_atol

    op = {tuple(term): complex(value) for term, value in operator.terms.items()}
    present_at_block_start = set(op)
    dropped: set[Term] = set()
    regenerated: set[Term] = set()
    for k in range(len(layers) - 1, -1, -1):
        _, string, coeff, theta = layers[k]
        op = _rotate(op, string, coeff, theta)
        regenerated |= dropped & set(op)
        if naive_regeneration:
            op = {term: value for term, value in op.items() if term not in dropped}
        if k == 0 or labels[k - 1] != labels[k]:
            survivors = {
                term: value
                for term, value in op.items()
                if term in present_at_block_start or kept(term, value)
            }
            dropped |= set(op) - set(survivors)
            op = survivors
            present_at_block_start = set(op)
    if stats is not None:
        stats["dropped"] = len(dropped)
        stats["regenerated"] = len(regenerated)
    return op


def expectation(op: dict[Term, complex], initial_state: Sequence[int]) -> float:
    """``<psi|op|psi>`` for the Fock state with ``initial_state`` occupied."""
    occupied = set(initial_state)
    total = 0.0j
    for term, value in op.items():
        if not is_paired(term):
            continue
        phase = 1.0 + 0.0j
        for j in range(0, len(term), 2):
            phase *= 1j * (-1.0 if term[j] // 2 in occupied else 1.0)
        total += value * phase
    return total.real


def energy(
    operator: MajoranaOperator,
    circuit: Circuit,
    parameters: Sequence[float],
    initial_state: Sequence[int],
    **kwargs: Any,
) -> float:
    """Expectation value of the truncated evolved operator; ``kwargs`` go to [evolve][]."""
    return expectation(evolve(operator, circuit, parameters, **kwargs), initial_state)


def central_difference_gradient(
    operator: MajoranaOperator,
    circuit: Circuit,
    parameters: Sequence[float],
    initial_state: Sequence[int],
    step: float = 1e-6,
    **kwargs: Any,
) -> list[float]:
    """Finite-difference gradient of [energy][]; the truncation pattern is angle-independent."""
    grad = []
    for i in range(len(parameters)):
        plus = list(parameters)
        minus = list(parameters)
        plus[i] += step
        minus[i] -= step
        grad.append(
            (
                energy(operator, circuit, plus, initial_state, **kwargs)
                - energy(operator, circuit, minus, initial_state, **kwargs)
            )
            / (2 * step)
        )
    return grad
