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

"""Dict-based reference for standard, blockwise and max-sins Majorana truncation, in both pictures.

Independent of the engine: it works on the Hermitian operator algebra (complex coefficients on
sorted Majorana strings) rather than the engine's encoded real coefficients, and implements the
blockwise rule literally -- no structural cutoff inside a block, and at each block end every
monomial created during the block that is unpaired and over the cutoff is dropped. Standard
truncation is the same rule with every layer its own block.

Max-sins truncation is layered on top: every term carries the fewest sine factors on any path that
reached it (initial terms 0). In each layer a rotation between two present terms keeps both and
gives each the smaller of its own count and its partner's plus one; an absent partner takes its
source's count plus one and is created only if that is at most ``max_sins``. In Schrodinger a
paired partner is part of the reference state, so it counts as present with count 0.

Conventions, as monoprop documents them: a gate applies ``exp(+i theta H)`` for a Hermitian
generator ``H`` (one layer per generator term, in ``generator.terms`` order), the Heisenberg sweep
conjugates the observable by the last gate first, ``a_j = (gamma_{2j} + i gamma_{2j+1}) / 2`` so
``gamma_{2j} gamma_{2j+1} = i (1 - 2 n_j)``, and ``initial_state`` lists the occupied modes.

The Schrodinger picture evolves the reference state, scaled like the engine's state vector to
``2^N rho = prod_j (1 - i s_j gamma_{2j} gamma_{2j+1})`` (``s_j = 1 - 2 n_j``, so every paired
coefficient has magnitude 1 and ``upper_atol`` compares alike), forward, first gate first, and
contracts it with the observable by the trace. ``U rho U^dag`` is the Heisenberg conjugation at the negated angle. As in
the engine, the state starts with its paired terms of up to ``ceil(schrodinger_cutoff / 2)`` pairs,
and a paired term a rotation first reaches enters with its reference value rather than 0.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

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
    op: dict[Term, complex],
    string: Term,
    coeff: complex,
    theta: float,
    fresh_paired: Callable[[Term], complex] | None = None,
    frozen_pair: Callable[[Term, complex, Term, complex], bool] | None = None,
    blocked: set[Term] | None = None,
) -> dict[Term, complex]:
    """Conjugate ``op`` by ``exp(i theta h)`` with ``h = coeff * gamma_string`` Hermitian.

    For a term anticommuting with ``h``, ``U^dag O U = O U^2`` with
    ``U^2 = cos(2 theta |c|) + i sin(2 theta |c|) h / |c|``. Every partner is emitted; the caller
    filters what it keeps. With ``fresh_paired``, an absent paired partner first enters with that
    value, so it rotates as if it had been present all along. ``frozen_pair(term, value, partner,
    partner_value)``, for two present terms, withholds their exchange (both are only cos-scaled).
    A partner in ``blocked`` is never created (max-sins); its source is only cos-scaled.
    """
    if fresh_paired is not None:
        op = dict(op)
        for term in list(op):
            if _anticommute(term, string):
                partner, _ = _product(term, string)
                if partner not in op and is_paired(partner):
                    op[partner] = fresh_paired(partner)
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
        if (
            frozen_pair is not None
            and partner in op
            and frozen_pair(term, value, partner, op[partner])
        ):
            continue
        if blocked is not None and partner in blocked:
            continue
        out[partner] = out.get(partner, 0.0) + value * sin_unit * sign
    return out


def _emit_cut(
    abs_sin: float, cutoff: int, cutoff_type: str, upper_atol: float | None
) -> Callable[[Term, complex, Term, complex], bool]:
    """The engine's emit-time cut on a pair of present terms in a layer that applies the cutoff.

    Each term emits its partner unless the partner fails the cutoff and the sine term
    ``|sin| * |value|`` stays below ``upper_atol``; one emission records the full exchange.
    """

    def frozen(
        term: Term, value: complex, partner: Term, partner_value: complex
    ) -> bool:
        if passes(term, cutoff, cutoff_type) or passes(partner, cutoff, cutoff_type):
            return False
        largest = max(abs(value), abs(partner_value))
        return upper_atol is None or abs_sin * largest < upper_atol

    return frozen


def _sin_step(
    op: dict[Term, complex],
    string: Term,
    counts: dict[Term, int],
    max_sins: int,
    inc: int,
    *,
    paired_is_initial: bool,
    frozen_pair: Callable[[Term, complex, Term, complex], bool] | None,
) -> set[Term]:
    """Update ``counts`` in place for one layer and return the partners max-sins forbids creating.

    Reads pre-layer counts only: within a layer each term belongs to at most one rotation pair.
    """
    before = dict(counts)
    blocked: set[Term] = set()
    for term, value in op.items():
        if not _anticommute(term, string):
            continue
        partner, _ = _product(term, string)
        mine = before.get(term, 0)
        if partner in op:
            if frozen_pair is not None and frozen_pair(
                term, value, partner, op[partner]
            ):
                continue
            counts[term] = min(counts.get(term, mine), before.get(partner, 0) + inc)
        elif paired_is_initial and is_paired(partner):
            counts[partner] = 0
            counts[term] = min(counts.get(term, mine), inc)
        elif inc == 0 or mine + inc <= max_sins:
            counts[partner] = mine + inc
        else:
            blocked.add(partner)
    return blocked


def evolve(
    operator: MajoranaOperator | dict[Term, complex],
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
    schrodinger: bool = False,
    fresh_paired: Callable[[Term], complex] | None = None,
    engine_block_end: bool = False,
    max_sins: int | None = None,
    count_sins: bool = True,
    sin_counts: dict[Term, int] | None = None,
) -> dict[Term, complex]:
    """Evolve ``operator`` through ``circuit`` with standard or blockwise truncation.

    Args:
        operator: Initial observable (Heisenberg) or state (Schrodinger, see
            [initial_state_terms][]); its terms are never truncated.
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
        schrodinger: Apply ``U . U^dag`` first gate first, so a block ends at its last layer,
            instead of the Heisenberg ``U^dag . U`` last gate first.
        fresh_paired: Value an absent paired term enters with when a rotation first reaches it
            (the Schrodinger state's lazily materialized reference terms); ``None`` means 0.
        engine_block_end: Model the engine's emit-time cut in a block's last layer instead of the
            literal rule: two present terms both over the cutoff exchange weight there only if one
            of their sine terms reaches ``upper_atol``. Without ``upper_atol``, this differs from
            the literal rule only for unpaired over-cutoff terms present before the block.
        max_sins: Max-sins truncation limit (see the module docstring), or ``None``.
        count_sins: ``False`` makes every sine free (a partner inherits its source's count).
        sin_counts: The per-term counts, read and updated in place so a later call continues
            them; terms absent from it count 0. Only used with ``max_sins``.

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

    terms = operator if isinstance(operator, dict) else operator.terms
    op = {tuple(term): complex(value) for term, value in terms.items()}
    counts = sin_counts if sin_counts is not None else {}
    present_at_block_start = set(op)
    dropped: set[Term] = set()
    regenerated: set[Term] = set()
    n = len(layers)
    for k in range(n) if schrodinger else range(n - 1, -1, -1):
        _, string, coeff, theta = layers[k]
        angle = -theta if schrodinger else theta
        neighbour = k + 1 if schrodinger else k - 1
        block_end = not 0 <= neighbour < n or labels[neighbour] != labels[k]
        frozen_pair = None
        if engine_block_end and block_end:
            frozen_pair = _emit_cut(
                abs(math.sin(2.0 * angle * abs(coeff))), cutoff, cutoff_type, upper_atol
            )
        blocked = None
        if max_sins is not None:
            # A pair whose exchange is withheld merges no counts either.
            blocked = _sin_step(
                op,
                string,
                counts,
                max_sins,
                1 if count_sins else 0,
                paired_is_initial=fresh_paired is not None,
                frozen_pair=frozen_pair,
            )
        op = _rotate(op, string, coeff, angle, fresh_paired, frozen_pair, blocked)
        regenerated |= dropped & set(op)
        if naive_regeneration:
            op = {term: value for term, value in op.items() if term not in dropped}
        if block_end:
            survivors = {
                term: value
                for term, value in op.items()
                if term in present_at_block_start or kept(term, value)
            }
            dropped |= set(op) - set(survivors)
            op = survivors
            present_at_block_start = set(op)
            if max_sins is not None:
                for term in [t for t in counts if t not in op]:
                    del counts[term]
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


def reference_coefficient(term: Term, initial_state: Sequence[int]) -> complex:
    """Coefficient of the paired ``term`` in the expansion of ``2^N rho``."""
    occupied = set(initial_state)
    value = 1.0 + 0.0j
    for j in {i // 2 for i in term}:
        value *= -1j * (-1.0 if j in occupied else 1.0)
    return value


def initial_state_terms(
    num_modes: int, initial_state: Sequence[int], schrodinger_cutoff: int
) -> dict[Term, complex]:
    """``2^N rho``'s Majorana expansion, cut to ``ceil(schrodinger_cutoff / 2)`` pairs."""
    capped = min(schrodinger_cutoff, 2 * num_modes)
    max_pairs = capped // 2 + capped % 2
    out: dict[Term, complex] = {}
    for pairs in range(max_pairs + 1):
        for modes in itertools.combinations(range(num_modes), pairs):
            term = tuple(i for j in modes for i in (2 * j, 2 * j + 1))
            out[term] = reference_coefficient(term, initial_state)
    return out


def trace_contraction(state: dict[Term, complex], operator: MajoranaOperator) -> float:
    """``Tr[rho operator]`` for ``state = 2^N rho``, as ``Tr[gamma_t gamma_t] = 2^N (-1)^(k(k-1)/2)``."""
    total = 0.0j
    for term, value in operator.terms.items():
        key = tuple(term)
        if key in state:
            k = len(key)
            total += (
                state[key] * complex(value) * (-1.0 if (k * (k - 1) // 2) % 2 else 1.0)
            )
    return total.real


def energy(
    operator: MajoranaOperator,
    circuit: Circuit,
    parameters: Sequence[float],
    initial_state: Sequence[int],
    *,
    schrodinger_cutoff: int | None = None,
    **kwargs: Any,
) -> float:
    """Expectation value under truncated evolution; ``kwargs`` go to [evolve][].

    With ``schrodinger_cutoff`` the state is evolved (Schrodinger picture), otherwise the operator.
    """
    if schrodinger_cutoff is None:
        return expectation(
            evolve(operator, circuit, parameters, **kwargs), initial_state
        )
    num_modes = operator.num_modes
    state = initial_state_terms(num_modes, initial_state, schrodinger_cutoff)
    evolved = evolve(
        state,
        circuit,
        parameters,
        schrodinger=True,
        fresh_paired=lambda term: reference_coefficient(term, initial_state),
        **kwargs,
    )
    return trace_contraction(evolved, operator)


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
