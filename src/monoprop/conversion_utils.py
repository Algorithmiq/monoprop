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

"""Conversion tools module."""

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def _extend_pauli_string(pauli: str, qubits: Sequence[int], n_qubits: int) -> str:
    """Pad a local Pauli term with identities into a full ``n_qubits``-wide Pauli string."""
    if len(pauli) != len(qubits):
        raise ValueError("Pauli string and qubits must have the same length")

    pauli_list = ["I"] * n_qubits
    for p, qubit in zip(pauli, qubits, strict=True):
        pauli_list[qubit] = p
    return "".join(pauli_list)


def _pauli_to_majorana(pauli: str) -> tuple[tuple[int, ...], complex]:
    """Jordan-Wigner map a full-width Pauli string to a ``(Majorana indices, phase)`` pair."""
    # Scans right to left. A JW Majorana is Z...ZX or Z...ZY, so which index each letter emits
    # depends only on whether a Z string is still pending, tracked in flag_z.
    new_p = []
    flag_z = False
    coeff = 1 + 0j
    for i, p in reversed(list(enumerate(pauli))):
        if (p, flag_z) in {("Z", False), ("I", True)}:
            new_p.extend([2 * i + 1, 2 * i])
            coeff *= -1j  # cause Y X = -i Z and Z Y X =  Z * -i * Z
        elif p == "X" and not flag_z:
            new_p.append(2 * i)
            flag_z = True
        elif p == "X" and flag_z:
            new_p.append(2 * i + 1)
            flag_z = False
            coeff *= -1j  # cause Z Y = - i X
        elif p == "Y" and not flag_z:
            new_p.append(2 * i + 1)
            flag_z = True
        elif p == "Y" and flag_z:
            new_p.append(2 * i)
            flag_z = False
            coeff *= 1j  # cause Z X =  i X
    # Reversing needs no phase fix: the loop multiplied by phases rather than dividing.
    return tuple(reversed(new_p)), coeff


def _pauli_to_local_slots(string: str, qubits: Sequence[int]) -> tuple[int, ...]:
    """Pack a local Pauli term into native symplectic slots (no Jordan-Wigner string).

    Each qubit maps to its own two slots, independent of every other qubit:
    ``X_q -> {2q}``, ``Y_q -> {2q+1}``, ``Z_q -> {2q, 2q+1}`` (``I`` contributes nothing).
    This is the engine's ``Basis::Pauli`` encoding (mirrors ``slots_of_string`` in
    ``tests/cpp/PauliTestOracle.h``); a weight-``w`` Pauli occupies at most ``2w``
    slots, so the packed popcount is ``O(weight)`` and independent of the qubit count -- unlike
    the Jordan-Wigner image (``_pauli_to_majorana``), whose ``Z`` prefix makes a single
    ``X_q`` span ``2q+1`` slots.

    Returns:
        The sorted tuple of slot indices encoding the term.
    """
    slots: list[int] = []
    for letter, qubit in zip(string, qubits, strict=True):
        if letter == "X":
            slots.append(2 * qubit)
        elif letter == "Y":
            slots.append(2 * qubit + 1)
        elif letter == "Z":
            slots.extend((2 * qubit, 2 * qubit + 1))
    return tuple(sorted(slots))


def _local_slots_to_pauli(slots: Sequence[int]) -> tuple[str, tuple[int, ...]]:
    """Decode native symplectic slots back to a local Pauli term.

    Inverse of ``_pauli_to_local_slots`` (mirrors ``letter_from_bitset`` in
    ``tests/cpp/PauliTestOracle.h``): for qubit ``q`` the slots ``2q`` (``u``) and
    ``2q+1`` (``v``) decode as ``(1,0)=X``, ``(0,1)=Y``, ``(1,1)=Z``.

    Returns:
        A ``(string, qubits)`` pair ready for [Pauli][monoprop.pauli.Pauli].
    """
    present = set(slots)
    letters: list[str] = []
    acting_qubits: list[int] = []
    for qubit in sorted({s // 2 for s in slots}):
        u = 2 * qubit in present
        v = 2 * qubit + 1 in present
        if u and not v:
            letters.append("X")
        elif v and not u:
            letters.append("Y")
        else:  # u and v
            letters.append("Z")
        acting_qubits.append(qubit)
    return "".join(letters), tuple(acting_qubits)


def _parity(perm: Sequence[int]) -> int:
    r"""Compute parity of a permutation.

    Returns:
        The value of $(-1)^{\sigma}$.

    Example:
        ```python
        >>> _parity([1, 2, 3, 4])
        1
        >>> _parity([2, 1, 3])
        -1

        ```
    """
    parity: int = 1
    for i, x in enumerate(perm):
        for y in perm[i + 1 :]:
            parity *= -1 if (x > y) else 1

    return parity


def _remove_repeated_pairs(term: tuple[int, ...]) -> tuple[int, ...]:
    """Cancel adjacent duplicate indices (``m_i m_i = 1``) in a sorted index tuple."""
    mut_term = list(term)
    i = 0
    while i < len(mut_term) - 1:
        if mut_term[i] == mut_term[i + 1]:
            mut_term[i] = mut_term[i + 1] = -1
            i += 2
        else:
            i += 1
    return tuple(el for el in mut_term if el != -1)


def _n_product(
    term: list[tuple[int, str]] | list[tuple[int, int]],
    plus_inds: list[int],
    minus_inds: list[int],
) -> Iterator[tuple[tuple[int, ...], complex]]:
    """Expand a product of ladder operators into its ``(Majorana indices, coefficient)`` terms."""
    ind: tuple[int, ...]
    term_len = len(term)
    for ind in it.product(*[[2 * el[0], 2 * el[0] + 1] for el in term]):
        plus_parity = sum(ind[i] % 2 for i in plus_inds)
        minus_parity = sum(ind[i] % 2 for i in minus_inds)
        coeff = (
            _parity(ind) * ((-1j) ** plus_parity) * ((1j) ** minus_parity) / 2**term_len
        )
        yield _remove_repeated_pairs(tuple(sorted(ind))), coeff
