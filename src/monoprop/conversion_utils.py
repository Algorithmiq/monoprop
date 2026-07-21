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
    """Extend a Pauli string to a full Pauli string including identities.

    Args:
        pauli: String representing a shrunken Pauli operator.
        qubits: Qubits specifying where the Pauli operators are applied.
        n_qubits: Number of qubits in the full system.

    Returns:
        Extended Pauli string to a n_qubits-system.
    """
    if len(pauli) != len(qubits):
        raise ValueError("Pauli string and qubits must have the same length")

    pauli_list = ["I"] * n_qubits
    for p, qubit in zip(pauli, qubits, strict=True):
        pauli_list[qubit] = p
    return "".join(pauli_list)


def _pauli_to_majorana(pauli: str) -> tuple[tuple[int, ...], complex]:
    # Jordan-Wigner map a full-width Pauli string to a Majorana monomial: returns the
    # (Majorana index tuple, phase coefficient) pair, not a fermionic operator.
    # the algorithms starts with last qubit and checks it's Pauli. Knowing, that
    # JW Majoranas are Z...ZX or Z...ZY, we can determine Majorana to be added
    # by tracking if currently the Z should be applied and what we are seeing
    new_p = []
    flag_z = False  # flag to keep track of the last Z
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
    # no need to fix coeff for reversing because we were multiplying by phases above, not dividing
    return tuple(reversed(new_p)), coeff


def _majorana_to_pauli(
    majorana: tuple[int, ...], *, n_qubits: int
) -> tuple[str, complex]:
    """Invert :func:`_pauli_to_majorana` for a fixed system width.

    Args:
        majorana: Majorana-index tuple.
        n_qubits: expected number of qubits.

    Returns:
        A tuple ``(pauli, coeff)`` where ``pauli`` maps back to ``majorana`` under
        :func:`_pauli_to_majorana`, and ``coeff`` is the complex conjugate of the
        coefficient returned by :func:`_pauli_to_majorana` for that ``pauli``.
    """
    majorana_set = set(majorana)
    if any(el >= 2 * n_qubits for el in majorana_set):
        raise ValueError(
            f"Majorana indices {majorana} out of range for {n_qubits} qubits"
        )

    pauli = ["I"] * n_qubits
    flag_z = False
    forward_coeff = 1 + 0j

    for i in range(n_qubits - 1, -1, -1):
        has_even = (2 * i) in majorana_set
        has_odd = (2 * i + 1) in majorana_set

        if not flag_z:
            if has_even and has_odd:
                pauli[i] = "Z"
                forward_coeff *= -1j
            elif has_even and not has_odd:
                pauli[i] = "X"
                flag_z = True
            elif not has_even and has_odd:
                pauli[i] = "Y"
                flag_z = True
            # the only possibility left is has_even == has_odd == False but this is just
            # setting identity again
        elif has_even and has_odd:
            # pauli[i] = "I" but this is already set
            forward_coeff *= -1j
        elif not has_even and not has_odd:
            pauli[i] = "Z"
        elif has_even:
            pauli[i] = "Y"
            flag_z = False
            forward_coeff *= 1j
        else:
            pauli[i] = "X"
            flag_z = False
            forward_coeff *= -1j

    return "".join(pauli), forward_coeff.conjugate()


def _parity(perm: Sequence[int]) -> int:
    r"""Compute parity of a permutation.

    Args:
        perm: sequence of integers.

    Returns:
        The value of $(-1)^{\sigma}$.

    Example:
        ```python
        >>> _parity([1, 2, 3, 4])
        1
        >>> _parity([2, 1, 3])
        -1

        ```

    Notes:
        Uses the technique described here: https://math.stackexchange.com/a/1170666
    """
    parity: int = 1
    for i, x in enumerate(perm):
        for y in perm[i + 1 :]:
            parity *= -1 if (x > y) else 1

    return parity


def _remove_repeated_pairs(term: tuple[int, ...]) -> tuple[int, ...]:
    # assumes elements are sorted
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
    ind: tuple[int, ...]
    term_len = len(term)
    for ind in it.product(*[[2 * el[0], 2 * el[0] + 1] for el in term]):
        plus_parity = sum(ind[i] % 2 for i in plus_inds)
        minus_parity = sum(ind[i] % 2 for i in minus_inds)
        coeff = (
            _parity(ind) * ((-1j) ** plus_parity) * ((1j) ** minus_parity) / 2**term_len
        )
        yield _remove_repeated_pairs(tuple(sorted(ind))), coeff
