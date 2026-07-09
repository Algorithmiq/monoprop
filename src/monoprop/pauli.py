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

"""Pauli term and operator data structures."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from .conversion_utils import _extend_pauli_string, _pauli_to_fermi
from .majorana import MajoranaOperator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_VALID_PAULI_CHARS = frozenset("IXYZ")


class Pauli:
    """A single Pauli term: Pauli letters placed on specific qubits.

    A term is the atom a :class:`PauliOperator` is built from and the generator an
    :class:`~monoprop.circuit.Exp` gate exponentiates. The placement is *local* -- the
    string names only the qubits the term acts on -- so the same term can appear in operators
    of any width; the total ``num_qubits`` lives on the operator, not the term.

    Terms are canonicalized on construction: identity (``I``) letters are dropped and the
    remaining ``(qubit, letter)`` pairs are sorted by qubit, so ``Pauli("XY", (1, 0))`` and
    ``Pauli("YX", (0, 1))`` compare equal and hash alike -- an immutable value object usable
    as a dictionary key (as :attr:`PauliOperator.terms` does).

    Attributes:
        string: The non-identity Pauli letters, ordered to match :attr:`qubits`.
        qubits: The (sorted, distinct) qubit indices the letters act on.
    """

    __slots__ = ("qubits", "string")

    def __init__(self, string: str, qubits: int | Sequence[int] | None = None) -> None:
        """Initialize the Pauli term.

        Args:
            string: Pauli letters (each one of ``I``, ``X``, ``Y``, ``Z``).
            qubits: Qubit index or indices the letters act on. Defaults to
                ``range(len(string))`` (i.e. a full-width string on qubits ``0..len-1``).

        Raises:
            ValueError: On invalid characters, a string/qubits length mismatch, or duplicate
                qubit indices.
        """
        if qubits is None:
            qubits = range(len(string))
        elif isinstance(qubits, int):
            qubits = (qubits,)
        qubit_tuple = tuple(int(q) for q in qubits)

        invalid = set(string) - _VALID_PAULI_CHARS
        if invalid:
            raise ValueError(
                f"Invalid characters in Pauli string: {invalid}. Only I, X, Y, Z are allowed."
            )
        if len(string) != len(qubit_tuple):
            raise ValueError(
                f"Pauli string {string!r} and qubits {qubit_tuple} must have the same length."
            )
        if len(set(qubit_tuple)) != len(qubit_tuple):
            raise ValueError(f"Duplicate qubit indices in Pauli term: {qubit_tuple}.")

        pairs = sorted(
            (q, p) for q, p in zip(qubit_tuple, string, strict=True) if p != "I"
        )
        self.qubits = tuple(q for q, _ in pairs)
        self.string = "".join(p for _, p in pairs)

    def __eq__(self, other: object) -> bool:
        """Two Pauli terms are equal when their letters and qubits match (post-canonicalization)."""
        if not isinstance(other, Pauli):
            return NotImplemented
        return self.string == other.string and self.qubits == other.qubits

    def __hash__(self) -> int:
        """Hash on the canonical ``(string, qubits)`` so equal terms share a bucket."""
        return hash((self.string, self.qubits))

    def __repr__(self) -> str:
        """Return a string representation such as ``Pauli('ZZ', (0, 1))``."""
        return f"{self.__class__.__name__}({self.string!r}, {self.qubits})"


class PauliOperator:
    """A weighted sum of Pauli terms.

    Constructed from a ``{term: coefficient}`` mapping, where each key is a :class:`Pauli`
    term (or, equivalently, a raw full-width Pauli string like ``"ZZ"``, which is read as a
    term on qubits ``0..len-1``). The total qubit count lives here, on the operator, and is
    required so a propagator can be built from it directly. A gate generator is also authored
    as a :class:`PauliOperator` (wrapped in :class:`~monoprop.circuit.Exp`) -- bare
    :class:`Pauli` terms are not accepted by ``Exp``, since the operator is what carries the
    qubit count.
    """

    def __init__(
        self,
        terms: Mapping[Pauli | str, complex],
        num_qubits: int,
        threshold: float = 1e-12,
    ) -> None:
        """Initialize the Pauli operator from a term mapping.

        Args:
            terms: Mapping from :class:`Pauli` terms (or raw full-width strings) to their
                coefficients.
            num_qubits: Total number of qubits the operator acts on. Required: an operator
                carries its own qubit count so a propagator can be built from it directly.
                Every term must act within ``0..num_qubits-1``.
            threshold: Terms with ``|coefficient| < threshold`` are discarded.

        Raises:
            ValueError: If a term acts on a qubit index ``>= num_qubits``.
        """
        accumulated: dict[Pauli, complex] = defaultdict(complex)
        for key, coeff in terms.items():
            pauli = key if isinstance(key, Pauli) else Pauli(key)
            accumulated[pauli] += coeff
        self.terms: dict[Pauli, complex] = {
            pauli: coef for pauli, coef in accumulated.items() if abs(coef) >= threshold
        }
        self.num_qubits = num_qubits
        if num_qubits is not None:
            for pauli in self.terms:
                if pauli.qubits and max(pauli.qubits) >= num_qubits:
                    raise ValueError(
                        f"Pauli term {pauli} acts on a qubit index >= num_qubits="
                        f"{num_qubits}."
                    )

    @classmethod
    def _from_terms(
        cls,
        strings: Sequence[Pauli | str],
        coefficients: Sequence[complex],
        num_qubits: int | None = None,
        threshold: float = 1e-12,
    ) -> PauliOperator:
        """Build from parallel ``strings``/``coefficients`` lists (internal)."""
        accumulated: dict[Pauli, complex] = defaultdict(complex)
        for string, coeff in zip(strings, coefficients, strict=True):
            pauli = string if isinstance(string, Pauli) else Pauli(string)
            accumulated[pauli] += coeff
        return cls(accumulated, num_qubits, threshold)

    def __len__(self) -> int:
        """Number of terms in the operator."""
        return len(self.terms)

    def __str__(self) -> str:
        """Return a string representation of the operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_qubits} qubits"
        if n <= 8:
            terms = ", ".join(f"{c}*{p!r}" for p, c in self.terms.items())
            out += f": {terms}"
        out += ")"
        return out

    def isclose(self, other: object, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
        """Check if two PauliOperators are closely equal (same terms and coefficients).

        Args:
            other: Another PauliOperator to compare with.
            rtol: Relative tolerance for coefficient comparison.
            atol: Absolute tolerance for coefficient comparison.

        Returns:
            True if the operators have the same qubit count and matching terms, else False.

        Raises:
            TypeError: If ``other`` is not a :class:`PauliOperator`.
        """
        if not isinstance(other, PauliOperator):
            raise TypeError(
                f"Cannot compare PauliOperator with {type(other).__name__}."
            )
        if (
            self.num_qubits != other.num_qubits
            or self.terms.keys() != other.terms.keys()
        ):
            return False
        return all(
            np.isclose(coef, other.terms[pauli], rtol=rtol, atol=atol)
            for pauli, coef in self.terms.items()
        )

    def get_majorana_operator(self) -> MajoranaOperator:
        """Convert the Pauli operator to a MajoranaOperator via Jordan-Wigner.

        Each local term is extended to the full ``num_qubits`` width (identities filled in)
        before the Jordan-Wigner map, so the resulting Majorana indices are global.

        Raises:
            ValueError: If ``num_qubits`` is unset.
        """
        if self.num_qubits is None:
            raise ValueError(
                "PauliOperator.get_majorana_operator() needs num_qubits; construct the "
                "operator with an explicit num_qubits."
            )
        majoranas: list[Sequence[int]] = []
        coefficients: list[complex] = []
        for pauli, coeff in self.terms.items():
            extended = _extend_pauli_string(pauli.string, pauli.qubits, self.num_qubits)
            majorana, jw_coeff = _pauli_to_fermi(extended)
            majoranas.append(majorana)
            coefficients.append(jw_coeff * coeff)
        return MajoranaOperator._from_terms(majoranas, coefficients, self.num_qubits)
