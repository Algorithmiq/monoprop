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

"""Majorana term and operator data structures."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class Majorana:
    """A single Majorana monomial: the ordered product ``gamma_{i_1} ... gamma_{i_w}``.

    A term is the atom a :class:`MajoranaOperator` is built from and the generator an
    :class:`~monoprop.circuit.ExpGate` gate exponentiates. Indices are sorted on construction
    (matching the operator's canonicalization) and must be distinct and non-negative. A
    repeated index is rejected because ``gamma_i^2 = 1`` would silently change the monomial's
    weight -- almost always a mistake rather than an intended simplification.

    An immutable value object: equal indices compare equal and hash alike, so a term can be
    used as a dictionary key (as :attr:`MajoranaOperator.terms` does).

    Attributes:
        indices: The sorted, distinct Majorana indices of the monomial.
    """

    __slots__ = ("indices",)

    def __init__(self, *indices: int) -> None:
        """Initialize the Majorana monomial from its indices.

        Args:
            *indices: The Majorana indices, in any order (they are sorted).

        Raises:
            ValueError: If any index is negative or an index is repeated.
        """
        ordered = tuple(sorted(int(i) for i in indices))
        if ordered and ordered[0] < 0:
            raise ValueError(f"Majorana indices must be non-negative; got {ordered}.")
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"Majorana indices must be distinct; got {ordered}.")
        self.indices = ordered

    def __eq__(self, other: object) -> bool:
        """Two Majorana terms are equal when their sorted indices match."""
        if not isinstance(other, Majorana):
            return NotImplemented
        return self.indices == other.indices

    def __hash__(self) -> int:
        """Hash on the sorted indices so equal terms share a bucket."""
        return hash(self.indices)

    def __repr__(self) -> str:
        """Return a string representation such as ``Majorana(4, 5)``."""
        return f"{self.__class__.__name__}({', '.join(map(str, self.indices))})"


class MajoranaOperator:
    """A weighted sum of Majorana monomials.

    Constructed from a ``{term: coefficient}`` mapping, where each key is a
    :class:`Majorana` term (or, equivalently, a raw index tuple). Terms are normalized:
    indices within each monomial are sorted and duplicate monomials are summed. The resulting
    :attr:`terms` mapping (Majorana-index tuple to complex coefficient) is what the propagator
    hands to the C++ engine.
    """

    def __init__(
        self,
        terms: Mapping[Majorana | Sequence[int], complex],
        num_modes: int,
    ) -> None:
        """Initialize the Majorana operator from a term mapping.

        Args:
            terms: Mapping from :class:`Majorana` terms (or raw index tuples) to coefficients.
            num_modes: Number of modes in the system. Required: an operator carries its own
                mode count so a propagator can be built from it directly. A gate generator is
                also authored as a :class:`MajoranaOperator` (wrapped in
                :class:`~monoprop.circuit.ExpGate`) -- bare :class:`Majorana` terms are not accepted
                by ``ExpGate``, since the operator is what carries the mode count.
        """
        # Route raw index tuples through Majorana so they get the same non-negative/distinct
        # validation a Majorana key already carries (a bare tuple would otherwise slip past it).
        majoranas = [
            (key if isinstance(key, Majorana) else Majorana(*key)).indices
            for key in terms
        ]
        self.num_modes = num_modes
        self.terms = self._accumulate(majoranas, list(terms.values()))

    @classmethod
    def _from_terms(
        cls,
        majoranas: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
        num_modes: int | None = None,
    ) -> MajoranaOperator:
        """Build from parallel ``majoranas``/``coefficients`` lists (internal).

        Unlike the dict constructor this accepts colliding monomials and sums them, which the
        Jordan-Wigner and fermionic conversions (:meth:`get_majorana_operator`) rely on.
        """
        obj = cls.__new__(cls)
        obj.num_modes = num_modes
        obj.terms = cls._accumulate(majoranas, coefficients)
        return obj

    @staticmethod
    def _accumulate(
        majoranas: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
    ) -> dict[tuple[int, ...], complex]:
        """Sort and sum duplicates of a set of Majorana terms."""
        accumulated: dict[tuple[int, ...], complex] = defaultdict(complex)
        for majorana, coefficient in zip(majoranas, coefficients, strict=True):
            key = tuple(sorted(int(i) for i in majorana))
            accumulated[key] += coefficient
        return dict(accumulated)

    def get_majorana_operator(self) -> MajoranaOperator:
        """Return ``self`` (satisfies the operator-conversion protocol)."""
        return self

    def __len__(self) -> int:
        """Number of distinct Majorana monomial terms."""
        return len(self.terms)

    def __repr__(self) -> str:
        """Return a string representation of the operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_modes} modes"
        if n <= 8:
            terms = ", ".join(f"{coef}*{list(key)}" for key, coef in self.terms.items())
            out += f": {terms}"
        out += ")"
        return out

    def isclose(self, other: object, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
        """Check if two MajoranaOperators are closely equal (same terms and coefficients).

        Args:
            other: Another MajoranaOperator to compare with.
            rtol: Relative tolerance for coefficient comparison.
            atol: Absolute tolerance for coefficient comparison.

        Returns:
            True if the operators have the same mode count and matching terms, else False.

        Raises:
            TypeError: If ``other`` is not a :class:`MajoranaOperator`.
        """
        if not isinstance(other, MajoranaOperator):
            raise TypeError(
                f"Cannot compare MajoranaOperator with {type(other).__name__}."
            )
        if self.num_modes != other.num_modes or self.terms.keys() != other.terms.keys():
            return False
        return all(
            np.isclose(coef, other.terms[key], rtol=rtol, atol=atol)
            for key, coef in self.terms.items()
        )
