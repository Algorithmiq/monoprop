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
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from monoprop.conversion_utils import _parity, _remove_repeated_pairs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class Majorana:
    """A single Majorana monomial: the ordered product ``m_{i_1} ... m_{i_w}``.

    A term is the atom a :class:`MajoranaOperator` is built from and the generator an
    :class:`~monoprop.circuit.ExpGate` gate exponentiates. Indices must be sorted,
    distinct, and non-negative. A
    repeated index is rejected because ``m_i^2 = 1`` would silently change the monomial's
    weight -- almost always a mistake rather than an intended simplification. Use
    :meth:`from_unsorted` to create a canonical term from an unsorted and/or repeated index sequence.

    An immutable value object: equal indices compare equal and hash alike, so a term can be
    used as a dictionary key (as :attr:`MajoranaOperator.terms` does).

    Attributes:
        indices: The sorted, distinct Majorana indices of the monomial.
    """

    __slots__ = ("indices",)

    def __init__(self, *indices: int) -> None:
        """Initialize the Majorana monomial from its indices.

        Args:
            *indices: The Majorana indices, in sorted ascending order.

        Raises:
            ValueError: If indices are not sorted, or any index is negative or repeated.
        """
        if any(i < 0 for i in indices):
            raise ValueError(f"Majorana indices must be non-negative; got {indices}.")
        is_strictly_increasing = all(x < y for x, y in pairwise(indices))
        if not is_strictly_increasing:
            raise ValueError(
                f"Majorana indices must be distinct and sorted; got {indices}."
            )

        self.indices = indices

    @classmethod
    def from_unsorted(cls, *indices: int) -> tuple[Majorana, float]:
        """Create a canonical term from an unsorted index sequence.

        Returns the sorted :class:`Majorana` term together with the sign coming
        from reordering the indices under Majorana anticommutation.

        Args:
            indices: Majorana indices in arbitrary order.

        Returns:
            ``(term, sign)``, where ``sign`` is ``+1.0`` or ``-1.0``.

        Repeated indices are canceled in pairs using ``m_i^2 = 1``.

        Raises:
            ValueError: If any index is negative.
        """
        if any(i < 0 for i in indices):
            # better to check as negative indices might cancel out
            # (and should still not be allowed)
            raise ValueError(f"Majorana indices must be non-negative; got {indices}.")
        sorted_values = _remove_repeated_pairs(tuple(sorted(indices)))
        sign = float(_parity(indices))
        return cls(*sorted_values), sign

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
        # Route raw index tuples through phase-aware canonicalization in _accumulate.
        majoranas = [
            key.indices if isinstance(key, Majorana) else tuple(key) for key in terms
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
        """Canonicalize (with fermionic sign) and sum duplicates of Majorana terms."""
        accumulated: dict[tuple[int, ...], complex] = defaultdict(complex)
        for majorana, coefficient in zip(majoranas, coefficients, strict=True):
            canonical, sign = Majorana.from_unsorted(*majorana)
            accumulated[canonical.indices] += coefficient * sign
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

    def __eq__(self, other: object) -> bool:
        """Equal when num_modes and term coefficients match exactly."""
        if not isinstance(other, MajoranaOperator):
            return NotImplemented
        return self.num_modes == other.num_modes and self.terms == other.terms

    __hash__ = None  # type: ignore[assignment]  # value-equal but mutable

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
        if self.num_modes != other.num_modes:
            return False
        return all(
            np.isclose(
                self.terms.get(key, 0), other.terms.get(key, 0), rtol=rtol, atol=atol
            )
            for key in self.terms | other.terms
        )
