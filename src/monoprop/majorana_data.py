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
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True, init=False)
class Majorana:
    """A single Majorana monomial: the ordered product ``gamma_{i_1} ... gamma_{i_w}``.

    A term is the atom a :class:`MajoranaOperator` is built from and the generator an
    :class:`~monoprop.circuit.Exp` gate exponentiates. Indices are sorted on
    construction (matching the operator's canonicalization); repeated indices are rejected
    (``gamma_i^2 = 1`` would silently change the monomial's weight), as are negative indices.

    Attributes:
        indices: The sorted, distinct Majorana indices of the monomial.
    """

    indices: tuple[int, ...]

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
            raise ValueError(
                f"Majorana indices must be distinct; got {ordered}. A repeated index "
                "(gamma_i^2 = 1) is almost always a mistake."
            )
        object.__setattr__(self, "indices", ordered)

    def __repr__(self) -> str:
        """Return a string representation such as ``Majorana(4, 5)``."""
        return f"{self.__class__.__name__}({', '.join(map(str, self.indices))})"


class MajoranaOperator:
    """A weighted sum of Majorana monomials.

    Constructed from a ``{term: coefficient}`` mapping, where each key is a
    :class:`Majorana` term (or, equivalently, a raw index tuple). Terms are normalized:
    indices within each monomial are sorted, duplicate monomials are summed, and terms whose
    ``|coefficient|`` falls below ``threshold`` are dropped. The resulting :attr:`terms`
    mapping (Majorana-index tuple to complex coefficient) is what the propagator hands to the
    C++ engine.
    """

    def __init__(
        self,
        terms: Mapping[Majorana | Sequence[int], complex],
        num_modes: int | None = None,
        threshold: float = 1e-12,
    ) -> None:
        """Initialize the Majorana operator from a term mapping.

        Args:
            terms: Mapping from :class:`Majorana` terms (or raw index tuples) to coefficients.
            num_modes: Number of modes in the system, or ``None`` when the operator is used
                only as a gate generator (the mode count is then supplied by the propagator).
            threshold: Terms with ``|coefficient| < threshold`` are discarded.
        """
        majoranas = [key.indices if isinstance(key, Majorana) else key for key in terms]
        self.num_modes = num_modes
        self.terms = self._accumulate(majoranas, list(terms.values()), threshold)

    @classmethod
    def _from_terms(
        cls,
        majoranas: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
        num_modes: int | None = None,
        threshold: float = 1e-12,
    ) -> MajoranaOperator:
        """Build from parallel ``majoranas``/``coefficients`` lists (internal).

        Unlike the dict constructor this accepts colliding monomials and sums them, which the
        Jordan-Wigner and fermionic conversions (:meth:`get_majorana_operator`) rely on.
        """
        obj = cls.__new__(cls)
        obj.num_modes = num_modes
        obj.terms = cls._accumulate(majoranas, coefficients, threshold)
        return obj

    @staticmethod
    def _accumulate(
        majoranas: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
        threshold: float,
    ) -> dict[tuple[int, ...], complex]:
        """Sort, sum duplicates, and threshold a set of Majorana terms."""
        accumulated: dict[tuple[int, ...], complex] = defaultdict(complex)
        for majorana, coefficient in zip(majoranas, coefficients, strict=True):
            key = tuple(sorted(int(i) for i in majorana))
            accumulated[key] += coefficient
        return {
            key: coef for key, coef in accumulated.items() if abs(coef) >= threshold
        }

    def get_majorana_operator(self) -> MajoranaOperator:
        """Return ``self`` (satisfies the operator-conversion protocol)."""
        return self

    def __len__(self) -> int:
        """Number of distinct Majorana monomial terms."""
        return len(self.terms)

    def __str__(self) -> str:
        """Return a string representation of the operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_modes} modes"
        if n <= 8:
            terms = ", ".join(f"{coef}*{list(key)}" for key, coef in self.terms.items())
            out += f": {terms}"
        out += ")"
        return out

    def is_identity(self) -> bool:
        """Check if the operator is the identity (has no terms)."""
        return len(self) == 0
