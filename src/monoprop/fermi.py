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

"""Module for Fermi data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .conversion_utils import _n_product
from .majorana import MajoranaOperator

if TYPE_CHECKING:
    from collections.abc import Sequence


_VALID_FERMI_CHARS = frozenset("+-")


class FermiString:
    """An ordered product of fermionic ladder operators."""

    def __init__(self, expression: Sequence[tuple[int, str]] | FermiString) -> None:
        """Initialize the Fermi string.

        Args:
            expression: A sequence of ``(mode index, '+' or '-')`` pairs, with non-negative
                indices, or another [FermiString][] to copy.
        """
        if isinstance(expression, FermiString):
            self.expression = expression.expression
        else:
            self._validate_signal(expression)
            self.expression = tuple(expression)

    def _validate_signal(self, ferm_expression: Sequence[tuple[int, str]]) -> None:
        """Reject negative mode indices and operator characters other than ``+``/``-``."""
        for idx, op in ferm_expression:
            if idx < 0:
                raise ValueError(f"Invalid index {idx}: must be non-negative")
        invalid = {op for _, op in ferm_expression if op not in _VALID_FERMI_CHARS}
        if invalid:
            raise ValueError(f"Invalid operator(s) {invalid!r}: must be '+' or '-'")

    def _canonicalize(self) -> tuple[tuple[tuple[int, str], ...], int]:
        """Return the expression ordered ``+`` before ``-`` (each ascending) and the swap sign."""
        expr = list(self.expression)

        def key(op: tuple[int, str]) -> tuple[int, int]:
            idx, kind = op
            return (0 if kind == "+" else 1, idx)

        parity = 0

        # insertion sort so we can count swaps
        for i in range(1, len(expr)):
            j = i
            while j > 0 and key(expr[j]) < key(expr[j - 1]):
                expr[j], expr[j - 1] = expr[j - 1], expr[j]
                parity ^= 1
                j -= 1

        sign = -1 if parity else 1

        return tuple(expr), sign

    def __repr__(self) -> str:
        """Return a string representation of the Fermi string."""
        terms = " ".join(f"c_{idx}^{op}" for idx, op in self.expression)
        return f"{self.__class__.__name__}({terms})"

    def __eq__(self, value: object) -> bool:
        """Check equality of two FermiStrings."""
        if not isinstance(value, FermiString):
            return NotImplemented
        return self.expression == value.expression

    def __hash__(self) -> int:
        """Return a hash of the Fermi string."""
        return hash(self.expression)


def _fermi_string_to_majorana_terms(
    ferm_string: FermiString,
) -> list[tuple[Sequence[int], complex]]:
    """Expand a fermi string into ``(majorana indices, coefficient)`` terms."""
    expr = list(ferm_string.expression)
    plus_inds = [i for i, (_, op) in enumerate(expr) if op == "+"]
    minus_inds = [i for i, (_, op) in enumerate(expr) if op == "-"]
    return list(_n_product(expr, plus_inds, minus_inds))


class FermiOperator:
    """A weighted sum of [FermiString][] terms."""

    def __init__(
        self,
        terms: Sequence[FermiString] | Sequence[Sequence[tuple[int, str]]],
        coefficients: Sequence[complex],
        num_modes: int | None = None,
    ) -> None:
        """Initialize the fermi operator.

        Args:
            terms: The [FermiString][] terms, or ``(index, '+'/'-')`` sequences to build them.
            coefficients: One coefficient per term, in the same order.
            num_modes: Inferred from the largest index in ``terms`` when ``None``.
        """
        self.terms = [
            t if isinstance(t, FermiString) else FermiString(t) for t in terms
        ]
        self.coefficients = list(coefficients)
        self.num_modes = (
            num_modes
            if num_modes is not None
            else max((idx for f in self.terms for idx, _ in f.expression)) + 1
        )

    @classmethod
    def from_dict(
        cls, terms_dict: dict[tuple[tuple[int, str], ...], complex]
    ) -> FermiOperator:
        """Construct a FermiOperator from a dictionary."""
        terms = []
        coefficients = []
        for key, value in terms_dict.items():
            terms.append(FermiString(key))
            coefficients.append(value)
        return cls(terms=terms, coefficients=coefficients)

    def __len__(self) -> int:
        """Number of terms in the operator."""
        return len(self.terms)

    def __repr__(self) -> str:
        """Return a string representation of the fermionic operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_modes} modes"
        if n <= 8:
            terms = ", ".join(f"{c}*{s}" for c, s in zip(self.coefficients, self.terms))
            out += f": {terms}"
        out += ")"
        return out

    def __eq__(self, other: object) -> bool:
        """Equal when terms, coefficients, and num_modes match exactly."""
        if not isinstance(other, FermiOperator):
            return NotImplemented
        return (
            self.num_modes == other.num_modes
            and self.terms == other.terms
            and self.coefficients == other.coefficients
        )

    __hash__ = None  # type: ignore[assignment]  # value-equal but mutable

    def _as_dict(self) -> dict[tuple, complex]:
        """Return ``{canonical expression: coefficient}``, with the reordering sign folded in."""
        result = {}
        for term, coeff in zip(self.terms, self.coefficients):
            cano, sign = term._canonicalize()
            result[cano] = coeff * sign
        return result

    def isclose(self, other: object, rtol: float = 1e-05, atol: float = 1.0e-8) -> bool:
        """Check that two operators are almost equal, term-wise.

        Terms are compared after canonicalization, with ``numpy.isclose`` at ``rtol``/``atol``;
        a differing mode count is False, not an error.

        Raises:
            TypeError: If ``other`` is not a [FermiOperator][].
        """
        if not isinstance(other, FermiOperator):
            raise TypeError(
                f"Cannot compare FermiOperator with {type(other).__name__}."
            )
        if self.num_modes != other.num_modes:
            return False

        lhs = self._as_dict()
        rhs = other._as_dict()

        return all(
            np.isclose(lhs.get(term, 0) - rhs.get(term, 0), 0, rtol=rtol, atol=atol)
            for term in lhs | rhs
        )

    def get_majorana_operator(self) -> MajoranaOperator:
        """Convert the fermi operator to a MajoranaOperator."""
        majoranas: list[Sequence[int]] = []
        coefficients: list[complex] = []
        for term, c in zip(self.terms, self.coefficients):
            for majorana, val in _fermi_string_to_majorana_terms(term):
                majoranas.append(majorana)
                coefficients.append(val * c)
        return MajoranaOperator._from_terms(majoranas, coefficients, self.num_modes)
