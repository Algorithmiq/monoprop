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

"""Majorana operator data structure."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class MajoranaOperator:
    """A weighted sum of Majorana monomials.

    Terms are normalized on construction: the indices within each monomial are sorted,
    duplicate monomials are summed, and terms whose ``|coefficient|`` falls below
    ``threshold`` are dropped. The resulting :attr:`terms` mapping (Majorana-index tuple
    to complex coefficient) is what the propagator hands to the C++ engine.
    """

    def __init__(
        self,
        majoranas: Sequence[Sequence[int]],
        coefficients: Sequence[complex],
        num_modes: int | None,
        threshold: float = 1e-12,
    ) -> None:
        """Initialize the Majorana operator.

        Args:
            majoranas: One Majorana-index sequence per term.
            coefficients: Coefficient for each term (matched to ``majoranas``).
            num_modes: Number of modes in the system.
            threshold: Terms with ``|coefficient| < threshold`` are discarded.
        """
        self.num_modes = num_modes
        accumulated: dict[tuple[int, ...], complex] = defaultdict(complex)
        for majorana, coefficient in zip(majoranas, coefficients, strict=True):
            key = tuple(sorted(int(i) for i in majorana))
            accumulated[key] += coefficient
        self.terms: dict[tuple[int, ...], complex] = {
            key: coef for key, coef in accumulated.items() if abs(coef) >= threshold
        }

    @classmethod
    def from_dict(
        cls, terms_dict: dict[tuple[int, ...], complex], num_modes: int | None = None
    ) -> MajoranaOperator:
        """Construct a MajoranaOperator from a Majorana-index -> coefficient mapping.

        ``num_modes`` may be omitted (``None``) when the operator is used only as a gate
        generator, where the mode count is supplied by the propagator.
        """
        return cls(list(terms_dict.keys()), list(terms_dict.values()), num_modes)

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
