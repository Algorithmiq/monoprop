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

"""Module for Monomial data structures."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


@dataclass
class Monomial:
    """Monomial term."""

    set_bits: list[int]
    coefficient: complex


@dataclass
class MonomialCircuit:
    """Monomial circuit."""

    majoranas: list[tuple[int, ...]] | ndarray
    parameters: list[float] | ndarray
    gen_coeffs: list[float] | ndarray
    param_inds: list[int] | ndarray
    identical_params: list[int] | ndarray | None = None


class MonomialOperator:
    """Monomial operator."""

    def __init__(
        self,
        terms: list[Monomial],
        num_modes: int,
        threshold: float = 1e-12,
    ) -> None:
        """Initialize the monomial operator."""
        self.num_modes = num_modes
        self.terms = defaultdict(complex)
        for term in terms:
            key = tuple(sorted(term.set_bits))
            self.terms[key] += term.coefficient
        # Remove terms below the threshold
        self.terms = {
            key: coef for key, coef in self.terms.items() if abs(coef) >= threshold
        }

    @classmethod
    def from_dict(
        cls, terms_dict: dict[tuple[int, ...], complex], num_modes: int
    ) -> MonomialOperator:
        """Construct a MonomialOperator from a dictionary."""
        terms = [Monomial(list(key), coef) for key, coef in terms_dict.items()]
        return cls(terms, num_modes)

    def __len__(self) -> int:
        """Number of terms in the operator."""
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
        """Check if the operator is the identity."""
        return len(self) == 0
