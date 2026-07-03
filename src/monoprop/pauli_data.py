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

"""Module for Pauli data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .conversion_utils import _pauli_to_fermi
from .majorana_data import MajoranaOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

_VALID_PAULI_CHARS = frozenset("IXYZ")


class PauliString:  # noqa: PLW1641
    """A single Pauli string composed of I, X, Y, Z characters."""

    def __init__(self, string: str) -> None:
        """Initialize the Pauli string.

        Args:
            string: A string of characters representing the Pauli operators on
                each qubit. Each character must be one of 'I', 'X', 'Y', 'Z'.

        Raises:
            ValueError: If the string contains invalid characters.
        """
        invalid = set(string) - _VALID_PAULI_CHARS
        if invalid:
            raise ValueError(
                f"Invalid characters in Pauli string: {invalid}. "
                "Only I, X, Y, Z are allowed."
            )
        self.string = string

    def __len__(self) -> int:
        """Return the number of qubits (length of the string)."""
        return len(self.string)

    def __repr__(self) -> str:
        """Return a string representation of the Pauli string."""
        return f"{self.__class__.__name__}('{self.string}')"

    def __eq__(self, other: object) -> bool:
        """Check equality with another PauliString."""
        if not isinstance(other, PauliString):
            return NotImplemented
        return self.string == other.string


class PauliOperator:
    """A weighted sum of Pauli strings.

    Note: The Pauli Operator needs to be Hermitian, therefore, only real coefficients are allowed.

    Attributes:
        strings: Pauli strings, each composed solely of I, X, Y, Z characters.
            All strings must share the same length (number of qubits).
        coefficients: Float coefficients corresponding to each Pauli string.
    """

    def __init__(
        self,
        strings: list[str],
        coefficients: list[float] | np.ndarray,
        num_qubits: int | None,
    ) -> None:
        """Initialize the Pauli operator.

        Args:
            strings: List of Pauli strings (characters must be I, X, Y, Z).
            coefficients: List of float coefficients, one per Pauli string.
            num_qubits: Number of qubits the operator acts on. Every Pauli string must
                have exactly this length. Passed explicitly rather than inferred from the
                string length; ``None`` leaves it unspecified (the strings must then still
                share a common length).

        Raises:
            ValueError: If validation fails on characters, string lengths (against each
                other or against ``num_qubits``), or the number of coefficients.
        """
        pauli_strings = self._validate_inputs(strings, coefficients, num_qubits)

        self.strings = pauli_strings
        self.coefficients = list(coefficients)
        self.num_qubits = num_qubits

    def _validate_inputs(
        self,
        strings: list[str],
        coefficients: list[float] | np.ndarray,
        num_qubits: int | None,
    ) -> list[PauliString]:
        if len(strings) != len(coefficients):
            raise ValueError(
                f"Number of strings ({len(strings)}) must match "
                f"number of coefficients ({len(coefficients)})."
            )
        pauli_strings = [PauliString(s) for s in strings]
        # Determine the common length the strings must share: num_qubits when given,
        # otherwise the first string's length (never inferred *as* num_qubits).
        expected_len = num_qubits if num_qubits is not None else None
        if expected_len is None and pauli_strings:
            expected_len = len(pauli_strings[0])
        for i, s in enumerate(pauli_strings):
            if len(s) != expected_len:
                raise ValueError(
                    f"All Pauli strings must have the same length {expected_len} "
                    f"(num_qubits={num_qubits}), but string at index {i} has length "
                    f"{len(s)}."
                )
        return pauli_strings

    def get_pauli_labels(self) -> list[str]:
        """Get a list of Pauli Strings labels."""
        return [ps.string for ps in self.strings]

    def __len__(self) -> int:
        """Number of terms in the operator."""
        return len(self.strings)

    def __str__(self) -> str:
        """Return a string representation of the operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_qubits} qubits"
        if n <= 8:
            terms = ", ".join(
                f"{c}*{s}" for s, c in zip(self.strings, self.coefficients)
            )
            out += f": {terms}"
        out += ")"
        return out

    def isclose(self, other: object, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
        """Check if two PauliOperators are closely equal.

        Args:
            other: Another PauliOperator to compare with.
            rtol: Relative tolerance for coefficient comparison.
            atol: Absolute tolerance for coefficient comparison.

        Returns:
            True if the operators are closely equal, False otherwise.

        Raises:
            TypeError: If ``other`` is not a :class:`PauliOperator`.
        """
        if not isinstance(other, PauliOperator):
            raise TypeError(
                f"Cannot compare PauliOperator with {type(other).__name__}."
            )
        if self.num_qubits != other.num_qubits:
            return False
        if len(self) != len(other):
            return False
        for s1, c1, s2, c2 in zip(
            self.strings, self.coefficients, other.strings, other.coefficients
        ):
            if s1.string != s2.string or not np.isclose(c1, c2, rtol=rtol, atol=atol):
                return False
        return True

    @classmethod
    def from_dict(
        cls, operator: dict[str, float], num_qubits: int | None = None
    ) -> PauliOperator:
        """Create a PauliOperator from a Pauli-string -> coefficient mapping.

        Mirrors :meth:`~monoprop.majorana_data.MajoranaOperator.from_dict`: ``num_qubits``
        may be omitted (``None``) when it is not needed, but is validated against the
        string lengths when given.

        Args:
            operator: A dictionary mapping Pauli strings (composed of I, X, Y, Z)
                to their corresponding float coefficients.
            num_qubits: Number of qubits the operator acts on (see :meth:`__init__`).

        Returns:
            A PauliOperator instance representing the given operator.
        """
        strings = list(operator.keys())
        coefficients = list(operator.values())
        return cls(strings, coefficients, num_qubits)

    def to_dict(self) -> dict[str, float]:
        """Create a dictionary from PauliOperator."""
        keys = [ps.string for ps in self.strings]
        values = self.coefficients
        return dict(zip(keys, values))

    def get_majorana_operator(self) -> MajoranaOperator:
        """Convert the Pauli operator to a MajoranaOperator via Jordan-Wigner."""
        majoranas: list[Sequence[int]] = []
        coefficients: list[complex] = []
        for pauli_string, coeff in zip(self.strings, self.coefficients):
            majorana, jw_coeff = _pauli_to_fermi(pauli_string.string)
            majoranas.append(majorana)
            coefficients.append(jw_coeff * coeff)
        return MajoranaOperator(majoranas, coefficients, self.num_qubits)
