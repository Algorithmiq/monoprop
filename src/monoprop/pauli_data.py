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

import itertools as it
from typing import TYPE_CHECKING

import numpy as np

from .conversion_utils import _pauli_to_fermi
from .monomial_data import Monomial, MonomialOperator

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def _pauli_to_monomial(pauli: PauliString) -> Monomial:
    """Convert a Pauli String to a MonomialOperator using Jordan Wigner Mapping."""
    majoranas, coefficient = _pauli_to_fermi(pauli.string)
    return Monomial(set_bits=np.array(majoranas), coefficient=coefficient)


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
    ) -> None:
        """Initialize the Pauli operator.

        Args:
            strings: List of Pauli strings (characters must be I, X, Y, Z).
            coefficients: List of float coefficients, one per Pauli string.

        Raises:
            ValueError: If validation fails on characters, string lengths, or
                the number of coefficients.
        """
        pauli_strings = self._validate_inputs(strings, coefficients)

        self.strings = pauli_strings
        self.coefficients = list(coefficients)
        self.num_qubits = len(pauli_strings[0]) if len(pauli_strings) > 0 else None

    def _validate_inputs(
        self, strings: list[str], coefficients: list[float] | np.ndarray
    ) -> list[PauliString]:
        if len(strings) != len(coefficients):
            raise ValueError(
                f"Number of strings ({len(strings)}) must match "
                f"number of coefficients ({len(coefficients)})."
            )
        if len(strings) == 0:
            return []
        pauli_strings = [PauliString(s) for s in strings]
        expected_len = len(pauli_strings[0])
        for i, s in enumerate(pauli_strings):
            if len(s) != expected_len:
                raise ValueError(
                    f"All Pauli strings must have the same length ({expected_len}), "
                    f"but string at index {i} has length {len(s)}."
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
    def from_dict(cls, operator: dict[str, float]) -> PauliOperator:
        """Create a PauliOperator from a dictionary of Pauli strings and coefficients.

        Args:
            operator: A dictionary mapping Pauli strings (composed of I, X, Y, Z)
                to their corresponding float coefficients.

        Returns:
            A PauliOperator instance representing the given operator.
        """
        strings = list(operator.keys())
        coefficients = list(operator.values())
        return cls(strings, coefficients)

    def to_dict(self) -> dict[str, float]:
        """Create a dictionary from PauliOperator."""
        keys = [ps.string for ps in self.strings]
        values = self.coefficients
        return dict(zip(keys, values))

    def get_monomial_operator(self) -> MonomialOperator:
        """Convert the Pauli operator to a MonomialOperator."""
        terms = [_pauli_to_monomial(s) for s in self.strings]
        for term, coeff in zip(terms, self.coefficients):
            term.coefficient *= coeff
        return MonomialOperator(terms, self.num_qubits)


class PauliEvGate:
    """A Pauli evolution gate acting on specified qubits.

    Each qubit has an associated :class:`PauliOperator` and a float  coefficient
    that scales the corresponding evolution term.

    Attributes:
        qubits: Qubit indices on which the gate acts.
        paulis: Pauli operators, one per qubit.
        coefficients: Float coefficients, one per qubit.
    """

    def __init__(
        self,
        qubits: list[int],
        paulis: PauliOperator,
        parameter: float,
    ) -> None:
        """Initialize the Pauli evolution gate.

        Args:
            qubits: List of qubit indices.
            paulis: List of :class:`PauliOperator` instances, one per qubit.
            parameter: Real parameter for the evolution gate.

        Raises:
            ValueError: If the lengths of qubits and paulis differ or if qubits is empty.
        """
        self._validate_inputs(qubits, paulis)

        self.qubits = list(qubits)
        self.paulis = paulis
        self.parameter = parameter

    @staticmethod
    def _validate_inputs(qubits: list[int], paulis: PauliOperator) -> None:
        if not qubits:
            raise ValueError("At least one qubit is required.")

        if not _is_iterable_of_commuting_paulis(paulis.get_pauli_labels()):
            raise ValueError("Pauli strings in PauliOperator do not commute.")

    def __len__(self) -> int:
        """Number of qubit entries in the gate."""
        return len(self.qubits)

    def __repr__(self) -> str:
        """Return a string representation of the gate."""
        return f"{self.__class__.__name__}({len(self)} qubits: {self.qubits}, parameter: {self.parameter})"

    def is_identity(self) -> bool:
        """Check if the gate is an identity operation."""
        return len(self) == 0 or all(s.string == "I" for s in self.paulis.strings)  # type: ignore

    def isclose(self, other: object, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
        """Check if two PauliEvGates are closely equal.

        Args:
            other: Another PauliEvGate to compare with.
            rtol: Relative tolerance for parameter comparison.
            atol: Absolute tolerance for parameter comparison.

        Returns:
            True if the gates are closely equal, False otherwise.

        Raises:
            TypeError: If ``other`` is not a :class:`PauliEvGate`.
        """
        if not isinstance(other, PauliEvGate):
            raise TypeError(f"Cannot compare PauliEvGate with {type(other).__name__}.")
        if self.qubits != other.qubits:
            return False
        if not self.paulis.isclose(other.paulis, rtol=rtol, atol=atol):
            return False
        return bool(np.isclose(self.parameter, other.parameter, rtol=rtol, atol=atol))


class PauliEvCircuit:
    """A circuit composed of Pauli evolution gates with an initial state.

    Attributes:
        gates: Ordered list of :class:`PauliEvGate` instances.
        initial_state: List of ``(axis, qubit)`` pairs specifying the initial
            single-qubit states, where *axis* is one of ``'X'``, ``'Y'``, ``'Z'``
            and *qubit* is the qubit index.
    """

    def __init__(
        self,
        gates: list[PauliEvGate],
        initial_state: list[int],
        num_qubits: int,
    ) -> None:
        """Initialize the Pauli evolution circuit.

        Args:
            gates: List of :class:`PauliEvGate` instances.
            initial_state: List of qubit indices specifying the initial
                single-qubit states where it is placed an X gate.
            num_qubits: Total number of qubits in the circuit.

        """
        self.gates = [gate for gate in gates if not gate.is_identity()]
        self.initial_state = initial_state
        self.num_qubits = num_qubits

    def __len__(self) -> int:
        """Number of gates in the circuit."""
        return len(self.gates)

    def __repr__(self) -> str:
        """Return a string representation of the circuit."""
        return (
            f"{self.__class__.__name__}("
            f"{len(self)} gates, "
            f"initial_state={self.initial_state})"
        )

    def isclose(self, other: object, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
        """Check if two PauliEvCircuits are closely equal.

        Args:
            other: Another PauliEvCircuit to compare with.
            rtol: Relative tolerance for parameter comparison.
            atol: Absolute tolerance for parameter comparison.

        Returns:
            True if the circuits are closely equal, False otherwise.

        Raises:
            TypeError: If ``other`` is not a :class:`PauliEvCircuit`.
        """
        if not isinstance(other, PauliEvCircuit):
            raise TypeError(
                f"Cannot compare PauliEvCircuit with {type(other).__name__}."
            )
        if self.num_qubits != other.num_qubits:
            return False
        if self.initial_state != other.initial_state:
            return False
        if len(self) != len(other):
            return False
        return all(
            gate1.isclose(gate2, rtol=rtol, atol=atol)
            for gate1, gate2 in zip(self.gates, other.gates)
        )


def _is_iterable_of_commuting_paulis(paulis: Iterable[str]) -> bool:
    """Verifies if the Pauli operators pairwise commute.

    Args:
        paulis: list of Pauli operators to check.

    Returns:
        Whether all the Pauli operators pairwise commute.
    """
    for pauli_string1, pauli_string2 in it.combinations(paulis, r=2):
        if (
            sum(
                p1 not in {p2, "I"} and p2 != "I"
                for p1, p2 in zip(pauli_string1, pauli_string2, strict=True)
            )
            % 2
        ):
            return False
    return True
