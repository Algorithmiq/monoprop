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
from .monomial_data import Monomial, MonomialOperator, MonomialSequence

if TYPE_CHECKING:
    from collections.abc import Sequence


_VALID_FERMI_CHARS = frozenset("+-")


class FermiString:
    """Class representing a Fermi string."""

    def __init__(self, expression: Sequence[tuple[int, str]] | FermiString) -> None:
        """Initialize the Fermi string.

        Args:
            expression: List of (index, char +/-) pairs representing the Fermi string.
                The index should be a non-negative integer, and the char should be either '+' or '-'.
        """
        if isinstance(expression, FermiString):
            self.expression = expression.expression
        else:
            self._validate_signal(expression)
            self.expression = tuple(expression)

    def _validate_signal(self, ferm_expression: Sequence[tuple[int, str]]) -> None:
        for idx, op in ferm_expression:
            if idx < 0:
                raise ValueError(f"Invalid index {idx}: must be non-negative")
        invalid = {op for _, op in ferm_expression if op not in _VALID_FERMI_CHARS}
        if invalid:
            raise ValueError(f"Invalid operator(s) {invalid!r}: must be '+' or '-'")

    def _canonicalize(self) -> tuple(tuple, int):
        """Return the FermiString in a predefined order and a permutation sign."""
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
        terms = " ".join(f"a_{idx}^{op}" for idx, op in self.expression)
        return f"{self.__class__.__name__}({terms})"

    def __eq__(self, value: object) -> bool:
        """Check equality of two FermiStrings."""
        if not isinstance(value, FermiString):
            return NotImplemented
        return self.expression == value.expression

    def __hash__(self) -> int:
        """Return a hash of the Fermi string."""
        return hash(self.expression)


def _from_fermi_to_monomial(ferm_string: FermiString) -> list[Monomial]:
    """Convert a fermi string to a list of Majorana operators."""
    expr = list(ferm_string.expression)
    plus_inds = [i for i, (_, op) in enumerate(expr) if op == "+"]
    minus_inds = [i for i, (_, op) in enumerate(expr) if op == "-"]
    return [
        Monomial(np.array(key), val)
        for key, val in _n_product(expr, plus_inds, minus_inds)
    ]


class FermiOperator:
    """Class representing a summed fermi operator."""

    def __init__(
        self,
        terms: Sequence[FermiString] | Sequence[Sequence[tuple[int, str]]],
        coefficients: Sequence[complex],
        num_modes: int | None = None,
    ) -> None:
        """Initialize the fermi operator.

        Args:
            terms: List of FermiString objects representing the operator.
            coefficients: List of coefficients corresponding to the terms.
            num_modes: Optional number of modes. If not provided, it will be inferred from the terms.

        Raises:
            ValueError: If any index is out of bounds or if there are duplicate indices.
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

    def __str__(self) -> str:
        """Return a string representation of the fermionic operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_modes} modes"
        if n <= 8:
            terms = ", ".join(f"{c}*{s}" for c, s in zip(self.coefficients, self.terms))
            out += f": {terms}"
        out += ")"
        return out

    def _as_dict(self) -> dict[tuple, complex]:
        """Return the operator as a dictionary."""
        result = {}
        for term, coeff in zip(self.terms, self.coefficients):
            cano, sign = term._canonicalize()
            result[cano] = coeff * sign
        return result

    def isclose(
        self, other: FermiOperator, rtol: float = 1e-05, atol: float = 1.0e-8
    ) -> bool:
        """Check that two operators are almost equal, term-wise.

        Args:
            other: the other FermiOperator to compare to.
            rtol: the relative tolerance parameter.
            atol: the absolute tolerance parameter.

        Returns:
            A boolean.
        """
        if self.num_modes != other.num_modes:
            return False

        lhs = self._as_dict()
        rhs = other._as_dict()

        return all(
            abs(lhs.get(term, 0) - rhs.get(term, 0))
            <= atol + rtol * abs(rhs.get(term, 0))
            for term in lhs.keys() | rhs.keys()
        )

    def get_monomial_operator(self) -> MonomialOperator:
        """Convert the fermi operator to a MonomialOperator."""
        terms = [
            Monomial(m.set_bits, m.coefficient * c)
            for term, c in zip(self.terms, self.coefficients)
            for m in _from_fermi_to_monomial(term)
        ]
        return MonomialOperator(terms, self.num_modes)


class MajoranaOperator:
    """Class representing a Majorana operator."""

    def __init__(
        self,
        majoranas: list[tuple[int, ...]],
        coefficients: list[complex],
        num_modes: int | None,
    ) -> None:
        """Initialize the Majorana operator.

        Args:
            majoranas: List of tuples representing the indices of the Majorana operators.
            coefficients: List of coefficients corresponding to the Majorana terms.
            num_modes: Number of modes in the system.
        """
        self.majoranas = list(majoranas)
        self.coefficients = list(coefficients)
        self.num_modes = num_modes

    def __len__(self) -> int:
        """Number of terms in the operator."""
        return len(self.majoranas)

    def __str__(self) -> str:
        """Return a string representation of the Majorana operator."""
        n = len(self)
        out = f"{self.__class__.__name__}({n} terms, {self.num_modes} modes"
        if n <= 8:
            terms = ", ".join(
                f"{c}*{m!r}" for c, m in zip(self.coefficients, self.majoranas)
            )
            out += f": {terms}"
        out += ")"
        return out

    def get_monomial_operator(self) -> MonomialOperator:
        """Convert the Majorana operator to a MonomialOperator."""
        terms = [
            Monomial(np.array(m), c) for m, c in zip(self.majoranas, self.coefficients)
        ]
        return MonomialOperator(terms, self.num_modes)


class FermiEvGate:
    """Class representing a fermi evolution gate."""

    def __init__(
        self,
        generator: FermiOperator | MajoranaOperator,
        parameter: complex,
    ) -> None:
        """Initialize the fermi evolution gate.

        Args:
            generator: The generator of the evolution, either as a FermiOperator or a MajoranaOperator.
            parameter: The parameter for the evolution.
        """
        self.generator: MonomialOperator = generator.get_monomial_operator()
        self.parameter = parameter

    def __len__(self) -> int:
        """Number of terms in the generator."""
        return len(self.generator)

    def __repr__(self) -> str:
        """Return a string representation of the gate."""
        return (
            f"{self.__class__.__name__}({len(self)} terms, parameter={self.parameter})"
        )


class FermiCircuit:
    """Class representing a fermi circuit."""

    def __init__(
        self,
        initial_state: list[int],
        gates: list[FermiEvGate],
    ) -> None:
        """Initialize the fermi circuit.

        Args:
            initial_state: List of mode indices representing the initial state.
            gates: List of FermiEvGate objects representing the gates in the circuit.
        """
        initial_state = sorted(initial_state)

        self._validate_inputs(initial_state)

        self.initial_state = initial_state
        self.gates = [gate for gate in gates if not gate.generator.is_identity()]

    @staticmethod
    def _validate_inputs(initial_state: list[int]) -> None:
        """Validate the inputs for the fermi circuit."""
        set_initial = set(initial_state)
        if len(set_initial) != len(initial_state):
            raise ValueError("Duplicate indices in initial state")

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

    def get_monomial_sequence(self) -> MonomialSequence:
        """Convert the fermi circuit to a MonomialSequence."""
        majoranas, gen_coeffs, parameters, param_inds = [], [], [], []
        for i, gate in enumerate(self.gates):
            parameters.append(gate.parameter)
            for monomial, coefficient in gate.generator.terms.items():
                w = len(monomial)
                antiherm = -coefficient / (1j) ** (w * (w - 1) / 2)
                majoranas.append(monomial)
                gen_coeffs.append(float(np.real(antiherm)))
                param_inds.append(i)

        return MonomialSequence(
            initial_state=self.initial_state,
            majoranas=majoranas,
            parameters=parameters,
            gen_coeffs=gen_coeffs,
            param_inds=param_inds,
        )
