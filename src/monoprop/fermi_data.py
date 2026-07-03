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

from .circuit import Gate, MajoranaCircuit, _antihermitian_gen_coeff
from .conversion_utils import _n_product
from .majorana_data import MajoranaOperator

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


def _fermi_string_to_majorana_terms(
    ferm_string: FermiString,
) -> list[tuple[Sequence[int], complex]]:
    """Expand a fermi string into ``(majorana indices, coefficient)`` terms."""
    expr = list(ferm_string.expression)
    plus_inds = [i for i, (_, op) in enumerate(expr) if op == "+"]
    minus_inds = [i for i, (_, op) in enumerate(expr) if op == "-"]
    return list(_n_product(expr, plus_inds, minus_inds))


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

    def get_majorana_operator(self) -> MajoranaOperator:
        """Convert the fermi operator to a MajoranaOperator."""
        majoranas: list[Sequence[int]] = []
        coefficients: list[complex] = []
        for term, c in zip(self.terms, self.coefficients):
            for majorana, val in _fermi_string_to_majorana_terms(term):
                majoranas.append(majorana)
                coefficients.append(val * c)
        return MajoranaOperator(majoranas, coefficients, self.num_modes)


class FermiGate:
    """A fermionic evolution gate; a pure generator, the qubit/Majorana analogue.

    Like :class:`~monoprop.circuit.MajoranaGate` and
    :class:`~monoprop.circuit.PauliGate` it carries no parameter index -- the driving angle
    is supplied by position via the circuit's ``parameters`` and ``parameter_mapping``. Its
    generator is normalized to a :class:`~monoprop.majorana_data.MajoranaOperator` on
    construction.
    """

    def __init__(self, generator: FermiOperator | MajoranaOperator) -> None:
        """Initialize the fermi gate.

        Args:
            generator: The generator of the evolution, either a
                :class:`FermiOperator` or a :class:`~monoprop.majorana_data.MajoranaOperator`.
        """
        self.generator: MajoranaOperator = generator.get_majorana_operator()

    def __len__(self) -> int:
        """Number of terms in the generator."""
        return len(self.generator)

    def __repr__(self) -> str:
        """Return a string representation of the gate."""
        return f"{self.__class__.__name__}({len(self)} terms)"


def _fermi_gate_to_gate(gate: FermiGate) -> Gate:
    """Convert a fermi gate's generator into a native :class:`~monoprop.circuit.Gate`.

    The generator's monomials are antihermitian-normalized to real structural coefficients
    and packed into a :class:`~monoprop.majorana_data.MajoranaOperator`; zero-coefficient
    (identity) monomials are dropped by the operator's construction threshold.
    """
    majoranas: list[tuple[int, ...]] = []
    coefficients: list[complex] = []
    for monomial, coefficient in gate.generator.terms.items():
        # Validates the fermionic generator is Hermitian (raises otherwise).
        majoranas.append(tuple(monomial))
        coefficients.append(complex(_antihermitian_gen_coeff(monomial, coefficient)))
    return Gate(MajoranaOperator(majoranas, coefficients, gate.generator.num_modes))


class FermiCircuit(MajoranaCircuit):
    """A variational circuit of :class:`FermiGate` generators.

    Each gate's generator is converted to a native :class:`~monoprop.circuit.Gate` at
    construction (its monomials antihermitian-normalized into a
    :class:`~monoprop.majorana_data.MajoranaOperator`), so a ``FermiCircuit`` *is* a
    :class:`~monoprop.circuit.MajoranaCircuit` and is consumed directly by
    :class:`~monoprop.majorana_propagator.MajoranaPropagator`. The original fermionic gates
    remain available on :attr:`fermi_gates`.

    Attributes:
        fermi_gates: The original :class:`FermiGate` generators, in application order.
    """

    def __init__(
        self,
        gates: Sequence[FermiGate] = (),
        parameters: Sequence[float] = (),
        parameter_mapping: Sequence[int] | None = None,
        initial_state: Sequence[int] = (),
    ) -> None:
        """Initialize the fermi circuit.

        Args:
            gates: The :class:`FermiGate` generators, in application order.
            parameters: The angle values driving the gates.
            parameter_mapping: Per-gate angle index, or ``None`` for the identity mapping.
            initial_state: Mode indices of the reference (occupied) state.

        Raises:
            ValueError: If ``initial_state`` contains duplicate indices.

        Note:
            Gates whose generator is the identity (no Majorana terms, e.g. a
            zero-coefficient generator) apply no rotation and are dropped. With the default
            identity mapping the corresponding parameter is dropped too; pass an explicit
            ``parameter_mapping`` to keep every gate.
        """
        occupied = sorted(int(i) for i in initial_state)
        if len(set(occupied)) != len(occupied):
            raise ValueError("Duplicate indices in initial state")

        fermi_gates = tuple(gates)
        majorana_gates = tuple(_fermi_gate_to_gate(g) for g in fermi_gates)
        parameters = tuple(parameters)

        if parameter_mapping is None:
            # Identity mapping: drop identity generators and their aligned parameter.
            kept = [i for i, gate in enumerate(majorana_gates) if gate.generator.terms]
            fermi_gates = tuple(fermi_gates[i] for i in kept)
            majorana_gates = tuple(majorana_gates[i] for i in kept)
            if parameters:
                parameters = tuple(parameters[i] for i in kept)

        object.__setattr__(self, "fermi_gates", fermi_gates)
        MajoranaCircuit.__init__(
            self,
            gates=majorana_gates,
            parameters=parameters,
            parameter_mapping=parameter_mapping,
            initial_state=tuple(occupied),
        )

    def __repr__(self) -> str:
        """Return a string representation of the circuit."""
        return (
            f"{self.__class__.__name__}("
            f"{len(self)} gates, "
            f"initial_state={list(self.initial_state)})"
        )
