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

"""Authoring types for Majorana/qubit gate sequences.

Gates are described with a :class:`Parameter` handle model: each :class:`Gate`
bundles the generated Majorana monomials driven by a single variational angle, and
sharing a parameter across gates is expressed by reusing the same :class:`Parameter`
handle (rather than a hand-maintained ``param_inds`` list). The propagator compiles
these into the dense ``(majoranas, gen_coeffs, parameter_mapping)`` arrays the C++
engine consumes; the graph then owns that gate information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from .monomial_data import MonomialCircuit
    from .pauli_data import PauliEvCircuit, PauliOperator

_parameter_counter = count()


@dataclass(frozen=True, slots=True)
class Parameter:
    """Handle for a single variational angle.

    Identity is by object (not by ``name``): reuse the *same* ``Parameter`` instance
    across gates to tie their angle. Two parameters created separately are distinct
    axes even if they share a name.

    Attributes:
        name: Optional human-readable label.
    """

    name: str | None = None
    _id: int = field(default_factory=lambda: next(_parameter_counter))


@dataclass(frozen=True, slots=True)
class Term:
    """One generated Majorana monomial and its structural coefficient.

    Attributes:
        majorana: Majorana indices of the monomial.
        gen_coeff: Generator coefficient ``g`` (the layer angle is
            ``parameters[param_index] * gen_coeff``).
    """

    majorana: tuple[int, ...]
    gen_coeff: float


@dataclass(frozen=True, slots=True)
class Gate:
    """The bundle of Majorana monomials driven by a single variational angle.

    Attributes:
        param: The variational parameter handle shared by all ``terms``.
        terms: The generated monomials.
    """

    param: Parameter
    terms: tuple[Term, ...]


@dataclass(frozen=True, slots=True)
class QubitGate:
    """A qubit (Pauli) evolution gate parameterised by a handle.

    The qubit analogue of :class:`Gate`; ``QubitPropagator`` maps it into Majorana
    :class:`Gate` objects via the Jordan-Wigner transform, preserving the parameter
    handle so cross-gate sharing survives the mapping.

    Attributes:
        param: The variational parameter handle.
        qubits: Qubit indices the gate acts on.
        paulis: Commuting Pauli operators applied on ``qubits``.
    """

    param: Parameter
    qubits: tuple[int, ...]
    paulis: PauliOperator


class ParameterVector:
    """Ordered registry of :class:`Parameter` handles.

    Registration order defines the canonical axis of the flat parameter vector (and of
    the returned gradient). Each distinct handle maps to one dense index.
    """

    def __init__(self, params: Iterable[Parameter] = ()) -> None:
        """Initialize the registry, registering ``params`` in order."""
        self._params: list[Parameter] = []
        self._index: dict[Parameter, int] = {}
        for param in params:
            self.register(param)

    def new(self, name: str | None = None) -> Parameter:
        """Create a fresh parameter, register it, and return the handle."""
        param = Parameter(name)
        self.register(param)
        return param

    def register(self, param: Parameter) -> int:
        """Register ``param`` if unseen; return its dense index."""
        if param not in self._index:
            self._index[param] = len(self._params)
            self._params.append(param)
        return self._index[param]

    def index(self, param: Parameter) -> int:
        """Return the dense index of a registered ``param``."""
        return self._index[param]

    def __len__(self) -> int:
        """Number of distinct registered parameters."""
        return len(self._params)

    def __iter__(self) -> Iterator[Parameter]:
        """Iterate parameters in registration (axis) order."""
        return iter(self._params)

    def bind(self, values: Mapping[Parameter, float] | Sequence[float]) -> list[float]:
        """Resolve parameter values into a dense vector in canonical axis order.

        Args:
            values: Either a mapping from handle to value, or a sequence already in
                canonical order.

        Returns:
            The dense parameter vector.

        Raises:
            ValueError: If a sequence has the wrong length.
        """
        if hasattr(values, "keys"):
            return [float(values[param]) for param in self._params]  # type: ignore[index]
        vector = [float(v) for v in values]  # type: ignore[union-attr]
        if len(vector) != len(self._params):
            raise ValueError(
                f"Expected {len(self._params)} parameters, got {len(vector)}."
            )
        return vector


def to_engine_arrays(
    gates: Sequence[Gate],
    params: ParameterVector,
) -> tuple[list[tuple[int, ...]], list[float], list[int]]:
    """Compile Majorana gates into the dense engine arrays.

    Args:
        gates: Majorana gates to compile, in application order.
        params: Registry giving each gate's parameter its dense index.

    Returns:
        A tuple ``(majoranas, gen_coeffs, parameter_mapping)`` for the C++ engine.
    """
    majoranas: list[tuple[int, ...]] = []
    gen_coeffs: list[float] = []
    parameter_mapping: list[int] = []
    for gate in gates:
        param_index = params.index(gate.param)
        for term in gate.terms:
            majoranas.append(tuple(term.majorana))
            gen_coeffs.append(float(term.gen_coeff))
            parameter_mapping.append(param_index)
    return majoranas, gen_coeffs, parameter_mapping


def gates_from_monomial_circuit(
    circuit: MonomialCircuit,
) -> tuple[list[Gate], ParameterVector]:
    """Regroup a dense :class:`MonomialCircuit` into gates + a parameter vector.

    Consecutive monomials sharing a ``param_ind`` become one :class:`Gate`; equal
    ``param_ind`` values reuse the same :class:`Parameter` handle (so weight-tying is
    preserved). Dense indices are kept identical to the original ``param_inds`` so the
    ``parameters`` vector still lines up.

    Args:
        circuit: The transport circuit to convert.

    Returns:
        The gates (in order) and the :class:`ParameterVector` indexing them.
    """
    param_inds = [int(p) for p in circuit.param_inds]
    params = ParameterVector()
    handles = [params.new() for _ in range(max(param_inds) + 1 if param_inds else 0)]

    gates: list[Gate] = []
    current_index: int | None = None
    current_terms: list[Term] = []
    for maj, coeff, pidx in zip(
        circuit.majoranas, circuit.gen_coeffs, param_inds, strict=True
    ):
        if current_terms and pidx != current_index:
            gates.append(Gate(handles[current_index], tuple(current_terms)))
            current_terms = []
        current_index = pidx
        current_terms.append(Term(tuple(maj), float(coeff)))
    if current_terms:
        gates.append(Gate(handles[current_index], tuple(current_terms)))

    return gates, params


def gates_from_pauli_circuit(
    circuit: PauliEvCircuit,
) -> tuple[list[QubitGate], ParameterVector]:
    """Regroup a :class:`PauliEvCircuit` into qubit gates + a parameter vector.

    One :class:`QubitGate` per Pauli evolution gate, each with its own fresh parameter
    handle (matching the one-parameter-per-gate transport layout).

    Args:
        circuit: The Pauli evolution circuit to convert.

    Returns:
        The qubit gates (in order) and the :class:`ParameterVector` indexing them.
    """
    params = ParameterVector()
    gates = [
        QubitGate(params.new(), tuple(gate.qubits), gate.paulis)
        for gate in circuit.gates
    ]
    return gates, params
