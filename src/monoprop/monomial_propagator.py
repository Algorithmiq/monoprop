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

"""Majorana and qubit propagators.

Both classes wrap the same compiled C++ Majorana simulator. Gate information (the
Majorana generators, their coefficients, and the parameter each drives) is owned by
the propagation graph, so evaluation methods take only ``parameters``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from monoprop._dispatch import dispatch

from .circuit import Gate, ParameterVector, Term, to_engine_arrays
from .conversion_utils import _extend_pauli_string, _pauli_to_fermi
from .monomial_data import MonomialOperator
from .utils import jordan_wigner_basis_change, validate_basis_change

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mpi4py import MPI

    from .circuit import Parameter, QubitGate
    from .pauli_data import PauliOperator
    from .quantum_data import IQuantumOperator

    ParameterValues = Sequence[float] | Mapping[Parameter, float] | np.ndarray | None

logger = logging.getLogger(__name__)


class MajoranaPropagator:
    """Classical simulator for Majorana operators.

    The propagation graph owns the gate information; evaluation methods
    (:meth:`expectation_value`, :meth:`gradient`, ...) take only ``parameters``.
    """

    def __init__(
        self,
        initial_operator: IQuantumOperator | MonomialOperator,
        initial_state: list[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None = None,
        cutoff_type: str = "length",
        lower_atol: None | float = None,
        upper_atol: None | float = None,
        basis_change: None | list[list[int]] = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        """Initialize the propagator.

        Args:
            initial_operator: Initial operator, either a :class:`MonomialOperator` or an
                object implementing ``get_monomial_operator()``.
            initial_state: Slater determinant (occupied mode indices) for the initial state.
            cutoff: Truncation parameter (meaning depends on ``cutoff_type``).
            schrodinger_cutoff: Optional Schrodinger-picture cutoff (enables that picture).
            cutoff_type: ``"length"`` (Majorana length) or ``"support"`` (orbital support).
            lower_atol: Optional lower coefficient-truncation tolerance.
            upper_atol: Optional upper coefficient-retention tolerance.
            basis_change: Optional Majorana basis change used by the cutoff function.
            comm: Optional MPI communicator (must outlive the propagator).
        """
        monomial_operator: MonomialOperator = (
            initial_operator
            if isinstance(initial_operator, MonomialOperator)
            else initial_operator.get_monomial_operator()
        )
        num_modes = monomial_operator.num_modes
        logger.debug(
            "__init__. num_modes=%d, cutoff=%d, schrodinger_cutoff=%s",
            num_modes,
            cutoff,
            schrodinger_cutoff,
        )
        validate_basis_change(basis_change, num_modes)

        self._comm = comm
        self._params = ParameterVector()
        self._simulator = dispatch(num_modes)(
            initial_operator=monomial_operator.terms,
            cutoff=cutoff,
            slater_determinant=list(initial_state),
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            cutoff_type=cutoff_type,
            basis_change=basis_change,
            comm=comm,
        )

    # -- gate ingestion ---------------------------------------------------------

    def _majorana_gates(self, gates: Sequence[Gate]) -> Sequence[Gate]:
        """Hook for subclasses to map their gate type to Majorana gates."""
        return gates

    def propagate_build_graph(
        self,
        gates: Sequence[Gate | QubitGate],
        parameters: ParameterValues = None,
        *,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Append gates to the propagation graph.

        The graph records each layer's gate information. When extending a non-empty graph
        with coefficient-informed truncation, pass ``parameters`` covering the whole
        accumulated graph plus these new gates; the seed is regenerated internally.

        Args:
            gates: Gates to append (Majorana :class:`~monoprop.circuit.Gate`, or
                :class:`~monoprop.circuit.QubitGate` for :class:`QubitPropagator`).
            parameters: Optional full parameter vector (see above).
            only_rotate_len_k: If > 0, rotate monomials of length <= k even if they
                anticommute.
        """
        majorana_gates = self._majorana_gates(gates)
        for gate in majorana_gates:
            self._params.register(gate.param)
        majoranas, gen_coeffs, parameter_mapping = to_engine_arrays(
            majorana_gates, self._params
        )
        bound = None if parameters is None else self._bind(parameters)
        self._simulator.propagate_build_graph(
            majoranas,
            parameter_mapping,
            gen_coeffs,
            bound,
            only_rotate_len_k,
        )

    def propagate(
        self,
        gates: Sequence[Gate | QubitGate],
        parameters: ParameterValues,
        *,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Evolve and contract immediately, without storing a graph.

        Args:
            gates: Gates to apply.
            parameters: Parameter values for the gates.
            only_rotate_len_k: See :meth:`propagate_build_graph`.
        """
        majorana_gates = self._majorana_gates(gates)
        local = ParameterVector()
        for gate in majorana_gates:
            local.register(gate.param)
        majoranas, gen_coeffs, parameter_mapping = to_engine_arrays(
            majorana_gates, local
        )
        self._simulator.propagate(
            majoranas,
            parameter_mapping,
            gen_coeffs,
            local.bind(parameters),
            only_rotate_len_k,
        )

    def pare(self, threshold: float | None = 1e-10) -> None:
        """Build and cache a pared execution plan over the current graph.

        Args:
            threshold: Edge-retention cutoff. ``None`` clears any cached plan.
        """
        self._simulator.pare(threshold)

    # -- evaluation -------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Number of distinct variational parameters seen while building the graph."""
        return len(self._params)

    def expectation_value(
        self,
        parameters: ParameterValues = None,
    ) -> float:
        """Compute the expectation value at ``parameters``."""
        return self._simulator.expectation_value(self._bind(parameters))

    def expectation_value_and_gradient(
        self,
        parameters: ParameterValues = None,
    ) -> tuple[float, np.ndarray]:
        """Compute the expectation value and gradient at ``parameters``."""
        value, grad = self._simulator.expectation_value_and_gradient(
            self._bind(parameters)
        )
        return value, np.asarray(grad, dtype=np.float64)

    def gradient(
        self,
        parameters: ParameterValues = None,
    ) -> np.ndarray:
        """Compute the gradient at ``parameters``."""
        return self.expectation_value_and_gradient(parameters)[1]

    def expectation_value_functional(self) -> Callable[..., float]:
        """Return a reusable callable computing the expectation value from parameters."""
        fn = self._simulator.expectation_value_functional()
        return lambda parameters=None: fn(self._bind(parameters))

    def expectation_value_and_gradient_functional(self) -> Callable[..., tuple]:
        """Return a reusable callable computing (expectation value, gradient)."""
        fn = self._simulator.expectation_value_and_gradient_functional()

        def _call(parameters=None):  # noqa: ANN001, ANN202
            value, grad = fn(self._bind(parameters))
            return value, np.asarray(grad, dtype=np.float64)

        return _call

    def contract_partially(
        self,
        parameters: ParameterValues = None,
        *,
        inplace: bool = True,
    ) -> np.ndarray:
        """Contract the graph into the operator/state at ``parameters``.

        Args:
            parameters: Parameter values.
            inplace: If True, update internal state (consuming the graph); otherwise
                return the evolved coefficients without modifying state.

        Returns:
            The evolved coefficients (core term excluded).
        """
        return np.asarray(
            self._simulator.contract_partially(self._bind(parameters), inplace)
        )

    def evolved_operator_dict(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> dict[tuple[int, ...], complex]:
        """Return the evolved operator/state as a dict, without modifying state."""
        return self._simulator.evolved_operator_dict(self._bind(parameters), atol)

    def evolved_operator(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> dict[tuple[int, ...], complex]:
        """Return the evolved operator (Heisenberg picture only)."""
        if self._simulator.schrodinger:
            raise ValueError(
                "Cannot call evolved_operator in Schrodinger picture. "
                "Use evolved_operator_dict instead."
            )
        return self.evolved_operator_dict(parameters, atol=atol)

    def update_coeffs(self, new_operator: dict[tuple[int, ...], complex]) -> None:
        """Replace the initial-operator coefficients (existing terms only)."""
        self._simulator.update_initial_operator(new_operator)

    # -- introspection ----------------------------------------------------------

    def size(self) -> int:
        """Number of Majorana terms currently tracked."""
        return self._simulator.size()

    def graph_size(self) -> tuple[int, int]:
        """(n_cos_indices, n_cycles) of the evolution graph."""
        return self._simulator.graph_size()

    @property
    def num_modes(self) -> int:
        """Number of fermionic modes."""
        return self._simulator.num_modes

    @property
    def graph_layers(self) -> int:
        """Number of evolved Majoranas (graph layers)."""
        return self._simulator.graph_layers()

    @property
    def schrodinger(self) -> bool:
        """Whether the simulator is in Schrodinger picture."""
        return self._simulator.schrodinger

    @property
    def cutoff(self) -> int:
        """Current cutoff value."""
        return self._simulator.cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff: int) -> None:
        self._simulator.cutoff = new_cutoff

    @property
    def lower_atol(self) -> None | float:
        """Current lower absolute tolerance."""
        return self._simulator.lower_atol

    @lower_atol.setter
    def lower_atol(self, new_lower_atol: None | float) -> None:
        self._simulator.lower_atol = new_lower_atol

    @property
    def upper_atol(self) -> None | float:
        """Current upper absolute tolerance."""
        return self._simulator.upper_atol

    @upper_atol.setter
    def upper_atol(self, new_upper_atol: None | float) -> None:
        self._simulator.upper_atol = new_upper_atol

    @property
    def cutoff_type(self) -> str:
        """Current cutoff type."""
        return self._simulator.cutoff_type

    @cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        self._simulator.cutoff_type = new_cutoff_type

    @property
    def basis_change(self) -> None | list[list[int]]:
        """Current basis change."""
        return self._simulator.basis_change

    @basis_change.setter
    def basis_change(self, new_basis_change: None | list[list[int]]) -> None:
        validate_basis_change(new_basis_change, self.num_modes)
        self._simulator.basis_change = new_basis_change

    # -- helpers ----------------------------------------------------------------

    def _bind(self, parameters: ParameterValues) -> list[float]:
        """Resolve ``parameters`` into a dense vector in canonical axis order."""
        if parameters is None:
            return []
        if hasattr(parameters, "keys"):
            return self._params.bind(parameters)
        return [float(v) for v in parameters]


class QubitPropagator(MajoranaPropagator):
    """Propagator for qubit (Pauli) operators, mapped to Majoranas via Jordan-Wigner.

    Accepts a :class:`~monoprop.pauli_data.PauliOperator` and
    :class:`~monoprop.circuit.QubitGate` gates; the Jordan-Wigner basis change is set
    automatically so cutoffs act on Pauli weight. Outputs remain in the Majorana basis.
    """

    def __init__(
        self,
        initial_operator: PauliOperator,
        initial_state: list[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None = None,
        cutoff_type: str = "length",
        lower_atol: None | float = None,
        upper_atol: None | float = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        """Initialize the qubit propagator (see :class:`MajoranaPropagator`)."""
        num_qubits = initial_operator.num_qubits
        self._num_qubits = num_qubits
        super().__init__(
            initial_operator.get_monomial_operator(),
            initial_state,
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            cutoff_type=cutoff_type,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            basis_change=jordan_wigner_basis_change(num_qubits),
            comm=comm,
        )

    def _majorana_gates(self, gates: Sequence[QubitGate]) -> list[Gate]:  # type: ignore[override]
        """Map qubit gates to Majorana gates via Jordan-Wigner, preserving handles."""
        majorana_gates: list[Gate] = []
        for qubit_gate in gates:
            terms: list[Term] = []
            for pauli, coefficient in zip(
                qubit_gate.paulis.strings,
                qubit_gate.paulis.coefficients,
                strict=True,
            ):
                extended = _extend_pauli_string(
                    pauli.string, qubit_gate.qubits, self._num_qubits
                )
                majorana, fermi_coeff = _pauli_to_fermi(extended)
                weight = len(majorana)
                gen_coeff = (
                    -coefficient * fermi_coeff / (1j) ** (weight * (weight - 1) / 2)
                )
                terms.append(Term(tuple(majorana), float(np.real(gen_coeff))))
            majorana_gates.append(Gate(qubit_gate.param, tuple(terms)))
        return majorana_gates
