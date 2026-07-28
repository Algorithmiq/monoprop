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

"""Majorana propagator: the fermionic front-end of the shared monomial-propagation engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .majorana import MajoranaOperator
from .monomial_propagator import MonomialPropagator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, ExpGate
    from .monomial_propagator import ParameterValues
    from .quantum_data import IQuantumOperator


class MajoranaPropagator(MonomialPropagator[MajoranaOperator]):
    """Classical simulator for Majorana operators.

    Accepts a [MajoranaOperator][monoprop.majorana.MajoranaOperator] (or any object implementing
    ``get_majorana_operator()``, such as a [FermiOperator][monoprop.fermi.FermiOperator]) and a
    [Circuit][monoprop.circuit.Circuit] of Majorana/fermionic gates. See
    [MonomialPropagator][monoprop.monomial_propagator.MonomialPropagator] for the shared surface.
    """

    def __init__(
        self,
        initial_operator: IQuantumOperator | MajoranaOperator,
        initial_state: Sequence[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None = None,
        cutoff_type: str = "length",
        lower_atol: None | float = None,
        upper_atol: None | float = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        """Initialize the propagator.

        Args:
            initial_operator: A [MajoranaOperator][monoprop.majorana.MajoranaOperator] or any object
                implementing ``get_majorana_operator()``.
            initial_state: Slater determinant, as occupied mode indices.
            cutoff: Bound on the complexity of the Majorana monomials retained during evolution,
                read according to ``cutoff_type``. A *fully paired* monomial -- support made up
                entirely of complete pairs ``(m_{2j-1} m_{2j})`` -- is kept regardless: only paired
                monomials contribute against a computational-basis state or Slater determinant.
            schrodinger_cutoff: ``None`` (default) keeps the Heisenberg picture; an integer selects
                the Schrodinger picture and bounds the evolved state -- including its initialization
                from ``initial_state`` -- by the same notion as ``cutoff_type``. Choose it at least
                as large as ``cutoff`` for comparable accuracy.
            cutoff_type: ``"length"`` (default) bounds the number of Majorana operators in a
                monomial; ``"support"`` bounds the distinct orbitals it acts on. The fully-paired
                exception applies on top of either.
            lower_atol: Monomials with ``|coeff| < lower_atol`` are discarded during evolution.
            upper_atol: Monomials with ``|coeff| > upper_atol`` are kept regardless of complexity.
            comm: Optional MPI communicator (must outlive the simulator).
        """
        majorana_operator: MajoranaOperator = (
            initial_operator
            if isinstance(initial_operator, MajoranaOperator)
            else initial_operator.get_majorana_operator()
        )
        self._init_simulator(
            majorana_operator,
            initial_state,
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            cutoff_type=cutoff_type,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            basis_change=None,
            comm=comm,
        )

    def _circuit_gates(self, circuit: Circuit) -> Sequence[ExpGate]:
        """Accept a Majorana/fermionic circuit and return its gates; reject a qubit one."""
        if circuit.family == "pauli":
            raise TypeError(
                "MajoranaPropagator cannot consume a qubit circuit; its gates are Pauli. "
                "Use PauliPropagator for qubit circuits."
            )
        return circuit.gates

    def evolved_operator(
        self, parameters: ParameterValues = None, *, atol: float = 1e-12
    ) -> MajoranaOperator:
        """Return the evolved operator as a :class:`~monoprop.majorana.MajoranaOperator`.

        Args:
            parameters: Variational parameter values (see :meth:`expectation_value`).
            atol: Absolute tolerance for filtering small coefficients; terms with
                ``|coeff| < atol`` are dropped. Defaults to ``1e-12``; set to ``0.0`` to
                keep all terms.

        Returns:
            The evolved operator (Heisenberg picture) or the evolved state (Schrodinger
            picture) as a :class:`~monoprop.majorana.MajoranaOperator`.
        """
        terms = self._simulator.evolved_operator(self._bind(parameters), atol)
        return MajoranaOperator(terms, self.num_modes)

    def update_initial_operator(self, new_operator: MajoranaOperator) -> None:
        """Replace coefficients of the *initial operator* (existing terms only).

        Args:
            new_operator: :class:`~monoprop.majorana.MajoranaOperator` whose terms
                replace the matching initial-operator coefficients.

        Raises:
            RuntimeError: If a term in ``new_operator`` is not present in the current
                initial operator.
        """
        self._simulator.update_initial_operator(new_operator.terms)

    @MonomialPropagator.cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        """Set the cutoff type (``"length"`` or ``"support"``)."""
        self._simulator.cutoff_type = new_cutoff_type
