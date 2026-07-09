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

"""Majorana propagator.

Concrete :class:`~monoprop.monomial_propagator.MonomialPropagator` that accepts Majorana (or
fermionic) operators and gates. Gate information (the Majorana generators, their coefficients,
and the parameter each drives) is owned by the propagation graph, so evaluation methods take
only ``parameters``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .majorana import MajoranaOperator
from .monomial_propagator import MonomialPropagator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, Exp
    from .quantum_data import IQuantumOperator


class MajoranaPropagator(MonomialPropagator):
    """Classical simulator for Majorana operators.

    Accepts a :class:`~monoprop.majorana.MajoranaOperator` (or any object implementing
    ``get_majorana_operator()``, such as a :class:`~monoprop.fermi.FermiOperator`)
    observable and a :class:`~monoprop.circuit.Circuit` of Majorana/fermionic
    :class:`~monoprop.circuit.Exp` gates. See
    :class:`~monoprop.monomial_propagator.MonomialPropagator` for the shared building,
    evaluation, and introspection surface.
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

        Creates a simulator for quantum-system evolution in the Majorana
        representation. Both Heisenberg (operator evolution, the default) and
        Schrodinger (state evolution) pictures are supported, with configurable
        truncation.

        Args:
            initial_operator: Initial operator, either a
                :class:`~monoprop.majorana.MajoranaOperator` or an object
                implementing ``get_majorana_operator()``.
            initial_state: Slater determinant (occupied mode indices) for the initial
                state.
            cutoff: Truncation parameter controlling the maximum complexity of the
                Majorana monomials retained during evolution; its meaning depends on
                ``cutoff_type``. Higher values increase accuracy at greater cost. A
                *fully paired* monomial -- one whose support consists entirely of
                complete pairs ``(m_{2j-1} m_{2j})`` on a mode -- is always kept
                regardless of this cutoff, because only paired monomials can contribute
                to an expectation value against a computational-basis state or Slater
                determinant; discarding them would throw away signal.
            schrodinger_cutoff: Optional cutoff for Schrodinger-picture evolution. If
                provided, enables the Schrodinger picture; if ``None``, the Heisenberg
                picture is used.
            cutoff_type: Truncation scheme (the fully-paired exception above always
                applies on top of either). ``"length"`` (default) keeps monomials
                whose length -- the number of Majorana operators -- does not exceed
                ``cutoff``; ``"support"`` keeps monomials acting on at most ``cutoff``
                distinct orbitals (their orbital support).
            lower_atol: Optional lower absolute-tolerance threshold for coefficient
                truncation. Monomials with ``|coeff| < lower_atol`` are discarded during
                evolution to improve performance.
            upper_atol: Optional upper absolute-tolerance threshold. Monomials with
                ``|coeff| > upper_atol`` are always retained regardless of their
                complexity, overriding cutoff-based truncation.
            comm: Optional MPI communicator. The communicator must remain valid for the
                simulator's lifetime.
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

    def _circuit_gates(self, circuit: Circuit) -> Sequence[Exp]:
        """Validate the circuit's gate family and return its gates for expansion.

        A ``MajoranaPropagator`` rejects a qubit circuit; the shared conversion lives in
        :func:`~monoprop.circuit.expand_monomials`.
        """
        if circuit.family == "pauli":
            raise TypeError(
                "MajoranaPropagator cannot consume a qubit circuit; its gates are Pauli. "
                "Use PauliPropagator for qubit circuits."
            )
        return circuit.gates

    @MonomialPropagator.cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        """Set the cutoff type (``"length"`` or ``"support"``)."""
        self._simulator.cutoff_type = new_cutoff_type
