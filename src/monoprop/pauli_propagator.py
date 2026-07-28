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

"""Pauli propagator: the qubit front-end of the shared monomial-propagation engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .conversion_utils import _local_slots_to_pauli
from .monomial_propagator import MonomialPropagator
from .pauli import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, ExpGate
    from .monomial_propagator import ParameterValues


class PauliPropagator(MonomialPropagator[PauliOperator]):
    """Classical simulator for qubit (Pauli) operators.

    Accepts a [PauliOperator][monoprop.pauli.PauliOperator] and a [Circuit][monoprop.circuit.Circuit] of
    qubit gates; see [MonomialPropagator][monoprop.monomial_propagator.MonomialPropagator] for the shared
    surface. The cutoff is qubit Pauli weight -- the number of qubits a retained term touches --
    so ``cutoff_type`` is fixed and read-only here.
    """

    def __init__(
        self,
        initial_operator: PauliOperator,
        initial_state: list[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None = None,
        lower_atol: None | float = None,
        upper_atol: None | float = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        """Initialize the qubit propagator.

        Args:
            initial_operator: Initial qubit operator; its ``num_qubits`` sizes the simulator.
            initial_state: Computational-basis reference (indices of qubits set to 1).
            cutoff: Maximum Pauli weight retained during evolution. The fully-paired exception
                described on [MajoranaPropagator.__init__][monoprop.majorana_propagator.MajoranaPropagator.__init__]
                still applies.
            schrodinger_cutoff: ``None`` (default) keeps the Heisenberg picture; an integer selects
                the Schrodinger picture and bounds the Pauli weight of the evolved state, including
                its initialization from ``initial_state``. Choose it at least as large as ``cutoff``
                for comparable accuracy.
            lower_atol: Monomials with ``|coeff| < lower_atol`` are discarded during evolution.
            upper_atol: Monomials with ``|coeff| > upper_atol`` are kept regardless of weight.
            comm: Optional MPI communicator (must outlive the propagator).
        """
        num_qubits = initial_operator.num_qubits
        # The engine takes the Schrodinger cutoff in gamma slots (two per qubit); this API in qubits.
        schrodinger_cutoff = (
            None if schrodinger_cutoff is None else 2 * schrodinger_cutoff
        )
        self._init_simulator(
            initial_operator.get_local_operator(),
            initial_state,
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            cutoff_type="support",
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            basis_change=None,
            comm=comm,
            basis="pauli",
        )
        # Must follow _init_simulator, which resets it to None.
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Number of qubits the propagator acts on."""
        # Always set in __init__; the base declares it Optional for the Majorana propagator.
        if self._num_qubits is None:
            raise RuntimeError("PauliPropagator has no qubit count set.")
        return self._num_qubits

    def evolved_operator(  # type: ignore[override]
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> PauliOperator:
        """Return the evolved operator as a [PauliOperator][monoprop.pauli.PauliOperator].

        Returns:
            The evolved qubit operator (Heisenberg picture) or evolved state (Schrodinger picture).
        """
        raw = super().evolved_operator(parameters, atol=atol)
        terms: dict[Pauli, complex] = {
            Pauli(*_local_slots_to_pauli(slots)): coeff for slots, coeff in raw.items()
        }
        return PauliOperator(terms, self.num_qubits)

    def _circuit_gates(self, circuit: Circuit) -> Sequence[ExpGate]:
        """Accept a qubit circuit and return its gates; reject a Majorana/fermionic one."""
        if circuit.family == "majorana":
            raise TypeError(
                "PauliPropagator requires a qubit circuit; its gates are Majorana/fermionic. "
                "Use MajoranaPropagator for those."
            )
        return circuit.gates

    def update_initial_operator(self, new_operator: PauliOperator) -> None:
        """Replace coefficients of the *initial operator* (existing terms only).

        Args:
            new_operator: :class:`~monoprop.pauli.PauliOperator` whose terms replace
                the matching initial-operator coefficients.

        Raises:
            RuntimeError: If a term in ``new_operator`` is not present in the current
                initial operator.
        """
        self._simulator.update_initial_operator(  # type: ignore[union-attr]
            new_operator.get_local_operator().terms
        )
