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

"""Pauli propagator.

Concrete [MonomialPropagator][monoprop.monomial_propagator.MonomialPropagator] that accepts qubit (Pauli)
operators and gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .monomial_propagator import MonomialPropagator
from .utils import jordan_wigner_basis_change

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, ExpGate
    from .pauli import PauliOperator


class PauliPropagator(MonomialPropagator):
    """Classical simulator for qubit (Pauli) operators.

    Accepts a [PauliOperator][monoprop.pauli.PauliOperator] observable and a
    [Circuit][monoprop.circuit.Circuit] of qubit (Pauli) [ExpGate][monoprop.circuit.ExpGate] gates.
    See [MonomialPropagator][monoprop.monomial_propagator.MonomialPropagator] for the shared building,
    evaluation, and introspection surface.

    The cutoff is measured as qubit Pauli weight (the number of qubits a retained term
    touches); ``cutoff_type`` is fixed and read-only on this class.
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

        See [MonomialPropagator][monoprop.monomial_propagator.MonomialPropagator] for the shared
        arguments. The cutoff is always measured as Pauli weight, so ``cutoff`` bounds the
        number of qubits a retained term touches.

        Args:
            initial_operator: Initial qubit operator as a
                [PauliOperator][monoprop.pauli.PauliOperator].
            initial_state: Computational-basis reference (indices of qubits set to 1).
            cutoff: Maximum Pauli weight (number of qubits touched) retained during
                evolution. The fully-paired exception described in
                [MonomialPropagator][monoprop.monomial_propagator.MonomialPropagator] still applies.
            schrodinger_cutoff: Optional cutoff for Schrodinger-picture evolution. If
                provided, enables the Schrodinger picture, starting in a n initial state
                with terms truncated with that parameter; if ``None``, the Heisenberg
                picture is used. It is recommended that ``schrodinger_cutoff`` be slightly
                larger than ``cutoff`` for comparable accuracy.
            lower_atol: Optional lower coefficient-truncation tolerance.
            upper_atol: Optional upper coefficient-retention tolerance.
            comm: Optional MPI communicator (must outlive the propagator).
        """
        # The PauliOperator carries its own qubit count (a required constructor argument), so
        # the propagator reads it directly rather than validating it here.
        num_qubits = initial_operator.num_qubits

        # we have to multiply the Schrodinger cutoff by 2, because the Majorana
        # cutoff is measured in terms of Majorana operators, while PauliPropagator
        # measures it in terms of qubits. Each qubit corresponds to 2 Majorana operators.
        schrodinger_cutoff = (
            None if schrodinger_cutoff is None else 2 * schrodinger_cutoff
        )

        self._init_simulator(
            initial_operator.get_majorana_operator(),
            initial_state,
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            cutoff_type="support",
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            basis_change=jordan_wigner_basis_change(num_qubits),
            comm=comm,
        )
        # The qubit count comes from the observable and is carried into Pauli gate expansion
        # via build_graph (_init_simulator initializes it to None).
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Number of qubits the propagator acts on."""
        # Always set in __init__ (which raises if the observable has no qubit count); the base
        # declares it Optional for the native Majorana propagator.
        if self._num_qubits is None:
            raise RuntimeError("PauliPropagator has no qubit count set.")
        return self._num_qubits

    def _circuit_gates(self, circuit: Circuit) -> Sequence[ExpGate]:
        """Accept a qubit circuit; its gates are expanded by the shared pipeline.

        A ``PauliPropagator`` rejects a Majorana/fermionic circuit. The Jordan-Wigner mapping
        and antihermitian normalization live in [expand_monomials][monoprop.circuit.expand_monomials];
        the propagator's ``num_qubits`` (from the observable) reaches the expander via
        ``self._num_qubits``.
        """
        if circuit.family == "majorana":
            raise TypeError(
                "PauliPropagator requires a qubit circuit; its gates are Majorana/fermionic. "
                "Use MajoranaPropagator for those."
            )
        return circuit.gates
