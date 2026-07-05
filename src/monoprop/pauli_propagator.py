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

Thin wrapper over :class:`~monoprop.majorana_propagator.MajoranaPropagator` that accepts
Pauli operators and gates, mapping them into the Majorana basis via Jordan-Wigner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .majorana_propagator import MajoranaPropagator
from .utils import jordan_wigner_basis_change

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, PauliExp
    from .pauli_data import PauliOperator


class PauliPropagator(MajoranaPropagator):
    """Propagator for qubit (Pauli) operators, mapped to Majoranas via Jordan-Wigner.

    Accepts a :class:`~monoprop.pauli_data.PauliOperator` observable and a
    :class:`~monoprop.circuit.Circuit` of :class:`~monoprop.circuit.PauliExp` gates; the
    Jordan-Wigner basis change is set automatically so cutoffs act on Pauli weight. Outputs
    remain in the Majorana basis.

    The cutoff type is fixed to ``"support"`` -- i.e. the qubit Pauli weight -- since
    that is the meaningful structural measure for qubit operators; ``"length"`` is not
    available on this class.
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

        See :class:`~monoprop.majorana_propagator.MajoranaPropagator` for the shared
        arguments. The cutoff is always measured as Pauli weight
        (``cutoff_type="support"``), so ``cutoff`` bounds the number of qubits a retained
        term touches.

        Args:
            initial_operator: Initial qubit operator as a
                :class:`~monoprop.pauli_data.PauliOperator`.
            initial_state: Computational-basis reference (indices of qubits set to 1).
            cutoff: Maximum Pauli weight (number of qubits touched) retained during
                evolution. The fully-paired exception described in
                :class:`~monoprop.majorana_propagator.MajoranaPropagator` still applies.
            schrodinger_cutoff: Optional Schrodinger-picture cutoff (enables that
                picture).
            lower_atol: Optional lower coefficient-truncation tolerance.
            upper_atol: Optional upper coefficient-retention tolerance.
            comm: Optional MPI communicator (must outlive the propagator).
        """
        num_qubits = initial_operator.num_qubits
        if num_qubits is None:
            raise ValueError(
                "The initial PauliOperator has num_qubits=None; PauliPropagator needs the "
                "qubit count. Construct the operator with an explicit num_qubits."
            )
        super().__init__(
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
        # Set after super().__init__ (which resets it to None); the qubit count comes from
        # the observable and is carried into PauliExp gate expansion via build_graph.
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Number of qubits the propagator acts on."""
        return self._num_qubits

    @property
    def cutoff_type(self) -> str:
        """Cutoff type, always ``"support"`` (Pauli weight) for a qubit propagator.

        Read-only: a qubit propagator's cutoff is fixed to Pauli weight. Use
        :class:`~monoprop.majorana_propagator.MajoranaPropagator` for length-based
        truncation.
        """
        return self._simulator.cutoff_type

    @property
    def basis_change(self) -> list[list[int]] | None:
        """The Jordan-Wigner basis change, fixed at construction (read-only).

        A qubit propagator's cutoff acts on Pauli weight via a fixed Jordan-Wigner basis;
        overwriting it would break that semantics, so unlike
        :class:`~monoprop.majorana_propagator.MajoranaPropagator` the setter is not exposed.
        """
        return self._simulator.basis_change

    def _circuit_gates(self, circuit: Circuit) -> Sequence[PauliExp]:
        """Accept a qubit (PauliExp) circuit; its gates are expanded by the shared pipeline.

        There is a single :class:`~monoprop.circuit.Circuit` type; the family is carried by
        the gates (see :attr:`~monoprop.circuit.Circuit.family`). A ``PauliPropagator`` rejects
        a Majorana/fermionic circuit. The Jordan-Wigner mapping and antihermitian
        normalization live in :func:`~monoprop.circuit.expand_monomials`; the propagator's
        ``num_qubits`` (from the observable) reaches the expander via ``self._num_qubits``.
        """
        if circuit.family == "majorana":
            raise TypeError(
                "PauliPropagator requires a qubit (PauliExp) circuit; its gates are "
                "Majorana/fermionic. Use MajoranaPropagator for those."
            )
        return circuit.gates
