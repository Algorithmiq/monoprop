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

import numpy as np

from .circuit import MajoranaGate, Term
from .conversion_utils import _extend_pauli_string, _pauli_to_fermi
from .majorana_propagator import MajoranaPropagator
from .utils import jordan_wigner_basis_change

if TYPE_CHECKING:
    from mpi4py import MPI

    from .circuit import Circuit
    from .pauli_data import PauliOperator


class PauliPropagator(MajoranaPropagator):
    """Propagator for qubit (Pauli) operators, mapped to Majoranas via Jordan-Wigner.

    Accepts a :class:`~monoprop.pauli_data.PauliOperator` and
    :class:`~monoprop.circuit.PauliGate` gates; the Jordan-Wigner basis change is set
    automatically so cutoffs act on Pauli weight. Outputs remain in the Majorana basis.

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
        self._num_qubits = num_qubits
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

    @property
    def cutoff_type(self) -> str:
        """Cutoff type, always ``"support"`` (Pauli weight) for a qubit propagator."""
        return self._simulator.cutoff_type

    @cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        if new_cutoff_type != "support":
            raise ValueError(
                "PauliPropagator only supports the 'support' cutoff type (Pauli "
                f"weight); got {new_cutoff_type!r}. Use MajoranaPropagator for "
                "length-based truncation."
            )
        self._simulator.cutoff_type = new_cutoff_type

    def _majorana_gates(self, circuit: Circuit) -> list[MajoranaGate]:
        """Map each qubit gate to one Majorana gate via Jordan-Wigner (1:1, mapping-preserving)."""
        majorana_gates: list[MajoranaGate] = []
        for pauli_gate in circuit.gates:
            terms: list[Term] = []
            for pauli, coefficient in zip(
                pauli_gate.paulis.strings,
                pauli_gate.paulis.coefficients,
                strict=True,
            ):
                extended = _extend_pauli_string(
                    pauli.string, pauli_gate.qubits, self._num_qubits
                )
                majorana, fermi_coeff = _pauli_to_fermi(extended)
                weight = len(majorana)
                gen_coeff = (
                    -coefficient * fermi_coeff / (1j) ** (weight * (weight - 1) / 2)
                )
                terms.append(Term(tuple(majorana), float(np.real(gen_coeff))))
            majorana_gates.append(MajoranaGate(tuple(terms)))
        return majorana_gates
