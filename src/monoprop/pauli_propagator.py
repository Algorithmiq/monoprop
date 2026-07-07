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

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from .majorana_propagator import MajoranaPropagator
from .pauli_data import _SLOTS_BY_LETTER, Pauli, PauliOperator
from .utils import jordan_wigner_basis_change

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, Exp

    ParameterValues = Circuit | Sequence[float] | np.ndarray | None

#: Inverse of :data:`~monoprop.pauli_data._SLOTS_BY_LETTER`: the set of per-qubit gamma-slot
#: offsets present decodes back to the qubit's Pauli letter (``{0} -> X``, ``{1} -> Y``,
#: ``{0, 1} -> Z``). Used to decode a native evolved operator into a :class:`PauliOperator`.
_LETTER_BY_SLOTS: dict[frozenset[int], str] = {
    frozenset(offsets): letter for letter, offsets in _SLOTS_BY_LETTER.items()
}


class PauliPropagator(MajoranaPropagator):
    """Propagator for qubit (Pauli) operators.

    Accepts a :class:`~monoprop.pauli_data.PauliOperator` observable and a
    :class:`~monoprop.circuit.Circuit` of qubit (Pauli) :class:`~monoprop.circuit.Exp` gates.
    Two engine backings are available via ``engine_basis``:

    - ``"pauli"`` (the default): the **native** Pauli engine. The observable and gates are
      placed directly on gamma slots (``X_q -> slot 2q``, ``Y_q -> slot 2q+1``,
      ``Z_q -> both``) with no Jordan-Wigner phase; the C++ core runs in its Pauli basis, so
      no basis change is needed for the cutoff to measure Pauli weight. Use
      :meth:`evolved_pauli_operator` to read the evolved operator back as a
      :class:`~monoprop.pauli_data.PauliOperator`.
    - ``"majorana-jw"``: the Jordan-Wigner fallback (kept for A/B comparison). The observable
      is mapped to Majoranas and a Jordan-Wigner basis change is set automatically so cutoffs
      still act on Pauli weight. Outputs remain in the Majorana basis.

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
        engine_basis: Literal["pauli", "majorana-jw"] = "pauli",
        shards: int = 0,
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
            engine_basis: Engine backing. ``"pauli"`` (default) runs the native Pauli engine
                (no Jordan-Wigner); ``"majorana-jw"`` uses the Jordan-Wigner Majorana fallback.
            shards: Intra-process shard count (default 0 ⇒ ``monoprop_SHARDS`` env, else 1). >1
                partitions the operator across that many single-threaded shards pinned one-per-core
                for near-linear thread scaling; results are deterministic per shard count but differ
                from a single partition at the ULP level. Requires a single MPI rank.

        Raises:
            ValueError: If ``engine_basis`` is not ``"pauli"`` or ``"majorana-jw"``.
        """
        if engine_basis not in ("pauli", "majorana-jw"):
            raise ValueError(
                f"engine_basis must be 'pauli' or 'majorana-jw'; got {engine_basis!r}."
            )
        # The PauliOperator carries its own qubit count (a required constructor argument), so
        # the propagator reads it directly rather than validating it here.
        num_qubits = initial_operator.num_qubits
        if engine_basis == "pauli":
            # Native Pauli path: place the observable directly on gamma slots (no Jordan-Wigner)
            # and run the C++ core in its Pauli basis. No basis change is needed -- the Pauli
            # "support" cutoff already measures qubit weight.
            self._init_engine(
                initial_operator.get_symplectic_terms(),
                num_qubits,
                initial_state,
                cutoff=cutoff,
                schrodinger_cutoff=schrodinger_cutoff,
                cutoff_type="support",
                lower_atol=lower_atol,
                upper_atol=upper_atol,
                basis_change=None,
                comm=comm,
                engine_basis="pauli",
                shards=shards,
            )
        else:
            # Jordan-Wigner fallback (kept for A/B): map the observable to Majoranas and set the
            # JW basis change so the "support" cutoff still measures Pauli weight. The C++ core
            # runs in its Majorana basis (engine_basis="majorana", so `basis` reports "majorana");
            # outputs stay in the Majorana basis, so evolved_pauli_operator /
            # update_initial_pauli_operator raise -- read them via the MajoranaPropagator API.
            majorana_operator = initial_operator.get_majorana_operator()
            self._init_engine(
                majorana_operator.terms,
                majorana_operator.num_modes,
                initial_state,
                cutoff=cutoff,
                schrodinger_cutoff=schrodinger_cutoff,
                cutoff_type="support",
                lower_atol=lower_atol,
                upper_atol=upper_atol,
                basis_change=jordan_wigner_basis_change(num_qubits),
                comm=comm,
                engine_basis="majorana",
                shards=shards,
            )
        # Set after engine init (which resets it to None); the qubit count comes from the
        # observable and is carried into Pauli gate expansion via build_graph / propagate.
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Number of qubits the propagator acts on."""
        # Always set in __init__ (which raises if the observable has no qubit count); the base
        # declares it Optional for the native Majorana propagator.
        if self._num_qubits is None:
            raise RuntimeError("PauliPropagator has no qubit count set.")
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
        """The basis change fixed at construction (read-only).

        In the native Pauli engine (``engine_basis="pauli"``) there is **no** basis change --
        the Pauli-basis core measures qubit weight directly -- so this is ``None``. In the
        Jordan-Wigner fallback (``engine_basis="majorana-jw"``) it is the fixed Jordan-Wigner
        basis that makes the ``"support"`` cutoff act on Pauli weight. Either way it is
        read-only: unlike :class:`~monoprop.majorana_propagator.MajoranaPropagator` the setter
        is not exposed, since overwriting it would break the qubit-weight semantics.
        """
        return self._simulator.basis_change

    def evolved_pauli_operator(
        self, parameters: ParameterValues = None
    ) -> PauliOperator:
        """Return the evolved operator as a :class:`~monoprop.pauli_data.PauliOperator`.

        Native mode only (``engine_basis="pauli"``). Contracts the graph at ``parameters`` and
        decodes each stored gamma-slot term back to a qubit Pauli string: slots are grouped by
        qubit ``q = slot // 2`` with per-qubit offset ``slot % 2``, and the offset set maps to
        a letter via the inverse of the native encoding (``{0} -> X``, ``{1} -> Y``,
        ``{0, 1} -> Z``). Coefficients are real (the engine's identity decode). Small terms are
        dropped at the ``1e-12`` tolerance the binder rounds to.

        Args:
            parameters: Variational parameter values (see
                :meth:`~monoprop.majorana_propagator.MajoranaPropagator.expectation_value`).

        Returns:
            The evolved operator as a :class:`~monoprop.pauli_data.PauliOperator` on
            :attr:`num_qubits` qubits.

        Raises:
            NotImplementedError: In the ``"majorana-jw"`` fallback (the evolved operator is a
                Majorana operator there; use
                :meth:`~monoprop.majorana_propagator.MajoranaPropagator.evolved_operator`).
        """
        if self._engine_basis != "pauli":
            raise NotImplementedError(
                "evolved_pauli_operator is only available in the native Pauli engine "
                "(engine_basis='pauli'); in the 'majorana-jw' fallback the evolved operator "
                "is in the Majorana basis -- use evolved_operator instead."
            )
        evolved = self.evolved_operator(parameters)
        terms: dict[Pauli, complex] = {}
        for slots, coeff in evolved.items():
            offsets_by_qubit: dict[int, set[int]] = defaultdict(set)
            for slot in slots:
                offsets_by_qubit[slot // 2].add(slot % 2)
            qubits = sorted(offsets_by_qubit)
            string = "".join(
                _LETTER_BY_SLOTS[frozenset(offsets_by_qubit[q])] for q in qubits
            )
            terms[Pauli(string, tuple(qubits))] = complex(coeff).real
        return PauliOperator(terms, num_qubits=self.num_qubits)

    def update_initial_pauli_operator(self, op: PauliOperator) -> None:
        """Replace coefficients of the initial Pauli operator (native mode, existing terms).

        Re-weights the initial operator the graph is evaluated against, without touching the
        evolution graph. Only terms already present can be updated (see
        :meth:`~monoprop.majorana_propagator.MajoranaPropagator.update_initial_operator`); the
        Pauli terms are placed on their gamma slots via
        :meth:`~monoprop.pauli_data.PauliOperator.get_symplectic_terms`.

        Args:
            op: The new initial qubit operator.

        Raises:
            NotImplementedError: In the ``"majorana-jw"`` fallback (update the Majorana operator
                directly via
                :meth:`~monoprop.majorana_propagator.MajoranaPropagator.update_initial_operator`).
        """
        if self._engine_basis != "pauli":
            raise NotImplementedError(
                "update_initial_pauli_operator is only available in the native Pauli engine "
                "(engine_basis='pauli'); use update_initial_operator in the 'majorana-jw' "
                "fallback."
            )
        self._simulator.update_initial_operator(op.get_symplectic_terms())

    def _circuit_gates(self, circuit: Circuit) -> Sequence[Exp]:
        """Accept a qubit circuit; its gates are expanded by the shared pipeline.

        There is a single :class:`~monoprop.circuit.Circuit` type; the family is carried by
        the gates (see :attr:`~monoprop.circuit.Circuit.family`). A ``PauliPropagator`` rejects
        a Majorana/fermionic circuit. The Jordan-Wigner mapping and antihermitian
        normalization live in :func:`~monoprop.circuit.expand_monomials`; the propagator's
        ``num_qubits`` (from the observable) reaches the expander via ``self._num_qubits``.
        """
        if circuit.family == "majorana":
            raise TypeError(
                "PauliPropagator requires a qubit circuit; its gates are Majorana/fermionic. "
                "Use MajoranaPropagator for those."
            )
        return circuit.gates
