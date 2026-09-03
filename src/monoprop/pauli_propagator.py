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

from .conversion_utils import _local_slots_to_pauli, _pauli_to_local_slots
from .monomial_propagator import MonomialPropagator
from .pauli import Pauli, PauliOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, ExpGate
    from .monomial_propagator import OperatorTerm, ParameterValues


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
        lower_atol: float | None = None,
        upper_atol: float | None = None,
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
        # The engine takes the Schrodinger cutoff in slots (two per qubit); this API in qubits.
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

    @property
    def num_qubits(self) -> int:
        """Number of qubits the propagator acts on."""
        return self._system_size

    def evolved_operator(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> PauliOperator:
        """Return the evolved operator as a [PauliOperator][monoprop.pauli.PauliOperator].

        Equivalent to
        [contract_partially][monoprop.monomial_propagator.MonomialPropagator.contract_partially]
        with ``inplace=False``, without touching the simulator state. The engine hands back raw
        symplectic-slot keys; this decodes them into qubit Pauli terms.

        Args:
            parameters: Variational parameter values (see
                [expectation_value][monoprop.monomial_propagator.MonomialPropagator.expectation_value]).
            atol: Terms with ``|coeff| < atol`` are dropped; ``0.0`` keeps all of them.

        Returns:
            The evolved qubit operator (Heisenberg picture) or evolved state (Schrodinger picture).
        """
        raw = self._simulator.evolved_operator(self._bind(parameters), atol)  # type: ignore[attr-defined]
        terms: dict[Pauli, complex] = {
            Pauli(*_local_slots_to_pauli(slots)): coeff for slots, coeff in raw.items()
        }
        return PauliOperator(terms, self.num_qubits)

    def _term_slots(self, term: OperatorTerm) -> tuple[int, ...]:
        """Encode a qubit Pauli term into the engine's symplectic slots.

        Only [Pauli][monoprop.pauli.Pauli] terms are accepted: the slots are an engine-internal
        encoding, so a raw slot sequence is rejected rather than passed through.

        Args:
            term: A [Pauli][monoprop.pauli.Pauli] term.

        Returns:
            The term's symplectic slot indices.

        Raises:
            TypeError: If the term is not a [Pauli][monoprop.pauli.Pauli].
        """
        if not isinstance(term, Pauli):
            raise TypeError(
                f"Pauli terms are Pauli objects; got {type(term).__name__}."
            )
        return _pauli_to_local_slots(term.string, term.qubits)

    def _circuit_gates(self, circuit: Circuit) -> Sequence[ExpGate]:
        """Accept a qubit circuit; its gates are expanded by the shared pipeline.

        A ``PauliPropagator`` rejects a Majorana/fermionic circuit.
        """
        if circuit.family == "majorana":
            raise TypeError(
                "PauliPropagator requires a qubit circuit; its gates are Majorana/fermionic. "
                "Use MajoranaPropagator for those."
            )
        return circuit.gates

    @property
    def n_gates(self) -> int:
        """Number of authoring gates ingested into the graph.

        A single-Pauli-term gate expands to one graph layer; a multi-term gate expands to
        several layers sharing one gate, so ``n_gates <= graph_layers``. Stays correct after
        a graph prefix is consumed by
        [contract_partially][monoprop.monomial_propagator.MonomialPropagator.contract_partially] /
        [propagate][monoprop.monomial_propagator.MonomialPropagator.propagate].
        """
        return super().n_gates

    @property
    def parameter_mapping(self) -> list[int]:
        """The parameter mapping owned by the graph, one entry per graph layer.

        Entry ``i`` is the variational-parameter index driving the ``i``-th graph layer (a
        generated Pauli operator), in the same order as the parameter vector passed to
        [expectation_value][monoprop.monomial_propagator.MonomialPropagator.expectation_value].
        This is the graph's native (per-Pauli operators)
        mapping, which is finer-grained than the per-gate mapping of the authoring
        [Circuit][monoprop.circuit.Circuit] when gates bundle several Pauli terms.
        """
        return super().parameter_mapping

    @parameter_mapping.setter
    def parameter_mapping(self, mapping: Sequence[int]) -> None:
        """Re-wire which parameter drives each gate/layer, without rebuilding the graph.

        The graph structure depends only on the generators, not the parameter labels, so
        this is a cheap relabel -- use it to tie or untie parameters on an already-built
        graph. The mapping may be given at either granularity and must be contiguous
        ``0..n-1``:

        - **per graph layer** (length [graph_layers][], in the parameter-vector order):
          relabels each generated Pauli monomial layer directly.
        - **per gate** (length [n_gates][], indexed by gate): expanded to per-layer via
          each layer's gate, so a multi-term Pauli gate's layers stay tied. This is the
          granularity of the authoring [Circuit][monoprop.circuit.Circuit]'s mapping.

        When the two lengths coincide the per-layer reading is used. Functionals created
        earlier keep the mapping they were built with; rebuild a functional to pick up the
        new one.
        """
        MonomialPropagator.parameter_mapping.fset(self, mapping)

    def build_graph(
        self,
        circuit: Circuit,
        *,
        seed_parameters: ParameterValues = None,
        only_rotate_len_k: int | None = None,
    ) -> None:
        """Append a circuit to the propagation graph.

        Builds (or extends) the reusable evolution graph, recording each layer's gate
        information (the parameter that drives it and its generator coefficient) so that
        later evaluation takes only ``parameters``. The circuit's angle indices are local
        (``0``-based); when extending a non-empty graph they are shifted up onto the
        accumulated parameter axis automatically, so each call's circuit is authored
        independently.

        Args:
            circuit: Gates to append, as a [Circuit][monoprop.circuit.Circuit].
            seed_parameters: The full parameter vector covering the whole accumulated graph,
                used to regenerate the coefficient seed (by contracting the existing graph) so
                coefficient truncation sees realistic coefficients when extending. Only needed
                when extending a non-empty graph *with* coefficient-informed truncation; on the
                first (or a single) call it defaults to the circuit's own parameters. When
                omitted while extending, the new layers are built structurally (coefficient
                truncation is skipped for them); the engine validates the length of an explicit
                seed.
            only_rotate_len_k: If provided, apply gates to Pauli terms of length <= k in the
                evolved operator even if they anticommute. Length is counted in the engine's
                slots, not in qubits: ``X`` or ``Y`` on a qubit costs one slot and ``Z``
                costs two, so ``k`` ranges up to ``2 * num_qubits``.
        """
        super().build_graph(
            circuit,
            seed_parameters=seed_parameters,
            only_rotate_len_k=only_rotate_len_k,
        )

    def update_initial_operator(self, new_operator: PauliOperator) -> None:
        """Replace the *initial operator* (existing terms only).

        Re-weights the initial operator the graph is evaluated against, without touching
        the evolution graph or rebuilding the simulator. Only the initial operator is
        affected -- the gates and their generator coefficients are unchanged. Unlike the
        base method, which takes the engine's raw symplectic-slot keys, this accepts qubit Pauli
        terms and encodes them via
        [get_local_operator][monoprop.pauli.PauliOperator.get_local_operator]. Functionals created
        earlier are invalidated; see
        [update_initial_operator][monoprop.monomial_propagator.MonomialPropagator.update_initial_operator].

        Args:
            new_operator: A [PauliOperator][monoprop.pauli.PauliOperator] whose terms replace the
                matching initial-operator coefficients.

        Raises:
            RuntimeError: In the Heisenberg picture, if a term in ``new_operator`` is not
                present in the current initial operator.
        """
        self._simulator.update_initial_operator(  # type: ignore[union-attr]
            new_operator.get_local_operator().terms
        )

    def size(self) -> int:
        """Number of Pauli operators currently tracked.

        Returns:
            The number of distinct Pauli operators in the simulator's current
            representation.
        """
        return super().size()

    @property
    def graph_layers(self) -> int:
        """Number of evolved Pauli operators (graph layers)."""
        return super().graph_layers

    @property
    def num_modes(self) -> int:
        """Number of qubits the propagator acts on -- the same value as [num_qubits][].

        The Pauli basis counts one mode per qubit (two slots each), so the inherited
        mode-count name and [num_qubits][] never disagree here; prefer [num_qubits][] in
        qubit-facing code.
        """
        return super().num_modes
