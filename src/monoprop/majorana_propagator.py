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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from .majorana import Majorana, MajoranaOperator
from .monomial_propagator import MonomialPropagator

if TYPE_CHECKING:
    import numpy as np
    from mpi4py import MPI

    from .circuit import Circuit, ExpGate
    from .monomial_propagator import OperatorTerm, ParameterValues
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
                as large as ``cutoff`` for comparable accuracy. It also sizes the initial paired
                basis, which holds ``sum(C(num_modes, k) for k in range(pairs + 1))`` monomials for
                ``pairs = ceil(schrodinger_cutoff / 2)`` — combinatorial in the cutoff and
                upper-bounded by ``2**num_modes``, so a value approaching ``2 * num_modes`` is
                rejected with a ``RuntimeError`` rather than started.
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
        """Return the evolved operator as a [MajoranaOperator][monoprop.majorana.MajoranaOperator].

        Equivalent to
        [contract_partially][monoprop.monomial_propagator.MonomialPropagator.contract_partially]
        with ``inplace=False``, without touching the simulator state. Unlike the base method, which
        hands back the engine's raw index tuples, this wraps them into a Majorana operator carrying
        the propagator's mode count.

        Args:
            parameters: Variational parameter values (see
                [expectation_value][monoprop.monomial_propagator.MonomialPropagator.expectation_value]).
            atol: Terms with ``|coeff| < atol`` are dropped; ``0.0`` keeps all of them.

        Returns:
            The evolved operator (Heisenberg picture) or evolved state (Schrodinger picture).
        """
        terms = self._simulator.evolved_operator(self._bind(parameters), atol)
        return MajoranaOperator(terms, self.num_modes)

    def _term_slots(self, term: OperatorTerm) -> tuple[int, ...]:
        """Encode a Majorana term into the engine's index tuple.

        Majorana indices *are* the engine's keys, so this only unwraps the term. A raw sequence goes
        through [Majorana][monoprop.majorana.Majorana], which rejects a non-canonical product rather
        than drop the anticommutation sign a lookup cannot carry; for one of those, use
        [Majorana.from_unsorted][monoprop.majorana.Majorana.from_unsorted] and apply its sign.

        Args:
            term: A [Majorana][monoprop.majorana.Majorana] term, or a raw index sequence.

        Returns:
            The term's Majorana indices.
        """
        if isinstance(term, Majorana):
            return term.indices
        # Iterable rather than Sequence: a NumPy array of indices is not a Sequence.
        if not isinstance(term, Iterable):
            raise TypeError(
                "Majorana terms are Majorana objects or index sequences; got "
                f"{type(term).__name__}."
            )
        return Majorana(*term).indices

    def update_initial_operator(self, new_operator: MajoranaOperator) -> None:
        """Replace the *initial operator* (existing terms only).

        Re-weights the initial operator the graph is evaluated against, without touching
        the evolution graph or rebuilding the simulator. Only the initial operator is
        affected -- the gates and their generator coefficients are unchanged. Functionals created
        earlier are invalidated; see
        [update_initial_operator][monoprop.monomial_propagator.MonomialPropagator.update_initial_operator].

        Args:
            new_operator: A [MajoranaOperator][monoprop.majorana.MajoranaOperator] whose terms
                replace the matching initial-operator coefficients.

        Raises:
            RuntimeError: In the Heisenberg picture, if a term in ``new_operator`` is not
                present in the current initial operator.
        """
        self._simulator.update_initial_operator(new_operator.terms)

    @MonomialPropagator.cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        """Set the cutoff type (``"length"`` or ``"support"``)."""
        self._simulator.cutoff_type = new_cutoff_type

    @property
    def n_gates(self) -> int:
        """Number of authoring gates ingested into the graph.

        A single-Majorana-monomial gate expands to one graph layer; a multi-monomial gate
        expands to several layers sharing one gate, so ``n_gates <= graph_layers``. Stays
        correct after a graph prefix is consumed by
        [contract_partially][monoprop.monomial_propagator.MonomialPropagator.contract_partially] /
        [propagate][monoprop.monomial_propagator.MonomialPropagator.propagate].
        """
        return super().n_gates

    @property
    def parameter_mapping(self) -> list[int]:
        """The parameter mapping owned by the graph, one entry per graph layer.

        Entry ``i`` is the variational-parameter index driving the ``i``-th graph layer (a
        generated Majorana monomial), in the same order as the parameter vector passed to
        [expectation_value][monoprop.monomial_propagator.MonomialPropagator.expectation_value].
        This is the graph's native (per-monomial) mapping, which
        is finer-grained than the per-gate mapping of the authoring
        [Circuit][monoprop.circuit.Circuit] when gates bundle several monomials.
        """
        return super().parameter_mapping

    @parameter_mapping.setter
    def parameter_mapping(self, mapping: Sequence[int]) -> None:
        """Re-wire which parameter drives each gate/layer, without rebuilding the graph.

        The graph structure depends only on the generators, not the parameter labels, so
        this is an efficient relabel -- use it to tie or untie parameters on an already-built
        graph. The mapping may be given at either granularity and must be contiguous
        ``0..n-1``:

        - **per graph layer** (length [graph_layers][]): entry ``i`` is the parameter index
          for layer ``i``.
        - **per gate** (length [n_gates][]): entry ``g`` is the parameter index for gate
          ``g``, and every layer that gate generated inherits it -- so the layers of a
          multi-monomial gate necessarily share one parameter. This is the granularity of
          the authoring [Circuit][monoprop.circuit.Circuit]'s mapping.

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
            only_rotate_len_k: If provided, apply gates to Majorana monomials of length <= k
                in the evolved operator even if they anticommute. Useful when many
                free-fermionic gates (generators that are length-2 Majorana monomials) are
                applied before expectation-value estimation in Schrodinger-picture
                simulations.
        """
        super().build_graph(
            circuit,
            seed_parameters=seed_parameters,
            only_rotate_len_k=only_rotate_len_k,
        )

    def size(self) -> int:
        """Number of Majorana monomials currently tracked.

        Returns:
            The number of distinct Majorana monomial terms in the simulator's current
            representation.
        """
        return super().size()

    @property
    def graph_layers(self) -> int:
        """Number of evolved Majorana monomials (graph layers)."""
        return super().graph_layers

    @property
    def num_modes(self) -> int:
        """Number of fermionic modes for the simulator."""
        return super().num_modes
