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
    from .monomial_propagator import ParameterValues
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

    @property
    def n_gates(self) -> int:
        """Number of authoring gates ingested into the graph.

        A single-Pauli-term gate expands to one graph layer; a multi-term gate expands to
        several layers sharing one gate, so ``n_gates <= graph_layers``. Stays correct after
        a graph prefix is consumed by [contract_partially][] / [propagate][].
        """
        return super().n_gates

    @property
    def parameter_mapping(self) -> list[int]:
        """The parameter mapping owned by the graph, one entry per graph layer.

        Entry ``i`` is the variational-parameter index driving the ``i``-th graph layer (a
        generated Pauli operator), in the same order as the parameter vector
        passed to [expectation_value][]. This is the graph's native (per-Pauli operators)
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
          relabels each Jordan-Wigner-mapped Majorana monomial layer directly.
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
            only_rotate_len_k: If provided, apply gates to Pauli operators of length <= k
                in the evolved operator (the qubit gates are Jordan-Wigner-mapped to Pauli
                operators first, so ``k`` ranges up to ``num_qubits``) even if they
                anticommute. In here, length is understood as the number of X and Z
                operators that are required to express a given Pauli term.
        """
        super().build_graph(
            circuit,
            seed_parameters=seed_parameters,
            only_rotate_len_k=only_rotate_len_k,
        )

    def evolved_operator(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> dict[tuple[int, ...], complex]:
        """Return the evolved operator/state as a dict, without modifying state.

        Equivalent to [contract_partially][] with ``inplace=False``, returned as a mapping
        keyed by Majorana indices -- the underlying representation is Majorana even for a
        qubit propagator (qubit gates are Jordan-Wigner-mapped to Majorana monomials) --
        without touching the simulator state.

        Args:
            parameters: Variational parameter values (see [expectation_value][]).
            atol: Absolute tolerance for filtering small coefficients; terms with
                ``|coeff| < atol`` are dropped. Defaults to ``1e-12``; set to ``0.0`` to
                keep all terms.

        Returns:
            The evolved operator (Heisenberg picture) or the evolved state (Schrodinger
            picture) as a dict mapping Majorana-index tuples to complex coefficients.
        """
        return super().evolved_operator(parameters, atol=atol)

    def update_initial_operator(
        self, new_operator: dict[tuple[int, ...], complex]
    ) -> None:
        """Replace coefficients of the *initial operator* (existing terms only).

        Re-weights the initial operator the graph is evaluated against, without touching
        the evolution graph or rebuilding the simulator. Only the initial operator is
        affected -- the gates and their generator coefficients are unchanged -- and only
        Majorana terms already present in the initial operator (the Jordan-Wigner-mapped
        representation of the qubit operator) can be updated (no new terms are introduced).

        Args:
            new_operator: Mapping from Majorana-index tuples to their new complex
                coefficients.

        Raises:
            RuntimeError: If a term in ``new_operator`` is not present in the current
                initial operator.
        """
        super().update_initial_operator(new_operator)

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
