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

Wraps the compiled C++ Majorana simulator. Gate information (the Majorana generators,
their coefficients, and the parameter each drives) is owned by the propagation graph, so
evaluation methods take only ``parameters``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from monoprop._dispatch import dispatch

from .circuit import (
    Circuit,
    Exp,
    expand_monomials,
    validate_parameter_mapping,
)
from .majorana_data import MajoranaOperator
from .utils import validate_basis_change

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Self

    from mpi4py import MPI

    from .quantum_data import IQuantumOperator

    ParameterValues = Circuit | Sequence[float] | np.ndarray | None

logger = logging.getLogger(__name__)


class MajoranaPropagator:
    """Classical simulator for Majorana operators.

    The propagation graph owns the gate information; evaluation methods
    (:meth:`expectation_value`, :meth:`gradient`, ...) take only ``parameters``.

    .. note::
        **Incremental building and gate order.** In the Heisenberg picture the
        Heisenberg evolution applies gates back-to-front, so each
        :meth:`build_graph` / :meth:`propagate` call consumes its gate
        sequence in reverse. Splitting one circuit into forward chunks across several
        calls is therefore *not* equivalent to a single call with the whole sequence:
        the chunks are each reversed but not globally reordered. In the Schrodinger
        picture gates are applied front-to-back, so a forward split *is* equivalent.
        When you need incremental building to reproduce a single-call result, use the
        Schrodinger picture (or pass the full sequence in one call).
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
        basis_change: None | list[list[int]] = None,
        comm: MPI.Comm | None = None,
        shards: int = 0,
    ) -> None:
        """Initialize the propagator.

        Creates a simulator for quantum-system evolution in the Majorana
        representation. Both Heisenberg (operator evolution, the default) and
        Schrodinger (state evolution) pictures are supported, with configurable
        truncation.

        Args:
            initial_operator: Initial operator, either a
                :class:`~monoprop.majorana_data.MajoranaOperator` or an object
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
            basis_change: Optional basis transformation for the Majorana operators used
                by the cutoff function. If ``None``, the cutoff is measured in the
                standard Majorana representation. If provided, a list of ``2*num_modes``
                lists, each giving one basis vector as a set of Majorana indices.
            comm: Optional MPI communicator. The communicator must remain valid for the
                simulator's lifetime.
            shards: Intra-process shard count (default 0 ⇒ ``monoprop_SHARDS`` env, else 1).
                >1 partitions the operator across that many single-threaded shards pinned
                one-per-core; results are deterministic per shard count but differ from a single
                partition at the ULP level, as across MPI rank counts. Requires a single MPI rank.
        """
        majorana_operator: MajoranaOperator = (
            initial_operator
            if isinstance(initial_operator, MajoranaOperator)
            else initial_operator.get_majorana_operator()
        )
        # The operator carries its own mode count (a required constructor argument), so the
        # propagator reads it directly rather than validating it here.
        num_modes = majorana_operator.num_modes
        self._init_engine(
            majorana_operator.terms,
            num_modes,
            initial_state,
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            cutoff_type=cutoff_type,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            basis_change=basis_change,
            comm=comm,
            engine_basis="majorana",
            shards=shards,
        )

    def _init_engine(
        self,
        terms: dict[tuple[int, ...], complex],
        num_modes: int,
        initial_state: Sequence[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None,
        cutoff_type: str,
        lower_atol: float | None,
        upper_atol: float | None,
        basis_change: list[list[int]] | None,
        comm: MPI.Comm | None,
        engine_basis: str = "majorana",
        shards: int = 0,
    ) -> None:
        """Construct the dispatched C++ simulator and record the shared propagator state.

        Shared engine-init for both propagators: :class:`MajoranaPropagator` passes Majorana
        terms with ``engine_basis="majorana"``, while
        :class:`~monoprop.pauli_propagator.PauliPropagator` passes either JW-image Majorana
        terms (``"majorana"``) or native gamma-slot terms (``"pauli"``). The ``engine_basis``
        is stored on :attr:`_engine_basis` and threaded into gate expansion by
        :meth:`build_graph` / :meth:`propagate`; it is also handed to the C++ core as its
        operator ``basis``.

        Args:
            terms: The initial operator as an index-tuple -> coefficient mapping (Majorana or
                native gamma-slot indices, per ``engine_basis``).
            num_modes: The operator's mode / qubit-slot count (selects the C++ template).
            initial_state: Slater determinant (occupied mode indices) for the initial state.
            cutoff: See :meth:`__init__`.
            schrodinger_cutoff: See :meth:`__init__`.
            cutoff_type: See :meth:`__init__`.
            lower_atol: See :meth:`__init__`.
            upper_atol: See :meth:`__init__`.
            basis_change: See :meth:`__init__`.
            comm: See :meth:`__init__`.
            engine_basis: The C++ operator basis (``"majorana"`` or ``"pauli"``).
            shards: Intra-process shard count. 0 (default) reads the ``monoprop_SHARDS`` env var,
                falling back to a single partition. >1 partitions the operator across that many
                single-threaded shards pinned one-per-core (each an in-process ShmComm participant);
                results are deterministic per shard count but differ from a single partition at the
                ULP level, exactly as across MPI rank counts. Requires a single MPI rank.
        """
        logger.debug(
            "_init_engine. num_modes=%d, cutoff=%d, schrodinger_cutoff=%s, basis=%s",
            num_modes,
            cutoff,
            schrodinger_cutoff,
            engine_basis,
        )
        validate_basis_change(basis_change, num_modes)

        self._comm = comm
        self._n_params = 0
        self._engine_basis = engine_basis
        # System qubit count for expanding Pauli gates; set by PauliPropagator from the
        # observable. None for a native Majorana propagator (its gates need no qubit count).
        self._num_qubits: int | None = None
        self._initial_state = list(initial_state)
        # dispatch() is typed to return the base `type[_SimulatorAdapter]`, whose __init__ takes
        # extra positional args the generated per-mode subclasses fill in; the kwargs below match
        # the subclass __init__ that is actually returned.
        self._simulator = dispatch(num_modes)(  # type: ignore[call-arg]
            initial_operator=terms,
            cutoff=cutoff,
            slater_determinant=list(initial_state),
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            cutoff_type=cutoff_type,
            basis_change=basis_change,
            comm=comm,
            basis=engine_basis,
            shards=shards,
        )

    def __deepcopy__(self, memo: dict) -> "MajoranaPropagator":
        """Deep-copy the propagator into an independent instance.

        The heavy state (the dispatched simulator adapter and its C++ operator store) is
        deep-copied; the MPI communicator is shared as-is, which is correct both with MPI (the
        live mpi4py communicator) and without (``None``).
        """
        import copy as _copy

        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = value if key == "_comm" else _copy.deepcopy(value, memo)
        return new

    # -- gate ingestion ---------------------------------------------------------

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        initial_operator: IQuantumOperator | MajoranaOperator,
        **config: object,
    ) -> Self:
        """Construct a propagator from a circuit and propagate it in one step.

        Uses ``circuit.initial_state`` as the reference state and evolves ``circuit`` *in
        place* via :meth:`propagate` -- a one-shot contraction that stores **no** graph. The
        observable (``initial_operator``) and truncation settings (``cutoff`` and the rest of
        ``config``) are supplied separately -- they are not part of the circuit.

        Read the result off the evolved operator/state with :meth:`evolved_operator` (or
        :meth:`expectation_value`) taking **no** parameters -- the circuit's angles are already
        applied. This is the memory-lean path for a single evaluation; if you instead need a
        reusable graph to re-evaluate at many angles (or to take gradients), construct the
        propagator directly and call :meth:`build_graph`.

        Args:
            circuit: The circuit whose gates and initial state define the evolution.
            initial_operator: The observable to propagate.
            **config: Keyword arguments forwarded to the constructor (``cutoff``, etc.).

        Returns:
            A propagator with ``circuit`` already evolved into its state (no graph stored).
        """
        propagator = cls(initial_operator, list(circuit.initial_state), **config)  # type: ignore[arg-type]
        propagator.propagate(circuit)
        return propagator

    def _circuit_gates(self, circuit: Circuit) -> Sequence[Exp]:
        """Validate the circuit's gate family and return its gates for expansion.

        There is a single :class:`~monoprop.circuit.Circuit` type; the family is carried by
        the gates (see :attr:`~monoprop.circuit.Circuit.family`). A ``MajoranaPropagator``
        rejects a qubit circuit; the shared conversion lives in
        :func:`~monoprop.circuit.expand_monomials`.
        """
        if circuit.family == "pauli":
            raise TypeError(
                "MajoranaPropagator cannot consume a qubit circuit; its gates are Pauli. "
                "Use PauliPropagator for qubit circuits."
            )
        return circuit.gates

    def _check_initial_state(self, circuit: Circuit) -> None:
        """Reject a circuit whose reference state disagrees with the propagator's.

        A circuit's ``initial_state`` is advisory (the propagator was constructed with its
        own reference state); an empty one defers to the propagator. A non-empty one that
        names a different occupied set is almost certainly a mistake -- the circuit was
        authored against a different reference -- so fail loudly rather than silently
        evolving the wrong state. Occupied-index sets are order-insensitive.
        """
        if circuit.initial_state and sorted(circuit.initial_state) != sorted(
            self._initial_state
        ):
            raise ValueError(
                f"circuit.initial_state {list(circuit.initial_state)} does not match the "
                f"propagator's initial state {list(self._initial_state)}. Construct the "
                "propagator with this circuit's initial state (or via from_circuit)."
            )

    def build_graph(
        self,
        circuit: Circuit,
        *,
        seed_parameters: ParameterValues = None,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Append a circuit to the propagation graph.

        Builds (or extends) the reusable evolution graph, recording each layer's gate
        information (the parameter that drives it and its generator coefficient) so that
        later evaluation takes only ``parameters``. The circuit's angle indices are local
        (``0``-based); when extending a non-empty graph they are shifted up onto the
        accumulated parameter axis automatically, so each call's circuit is authored
        independently.

        Args:
            circuit: Gates to append, as a :class:`~monoprop.circuit.Circuit`.
            seed_parameters: The full parameter vector covering the whole accumulated graph,
                used to regenerate the coefficient seed (by contracting the existing graph) so
                coefficient truncation sees realistic coefficients when extending. Only needed
                when extending a non-empty graph *with* coefficient-informed truncation; on the
                first (or a single) call it defaults to the circuit's own parameters. When
                omitted while extending, the new layers are built structurally (coefficient
                truncation is skipped for them); the engine validates the length of an explicit
                seed.
            only_rotate_len_k: If > 0, apply gates to monomials of length <= k in the
                evolved operator even if they anticommute. Useful when many free-fermionic
                gates (generators that are length-2 Majorana monomials) are applied before
                expectation-value estimation in Schrodinger-picture simulations.
        """
        self._check_initial_state(circuit)
        # Resolve the coefficient seed handed to the engine (the operator coefficients the new
        # layers are contracted against while the graph is built, informing coefficient
        # truncation). The engine validates its length against the accumulated parameter axis.
        #  - An explicit seed_parameters is honored as given.
        #  - On the first build (empty graph) the circuit's own parameters are the whole axis.
        #  - When extending a non-empty graph without a seed the circuit's parameters cover only
        #    its local angles, not the accumulated axis, so there is no seed to give: build the
        #    new layers structurally (coefficient truncation applies only when a seed is
        #    supplied; pass seed_parameters to truncate an incremental extension).
        if seed_parameters is not None:
            seed = seed_parameters
        elif self.graph_layers == 0:
            seed = circuit.parameters
        else:
            seed = None
        gates = self._circuit_gates(circuit)
        num_qubits = self._num_qubits
        # Shift the circuit's local 0-based angle indices onto the accumulated axis.
        mapping = [self._n_params + m for m in circuit.resolved_mapping]
        self._n_params += circuit.n_parameters
        majoranas, gen_coeffs, per_monomial, gate_indices = expand_monomials(
            gates, mapping, num_qubits, self._engine_basis
        )
        # `seed` may be a NumPy array (an accepted ParameterValues type), so resolve to a list
        # first and treat an empty vector as "no seed" -- `if seed` would raise on an ndarray.
        bound = self._bind(seed) or None
        self._simulator.build_graph(
            majoranas,
            per_monomial,
            gen_coeffs,
            gate_indices,
            bound,
            only_rotate_len_k,
        )

    def propagate(
        self,
        circuit: Circuit,
        *,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Evolve and contract immediately, without storing a graph.

        More memory-efficient than :meth:`build_graph` because it does not retain
        the propagation graph; use it for a single contraction at the circuit's parameters
        rather than repeated re-evaluation.

        Args:
            circuit: Gates to apply and the angle values to apply them at, as a
                :class:`~monoprop.circuit.Circuit`.
            only_rotate_len_k: See :meth:`build_graph`.
        """
        self._check_initial_state(circuit)
        gates = self._circuit_gates(circuit)
        num_qubits = self._num_qubits
        majoranas, gen_coeffs, mapping, _gate_indices = expand_monomials(
            gates, circuit.resolved_mapping, num_qubits, self._engine_basis
        )
        self._simulator.propagate(
            majoranas,
            mapping,
            gen_coeffs,
            self._bind(circuit.parameters),
            only_rotate_len_k,
        )

    # -- evaluation -------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Number of distinct variational parameters seen while building the graph."""
        return self._n_params

    @property
    def n_gates(self) -> int:
        """Number of authoring gates ingested into the graph.

        A single-term gate expands to one graph layer; a multi-term gate expands to several
        layers sharing one gate, so ``n_gates <= graph_layers``. Stays correct after a graph
        prefix is consumed by :meth:`contract_partially` / :meth:`propagate`.
        """
        return self._simulator.n_gates()

    @property
    def parameter_mapping(self) -> list[int]:
        """The parameter mapping owned by the graph, one entry per graph layer.

        Entry ``i`` is the variational-parameter index driving the ``i``-th graph layer (a
        generated Majorana monomial), in the same order as the parameter vector passed to
        :meth:`expectation_value`. This is the graph's native (per-monomial) mapping, which
        is finer-grained than the per-gate mapping of the authoring
        :class:`~monoprop.circuit.Circuit` when gates bundle several monomials.
        """
        return list(self._simulator.parameter_mapping)

    @parameter_mapping.setter
    def parameter_mapping(self, mapping: Sequence[int]) -> None:
        """Re-wire which parameter drives each gate/layer, without rebuilding the graph.

        The graph structure depends only on the generators, not the parameter labels, so
        this is a cheap relabel -- use it to tie or untie parameters on an already-built
        graph. The mapping may be given at either granularity and must be contiguous
        ``0..n-1``:

        - **per graph layer** (length :attr:`graph_layers`, in the parameter-vector order):
          relabels each layer directly.
        - **per gate** (length :attr:`n_gates`, indexed by gate): expanded to per-layer via
          each layer's gate, so a multi-term gate's layers stay tied. This is the
          granularity of the authoring :class:`~monoprop.circuit.Circuit`'s mapping.

        When the two lengths coincide the per-layer reading is used. Functionals created
        earlier keep the mapping they were built with; rebuild a functional to pick up the
        new one.
        """
        resolved = [int(m) for m in mapping]
        n_layers, n_gates = self.graph_layers, self.n_gates
        if len(resolved) == n_layers:
            validate_parameter_mapping(resolved, n_layers, "graph layers")
        elif len(resolved) == n_gates:
            validate_parameter_mapping(resolved, n_gates, "gates")
        else:
            raise ValueError(
                f"parameter_mapping has {len(resolved)} entries; expected {n_layers} "
                f"(per graph layer) or {n_gates} (per gate)."
            )
        self._simulator.parameter_mapping = resolved
        self._n_params = max(resolved, default=-1) + 1

    def expectation_value(
        self,
        parameters: ParameterValues = None,
    ) -> float:
        """Compute the expectation value at ``parameters``.

        Replays the stored graph against the current initial operator and reference
        state. This is a convenience wrapper that builds and immediately evaluates an
        expectation-value functional with no paring.

        Args:
            parameters: Variational parameter values, as a sequence in parameter-index
                order, or a :class:`~monoprop.circuit.Circuit` (its parameters are used).
                ``None`` evaluates the current operator with an empty parameter vector.

        Returns:
            The expectation value as a float.
        """
        return self._simulator.expectation_value(self._bind(parameters))

    def expectation_value_and_gradient(
        self,
        parameters: ParameterValues = None,
    ) -> tuple[float, np.ndarray]:
        """Compute the expectation value and gradient at ``parameters``.

        Both quantities are computed in a single backward pass over the graph.

        Args:
            parameters: Variational parameter values (see :meth:`expectation_value`).

        Returns:
            A tuple ``(expectation_value, gradient)``, where ``gradient`` is a NumPy
            array in the canonical parameter-axis order.
        """
        value, grad = self._simulator.expectation_value_and_gradient(
            self._bind(parameters)
        )
        return value, np.asarray(grad, dtype=np.float64)

    def gradient(
        self,
        parameters: ParameterValues = None,
    ) -> np.ndarray:
        """Compute the gradient at ``parameters``.

        Args:
            parameters: Variational parameter values (see :meth:`expectation_value`).

        Returns:
            The gradient as a NumPy array of ``float64`` values, in canonical
            parameter-axis order.

        Note:
            Internally calls :meth:`expectation_value_and_gradient` and returns only its
            gradient component.
        """
        return self.expectation_value_and_gradient(parameters)[1]

    def expectation_value_functional(
        self, pare_threshold: float | None = None
    ) -> Callable[..., float]:
        """Return a reusable callable computing the expectation value from parameters.

        Returns a callable that accepts a parameter vector (a sequence, or ``None``) and
        returns the expectation value, replaying the current evolution graph. Build it once
        and call it repeatedly across many parameter values.

        Args:
            pare_threshold: Edge-retention cutoff for a masked execution plan built for
                this functional: edges whose contribution falls below the threshold are
                pared away so they are skipped during replay (a speed-up for sparse
                graphs, at the cost of some memory and accuracy). ``None`` (the default)
                disables paring.

        Returns:
            A callable ``fn(parameters=None) -> float``.
        """
        fn = self._simulator.expectation_value_functional(pare_threshold)
        return lambda parameters=None: fn(self._bind(parameters))

    def expectation_value_and_gradient_functional(
        self, pare_threshold: float | None = None
    ) -> Callable[..., tuple]:
        """Return a reusable callable computing (expectation value, gradient).

        Like :meth:`expectation_value_functional`, but the returned callable computes
        both the expectation value and the full parameter gradient in a single backward
        pass over the graph.

        Args:
            pare_threshold: See :meth:`expectation_value_functional`.

        Returns:
            A callable ``fn(parameters=None) -> (float, np.ndarray)``, where the
            gradient is in canonical parameter-axis order.
        """
        fn = self._simulator.expectation_value_and_gradient_functional(pare_threshold)

        def _call(parameters=None):  # noqa: ANN001, ANN202
            value, grad = fn(self._bind(parameters))
            return value, np.asarray(grad, dtype=np.float64)

        return _call

    def contract_partially(
        self,
        parameters: ParameterValues = None,
        *,
        inplace: bool = True,
    ) -> np.ndarray:
        """Contract the graph into the operator/state at ``parameters``.

        Permanently folds the stored gates, evaluated at ``parameters``, into the
        operand: the initial operator in the Heisenberg picture, or the reference state
        in the Schrodinger picture. This shrinks the graph that remains to be replayed,
        which is useful when a prefix of the circuit is fixed and its contribution can
        be baked in once instead of being replayed on every evaluation.

        Args:
            parameters: Variational parameter values (see :meth:`expectation_value`).
            inplace: If ``True`` (default), update the internal state, consuming the
                graph. If ``False``, leave the stored graph untouched and only return
                the contracted coefficients, so the same graph can be reused with other
                parameters.

        Returns:
            The evolved coefficients (core term excluded) as a NumPy array. In the
            Schrodinger picture these are the evolved-state coefficients; in the
            Heisenberg picture, the evolved-operator coefficients.
        """
        return np.asarray(
            self._simulator.contract_partially(self._bind(parameters), inplace)
        )

    def evolved_operator(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> dict[tuple[int, ...], complex]:
        """Return the evolved operator/state as a dict, without modifying state.

        Equivalent to :meth:`contract_partially` with ``inplace=False``, returned as a
        mapping keyed by Majorana indices and without touching the simulator state.

        Args:
            parameters: Variational parameter values (see :meth:`expectation_value`).
            atol: Absolute tolerance for filtering small coefficients; terms with
                ``|coeff| < atol`` are dropped. Defaults to ``1e-12``; set to ``0.0`` to
                keep all terms.

        Returns:
            The evolved operator (Heisenberg picture) or the evolved state (Schrodinger
            picture) as a dict mapping Majorana-index tuples to complex coefficients.
        """
        return self._simulator.evolved_operator(self._bind(parameters), atol)

    def update_initial_operator(
        self, new_operator: dict[tuple[int, ...], complex]
    ) -> None:
        """Replace coefficients of the *initial operator* (existing terms only).

        Re-weights the initial operator the graph is evaluated against, without touching
        the evolution graph or rebuilding the simulator. Only the initial operator is
        affected -- the gates and their generator coefficients are unchanged -- and only
        Majorana terms already present in the initial operator can be updated (no new
        terms are introduced).

        Args:
            new_operator: Mapping from Majorana-index tuples to their new complex
                coefficients.

        Raises:
            RuntimeError: If a term in ``new_operator`` is not present in the current
                initial operator.
        """
        self._simulator.update_initial_operator(new_operator)

    # -- introspection ----------------------------------------------------------

    def size(self) -> int:
        """Number of Majorana terms currently tracked.

        Returns:
            The number of distinct Majorana monomial terms in the simulator's current
            representation.
        """
        return self._simulator.size()

    def graph_size(self) -> tuple[int, int]:
        """Size metrics of the evolution graph.

        Returns:
            A tuple ``(n_cos_indices, n_cycles)``: the number of cosine indices and the
            number of cycles in the MP graph.
        """
        return self._simulator.graph_size()

    @property
    def num_modes(self) -> int:
        """Number of fermionic modes for the simulator."""
        return self._simulator.num_modes

    @property
    def basis(self) -> str:
        """The engine's operator basis: ``"majorana"`` (default) or ``"pauli"``.

        A plain :class:`MajoranaPropagator` is always ``"majorana"``;
        :class:`~monoprop.pauli_propagator.PauliPropagator` reports ``"pauli"`` in its native
        mode and ``"majorana"`` in the Jordan-Wigner fallback.
        """
        return self._simulator.basis

    @property
    def graph_layers(self) -> int:
        """Number of evolved Majoranas (graph layers)."""
        return self._simulator.graph_layers()

    @property
    def schrodinger(self) -> bool:
        """Whether the simulator is in the Schrodinger picture (else Heisenberg)."""
        return self._simulator.schrodinger

    @property
    def cutoff(self) -> int:
        """Current cutoff value for the simulation."""
        return self._simulator.cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff: int) -> None:
        self._simulator.cutoff = new_cutoff

    @property
    def lower_atol(self) -> None | float:
        """Current lower absolute tolerance for the cutoff function (``None`` if unset)."""
        return self._simulator.lower_atol

    @lower_atol.setter
    def lower_atol(self, new_lower_atol: None | float) -> None:
        self._simulator.lower_atol = new_lower_atol

    @property
    def upper_atol(self) -> None | float:
        """Current upper absolute tolerance for the cutoff function (``None`` if unset)."""
        return self._simulator.upper_atol

    @upper_atol.setter
    def upper_atol(self, new_upper_atol: None | float) -> None:
        self._simulator.upper_atol = new_upper_atol

    @property
    def cutoff_type(self) -> str:
        """Current cutoff type (``"length"`` or ``"support"``)."""
        return self._simulator.cutoff_type

    @cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        self._simulator.cutoff_type = new_cutoff_type

    @property
    def basis_change(self) -> None | list[list[int]]:
        """Current basis change for the cutoff function (``None`` if unset)."""
        return self._simulator.basis_change

    @basis_change.setter
    def basis_change(self, new_basis_change: None | list[list[int]]) -> None:
        validate_basis_change(new_basis_change, self.num_modes)
        self._simulator.basis_change = new_basis_change

    # -- helpers ----------------------------------------------------------------

    def _bind(self, parameters: ParameterValues) -> list[float]:
        """Resolve ``parameters`` into a dense vector in parameter-index order.

        Accepts a :class:`~monoprop.circuit.Circuit` (its ``parameters`` are used), a plain
        sequence of floats, or ``None`` (an empty vector).
        """
        if isinstance(parameters, Circuit):
            parameters = parameters.parameters
        if parameters is None:
            return []
        return [float(v) for v in parameters]
