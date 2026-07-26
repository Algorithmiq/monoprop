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

"""Monomial propagator base class.

Shared engine for the two concrete simulators
([MajoranaPropagator][monoprop.majorana_propagator.MajoranaPropagator] and
[PauliPropagator][monoprop.pauli_propagator.PauliPropagator]). Both wrap the
compiled C++ simulator and differ only in how they are constructed (which operator
family they accept); the graph building, evaluation, and introspection surface lives here.

Gate information (the generators, their coefficients, and the parameter each drives)
is owned by the propagation graph, so evaluation methods take only ``parameters``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from monoprop._dispatch import dispatch

from .circuit import (
    Circuit,
    ExpGate,
    expand_monomials,
    validate_parameter_mapping,
)
from .utils import validate_basis_change

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Self

    from mpi4py import MPI

    from .majorana import MajoranaOperator

    ParameterValues = Circuit | Sequence[float] | np.ndarray | None

logger = logging.getLogger(__name__)


class MonomialPropagator(ABC):
    """Abstract base for the classical monomial-propagation simulators.

    The propagation graph owns the gate information; evaluation methods
    ([expectation_value][], [gradient][], ...) take only ``parameters``.
    Concrete subclasses implement ``__init__`` (resolving their operator family to a
    [MajoranaOperator][monoprop.majorana.MajoranaOperator] and calling `_init_simulator`)
    and `_circuit_gates` (validating the circuit's gate family).

    .. note::
        **Incremental building and gate order.** In the Heisenberg picture the
        Heisenberg evolution applies gates back-to-front, so each
        [build_graph][] / [propagate][] call consumes its gate
        sequence in reverse. Splitting one circuit into forward chunks across several
        calls is therefore *not* equivalent to a single call with the whole sequence:
        the chunks are each reversed but not globally reordered. In the Schrodinger
        picture gates are applied front-to-back, so a forward split *is* equivalent.
        When you need incremental building to reproduce a single-call result, use the
        Schrodinger picture (or pass the full sequence in one call).
    """

    _comm: MPI.Comm | None
    _n_params: int
    _num_qubits: int | None
    _initial_state: list[int]
    _simulator: object

    def _init_simulator(
        self,
        majorana_operator: MajoranaOperator,
        initial_state: Sequence[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None,
        cutoff_type: str,
        lower_atol: None | float,
        upper_atol: None | float,
        basis_change: None | list[list[int]],
        comm: MPI.Comm | None,
    ) -> None:
        """Dispatch to the compiled per-mode simulator and record shared state.

        Called by each concrete subclass's ``__init__`` once it has resolved its operator
        family to a [MajoranaOperator][monoprop.majorana.MajoranaOperator]. The cutoff ``basis_change``
        is an internal detail chosen by the subclass (``None`` for a native Majorana
        propagator, Jordan-Wigner for a qubit one) -- it is not part of the public surface.
        The operator carries its own mode count (a required constructor argument), so the
        propagator reads it directly rather than validating it here.
        """
        num_modes = majorana_operator.num_modes
        logger.debug(
            "_init_simulator. num_modes=%d, cutoff=%d, schrodinger_cutoff=%s",
            num_modes,
            cutoff,
            schrodinger_cutoff,
        )
        validate_basis_change(basis_change, num_modes)

        self._comm = comm
        self._n_params = 0
        # System qubit count for expanding Pauli gates; set by PauliPropagator from the
        # observable. None for a native Majorana propagator (its gates need no qubit count).
        self._num_qubits = None
        self._initial_state = list(initial_state)
        # dispatch() is typed to return the base `type[_SimulatorAdapter]`, whose __init__ takes
        # extra positional args the generated per-mode subclasses fill in; the kwargs below match
        # the subclass __init__ that is actually returned.
        self._simulator = dispatch(num_modes)(  # type: ignore[call-arg]
            initial_operator=majorana_operator.terms,
            cutoff=cutoff,
            slater_determinant=list(initial_state),
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            cutoff_type=cutoff_type,
            basis_change=basis_change,
            comm=comm,
        )

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        initial_operator: object,
        **config: object,
    ) -> Self:
        """Construct a propagator from a circuit and propagate it in one step.

        Uses ``circuit.initial_state`` as the reference state and evolves ``circuit`` *in
        place* via [propagate][] -- a one-shot contraction that stores **no** graph. The
        observable (``initial_operator``) and truncation settings (``cutoff`` and the rest of
        ``config``) are supplied separately -- they are not part of the circuit.

        Read the result off the evolved operator/state with [evolved_operator][] (or
        [expectation_value][]) taking **no** parameters -- the circuit's angles are already
        applied. This is the memory-lean path for a single evaluation; if you instead need a
        reusable graph to re-evaluate at many angles (or to take gradients), construct the
        propagator directly and call [build_graph][].

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

    @abstractmethod
    def _circuit_gates(self, circuit: Circuit) -> Sequence[ExpGate]:
        """Validate the circuit's gate family and return its gates for expansion.

        There is a single [Circuit][monoprop.circuit.Circuit] type; the family is carried by
        the gates (see [family][monoprop.circuit.Circuit.family]). Each concrete propagator
        accepts one family and rejects the other; the shared conversion lives in
        [expand_monomials][monoprop.circuit.expand_monomials].
        """
        raise NotImplementedError

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

    def _validate_and_correct_only_rotate_len_k(
        self, only_rotate_len_k: int | None
    ) -> int:
        """Validate and correct the optional only_rotate_len_k argument.

        The argument is a positive integer in the range ``1..2*num_qubits`` (inclusive) or
        ``None``. If ``None``, it is replaced with ``0`` to indicate "no restriction" to the
        simulator. The simulator interprets ``0`` as "apply all gates to all monomials",
        so this is a convenient way to avoid passing an extra argument when no restriction
        is needed.

        Args:
            only_rotate_len_k: Optional length cutoff for gate application.

        Returns:
            The validated and corrected cutoff, with ``None`` replaced by ``0``.
        """
        if only_rotate_len_k is None:
            return 0
        if only_rotate_len_k <= 0 or (
            isinstance(self._num_qubits, int)
            and only_rotate_len_k > 2 * self._num_qubits
        ):
            raise ValueError(
                f"only_rotate_len_k={only_rotate_len_k} is out of range; must be 0 < k <= 2*num_qubits "
            )
        return only_rotate_len_k

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
            only_rotate_len_k: If provided, apply gates to terms of length <= k in the
                evolved operator even if they anticommute. Useful when many free-fermionic
                gates (generators that are length-2 terms) are applied before
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
        only_rotate_len_k = self._validate_and_correct_only_rotate_len_k(
            only_rotate_len_k
        )

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
            gates, mapping, num_qubits
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
        self, circuit: Circuit, *, only_rotate_len_k: int | None = None
    ) -> None:
        """Evolve and contract immediately, without storing a graph.

        More memory-efficient than [build_graph][] because it does not retain
        the propagation graph; use it for a single contraction at the circuit's parameters
        rather than repeated re-evaluation.

        Args:
            circuit: Gates to apply and the angle values to apply them at, as a
                [Circuit][monoprop.circuit.Circuit].
            only_rotate_len_k: See [build_graph][].
        """
        only_rotate_len_k = self._validate_and_correct_only_rotate_len_k(
            only_rotate_len_k
        )
        self._check_initial_state(circuit)
        gates = self._circuit_gates(circuit)
        num_qubits = self._num_qubits
        majoranas, gen_coeffs, mapping, _gate_indices = expand_monomials(
            gates, circuit.resolved_mapping, num_qubits
        )
        self._simulator.propagate(
            majoranas,
            mapping,
            gen_coeffs,
            self._bind(circuit.parameters),
            only_rotate_len_k,
        )

    @property
    def n_parameters(self) -> int:
        """Number of distinct variational parameters seen while building the graph."""
        return self._n_params

    @property
    def n_gates(self) -> int:
        """Number of authoring gates ingested into the graph.

        A single-term gate expands to one graph layer; a multi-term gate expands to several
        layers sharing one gate, so ``n_gates <= graph_layers``. Stays correct after a graph
        prefix is consumed by [contract_partially][] / [propagate][].
        """
        return self._simulator.n_gates()

    @property
    def parameter_mapping(self) -> list[int]:
        """The parameter mapping owned by the graph, one entry per graph layer.

        Entry ``i`` is the variational-parameter index driving the ``i``-th graph layer (a
        generated term), in the same order as the parameter vector passed to
        [expectation_value][]. This is the graph's native (per-term) mapping, which
        is finer-grained than the per-gate mapping of the authoring
        [Circuit][monoprop.circuit.Circuit] when gates bundle several terms.
        """
        return list(self._simulator.parameter_mapping)

    @parameter_mapping.setter
    def parameter_mapping(self, mapping: Sequence[int]) -> None:
        """Re-wire which parameter drives each gate/layer, without rebuilding the graph.

        The graph structure depends only on the generators, not the parameter labels, so
        this is a cheap relabel -- use it to tie or untie parameters on an already-built
        graph. The mapping may be given at either granularity and must be contiguous
        ``0..n-1``:

        - **per graph layer** (length [graph_layers][], in the parameter-vector order):
          relabels each layer directly.
        - **per gate** (length [n_gates][], indexed by gate): expanded to per-layer via
          each layer's gate, so a multi-term gate's layers stay tied. This is the
          granularity of the authoring [Circuit][monoprop.circuit.Circuit]'s mapping.

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
                order, or a [Circuit][monoprop.circuit.Circuit] (its parameters are used).
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
            parameters: Variational parameter values (see [expectation_value][]).

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
            parameters: Variational parameter values (see [expectation_value][]).

        Returns:
            The gradient as a NumPy array of ``float64`` values, in canonical
            parameter-axis order.

        Note:
            Internally calls [expectation_value_and_gradient][] and returns only its
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

        Like [expectation_value_functional][], but the returned callable computes
        both the expectation value and the full parameter gradient in a single backward
        pass over the graph.

        Args:
            pare_threshold: See [expectation_value_functional][].

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
            parameters: Variational parameter values (see [expectation_value][]).
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

        Equivalent to [contract_partially][] with ``inplace=False``, returned as a
        mapping keyed by term indices and without touching the simulator state.

        Args:
            parameters: Variational parameter values (see [expectation_value][]).
            atol: Absolute tolerance for filtering small coefficients; terms with
                ``|coeff| < atol`` are dropped. Defaults to ``1e-12``; set to ``0.0`` to
                keep all terms.

        Returns:
            The evolved operator (Heisenberg picture) or the evolved state (Schrodinger
            picture) as a dict mapping term-index tuples to complex coefficients.
        """
        return self._simulator.evolved_operator(self._bind(parameters), atol)

    def update_initial_operator(
        self, new_operator: dict[tuple[int, ...], complex]
    ) -> None:
        """Replace coefficients of the *initial operator* (existing terms only).

        Re-weights the initial operator the graph is evaluated against, without touching
        the evolution graph or rebuilding the simulator. Only the initial operator is
        affected -- the gates and their generator coefficients are unchanged -- and only
        terms already present in the initial operator can be updated (no new
        terms are introduced).

        Args:
            new_operator: Mapping from term-index tuples to their new complex
                coefficients.

        Raises:
            RuntimeError: If a term in ``new_operator`` is not present in the current
                initial operator.
        """
        self._simulator.update_initial_operator(new_operator)

    def size(self) -> int:
        """Number of terms currently tracked.

        Returns:
            The number of distinct terms in the simulator's current
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
    def graph_layers(self) -> int:
        """Number of evolved generators (graph layers)."""
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
        """Current cutoff type (``"length"`` or ``"support"``).

        Read-only on the base; [MajoranaPropagator][monoprop.majorana_propagator.MajoranaPropagator]
        exposes a setter since either scheme is valid there.
        """
        return self._simulator.cutoff_type

    def _bind(self, parameters: ParameterValues) -> list[float]:
        """Resolve ``parameters`` into a dense vector in parameter-index order.

        Accepts a [Circuit][monoprop.circuit.Circuit] (its ``parameters`` are used), a plain
        sequence of floats, or ``None`` (an empty vector).
        """
        if isinstance(parameters, Circuit):
            parameters = parameters.parameters
        if parameters is None:
            return []
        return [float(v) for v in parameters]
