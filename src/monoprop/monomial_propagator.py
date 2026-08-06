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

Shared engine for [MajoranaPropagator][monoprop.majorana_propagator.MajoranaPropagator] and
[PauliPropagator][monoprop.pauli_propagator.PauliPropagator], which pick the Majorana or Pauli
behavior of one compiled C++ engine through a runtime basis.

Gate information (the monomial generators, their coefficients, and the parameter each drives)
is owned by the propagation graph, so evaluation methods take only ``parameters``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from monoprop._dispatch import dispatch

from .circuit import (
    Circuit,
    ExpGate,
    expand_monomials,
    validate_parameter_mapping,
)
from .majorana import MajoranaOperator
from .pauli import PauliOperator
from .utils import validate_basis_change

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Self

    from mpi4py import MPI

    ParameterValues = Circuit | Sequence[float] | np.ndarray | None

logger = logging.getLogger(__name__)

T_op = TypeVar("T_op", MajoranaOperator, PauliOperator)


class MonomialPropagator(ABC, Generic[T_op]):
    """Abstract base for the classical monomial-propagation simulators.

    Subclasses implement ``__init__`` -- resolve their operator family to a
    [MajoranaOperator][monoprop.majorana.MajoranaOperator], then call ``_init_simulator`` -- and
    ``_circuit_gates``, which validates the circuit's gate family.

    Note:
        Heisenberg evolution consumes each [build_graph][] / [propagate][] call's gates
        back-to-front, so splitting one circuit across several calls is *not* equivalent to a
        single call; in the Schrodinger picture (front-to-back) it is.
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
        lower_atol: float | None,
        upper_atol: float | None,
        basis_change: list[list[int]] | None,
        comm: MPI.Comm | None,
        basis: str = "majorana",
    ) -> None:
        """Dispatch to the compiled per-mode simulator and record shared state.

        Args:
            majorana_operator: The observable; its ``num_modes`` sizes the simulator.
            initial_state: Reference state, as occupied mode/qubit indices.
            cutoff: Truncation parameter, read according to ``cutoff_type``.
            schrodinger_cutoff: State-truncation limit selecting Schrodinger; ``None`` = Heisenberg.
            cutoff_type: ``"length"`` or ``"support"``.
            lower_atol: Coefficient-truncation threshold, or ``None``.
            upper_atol: Coefficient-retention threshold, or ``None``.
            basis_change: Internal per-Majorana basis change (``2 * num_modes`` entries).
            comm: Optional MPI communicator (must outlive the propagator).
            basis: Engine basis -- ``"majorana"`` (default) or ``"pauli"``.
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
        # Qubit count for expanding Pauli gates; PauliPropagator overwrites it after this call.
        self._num_qubits = None
        self._initial_state = list(initial_state)
        # dispatch() returns the concrete adapter class for this mode count;
        # call it with keyword args matching the public constructor.
        self._simulator = dispatch(num_modes)(  # type: ignore[call-arg]
            initial_operator=majorana_operator.terms,
            cutoff=cutoff,
            initial_state=list(initial_state),
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            cutoff_type=cutoff_type,
            basis_change=basis_change,
            comm=comm,
            basis=basis,
        )

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        initial_operator: object,
        **config: object,
    ) -> Self:
        """Construct a propagator from a circuit and propagate it in one step.

        The circuit supplies both the gates and the initial state; ``**config`` is forwarded to the
        constructor (``cutoff``, etc.).

        Returns:
            A propagator with ``circuit`` already evolved via [propagate][] and **no** graph
            stored; use [build_graph][] instead when a reusable graph is wanted.
        """
        propagator = cls(initial_operator, list(circuit.initial_state), **config)  # type: ignore[arg-type]
        propagator.propagate(circuit)
        return propagator

    @abstractmethod
    def _circuit_gates(self, circuit: Circuit) -> Sequence[ExpGate]:
        """Validate the circuit's gate family and return its gates for expansion."""
        raise NotImplementedError

    def _check_initial_state(self, circuit: Circuit) -> None:
        """Reject a circuit whose reference state disagrees with the propagator's.

        An *unspecified* circuit state (``None``) defers to the propagator's, while ``()`` is the
        vacuum, not "unspecified". Occupied-index sets compare order-insensitively.
        """
        if circuit._state_given and sorted(circuit.initial_state) != sorted(
            self._initial_state
        ):
            raise ValueError(
                f"circuit.initial_state {list(circuit.initial_state)} does not match the "
                f"propagator's initial state {list(self._initial_state)}. Construct the "
                "propagator with this circuit's initial state (or via from_circuit)."
            )

    def _validate_only_rotate_len_k(self, only_rotate_len_k: int | None) -> int | None:
        """Validate ``only_rotate_len_k``.

        Must be positive, and at most ``2 * num_qubits`` when the propagator knows its qubit count
        (i.e. on a [PauliPropagator][monoprop.pauli_propagator.PauliPropagator]).

        Args:
            only_rotate_len_k: Optional length cutoff for gate application.

        Returns:
            The validated optional cutoff.
        """
        if only_rotate_len_k is not None and (
            only_rotate_len_k <= 0
            or (
                isinstance(self._num_qubits, int)
                and only_rotate_len_k > 2 * self._num_qubits
            )
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

        Builds (or extends) the reusable evolution graph, recording each layer's driving parameter
        and generator coefficient so later evaluation takes only ``parameters``. A circuit's angle
        indices are local (``0``-based) and shift onto the accumulated axis when extending.

        Args:
            circuit: Gates to append, as a [Circuit][monoprop.circuit.Circuit].
            seed_parameters: Full parameter vector for the whole accumulated graph; regenerates the
                coefficient seed (by contracting the existing graph) so truncation sees realistic
                coefficients. Needed only when extending a non-empty graph *with*
                coefficient-informed truncation. Defaults to the circuit's own parameters on the
                first call; omitted while extending, the new layers are built structurally. The
                engine validates the length of an explicit seed.
            only_rotate_len_k: If given, apply gates to monomials of length <= k even where they
                anticommute -- useful ahead of expectation-value estimation in the Schrodinger
                picture with many free-fermionic (length-2 Majorana) generators.
        """
        self._check_initial_state(circuit)
        only_rotate_len_k = self._validate_only_rotate_len_k(only_rotate_len_k)

        if seed_parameters is not None:
            seed = seed_parameters
        elif self.graph_layers == 0:
            seed = circuit.parameters
        else:
            seed = None
        gates = self._circuit_gates(circuit)
        num_qubits = self._num_qubits
        mapping = [self._n_params + m for m in circuit.resolved_mapping]
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
        # Advance the axis only once the graph owns the layers: expand_monomials, _bind and the C++
        # validation all raise, and a retry after such a failure must reuse the same indices.
        self._n_params += circuit.n_parameters

    def propagate(
        self, circuit: Circuit, *, only_rotate_len_k: int | None = None
    ) -> None:
        """Evolve and contract immediately, without storing a graph.

        Retains no graph, so it is cheaper than [build_graph][] but is one-shot, at the
        circuit's own parameters.

        Args:
            circuit: Gates to apply, and the angle values to apply them at.
            only_rotate_len_k: See [build_graph][].
        """
        only_rotate_len_k = self._validate_only_rotate_len_k(only_rotate_len_k)
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
        """Size of the current graph's parameter axis.

        Follows the graph rather than accumulating: [build_graph][] extends it, a re-wire through
        [parameter_mapping][] resets it, and an in-place [contract_partially][] shrinks it to the
        axis of the layers that are left.
        """
        return self._n_params

    @property
    def n_gates(self) -> int:
        """Number of distinct authoring gates currently in the graph.

        A multi-term gate expands to several layers sharing one gate, so ``n_gates <= graph_layers``.
        """
        return self._simulator.n_gates()

    @property
    def parameter_mapping(self) -> list[int]:
        """The parameter mapping owned by the graph, one entry per graph layer.

        Entry ``i`` is the parameter index driving graph layer ``i`` (one generated monomial), in
        the same order as the parameter vector passed to [expectation_value][]. This is the graph's
        native per-monomial mapping, finer-grained than the per-gate mapping of the authoring
        [Circuit][monoprop.circuit.Circuit] when gates bundle several monomials.
        """
        return list(self._simulator.parameter_mapping)

    @parameter_mapping.setter
    def parameter_mapping(self, mapping: Sequence[int]) -> None:
        """Re-wire which parameter drives each gate/layer, without rebuilding the graph.

        The mapping must be contiguous ``0..n-1``, and may be given per graph layer (length
        [graph_layers][], in parameter-vector order) or per gate (length [n_gates][],
        expanded so a multi-term gate's layers stay tied); when the two lengths coincide the
        per-layer reading wins. Functionals created earlier keep the mapping they were built with.
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

        Replays the stored graph against the current initial operator and reference state.

        Args:
            parameters: Variational parameter values, as a sequence in parameter-index order or a
                [Circuit][monoprop.circuit.Circuit] (whose parameters are used). ``None`` means an
                empty parameter vector.
        """
        return self._simulator.expectation_value(self._bind(parameters))

    def expectation_value_and_gradient(
        self,
        parameters: ParameterValues = None,
    ) -> tuple[float, np.ndarray]:
        """Compute the expectation value and gradient in a single backward pass over the graph.

        Args:
            parameters: Variational parameter values (see [expectation_value][]).

        Returns:
            ``(expectation_value, gradient)``, with ``gradient`` in parameter-axis order.
        """
        value, grad = self._simulator.expectation_value_and_gradient(
            self._bind(parameters)
        )
        return value, np.asarray(grad, dtype=np.float64)

    def gradient(
        self,
        parameters: ParameterValues = None,
    ) -> np.ndarray:
        """Compute the gradient at ``parameters``, as ``float64`` in parameter-axis order.

        Args:
            parameters: Variational parameter values (see [expectation_value][]).
        """
        return self.expectation_value_and_gradient(parameters)[1]

    def expectation_value_functional(
        self, pare_threshold: float | None = None
    ) -> Callable[..., float]:
        """Return a reusable callable computing the expectation value from parameters.

        Args:
            pare_threshold: Edge-retention cutoff for this functional's masked plan: edges
                contributing below it are pared away and skipped during replay, trading memory and
                accuracy for speed. ``None`` (default) disables paring.

        Returns:
            A callable ``fn(parameters=None) -> float``.
        """
        fn = self._simulator.expectation_value_functional(pare_threshold)
        return lambda parameters=None: fn(self._bind(parameters))

    def expectation_value_and_gradient_functional(
        self, pare_threshold: float | None = None
    ) -> Callable[..., tuple]:
        """Return a reusable callable computing (expectation value, gradient).

        Like [expectation_value_functional][], but one backward pass also yields the gradient.

        Args:
            pare_threshold: See [expectation_value_functional][].

        Returns:
            A callable ``fn(parameters=None) -> (float, np.ndarray)``, gradient in parameter order.
        """
        fn = self._simulator.expectation_value_and_gradient_functional(pare_threshold)

        def _call(parameters=None):  # noqa: ANN001, ANN202
            value, grad = fn(self._bind(parameters))
            return value, np.asarray(grad, dtype=np.float64)

        return _call

    def expval(
        self,
        parameters: ParameterValues = None,
    ) -> float:
        """Shorthand for [expectation_value][].

        See [expectation_value][] for full documentation.
        """
        return self.expectation_value(parameters)

    def grad(
        self,
        parameters: ParameterValues = None,
    ) -> np.ndarray:
        """Shorthand for [gradient][].

        See [gradient][] for full documentation.
        """
        return self.gradient(parameters)

    def expval_and_grad(
        self,
        parameters: ParameterValues = None,
    ) -> tuple[float, np.ndarray]:
        """Shorthand for [expectation_value_and_gradient][].

        See [expectation_value_and_gradient][] for full documentation.
        """
        return self.expectation_value_and_gradient(parameters)

    def expval_functional(
        self, pare_threshold: float | None = None
    ) -> Callable[..., float]:
        """Shorthand for [expectation_value_functional][].

        See [expectation_value_functional][] for full documentation.
        """
        return self.expectation_value_functional(pare_threshold)

    def expval_and_grad_functional(
        self, pare_threshold: float | None = None
    ) -> Callable[..., tuple]:
        """Shorthand for [expectation_value_and_gradient_functional][].

        See [expectation_value_and_gradient_functional][] for full documentation.
        """
        return self.expectation_value_and_gradient_functional(pare_threshold)

    def contract_partially(
        self,
        parameters: ParameterValues = None,
        *,
        inplace: bool = True,
    ) -> np.ndarray:
        """Contract the graph into the operator/state at ``parameters``.

        Folds the stored gates, evaluated at ``parameters``, into the operand -- the initial operator
        in the Heisenberg picture, the reference state in the Schrodinger picture.

        Args:
            parameters: Variational parameter values (see [expectation_value][]).
            inplace: ``True`` (default) consumes the graph into the internal state, which also
                rewinds [n_parameters][] to the axis of the layers left behind -- a following
                [build_graph][] numbers its angles from there. ``False`` only returns the
                coefficients, leaving both the graph and the axis untouched.

        Returns:
            The evolved coefficients as a NumPy array, core term excluded -- of the state in the
            Schrodinger picture, of the operator in the Heisenberg picture. The array carries no term
            labels, and across partitions it is each partition's block concatenated in partition
            order: the same values as an unpartitioned run, but not in a reproducible order, since the
            partition count is auto-picked from the host's core count. Use [evolved_operator][] when
            you need coefficients tied to their terms.
        """
        coeffs = np.asarray(
            self._simulator.contract_partially(self._bind(parameters), inplace)
        )
        if inplace:
            # The contraction consumed the layers it folded, so the parameter axis has to follow the
            # graph that is left; otherwise the next build_graph() would append at a stale index.
            self._n_params = max(self._simulator.parameter_mapping, default=-1) + 1
        return coeffs

    @abstractmethod
    def evolved_operator(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> T_op:
        """Return the evolved operator/state without modifying simulator state.

        Equivalent to [contract_partially][] with ``inplace=False``, decoded into terms. Each
        concrete front-end implements this over its own operator type -- the engine yields raw index
        tuples (Majorana indices, or symplectic slots in the Pauli basis), which the subclass wraps
        into a [MajoranaOperator][monoprop.majorana.MajoranaOperator] or
        [PauliOperator][monoprop.pauli.PauliOperator].

        Args:
            parameters: Variational parameter values (see [expectation_value][]).
            atol: Terms with ``|coeff| < atol`` are dropped; ``0.0`` keeps all of them.

        Returns:
            The evolved operator (Heisenberg picture) or evolved state (Schrodinger picture).
        """

    @abstractmethod
    def update_initial_operator(self, new_operator: T_op) -> None:
        """Replace the *initial operator* (existing terms only).

        A re-weight, not a rebuild: the graph, its gates, and their generator coefficients are kept.
        Each concrete front-end implements this over its own operator type, encoding the terms into
        the engine's raw index tuples.

        Args:
            new_operator: A [MajoranaOperator][monoprop.majorana.MajoranaOperator] or
                [PauliOperator][monoprop.pauli.PauliOperator], per the front-end, whose terms replace
                the matching initial-operator.

        Raises:
            RuntimeError: In the Heisenberg picture, if a term is absent from the current operator.
        """

    def size(self) -> int:
        """Number of distinct monomial terms in the simulator's current representation."""
        return self._simulator.size()

    def graph_size(self) -> tuple[int, int]:
        """Size metrics of the evolution graph.

        Returns:
            ``(n_cos_indices, n_cycles)``: *cosine-only* indices -- terms scaled by a cosine without
            being a rotation endpoint -- and rotation cycles. The cosine-only count is legitimately
            ``0`` when nothing is truncated: every anticommuting term's sine partner then survives.
        """
        return self._simulator.graph_size()

    @property
    def num_modes(self) -> int:
        """Number of modes the simulator acts on (qubits, in the Pauli basis)."""
        return self._simulator.num_modes

    @property
    def graph_layers(self) -> int:
        """Number of evolved monomials (graph layers)."""
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
    def lower_atol(self) -> float | None:
        """Current lower absolute tolerance for the cutoff function (``None`` if unset)."""
        return self._simulator.lower_atol

    @lower_atol.setter
    def lower_atol(self, new_lower_atol: float | None) -> None:
        self._simulator.lower_atol = new_lower_atol

    @property
    def upper_atol(self) -> float | None:
        """Current upper absolute tolerance for the cutoff function (``None`` if unset)."""
        return self._simulator.upper_atol

    @upper_atol.setter
    def upper_atol(self, new_upper_atol: float | None) -> None:
        self._simulator.upper_atol = new_upper_atol

    @property
    def cutoff_type(self) -> str:
        """Current cutoff type, ``"length"`` or ``"support"``; read-only outside the Majorana front-end."""
        return self._simulator.cutoff_type

    def _bind(self, parameters: ParameterValues) -> list[float]:
        """Resolve ``parameters`` into a dense vector in parameter-index order.

        Accepts a [Circuit][monoprop.circuit.Circuit] (its ``parameters``), a float sequence, or
        ``None`` (an empty vector).
        """
        if isinstance(parameters, Circuit):
            parameters = parameters.parameters
        if parameters is None:
            return []
        return [float(v) for v in parameters]
