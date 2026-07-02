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

"""Majorana and qubit propagators.

Both classes wrap the same compiled C++ Majorana simulator. Gate information (the
Majorana generators, their coefficients, and the parameter each drives) is owned by
the propagation graph, so evaluation methods take only ``parameters``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from monoprop._dispatch import dispatch

from .circuit import Gate, ParameterVector, Term, to_engine_arrays
from .conversion_utils import _extend_pauli_string, _pauli_to_fermi
from .monomial_data import MonomialOperator
from .utils import jordan_wigner_basis_change, validate_basis_change

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mpi4py import MPI

    from .circuit import Parameter, QubitGate
    from .pauli_data import PauliOperator
    from .quantum_data import IQuantumOperator

    ParameterValues = Sequence[float] | Mapping[Parameter, float] | np.ndarray | None

logger = logging.getLogger(__name__)


class MajoranaPropagator:
    """Classical simulator for Majorana operators.

    The propagation graph owns the gate information; evaluation methods
    (:meth:`expectation_value`, :meth:`gradient`, ...) take only ``parameters``.

    .. note::
        **Incremental building and gate order.** In the Heisenberg picture the
        Heisenberg evolution applies gates back-to-front, so each
        :meth:`propagate_build_graph` / :meth:`propagate` call consumes its gate
        sequence in reverse. Splitting one circuit into forward chunks across several
        calls is therefore *not* equivalent to a single call with the whole sequence:
        the chunks are each reversed but not globally reordered. In the Schrodinger
        picture gates are applied front-to-back, so a forward split *is* equivalent.
        When you need incremental building to reproduce a single-call result, use the
        Schrodinger picture (or pass the full sequence in one call).
    """

    def __init__(
        self,
        initial_operator: IQuantumOperator | MonomialOperator,
        initial_state: list[int] | np.ndarray,
        *,
        cutoff: int,
        schrodinger_cutoff: int | None = None,
        cutoff_type: str = "length",
        lower_atol: None | float = None,
        upper_atol: None | float = None,
        basis_change: None | list[list[int]] = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        """Initialize the propagator.

        Creates a simulator for quantum-system evolution in the Majorana
        representation. Both Heisenberg (operator evolution, the default) and
        Schrodinger (state evolution) pictures are supported, with configurable
        truncation.

        Args:
            initial_operator: Initial operator, either a :class:`MonomialOperator` or an
                object implementing ``get_monomial_operator()``.
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
        """
        monomial_operator: MonomialOperator = (
            initial_operator
            if isinstance(initial_operator, MonomialOperator)
            else initial_operator.get_monomial_operator()
        )
        num_modes = monomial_operator.num_modes
        logger.debug(
            "__init__. num_modes=%d, cutoff=%d, schrodinger_cutoff=%s",
            num_modes,
            cutoff,
            schrodinger_cutoff,
        )
        validate_basis_change(basis_change, num_modes)

        self._comm = comm
        self._params = ParameterVector()
        self._simulator = dispatch(num_modes)(
            initial_operator=monomial_operator.terms,
            cutoff=cutoff,
            slater_determinant=list(initial_state),
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            cutoff_type=cutoff_type,
            basis_change=basis_change,
            comm=comm,
        )

    # -- gate ingestion ---------------------------------------------------------

    def _majorana_gates(self, gates: Sequence[Gate]) -> Sequence[Gate]:
        """Hook for subclasses to map their gate type to Majorana gates."""
        return gates

    def propagate_build_graph(
        self,
        gates: Sequence[Gate | QubitGate],
        parameters: ParameterValues = None,
        *,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Append gates to the propagation graph.

        Builds (or extends) the reusable evolution graph, recording each layer's gate
        information (the parameter that drives it and its generator coefficient) so
        that later evaluation takes only ``parameters``. When extending a non-empty
        graph with coefficient-informed truncation, pass ``parameters`` covering the
        whole accumulated graph plus these new gates; the coefficient seed is
        regenerated internally by contracting the existing graph at those parameters.

        Args:
            gates: Gates to append (Majorana :class:`~monoprop.circuit.Gate`, or
                :class:`~monoprop.circuit.QubitGate` for :class:`QubitPropagator`).
            parameters: Optional full parameter vector (see above), given either as a
                sequence in canonical parameter order or as a mapping from
                :class:`~monoprop.circuit.Parameter` handle to value.
            only_rotate_len_k: If > 0, apply gates to monomials of length <= k in the
                evolved operator even if they anticommute. Useful when many
                free-fermionic gates (generators that are length-2 Majorana monomials)
                are applied before expectation-value estimation in Schrodinger-picture
                simulations.
        """
        majorana_gates = self._majorana_gates(gates)
        for gate in majorana_gates:
            self._params.register(gate.param)
        majoranas, gen_coeffs, parameter_mapping = to_engine_arrays(
            majorana_gates, self._params
        )
        bound = None if parameters is None else self._bind(parameters)
        self._simulator.propagate_build_graph(
            majoranas,
            parameter_mapping,
            gen_coeffs,
            bound,
            only_rotate_len_k,
        )

    def propagate(
        self,
        gates: Sequence[Gate | QubitGate],
        parameters: ParameterValues,
        *,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Evolve and contract immediately, without storing a graph.

        More memory-efficient than :meth:`propagate_build_graph` because it does not
        retain the propagation graph; use it for a single contraction at fixed
        parameters rather than repeated re-evaluation.

        Args:
            gates: Gates to apply.
            parameters: Parameter values for the gates, given either as a sequence in
                gate order or as a mapping from
                :class:`~monoprop.circuit.Parameter` handle to value.
            only_rotate_len_k: See :meth:`propagate_build_graph`.
        """
        majorana_gates = self._majorana_gates(gates)
        local = ParameterVector()
        for gate in majorana_gates:
            local.register(gate.param)
        majoranas, gen_coeffs, parameter_mapping = to_engine_arrays(
            majorana_gates, local
        )
        self._simulator.propagate(
            majoranas,
            parameter_mapping,
            gen_coeffs,
            local.bind(parameters),
            only_rotate_len_k,
        )

    # -- evaluation -------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Number of distinct variational parameters seen while building the graph."""
        return len(self._params)

    def expectation_value(
        self,
        parameters: ParameterValues = None,
    ) -> float:
        """Compute the expectation value at ``parameters``.

        Replays the stored graph against the current initial operator and reference
        state. This is a convenience wrapper that builds and immediately evaluates an
        expectation-value functional with no paring.

        Args:
            parameters: Variational parameter values, given either as a sequence in
                canonical parameter order or as a mapping from
                :class:`~monoprop.circuit.Parameter` handle to value. ``None`` evaluates
                the current operator with an empty parameter vector.

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

        Returns a callable that accepts a parameter vector (a sequence, a
        :class:`~monoprop.circuit.Parameter` mapping, or ``None``) and returns the
        expectation value, replaying the current evolution graph. Build it once and
        call it repeatedly across many parameter values.

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

    def evolved_operator_dict(
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
            The evolved operator (Heisenberg) or state (Schrodinger) as a dict mapping
            Majorana-index tuples to complex coefficients.
        """
        return self._simulator.evolved_operator_dict(self._bind(parameters), atol)

    def evolved_operator(
        self,
        parameters: ParameterValues = None,
        *,
        atol: float = 1e-12,
    ) -> dict[tuple[int, ...], complex]:
        """Return the evolved operator (Heisenberg picture only).

        Args:
            parameters: Variational parameter values (see :meth:`expectation_value`).
            atol: Absolute tolerance for filtering small coefficients (see
                :meth:`evolved_operator_dict`).

        Returns:
            The evolved operator as a dict mapping Majorana-index tuples to complex
            coefficients.

        Raises:
            ValueError: If the simulator is in the Schrodinger picture; use
                :meth:`evolved_operator_dict` there instead.
        """
        if self._simulator.schrodinger:
            raise ValueError(
                "Cannot call evolved_operator in Schrodinger picture. "
                "Use evolved_operator_dict instead."
            )
        return self.evolved_operator_dict(parameters, atol=atol)

    def update_coeffs(self, new_operator: dict[tuple[int, ...], complex]) -> None:
        """Replace the initial-operator coefficients (existing terms only).

        Allows dynamic re-weighting of the initial operator without rebuilding the
        simulator or the graph. Only Majorana terms already present in the operator can
        be updated.

        Args:
            new_operator: Mapping from Majorana-index tuples to their new complex
                coefficients.

        Raises:
            RuntimeError: If a term in ``new_operator`` is not present in the current
                operator.
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
        """Resolve ``parameters`` into a dense vector in canonical axis order."""
        if parameters is None:
            return []
        if hasattr(parameters, "keys"):
            return self._params.bind(parameters)
        return [float(v) for v in parameters]


class QubitPropagator(MajoranaPropagator):
    """Propagator for qubit (Pauli) operators, mapped to Majoranas via Jordan-Wigner.

    Accepts a :class:`~monoprop.pauli_data.PauliOperator` and
    :class:`~monoprop.circuit.QubitGate` gates; the Jordan-Wigner basis change is set
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

        See :class:`MajoranaPropagator` for the shared arguments. The cutoff is always
        measured as Pauli weight (``cutoff_type="support"``), so ``cutoff`` bounds the
        number of qubits a retained term touches.

        Args:
            initial_operator: Initial qubit operator as a
                :class:`~monoprop.pauli_data.PauliOperator`.
            initial_state: Computational-basis reference (indices of qubits set to 1).
            cutoff: Maximum Pauli weight (number of qubits touched) retained during
                evolution. The fully-paired exception described in
                :class:`MajoranaPropagator` still applies.
            schrodinger_cutoff: Optional Schrodinger-picture cutoff (enables that
                picture).
            lower_atol: Optional lower coefficient-truncation tolerance.
            upper_atol: Optional upper coefficient-retention tolerance.
            comm: Optional MPI communicator (must outlive the propagator).
        """
        num_qubits = initial_operator.num_qubits
        self._num_qubits = num_qubits
        super().__init__(
            initial_operator.get_monomial_operator(),
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
                "QubitPropagator only supports the 'support' cutoff type (Pauli "
                f"weight); got {new_cutoff_type!r}. Use MajoranaPropagator for "
                "length-based truncation."
            )
        self._simulator.cutoff_type = new_cutoff_type

    def _majorana_gates(self, gates: Sequence[QubitGate]) -> list[Gate]:  # type: ignore[override]
        """Map qubit gates to Majorana gates via Jordan-Wigner, preserving handles."""
        majorana_gates: list[Gate] = []
        for qubit_gate in gates:
            terms: list[Term] = []
            for pauli, coefficient in zip(
                qubit_gate.paulis.strings,
                qubit_gate.paulis.coefficients,
                strict=True,
            ):
                extended = _extend_pauli_string(
                    pauli.string, qubit_gate.qubits, self._num_qubits
                )
                majorana, fermi_coeff = _pauli_to_fermi(extended)
                weight = len(majorana)
                gen_coeff = (
                    -coefficient * fermi_coeff / (1j) ** (weight * (weight - 1) / 2)
                )
                terms.append(Term(tuple(majorana), float(np.real(gen_coeff))))
            majorana_gates.append(Gate(qubit_gate.param, tuple(terms)))
        return majorana_gates
