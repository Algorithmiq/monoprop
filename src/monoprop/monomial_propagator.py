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

"""Monomial Propagator module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np

from monoprop._dispatch import dispatch

from .monomial_data import MonomialCircuit, MonomialOperator
from .utils import (
    normalize_parameters,
    validate_basis_change,
    wrap_functional_call,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mpi4py import MPI

    from .quantum_data import IQuantumCircuit, IQuantumOperator

logger = logging.getLogger(__name__)


class ExpectationValueFunctional(Protocol):
    """Protocol for expectation value functional callables."""

    def __call__(self, parameters: list[float] | np.ndarray | None = None) -> float:
        """Compute expectation value for given parameters."""
        ...


class ExpectationValueAndGradientFunctional(Protocol):
    """Protocol for expectation value and gradient functional callables."""

    def __call__(
        self, parameters: list[float] | np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        """Compute expectation value and gradient for given parameters."""
        ...


class GradientFunctional(Protocol):
    """Protocol for gradient functional callables."""

    def __call__(
        self, parameters: list[float] | np.ndarray | None = None
    ) -> np.ndarray:
        """Compute gradient for given parameters."""
        ...


class MonomialPropagator:
    """Classical simulator for Majorana operators."""

    def __init__(
        self,
        initial_operator: IQuantumOperator | MonomialOperator,
        quantum_circuit: IQuantumCircuit | MonomialCircuit,
        cutoff: int,
        *,
        schrodinger_cutoff: int | None = None,
        cutoff_type: str = "length",
        lower_atol: None | float = None,
        upper_atol: None | float = None,
        basis_change: None | list[list[int]] = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        """Initialize the MonomialPropagator.

        Creates a new Monomial Propagator for quantum system evolution using
        the Monomial fermion representation. The simulator supports both Heisenberg
        and Schrödinger picture evolution with configurable truncation schemes.

        Args:
            initial_operator: Initial Operator represented as objects implementing
            the IQuantumOperator protocol.
            quantum_circuit: Quantum circuit representing the evolution. Can be provided
            as an object implementing the IQuantumCircuit protocol.
            cutoff: Truncation parameter controlling the maximum complexity of
                Monomial operators retained during evolution. Its meaning depends
                on ``cutoff_type`` (see below). Higher values increase accuracy but
                require more computational resources. Note that a *fully paired*
                monomial -- one whose support consists entirely of complete pairs
                (m_{2j-1} m_{2j}) on a mode -- is always kept regardless of this cutoff,
                because only paired monomials can contribute to an expectation value
                against a computational-basis state or Slater determinant; discarding
                them would throw away signal.
            schrodinger_cutoff: Optional cutoff parameter for Schrödinger picture
                evolution. If provided, enables Schrödinger picture mode; if None,
                uses Heisenberg picture (default behaviour).
            cutoff_type: Type of truncation scheme to apply (the fully-paired
                exception above always applies on top of either). Supported values:
                "length" (default) keeps monomials whose length -- the number of
                Majorana operators -- does not exceed ``cutoff``;
                "support" keeps monomials acting on at most ``cutoff`` distinct
                orbitals (the orbital support). Under the Jordan-Wigner mapping the
                support equals the qubit Pauli weight, so "support" truncates by the
                number of X/Y/Z factors.
            lower_atol: Optional lower absolute tolerance threshold for coefficient
                truncation. Monomial operators with coefficients below this value
                will be discarded during evolution to improve performance.
            upper_atol: Optional upper absolute tolerance threshold. Monomial operators
                with coefficients above this value will always be retained regardless
                of their complexity, overriding cutoff-based truncation.
            basis_change: Optional basis transformation for Majorana operators used
                in the cutoff function. If None, cutoff is based on standard Majorana
                representation. If provided, must be a list of 2*num_modes lists,
                where each inner list defines a basis vector in terms of Majorana indices.
            comm: Optional MPI communicator specifier. The communicator must remain valid for the simulator's lifetime.

                Example for fermion-to-qubit (Jordan-Wigner) transformation:

                .. code-block:: python

                    basis_change = [
                        [0],  # m_0 -> X_0
                        [1],  # m_1 -> Y_0
                        [0, 1, 2],  # m_2 -> Z_0 X_1
                        [0, 1, 3],  # m_3 -> Z_0 Y_1
                        ...,
                    ]

                This enables cutoff based on Pauli weight rather than Majorana length.
        """
        monomial_operator: MonomialOperator = (
            initial_operator
            if isinstance(initial_operator, MonomialOperator)
            else initial_operator.get_monomial_operator()
        )
        self.quantum_circuit: MonomialCircuit = (
            quantum_circuit
            if isinstance(quantum_circuit, MonomialCircuit)
            else quantum_circuit.get_monomial_circuit()
        )
        num_modes = monomial_operator.num_modes
        slater_determinant = self.quantum_circuit.initial_state
        logger.debug(
            "__init__. num_modes=%d, cutoff=%d, slater_determinant=%s, schrodinger_cutoff=%s",
            num_modes,
            cutoff,
            slater_determinant,
            schrodinger_cutoff,
        )
        cls = dispatch(num_modes)

        validate_basis_change(basis_change, num_modes)

        self._comm = comm
        self._simulator = cls(
            initial_operator=monomial_operator.terms,
            cutoff=cutoff,
            slater_determinant=slater_determinant,
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            upper_atol=upper_atol,
            cutoff_type=cutoff_type,
            basis_change=basis_change,
            comm=comm,
        )

    def __deepcopy__(self, memo):
        """Deep-copy the simulator into an independent instance.

        The heavy state (the bound simulator and its operator store) is deep-copied; the MPI
        communicator is shared as-is, which is correct both with MPI (the live mpi4py comm) and
        without (``None``). Inherited unchanged by ``MonomialPropagatorExtra``.
        """
        import copy as _copy

        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = (
                value if key == "_comm" else _copy.deepcopy(value, memo)
            )
        return new

    def _create_functional_wrapper(
        self,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
        *,
        pare_threshold: float | None = None,
        functional_type: str = "expectation_value",
    ) -> Callable:
        """Create a functional wrapper with validation and state capture.

        This helper method centralizes the common logic for creating expectation value and
        gradient functionals, including parameter validation and state capture
        for runtime validation of functional calls.

        Args:
            parameter_mapping: The parameter mapping.
            gen_coeffs: The generator coefficients.
            pare_threshold: Absolute value cutoff for retaining edges in the pared graph.
                If None, graph paring is disabled.
            functional_type: Type of functional to create ('expectation_value' or 'gradient').

        Returns:
            Callable: expectation value or gradient functional

        Raises:
            ValueError: If functional_type is unknown.
        """
        _, parameter_mapping, gen_coeffs = normalize_parameters(
            None, parameter_mapping, gen_coeffs
        )

        if functional_type == "expectation_value":
            underlying_fn = self._simulator.expectation_value_functional(
                parameter_mapping=parameter_mapping,
                gen_coeffs=gen_coeffs,
                pare_threshold=pare_threshold,
            )
        elif functional_type == "gradient":
            underlying_fn = self._simulator.expectation_value_and_gradient_functional(
                parameter_mapping=parameter_mapping,
                gen_coeffs=gen_coeffs,
                pare_threshold=pare_threshold,
            )
        else:
            raise ValueError(f"Unknown functional type: {functional_type}")

        return underlying_fn

    @property
    def num_modes(self) -> int:
        """Number of Fermionic modes.

        Returns:
            The number of Fermionic modes for the simulator.
        """
        return self._simulator.num_modes

    @property
    def graph_layers(self) -> int:
        """Number of evolved Majoranas (graph layers).

        Returns:
            The number of Majorana operators that have been evolved.
        """
        return self._simulator.graph_layers()

    @property
    def cutoff(self) -> int:
        """Current cutoff value for the simulation.

        Returns:
            The current cutoff value.
        """
        return self._simulator.cutoff

    @cutoff.setter
    def cutoff(self, new_cutoff: int) -> None:
        """Set the cutoff value for the simulation.

        Args:
            new_cutoff: The new cutoff value.
        """
        self._simulator.cutoff = new_cutoff

    @property
    def lower_atol(self) -> None | float:
        """Current lower absolute tolerance for the cutoff function.

        Returns:
            The current lower absolute tolerance, or None if not set.
        """
        return self._simulator.lower_atol

    @lower_atol.setter
    def lower_atol(self, new_lower_atol: None | float) -> None:
        """Set the lower absolute tolerance for the cutoff function.

        Args:
            new_lower_atol: The new lower absolute tolerance. If None, the lower atol is disabled.
        """
        self._simulator.lower_atol = new_lower_atol

    @property
    def upper_atol(self) -> None | float:
        """Current upper absolute tolerance for the cutoff function.

        Returns:
            The current upper absolute tolerance, or None if not set.
        """
        return self._simulator.upper_atol

    @upper_atol.setter
    def upper_atol(self, new_upper_atol: None | float) -> None:
        """Set the upper absolute tolerance for the cutoff function.

        Args:
            new_upper_atol: The new upper absolute tolerance. If None, the upper atol is disabled.
        """
        self._simulator.upper_atol = new_upper_atol

    @property
    def cutoff_type(self) -> str:
        """Current cutoff type for the simulation.

        Returns:
            The current cutoff type as a string.
        """
        return self._simulator.cutoff_type

    @cutoff_type.setter
    def cutoff_type(self, new_cutoff_type: str) -> None:
        """Set the cutoff type for the simulation.

        Args:
            new_cutoff_type: The new cutoff type.
        """
        self._simulator.cutoff_type = new_cutoff_type

    @property
    def basis_change(self) -> None | list[list[int]]:
        """Current basis change for the cutoff function.

        Returns:
            The current basis change, or None if not set.
        """
        return self._simulator.basis_change

    @basis_change.setter
    def basis_change(self, new_basis_change: None | list[list[int]]) -> None:
        """Set the basis change for the cutoff function.

        Args:
            new_basis_change: The new basis change. If None, no basis change is applied.

        Raises:
            ValueError: If the basis change is invalid.
        """
        validate_basis_change(new_basis_change, self.num_modes)
        self._simulator.basis_change = new_basis_change

    @property
    def schrodinger(self) -> bool:
        """Whether the simulator is in Schrödinger picture.

        Returns:
            True if the simulator is in Schrödinger picture, False if in Heisenberg picture.
        """
        return self._simulator.schrodinger

    def propagate(
        self,
        majoranas: list[tuple[int, ...]] | None = None,
        operator_coeffs: None | list[float] | np.ndarray = None,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
        parameters: list[float] | np.ndarray | None = None,
        *,
        evolve_with_coeffs: bool = False,
        only_rotate_len_k: int = 0,
    ) -> None:
        """Propagate the Operator by multiple Majorana operators.

        This method supports three propagation strategies:

        1. Build only the propagation graph by providing only ``majoranas``.
        2. Build the propagation graph with coefficient information by providing
           ``majoranas``, ``parameter_mapping``, ``gen_coeffs``, ``parameters``,
           and ``operator_coeffs``. Operator coefficients can be obtained from a
           prior call to ``contract_partially(inplace=False)`` if you want to
           preserve the graph.
        3. Propagate and contract immediately without building a graph by providing
           ``majoranas``, ``parameter_mapping``, ``gen_coeffs``, and
           ``parameters`` only (do not provide ``operator_coeffs``). This mode is
           more memory efficient because it does not store the propagation graph.

        Args:
            majoranas: List of Majorana operators to evolve.
            parameter_mapping: Optional mapping from variational parameters to
                generator indices. Must be provided together with ``gen_coeffs``
                and ``parameters``.
            gen_coeffs: Optional generator coefficients corresponding to each
                entry in ``parameter_mapping``. Must be provided together with
                ``parameter_mapping`` and ``parameters``.
            parameters: Optional parameter values for immediate evolution.
                Must be provided together with ``parameter_mapping`` and
                ``gen_coeffs``.
            operator_coeffs: Optional operator coefficients for the current
                state or operator.
            evolve_with_coeffs: Whether to evolve with coefficients. Defaults to False.
            only_rotate_len_k: If > 0, apply gates to monomials of length <= k in the evolved
                operator even if they anticommute. This is useful for when you apply many free
                fermionic gates (ie: gates generated by length 2 majorana monomials) before
                expectation value estimation in schrodinger picture simulations.

        Raises:
            ValueError: If the provided parameters are inconsistent or if there
                are already propagated Majorana operators when coefficient
                information is supplied.
        """
        majoranas = (
            majoranas if majoranas is not None else self.quantum_circuit.majoranas
        )

        if evolve_with_coeffs:
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs
            parameters = self.quantum_circuit.parameters

        self._simulator.propagate(
            majoranas=majoranas,
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            parameters=parameters,
            operator_coeffs=operator_coeffs,
            only_rotate_len_k=only_rotate_len_k,
        )

    def expectation_value_functional(
        self,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
        *,
        use_coeffs: bool = False,
        pare_threshold: float | None = 1e-10,
    ) -> ExpectationValueFunctional:
        """Create an expectation value functional for the current system state.

        Returns a callable function that computes the expectation value
        for given variational parameters using the current evolution graph.

        Args:
            parameter_mapping: Optional mapping from variational parameters to generator
                indices. If None, defaults to the quantum circuit's parameter indices.
            gen_coeffs: Optional generator coefficients corresponding to each parameter.
                If None, defaults to the quantum circuit's generator coefficients.
            use_coeffs: If True, use the quantum circuit's parameter mapping and generator
                coefficients to construct the expectation value functional.
            pare_threshold: Absolute value cutoff for retaining edges in the pared graph.
                If None, graph paring is disabled. Defaults to 1e-10.

        Returns:
            A callable that takes optional parameters and returns the expectation value as a float.

        Raises:
            ValueError: If parameter_mapping and gen_coeffs have different lengths,
                or if the lengths don't match the number of evolved Majoranas.
        """
        if use_coeffs:
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs

        ener_fn = self._create_functional_wrapper(
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            pare_threshold=pare_threshold,
            functional_type="expectation_value",
        )
        return wrap_functional_call(ener_fn)

    def expectation_value_and_gradient_functional(
        self,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
        *,
        use_coeffs: bool = False,
        pare_threshold: float | None = 1e-10,
    ) -> ExpectationValueAndGradientFunctional:
        """Create an expectation value and gradient functional for the current system state.

        Returns a callable function that computes both the expectation value
        and its gradient with respect to variational parameters using the current
        evolution graph.

        Args:
            parameter_mapping: Optional mapping from variational parameters to generator
                indices. If None, defaults to the quantum circuit's parameter indices.
            gen_coeffs: Optional generator coefficients corresponding to each parameter.
                If None, defaults to the quantum circuit's generator coefficients.
            use_coeffs: If True, use the quantum circuit's parameter mapping and generator
                coefficients to construct the expectation value and gradient functional.
            pare_threshold: Absolute value cutoff for retaining edges in the pared graph.
                If None, graph paring is disabled. Defaults to 1e-10.

        Returns:
            A callable that takes optional parameters and returns a tuple of
            (expectation_value, gradient) where expectation_value is a float and gradient is a numpy array.

        Raises:
            ValueError: If parameter_mapping and gen_coeffs have different lengths,
                or if the lengths don't match the number of evolved Majoranas.
        """
        if use_coeffs:
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs

        grad_fn = self._create_functional_wrapper(
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            pare_threshold=pare_threshold,
            functional_type="gradient",
        )
        return wrap_functional_call(grad_fn)

    def gradient_functional(
        self,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
        *,
        use_coeffs: bool = False,
        pare_threshold: float | None = 1e-10,
    ) -> GradientFunctional:
        """Create a gradient functional for the current system state.

        Returns a callable function that computes the gradient of the expectation value
        with respect to variational parameters using the current evolution graph.

        Args:
            parameter_mapping: Optional mapping from variational parameters to generator
                indices. If None, defaults to the quantum circuit's parameter indices.
            gen_coeffs: Optional generator coefficients corresponding to each parameter.
                If None, defaults to the quantum circuit's generator coefficients.
            use_coeffs: If True, use the quantum circuit's parameter mapping and generator
                coefficients to construct the gradient functional.
            pare_threshold: Absolute value cutoff for retaining edges in the pared graph.
                If None, graph paring is disabled. Defaults to 1e-10.

        Returns:
            A callable that takes optional parameters and returns the gradient
            as a numpy array of float64 values.

        Note:
            This method internally calls expectation_value_and_gradient_functional and
            extracts only the gradient component from the result.
        """
        if use_coeffs:
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs
        expval_grad_fn = self.expectation_value_and_gradient_functional(
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            pare_threshold=pare_threshold,
        )
        return wrap_functional_call(
            expval_grad_fn,
            lambda result: np.array(result[1], dtype=np.float64),
        )

    def expectation_value(
        self,
        *,
        use_coeffs: bool = False,
    ) -> float:
        """Compute the expectation value for the current system state.

        Evaluates the expectation value using the current evolution graph
        and the provided variational parameters. This is a convenience method
        that creates and immediately evaluates an expectation value functional.

        Args:
            use_coeffs: Whether to use the quantum circuit's parameter mapping and
                generator coefficients to evaluate the expectation value. If False, the expectation value is
                evaluated at the current parameter values without coefficient mapping.

        Returns:
            The expectation value as a float.

        Note:
            This method internally calls expectation_value_functional() with pare_threshold=None to
            avoid graph optimization overhead for single evaluations.
        """
        parameter_mapping = None
        gen_coeffs = None
        parameters = None
        if use_coeffs:
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs
            parameters = self.quantum_circuit.parameters

        parameters, _, _ = normalize_parameters(parameters, None, None)

        return self.expectation_value_functional(
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            pare_threshold=None,
        )(parameters)

    def expectation_value_and_gradient(
        self,
        parameters: list[float] | np.ndarray | None = None,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Get the expectation value and gradient for the current state.

        Args:
            parameters: The parameters.
            parameter_mapping: The parameter mapping.
            gen_coeffs: The generator coefficients.

        Returns:
            The expectation value and gradient.

        Note:
            Returns the result of calling the expectation-value-and-gradient functional with `parameters` and no paring.
        """
        if parameters is None:
            parameters = self.quantum_circuit.parameters
        if parameter_mapping is None:
            parameter_mapping = self.quantum_circuit.param_inds
        if gen_coeffs is None:
            gen_coeffs = self.quantum_circuit.gen_coeffs

        parameters, _, _ = normalize_parameters(parameters, None, None)
        return self.expectation_value_and_gradient_functional(
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            pare_threshold=None,
        )(parameters)

    def gradient(
        self,
        parameters: list[float] | np.ndarray | None = None,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Get the gradient for the current state.

        Args:
            parameters: The parameters.
            parameter_mapping: The parameter mapping.
            gen_coeffs: The generator coefficients.

        Returns:
            The gradient as a numpy array.

        Note:
            Returns the result of calling the expectation-value-and-gradient functional with `parameters` and no paring.
        """
        parameter_mapping = (
            parameter_mapping
            if parameter_mapping is not None
            else self.quantum_circuit.param_inds
        )
        gen_coeffs = (
            gen_coeffs if gen_coeffs is not None else self.quantum_circuit.gen_coeffs
        )
        parameters = (
            parameters if parameters is not None else self.quantum_circuit.parameters
        )

        parameters, parameter_mapping, gen_coeffs = normalize_parameters(
            parameters, parameter_mapping, gen_coeffs
        )

        return np.array(
            self.expectation_value_and_gradient(
                parameters=parameters,
                parameter_mapping=parameter_mapping,
                gen_coeffs=gen_coeffs,
            )[1],
            dtype=np.float64,
        )

    def contract_partially(
        self,
        parameters: list[float] | np.ndarray | None = None,
        parameter_mapping: list[int] | np.ndarray | None = None,
        gen_coeffs: list[float] | np.ndarray | None = None,
        *,
        ignore_coeffs: bool = True,
        inplace: bool = True,
    ) -> np.ndarray:
        """Contract evolution gates into the operator and update simulator state.

        Applies the evolution gates with specified parameters to the system by contracting
        them into the initial operator. In Heisenberg picture, gates are contracted into
        the initial operator. In Schrödinger picture, gates are contracted into the
        Hartree-Fock state. This operation modifies the simulator's internal state.

        Args:
            parameters: Optional variational parameter values for the gates to be
                contracted. If None, defaults to empty list.
            parameter_mapping: Optional mapping from parameters to gate indices.
                Must have same length as gen_coeffs. If None, defaults to empty list.
            gen_coeffs: Optional generator coefficients for each gate. Must have same
                length as parameter_mapping. If None, defaults to empty list.
            ignore_coeffs: Whether to ignore the generator coefficients. If True (default),
                the coefficients are not applied to the gates during contraction.
            inplace: Whether to modify the simulator state in place. If True (default),
                the simulator's internal graph is contracted. If False, the simulator graph
                is preserved and only the coefficients are returned.

        Returns:
            The updated operator coefficients as a numpy array. In Schrödinger picture,
            returns the evolved state. In Heisenberg picture, returns the evolved
            operator coefficients.

        Raises:
            ValueError: If parameter_mapping and gen_coeffs have different lengths,
                if parameter length doesn't match mapping requirements, or if mapping
                length exceeds the number of evolved Majoranas.
        """
        if not ignore_coeffs:
            parameters = self.quantum_circuit.parameters
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs

        parameters, parameter_mapping, gen_coeffs = normalize_parameters(
            parameters, parameter_mapping, gen_coeffs
        )

        return self._simulator.contract_partially(
            parameters=parameters,
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            inplace=inplace,
        )

    def evolved_operator_dict(
        self,
        atol: float = 1e-12,
        *,
        evolve_with_coeffs: bool = False,
    ) -> dict[tuple[int, ...], complex]:
        """Get the evolved state or operator.

        Applies contract_partially, but does not affect the state of the simulator,
        and returns the evolved operator.

        Args:
            atol: Absolute tolerance for filtering small coefficients. Terms with
                coefficients smaller than this threshold will be removed from the
                result. Defaults to 1e-12. Set to 0.0 to keep all terms.
            evolve_with_coeffs: If True, the operator is evolved with the generator
                coefficients. Defaults to False.

        Returns:
            The evolved operator.
        """
        # Convert None to empty lists (following expectation_value_functional pattern)
        parameters = None
        parameter_mapping = None
        gen_coeffs = None

        if evolve_with_coeffs:
            parameters = self.quantum_circuit.parameters
            parameter_mapping = self.quantum_circuit.param_inds
            gen_coeffs = self.quantum_circuit.gen_coeffs

        parameters, parameter_mapping, gen_coeffs = normalize_parameters(
            parameters, parameter_mapping, gen_coeffs
        )

        return self._simulator.evolved_operator_dict(
            parameters=parameters,
            parameter_mapping=parameter_mapping,
            gen_coeffs=gen_coeffs,
            atol=atol,
        )

    def evolved_operator(
        self,
        atol: float = 1e-12,
        *,
        evolve_with_coeffs: bool = False,
    ) -> dict[tuple[int, ...], complex]:
        """Get the evolved operator.

        Applies contract_partially, but does not affect the state of the simulator, and returns the evolved operator.

        Args:
            atol: Absolute tolerance for filtering small coefficients.
                Terms with coefficients smaller than this threshold will be removed from
                the result. Defaults to 1e-12. Set to 0.0 to keep all terms.
            evolve_with_coeffs: If True, the operator is evolved with the generator
                coefficients. Defaults to False.

        Returns:
            The evolved operator dictionary.
        """
        if self._simulator.schrodinger:
            raise ValueError(
                "Cannot call evolved_operator in Schrodinger picture. "
                "Use evolved_operator_dict instead."
            )

        return self.evolved_operator_dict(
            atol=atol,
            evolve_with_coeffs=evolve_with_coeffs,
        )

    def update_coeffs(
        self,
        new_operator: dict[tuple[int, ...], complex],
    ) -> None:
        """Update the initial operator with new coefficients.

        Replaces the coefficients of the initial operator with the provided values.
        This allows for dynamic modification of the system's operator during simulation.
        Only Majorana terms that already exist in the system can be updated.

        Args:
            new_operator: Dictionary mapping Majorana operator indices (as tuples)
                to their new complex coefficients. Keys are tuples of integer indices
                representing Majorana operators, values are the corresponding coefficients.

        Raises:
            RuntimeError: If an operator term in new_operator is not found
                in the current system.
        """
        self._simulator.update_initial_operator(new_operator)

    def size(self) -> int:
        """Get the number of Majorana operators in the current system.

        Returns the total number of distinct Majorana operator terms that are
        currently tracked in the simulator's internal representation.

        Returns:
            The number of Majorana operators currently in the system as an integer.
        """
        return self._simulator.size()

    def graph_size(self) -> tuple[int, int]:
        """Get the size metrics of the evolution graph.

        Returns information about the computational complexity of the current
        evolution graph, which is useful for performance monitoring and optimization.

        Returns:
            A tuple containing (n_cos_indices, n_cycles) where:

            - n_cos_indices: Number of cosine indices in the MP graph
            - n_cycles: Number of cycles in the MP graph
        """
        return self._simulator.graph_size()
