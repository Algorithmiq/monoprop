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

"""MPI tests for the Monomial Propagator.

Run with: mpirun -np <num_ranks> uv run pytest tests/test_mpi.py --with-mpi -m mpi -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.cases import load_problem

MPI = pytest.importorskip("mpi4py").MPI

from monoprop import MonomialPropagator  # noqa: E402


@pytest.fixture
def lih_fermionic_spin_exact(lazy_shared_datadir):
    """Load the LiH fermionic test problem from its msgpack fixture."""
    data_path = lazy_shared_datadir / "lih_fermionic_spin_exact.msgpack"
    return load_problem(data_path)


def _make_mp(problem, comm):
    """Create a standard MP instance from a fermionic test problem."""
    return MonomialPropagator(
        problem.operator,
        problem.initial_state,
        cutoff=2 * problem.n_modes,
        comm=comm,
    )


def _assert_expval(expval, expected, atol=1e-9):
    assert np.isclose(expval, expected, atol=atol), (
        f"Expectation value mismatch: {expval} vs {expected}"
    )


def _finite_difference_gradient(expval_fn, parameters, eps=1e-6):
    gradient = np.zeros_like(parameters)
    for index in range(len(parameters)):
        params_plus = parameters.copy()
        params_plus[index] += eps
        params_minus = parameters.copy()
        params_minus[index] -= eps
        gradient[index] = (expval_fn(params_plus) - expval_fn(params_minus)) / (2 * eps)
    return gradient


@pytest.mark.mpi
class TestMPISimulator:
    """MPI tests for the Monomial Propagator."""

    def test_evolve_expectation_value_functional(self, lih_fermionic_spin_exact):
        mp = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_fermionic_spin_exact.monomial_circuit)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.monomial_circuit.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_evolve_with_operator_coeffs(self, lih_fermionic_spin_exact):
        mp = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        operator_coeffs = mp.contract_partially(inplace=False)
        mp.propagate(
            lih_fermionic_spin_exact.monomial_circuit, operator_coeffs=operator_coeffs
        )
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.monomial_circuit.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_immediate_contraction(self, lih_fermionic_spin_exact):
        mp = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(
            lih_fermionic_spin_exact.monomial_circuit, ignore_circuit_parameters=False
        )
        _assert_expval(
            mp.expectation_value_functional(ignore_coeffs=True, pare_threshold=1e-10)(),
            lih_fermionic_spin_exact.exact_expval,
        )

    def test_schrodinger_picture(self, lih_fermionic_spin_exact):
        mp = MonomialPropagator(
            lih_fermionic_spin_exact.operator,
            lih_fermionic_spin_exact.initial_state,
            cutoff=2 * lih_fermionic_spin_exact.n_modes,
            schrodinger_cutoff=2 * lih_fermionic_spin_exact.n_modes,
            comm=MPI.COMM_WORLD,
        )
        mp.propagate(lih_fermionic_spin_exact.monomial_circuit)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.monomial_circuit.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_gradient_functional(self, lih_fermionic_spin_exact):
        """Test gradient_functional against finite difference."""
        mp = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_fermionic_spin_exact.monomial_circuit)

        expval_fn = mp.expectation_value_functional(pare_threshold=1e-10)
        grad_fn = mp.gradient_functional(pare_threshold=1e-10)

        rng = np.random.default_rng(42)
        xk = rng.random(size=len(lih_fermionic_spin_exact.monomial_circuit.parameters))

        analytical_grad = grad_fn(xk)
        fd_gradient = _finite_difference_gradient(expval_fn, xk)

        assert np.allclose(analytical_grad, fd_gradient, atol=1e-9)

    def test_expectation_value_and_gradient_functional(self, lih_fermionic_spin_exact):
        """Test expectation_value_and_gradient_functional returns consistent expectation value and gradient."""
        mp = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_fermionic_spin_exact.monomial_circuit)

        expval_grad_fn = mp.expectation_value_and_gradient_functional(
            pare_threshold=1e-10
        )
        expval_fn = mp.expectation_value_functional(pare_threshold=1e-10)
        grad_fn = mp.gradient_functional(pare_threshold=1e-10)

        combined_expval, combined_grad = expval_grad_fn(
            lih_fermionic_spin_exact.monomial_circuit.parameters
        )
        individual_expval = expval_fn(
            lih_fermionic_spin_exact.monomial_circuit.parameters
        )
        individual_grad = grad_fn(lih_fermionic_spin_exact.monomial_circuit.parameters)

        assert np.isclose(combined_expval, individual_expval, atol=1e-12)
        assert np.allclose(combined_grad, individual_grad, atol=1e-12)
        _assert_expval(combined_expval, lih_fermionic_spin_exact.exact_expval)

        grad_diff = np.max(
            np.abs(np.array(combined_grad) - lih_fermionic_spin_exact.exact_gradient)
        )
        assert np.allclose(
            combined_grad, lih_fermionic_spin_exact.exact_gradient, atol=1e-9
        ), f"Gradient vs exact mismatch: max diff = {grad_diff}"

    @pytest.mark.mpi(min_size=2)
    def test_custom_communicator_split(self, lih_fermionic_spin_exact):
        """Ensure custom communicators from mpi4py objects are respected."""
        rank = MPI.COMM_WORLD.Get_rank()
        sub_comm = MPI.COMM_WORLD.Split(color=rank % 2, key=rank)

        try:
            mp = _make_mp(lih_fermionic_spin_exact, sub_comm)
            mp.propagate(lih_fermionic_spin_exact.monomial_circuit)
            expval = mp.expectation_value_functional(
                pare_threshold=1e-10,
            )(lih_fermionic_spin_exact.monomial_circuit.parameters)
        finally:
            mp = None
            sub_comm.Free()

        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_mpi_comm_self(self, lih_fermionic_spin_exact):
        """Test single-rank communicator works correctly."""
        mp = _make_mp(lih_fermionic_spin_exact, MPI.COMM_SELF)
        mp.propagate(lih_fermionic_spin_exact.monomial_circuit)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.monomial_circuit.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)
