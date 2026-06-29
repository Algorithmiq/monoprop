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

from monoprop.monomial_data import MonomialCircuit, MonomialOperator

MPI = pytest.importorskip("mpi4py").MPI

from monoprop import MonomialPropagator, MPData  # noqa: E402


@pytest.fixture
def lih_fermionic_spin_exact(lazy_shared_datadir):
    """Load test data from msgpack file."""
    data_path = lazy_shared_datadir / "lih_fermionic_spin_exact.msgpack"
    return MPData.from_msgpack(filepath=data_path)


@pytest.fixture
def lih_hamiltonian(lih_fermionic_spin_exact):
    """Build the LiH Hamiltonian from test data."""
    return MonomialOperator.from_dict(
        terms_dict=lih_fermionic_spin_exact.fermionic_hamiltonian,
        num_modes=lih_fermionic_spin_exact.num_modes,
    )


@pytest.fixture
def lih_circuit(lih_fermionic_spin_exact):
    """Build the LiH monomial circuit from test data."""
    return MonomialCircuit(
        majoranas=lih_fermionic_spin_exact.majoranas,
        parameters=lih_fermionic_spin_exact.parameters,
        gen_coeffs=lih_fermionic_spin_exact.gen_coeffs,
        param_inds=lih_fermionic_spin_exact.param_inds,
    )


def _make_mp(hamiltonian, data, comm):
    """Create a standard MP instance from a pre-built hamiltonian."""
    return MonomialPropagator(
        hamiltonian,
        data.hartree_fock,
        2 * data.num_modes,
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

    def test_evolve_expectation_value_functional(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_circuit)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.actual_energy)

    def test_evolve_with_operator_coeffs(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, MPI.COMM_WORLD)
        operator_coeffs = mp.contract_partially(inplace=False)
        mp.propagate(lih_circuit, operator_coeffs=operator_coeffs)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.actual_energy)

    def test_immediate_contraction(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_circuit, ignore_circuit_parameters=False)
        _assert_expval(
            mp.expectation_value_functional(ignore_coeffs=True, pare_threshold=1e-10)(),
            lih_fermionic_spin_exact.actual_energy,
        )

    def test_schrodinger_picture(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        mp = MonomialPropagator(
            lih_hamiltonian,
            lih_fermionic_spin_exact.hartree_fock,
            2 * lih_fermionic_spin_exact.num_modes,
            schrodinger_cutoff=2 * lih_fermionic_spin_exact.num_modes,
            comm=MPI.COMM_WORLD,
        )
        mp.propagate(lih_circuit)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.actual_energy)

    def test_gradient_functional(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        """Test gradient_functional against finite difference."""
        mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_circuit)

        expval_fn = mp.expectation_value_functional(pare_threshold=1e-10)
        grad_fn = mp.gradient_functional(pare_threshold=1e-10)

        rng = np.random.default_rng(42)
        xk = rng.random(size=len(lih_fermionic_spin_exact.parameters))

        analytical_grad = grad_fn(xk)
        fd_gradient = _finite_difference_gradient(expval_fn, xk)

        assert np.allclose(analytical_grad, fd_gradient, atol=1e-9)

    def test_expectation_value_and_gradient_functional(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        """Test expectation_value_and_gradient_functional returns consistent expectation value and gradient."""
        mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(lih_circuit)

        expval_grad_fn = mp.expectation_value_and_gradient_functional(
            pare_threshold=1e-10
        )
        expval_fn = mp.expectation_value_functional(pare_threshold=1e-10)
        grad_fn = mp.gradient_functional(pare_threshold=1e-10)

        combined_expval, combined_grad = expval_grad_fn(
            lih_fermionic_spin_exact.parameters
        )
        individual_expval = expval_fn(lih_fermionic_spin_exact.parameters)
        individual_grad = grad_fn(lih_fermionic_spin_exact.parameters)

        assert np.isclose(combined_expval, individual_expval, atol=1e-12)
        assert np.allclose(combined_grad, individual_grad, atol=1e-12)
        _assert_expval(combined_expval, lih_fermionic_spin_exact.actual_energy)

        grad_diff = np.max(
            np.abs(np.array(combined_grad) - lih_fermionic_spin_exact.actual_gradient)
        )
        assert np.allclose(
            combined_grad, lih_fermionic_spin_exact.actual_gradient, atol=1e-9
        ), f"Gradient vs exact mismatch: max diff = {grad_diff}"

    @pytest.mark.mpi(min_size=2)
    def test_custom_communicator_split(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        """Ensure custom communicators from mpi4py objects are respected."""
        rank = MPI.COMM_WORLD.Get_rank()
        sub_comm = MPI.COMM_WORLD.Split(color=rank % 2, key=rank)

        try:
            mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, sub_comm)
            mp.propagate(lih_circuit)
            expval = mp.expectation_value_functional(
                pare_threshold=1e-10,
            )(lih_fermionic_spin_exact.parameters)
        finally:
            mp = None
            sub_comm.Free()

        _assert_expval(expval, lih_fermionic_spin_exact.actual_energy)

    def test_mpi_comm_self(
        self, lih_fermionic_spin_exact, lih_hamiltonian, lih_circuit
    ):
        """Test single-rank communicator works correctly."""
        mp = _make_mp(lih_hamiltonian, lih_fermionic_spin_exact, MPI.COMM_SELF)
        mp.propagate(lih_circuit)
        expval = mp.expectation_value_functional(
            pare_threshold=1e-10,
        )(lih_fermionic_spin_exact.parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.actual_energy)
