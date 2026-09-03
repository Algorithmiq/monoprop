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

"""MPI tests for the Majorana Propagator.

Run with: mpirun -np <num_ranks> uv run pytest tests/test_monoprop_mpi.py --with-mpi -m mpi -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.cases import load_problem

MPI = pytest.importorskip("mpi4py").MPI

from monoprop import MajoranaPropagator  # noqa: E402


@pytest.fixture
def lih_fermionic_spin_exact(lazy_shared_datadir):
    data_path = lazy_shared_datadir / "lih_fermionic_spin_exact.msgpack"
    return load_problem(data_path)


def _make_mp(problem, comm, *, schrodinger=False):
    mp = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        schrodinger_cutoff=2 * problem.n_modes if schrodinger else None,
        comm=comm,
    )
    circuit = problem.monomial_circuit.to_circuit()
    return mp, circuit


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
    def test_evolve_expectation_value_functional(self, lih_fermionic_spin_exact):
        mp, circuit = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.build_graph(circuit)
        expval = mp.expval_functional(pare_threshold=1e-10)(
            lih_fermionic_spin_exact.monomial_circuit.parameters
        )
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_evolve_build_graph_with_coeffs(self, lih_fermionic_spin_exact):
        mp, circuit = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        parameters = lih_fermionic_spin_exact.monomial_circuit.parameters
        mp.build_graph(circuit)
        expval = mp.expval_functional(pare_threshold=1e-10)(parameters)
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_immediate_contraction(self, lih_fermionic_spin_exact):
        mp, circuit = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.propagate(circuit)
        _assert_expval(
            mp.expval_functional(pare_threshold=1e-10)(),
            lih_fermionic_spin_exact.exact_expval,
        )

    def test_schrodinger_picture(self, lih_fermionic_spin_exact):
        mp, circuit = _make_mp(
            lih_fermionic_spin_exact, MPI.COMM_WORLD, schrodinger=True
        )
        mp.build_graph(circuit)
        expval = mp.expval_functional(pare_threshold=1e-10)(
            lih_fermionic_spin_exact.monomial_circuit.parameters
        )
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_gradient(self, lih_fermionic_spin_exact):
        """Check the analytic gradient against central finite differences."""
        mp, circuit = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.build_graph(circuit)

        expval_fn = mp.expval_functional(pare_threshold=1e-10)
        grad_fn = mp.expval_and_grad_functional(pare_threshold=1e-10)

        rng = np.random.default_rng(42)
        xk = rng.random(size=len(lih_fermionic_spin_exact.monomial_circuit.parameters))

        _, analytical_grad = grad_fn(xk)
        fd_gradient = _finite_difference_gradient(expval_fn, xk)

        assert np.allclose(analytical_grad, fd_gradient, atol=1e-9)

    def test_expectation_value_and_gradient_functional(self, lih_fermionic_spin_exact):
        """Combined expectation value + gradient is consistent and matches exact."""
        mp, circuit = _make_mp(lih_fermionic_spin_exact, MPI.COMM_WORLD)
        mp.build_graph(circuit)
        parameters = lih_fermionic_spin_exact.monomial_circuit.parameters

        expval_grad_fn = mp.expval_and_grad_functional(pare_threshold=1e-10)
        expval_fn = mp.expval_functional(pare_threshold=1e-10)

        combined_expval, combined_grad = expval_grad_fn(parameters)
        individual_expval = expval_fn(parameters)

        assert np.isclose(combined_expval, individual_expval, atol=1e-12)
        _assert_expval(combined_expval, lih_fermionic_spin_exact.exact_expval)

        assert np.allclose(
            combined_grad, lih_fermionic_spin_exact.exact_gradient, atol=1e-9
        )

    def test_evolved_operator_coefficients_is_rank_local(
        self, lih_fermionic_spin_exact
    ):
        """Each rank answers for the terms it owns, and the ranks' answers sum to the serial result.

        ``evolved_operator_coefficients`` is rank-local exactly as ``evolved_operator`` is: a term another
        rank owns reads back as 0. The identity is excluded from the query -- the core term is
        replicated on every rank, so the sum below would count it once per rank.
        """
        problem = lih_fermionic_spin_exact
        parameters = problem.monomial_circuit.parameters

        serial, serial_circuit = _make_mp(problem, MPI.COMM_SELF)
        serial.build_graph(serial_circuit)
        # Sorted, so every rank queries the same terms in the same order without relying on the
        # enumeration order being reproducible across processes.
        terms = sorted(
            t for t in serial.evolved_operator(parameters, atol=0.0).terms if t
        )
        assert terms
        expected = serial.evolved_operator_coefficients(terms, parameters)

        world, world_circuit = _make_mp(problem, MPI.COMM_WORLD)
        world.build_graph(world_circuit)
        local = world.evolved_operator_coefficients(terms, parameters)

        summed = np.zeros_like(local)
        MPI.COMM_WORLD.Allreduce(local, summed, op=MPI.SUM)

        np.testing.assert_allclose(summed, expected, atol=1e-9)

        if MPI.COMM_WORLD.size > 1:
            # Pin that the split is real, so the sum above is not just every rank answering in
            # full. Counted against the terms each rank *owns* rather than against its nonzero
            # coefficients, which would drift if an owned term evolved to exactly 0. Collective,
            # so the assertion holds or fails identically on every rank.
            local_terms = world.evolved_operator(parameters, atol=0.0).terms
            owned = sum(1 for t in local_terms if t)  # the identity is on every rank
            busiest = MPI.COMM_WORLD.allreduce(owned, op=MPI.MAX)
            assert busiest < len(terms)

    @pytest.mark.mpi(min_size=2)
    def test_custom_communicator_split(self, lih_fermionic_spin_exact):
        rank = MPI.COMM_WORLD.Get_rank()
        sub_comm = MPI.COMM_WORLD.Split(color=rank % 2, key=rank)

        try:
            mp, circuit = _make_mp(lih_fermionic_spin_exact, sub_comm)
            mp.build_graph(circuit)
            expval = mp.expval_functional(pare_threshold=1e-10)(
                lih_fermionic_spin_exact.monomial_circuit.parameters
            )
        finally:
            mp = None
            sub_comm.Free()

        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)

    def test_mpi_comm_self(self, lih_fermionic_spin_exact):
        mp, circuit = _make_mp(lih_fermionic_spin_exact, MPI.COMM_SELF)
        mp.build_graph(circuit)
        expval = mp.expval_functional(pare_threshold=1e-10)(
            lih_fermionic_spin_exact.monomial_circuit.parameters
        )
        _assert_expval(expval, lih_fermionic_spin_exact.exact_expval)
