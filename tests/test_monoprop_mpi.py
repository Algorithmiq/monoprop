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

"""Distributed propagation, checked against exact values or the same propagation run serially.

Every test carries the ``mpi`` marker and runs on ``comm_world``, the session's duplicate of
COMM_WORLD. ``just test-py-mpi`` runs them in each configuration CI covers; by hand:

    mpiexec --map-by :OVERSUBSCRIBE -n 4 uv run --no-sync pytest tests/test_monoprop_mpi.py --with-mpi -m mpi
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from monoprop import Circuit, ExpGate, MajoranaPropagator, PauliPropagator
from monoprop.pauli import Pauli, PauliOperator
from tests.reference_data import load_problem

MPI = pytest.importorskip("mpi4py").MPI

pytestmark = pytest.mark.mpi

# The C++ suite's near(): a reduction sums in an order that depends on the rank count.
RTOL = 1e-7
ATOL = 1e-9

# random_exact's observable has only terms of length >= 10: a lower cutoff leaves the energy 0 on
# every rank count, which cannot tell a broken exchange apart.
MAJORANA_CUTOFF = 10
PAULI_CUTOFF = 3
N_QUBITS = 6
PAULI_OBSERVABLES = (
    Pauli("ZZ", (0, 1)),
    Pauli("XY", (2, 3)),
    Pauli("Z", (4,)),
    Pauli("YX", (1, 5)),
    Pauli("ZZ", (2, 4)),
)


@pytest.fixture
def random_exact(lazy_shared_datadir):
    return load_problem(lazy_shared_datadir / "random_exact.msgpack")


def _majorana(problem, comm, *, cutoff=None, schrodinger=False):
    cutoff = 2 * problem.n_modes if cutoff is None else cutoff
    propagator = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=cutoff,
        schrodinger_cutoff=cutoff if schrodinger else None,
        comm=comm,
    )
    return propagator, problem.monomial_circuit.to_circuit()


def _pauli_circuit() -> Circuit:
    """Rotations about each axis between nearest-neighbour ZZ layers, enough to spread every observable."""
    gates = []
    for layer in range(3):
        axis = "XYZ"[layer % 3]
        gates += [
            ExpGate(PauliOperator({Pauli(axis, (q,)): 1.0}, N_QUBITS))
            for q in range(N_QUBITS)
        ]
        gates += [
            ExpGate(PauliOperator({Pauli("ZZ", (q, q + 1)): 1.0}, N_QUBITS))
            for q in range(layer % 2, N_QUBITS - 1, 2)
        ]
    return Circuit(tuple(gates), N_QUBITS)


def _pauli(comm, *, cutoff, schrodinger):
    """A Pauli propagator over ``PAULI_OBSERVABLES`` with the circuit built, and its angles."""
    circuit = _pauli_circuit()
    weights = np.linspace(0.3, 1.2, len(PAULI_OBSERVABLES))
    propagator = PauliPropagator(
        PauliOperator(dict(zip(PAULI_OBSERVABLES, weights)), N_QUBITS),
        initial_state=[0, 3],
        cutoff=cutoff,
        schrodinger_cutoff=cutoff if schrodinger else None,
        comm=comm,
    )
    propagator.build_graph(circuit)
    return propagator, np.random.default_rng(7).uniform(
        0, 2 * np.pi, circuit.n_parameters
    )


def _pauli_energy(comm, *, cutoff=PAULI_CUTOFF, schrodinger=False) -> float:
    propagator, parameters = _pauli(comm, cutoff=cutoff, schrodinger=schrodinger)
    return propagator.expectation_value(parameters)


def _finite_difference_gradient(expval_fn, parameters, eps=1e-6):
    gradient = np.zeros_like(parameters)
    for index in range(len(parameters)):
        params_plus = parameters.copy()
        params_plus[index] += eps
        params_minus = parameters.copy()
        params_minus[index] -= eps
        gradient[index] = (expval_fn(params_plus) - expval_fn(params_minus)) / (2 * eps)
    return gradient


def test_expval_functional_matches_exact(lih_problem, comm_world):
    propagator, circuit = _majorana(lih_problem, comm_world)
    propagator.build_graph(circuit)
    expval = propagator.expval_functional(pare_threshold=1e-10)(
        lih_problem.monomial_circuit.parameters
    )
    assert np.isclose(expval, lih_problem.exact_expval, atol=1e-9)


def test_immediate_contraction_matches_exact(lih_problem, comm_world):
    propagator, circuit = _majorana(lih_problem, comm_world)
    propagator.propagate(circuit)
    expval = propagator.expval_functional(pare_threshold=1e-10)()
    assert np.isclose(expval, lih_problem.exact_expval, atol=1e-9)


def test_schrodinger_matches_exact(lih_problem, comm_world):
    propagator, circuit = _majorana(lih_problem, comm_world, schrodinger=True)
    propagator.build_graph(circuit)
    expval = propagator.expval_functional(pare_threshold=1e-10)(
        lih_problem.monomial_circuit.parameters
    )
    assert np.isclose(expval, lih_problem.exact_expval, atol=1e-9)


def test_gradient_matches_finite_differences(lih_problem, comm_world):
    propagator, circuit = _majorana(lih_problem, comm_world)
    propagator.build_graph(circuit)
    expval_fn = propagator.expval_functional(pare_threshold=1e-10)
    grad_fn = propagator.expval_and_grad_functional(pare_threshold=1e-10)
    rng = np.random.default_rng(42)
    parameters = rng.random(size=len(lih_problem.monomial_circuit.parameters))
    _, gradient = grad_fn(parameters)
    assert np.allclose(
        gradient, _finite_difference_gradient(expval_fn, parameters), atol=1e-9
    )


def test_expval_and_grad_functional_matches_exact(lih_problem, comm_world):
    propagator, circuit = _majorana(lih_problem, comm_world)
    propagator.build_graph(circuit)
    parameters = lih_problem.monomial_circuit.parameters
    combined_expval, combined_grad = propagator.expval_and_grad_functional(
        pare_threshold=1e-10
    )(parameters)
    individual_expval = propagator.expval_functional(pare_threshold=1e-10)(parameters)
    assert np.isclose(combined_expval, individual_expval, atol=1e-12)
    assert np.isclose(combined_expval, lih_problem.exact_expval, atol=1e-9)
    assert np.allclose(combined_grad, lih_problem.exact_gradient, atol=1e-9)


def test_truncated_majorana_matches_serial(random_exact, comm_world):
    """A truncated evolution distributes to the serial energy and gradient."""
    parameters = random_exact.monomial_circuit.parameters
    results = []
    for comm in (None, comm_world):
        propagator, circuit = _majorana(random_exact, comm, cutoff=MAJORANA_CUTOFF)
        propagator.build_graph(circuit)
        results.append(propagator.expectation_value_and_gradient(parameters))
    (serial_energy, serial_gradient), (world_energy, world_gradient) = results
    assert abs(serial_energy) > 1e-6
    assert not np.isclose(serial_energy, random_exact.exact_expval, atol=1e-6)
    assert np.isclose(world_energy, serial_energy, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(world_gradient, serial_gradient, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
)
def test_truncated_pauli_matches_serial(schrodinger, comm_world):
    serial = _pauli_energy(None, schrodinger=schrodinger)
    assert not np.isclose(serial, _pauli_energy(None, cutoff=N_QUBITS), atol=1e-6)
    world = _pauli_energy(comm_world, schrodinger=schrodinger)
    assert np.isclose(world, serial, rtol=RTOL, atol=ATOL)


def test_update_initial_operator_matches_serial(comm_world):
    """Re-weighting reaches the terms each rank owns, so the distributed energy follows the serial one."""
    reweighted = PauliOperator(
        dict(zip(PAULI_OBSERVABLES, np.linspace(1.2, 0.3, len(PAULI_OBSERVABLES)))),
        N_QUBITS,
    )
    energies = []
    for comm in (None, comm_world):
        propagator, parameters = _pauli(comm, cutoff=PAULI_CUTOFF, schrodinger=False)
        propagator.update_initial_operator(reweighted)
        energies.append(propagator.expectation_value(parameters))
    serial, world = energies
    assert not np.isclose(serial, _pauli_energy(None), atol=1e-6)
    assert np.isclose(world, serial, rtol=RTOL, atol=ATOL)


def test_evolved_operator_coefficients_is_rank_local(lih_problem, comm_world):
    """Each rank answers for the terms it owns, and the ranks' answers sum to the serial result."""
    parameters = lih_problem.monomial_circuit.parameters

    serial, serial_circuit = _majorana(lih_problem, None)
    serial.build_graph(serial_circuit)
    # Sorted so every rank queries the same terms in the same order.
    terms = sorted(t for t in serial.evolved_operator(parameters, atol=0.0).terms if t)
    assert terms
    expected = serial.evolved_operator_coefficients(terms, parameters)

    world, world_circuit = _majorana(lih_problem, comm_world)
    world.build_graph(world_circuit)
    local = world.evolved_operator_coefficients(terms, parameters)

    summed = np.zeros_like(local)
    comm_world.Allreduce(local, summed, op=MPI.SUM)
    np.testing.assert_allclose(summed, expected, atol=1e-9)

    if comm_world.size > 1:
        # Counted against the terms each rank *owns* rather than against its nonzero coefficients.
        local_terms = world.evolved_operator(parameters, atol=0.0).terms
        owned = sum(1 for t in local_terms if t)  # the identity is on every rank
        busiest = comm_world.allreduce(owned, op=MPI.MAX)
        assert busiest < len(terms)


@pytest.mark.mpi(min_size=4)
def test_split_communicators_each_match_exact(lih_problem, comm_world):
    """Two halves of the world propagate independently; each half holds at least two ranks."""
    sub_comm = comm_world.Split(color=comm_world.rank % 2, key=comm_world.rank)
    propagator = None
    try:
        propagator, circuit = _majorana(lih_problem, sub_comm)
        propagator.build_graph(circuit)
        expval = propagator.expval_functional(pare_threshold=1e-10)(
            lih_problem.monomial_circuit.parameters
        )
    finally:
        propagator = None  # the communicator must outlive the propagator
        sub_comm.Free()
    assert np.isclose(expval, lih_problem.exact_expval, atol=1e-9)


@pytest.mark.mpi(min_size=3)
def test_linear_routing_refuses_a_three_rank_communicator(random_exact, comm_world):
    """Linear routing XORs rank indices, so construction refuses a rank count that is not a power of two."""
    if os.environ.get("monoprop_ROUTING", "") not in ("", "linear"):  # noqa: SIM112
        pytest.skip("monoprop_ROUTING selects splitmix, which routes any rank count")
    color = 0 if comm_world.rank < 3 else MPI.UNDEFINED
    three_ranks = comm_world.Split(color=color, key=comm_world.rank)
    if three_ranks == MPI.COMM_NULL:
        return  # outside the three-rank communicator there is nothing to construct
    try:
        with pytest.raises(RuntimeError, match="power-of-two rank count"):
            _majorana(random_exact, three_ranks)
    finally:
        three_ranks.Free()
