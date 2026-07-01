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

from __future__ import annotations

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from monoprop import MonomialPropagator
from tests.cases import CasesFermionicProblem


def _create_mp(operator, initial_circuit, n_modes, comm, *, schrodinger_cutoff=None):
    return MonomialPropagator(
        operator,
        initial_circuit,
        cutoff=2 * n_modes,
        schrodinger_cutoff=2 * n_modes
        if schrodinger_cutoff == "max"
        else schrodinger_cutoff,
        comm=comm,
    )


@parametrize_with_cases("problem", cases=CasesFermionicProblem)
@pytest.mark.parametrize("pare_threshold", [None, 1e-10])
@pytest.mark.parametrize("schrodinger_cutoff", [None, "max"])
@pytest.mark.parametrize("evolution_mode", ["deferred", "with_coeffs", "inplace"])
def test_infinite_cutoff(
    problem, pare_threshold, schrodinger_cutoff, evolution_mode, comm
):
    mp = _create_mp(
        problem.operator,
        problem.initial_state,
        problem.n_modes,
        comm,
        schrodinger_cutoff=schrodinger_cutoff,
    )
    quantum_circuit = problem.monomial_circuit
    match evolution_mode:
        case "deferred":
            mp.propagate(quantum_circuit=quantum_circuit)
            test_expval = mp.expectation_value_functional(
                pare_threshold=pare_threshold,
            )(quantum_circuit.parameters)

        case "with_coeffs":
            operator_coeffs = mp.contract_partially(inplace=False)
            mp.propagate(
                quantum_circuit,
                operator_coeffs=operator_coeffs,
                ignore_circuit_parameters=False,
            )
            test_expval = mp.expectation_value_functional(
                pare_threshold=pare_threshold,
            )(quantum_circuit.parameters)

        case "inplace":
            mp.propagate(
                quantum_circuit,
                ignore_circuit_parameters=False,
            )
            test_expval = mp.expectation_value_functional(
                pare_threshold=pare_threshold
            )([])

    assert np.isclose(test_expval, problem.exact_expval, atol=1e-12)


@parametrize_with_cases("problem", cases=CasesFermionicProblem)
@pytest.mark.parametrize("pare_threshold", [None, 1e-10])
@pytest.mark.parametrize("schrodinger", [False, True])
def test_gradient(problem, schrodinger, pare_threshold, comm):
    """Test expectation_value_and_gradient_functional against finite differences."""
    prng = np.random.default_rng(0)
    quantum_circuit = problem.monomial_circuit
    cutoff = 4
    mp = MonomialPropagator(
        problem.operator,
        problem.initial_state,
        cutoff=cutoff,
        schrodinger_cutoff=cutoff + 2 if schrodinger else None,
        comm=comm,
    )
    mp.propagate(quantum_circuit)
    ener_fn = mp.expectation_value_functional(
        pare_threshold=pare_threshold,
    )

    xk = prng.random(size=len(quantum_circuit.parameters))
    true_expval = ener_fn(xk)

    eps = 1e-6
    fd_gradient = np.zeros_like(xk)
    for i in range(len(xk)):
        xk_old = xk[i]
        xk[i] = xk_old + eps
        e_plus = ener_fn(xk)
        xk[i] = xk_old - eps
        e_minus = ener_fn(xk)
        fd_gradient[i] = (e_plus - e_minus) / (2 * eps)
        xk[i] = xk_old

    fn = mp.expectation_value_and_gradient_functional(
        pare_threshold=pare_threshold,
    )
    test_expval, test_gradient = fn(xk)
    assert np.isclose(test_expval, true_expval, atol=1e-6)
    assert np.allclose(test_gradient, fd_gradient, atol=1e-6)
    test_gradient2 = mp.gradient(xk)
    assert np.allclose(test_gradient2, fd_gradient, atol=1e-6)


@parametrize_with_cases("problem", cases=CasesFermionicProblem)
@pytest.mark.parametrize("schrodinger", [True, False])
def test_immediate_contraction(problem, schrodinger, comm):
    """Test propagate with immediate contraction returns correct expectation value."""
    n_modes = problem.n_modes

    mp = MonomialPropagator(
        problem.operator,
        problem.initial_state,
        cutoff=2 * n_modes,
        schrodinger_cutoff=2 * n_modes if schrodinger else None,
        lower_atol=1e-15,
        comm=comm,
    )
    mp.propagate(
        quantum_circuit=problem.monomial_circuit,
        ignore_circuit_parameters=False,
    )
    test_expval, gradient = mp.expectation_value_and_gradient([], ignore_coeffs=True)

    assert np.isclose(test_expval, problem.exact_expval, atol=1e-12)
    assert len(gradient) == 0

    test_expval = mp.expectation_value_functional(ignore_coeffs=True)([])
    assert np.isclose(test_expval, problem.exact_expval, atol=1e-12)
