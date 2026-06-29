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
from monoprop.monomial_data import MonomialCircuit, MonomialOperator
from tests.cases import CasesFermionicProblemOrbitalRotations


@pytest.fixture
def mp_kwargs(comm):
    """Common MP constructor arguments — parametrized over comm_self / comm_world."""
    return {"cutoff": 6, "schrodinger_cutoff": 8, "comm": comm}


@pytest.fixture
def serial_mp_kwargs(serial_comm):
    """Common MP constructor arguments — always single-rank."""
    return {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}


def _create_mp(operator, initial_state, **kwargs):
    return MonomialPropagator(operator, initial_state, **kwargs)


def test_basic_orbital_rotation(serial_comm):
    n_modes = 4

    majs = [(1, 2)]
    params = [np.pi / 4]
    param_inds = [0]
    gen_coeffs = [1.0]

    kwargs = {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}
    operator = MonomialOperator([], n_modes)
    monomial_circuit = MonomialCircuit(
        majoranas=majs,
        parameters=params,
        gen_coeffs=gen_coeffs,
        param_inds=param_inds,
    )
    mp_act = _create_mp(operator, [], **kwargs)
    initial_state = mp_act.evolved_operator_dict()

    mp_act.propagate(
        monomial_circuit,
        ignore_circuit_parameters=False,
    )
    rotated_state = mp_act.evolved_operator_dict()

    mp_orb = _create_mp(operator, [], **kwargs)
    mp_orb.propagate(
        monomial_circuit,
        ignore_circuit_parameters=False,
        only_rotate_len_k=4,
    )
    orb_rotated_state = mp_orb.evolved_operator_dict()

    for key in initial_state:
        if len(key) > 4:
            assert np.isclose(initial_state[key], orb_rotated_state[key], atol=1e-12)

    for key in rotated_state:
        if len(key) <= 4:
            assert np.isclose(rotated_state[key], orb_rotated_state[key], atol=1e-12)


@parametrize_with_cases(
    "problem", cases=CasesFermionicProblemOrbitalRotations, has_tag="only_rotate_len_k"
)
@pytest.mark.parametrize("inplace", [False, True])
def test_only_rotate_len_k(problem, inplace, serial_mp_kwargs):
    """Tests size/graph_size (rank-local) so uses serial_comm."""

    mp = _create_mp(problem.operator, problem.initial_state, **serial_mp_kwargs)
    mp_act = _create_mp(problem.operator, problem.initial_state, **serial_mp_kwargs)
    mp_act.propagate(problem.monomial_circuit)
    act_ener = mp_act.expectation_value_functional()(
        problem.monomial_circuit.parameters
    )

    orb = problem.split_only_rotate_len_k()
    qc_without_orbs = MonomialCircuit(
        majoranas=orb.majs,
        parameters=orb.parameters,
        gen_coeffs=orb.gen_coeffs,
        param_inds=orb.param_inds,
    )
    qc_with_orbs = MonomialCircuit(
        majoranas=orb.majs_orb,
        parameters=orb.parameters_orb,
        gen_coeffs=orb.gen_coeffs_orb,
        param_inds=orb.param_inds_orb,
    )
    if inplace:
        mp.propagate(
            qc_without_orbs,
            ignore_circuit_parameters=False,
        )
        mp.propagate(
            qc_with_orbs,
            ignore_circuit_parameters=False,
            only_rotate_len_k=4,
        )
        ener_fn = mp.expectation_value_functional(ignore_coeffs=True)
        test_expval = ener_fn([])
    else:
        mp.propagate(qc_without_orbs)
        mp.propagate(
            qc_with_orbs,
            only_rotate_len_k=4,
        )
        ener_fn = mp.expectation_value_functional()
        test_expval = ener_fn(problem.monomial_circuit.parameters)
        assert sum(mp.graph_size()) < sum(mp_act.graph_size())

    assert mp.size() < mp_act.size()
    assert np.isclose(test_expval, act_ener, atol=1e-12)
