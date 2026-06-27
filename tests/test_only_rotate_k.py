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


def _create_mp(operator, monomial_circuit, **kwargs):
    return MonomialPropagator(operator, monomial_circuit, **kwargs)


def test_basic_orbital_rotation(serial_comm):
    n_modes = 4

    majs = [(1, 2)]
    params = [np.pi / 4]
    param_inds = [0]
    gen_coeffs = [1.0]

    kwargs = {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}
    operator = MonomialOperator({}, n_modes)
    monomial_circuit = MonomialCircuit(
        initial_state=[],
        majoranas=majs,
        parameters=params,
        gen_coeffs=gen_coeffs,
        param_inds=param_inds,
    )
    mp_act = _create_mp(operator, monomial_circuit, **kwargs)
    initial_state = mp_act.evolved_operator_dict()

    mp_act.propagate(
        evolve_with_coeffs=True,
    )
    rotated_state = mp_act.evolved_operator_dict()

    mp_orb = _create_mp(operator, monomial_circuit, **kwargs)
    mp_orb.propagate(
        evolve_with_coeffs=True,
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

    mp = _create_mp(problem.operator, problem.monomial_circuit, **serial_mp_kwargs)
    mp_act = _create_mp(problem.operator, problem.monomial_circuit, **serial_mp_kwargs)
    mp_act.propagate()
    act_ener = mp_act.expectation_value_functional(
        use_coeffs=True,
    )(problem.monomial_circuit.parameters)

    orb = problem.split_only_rotate_len_k()

    if inplace:
        mp.propagate(
            parameter_mapping=orb.param_inds,
            majoranas=orb.majs,
            gen_coeffs=orb.gen_coeffs,
            parameters=orb.parameters,
        )
        mp.propagate(
            parameter_mapping=orb.param_inds_orb,
            majoranas=orb.majs_orb,
            gen_coeffs=orb.gen_coeffs_orb,
            parameters=orb.parameters_orb,
            only_rotate_len_k=4,
        )
        ener_fn = mp.expectation_value_functional(parameter_mapping=[], gen_coeffs=[])
        test_expval = ener_fn([])
    else:
        mp.propagate(orb.majs)
        mp.propagate(orb.majs_orb, only_rotate_len_k=4)
        ener_fn = mp.expectation_value_functional(
            use_coeffs=True,
        )
        test_expval = ener_fn(problem.monomial_circuit.parameters)
        assert sum(mp.graph_size()) < sum(mp_act.graph_size())

    assert mp.size() < mp_act.size()
    assert np.isclose(test_expval, act_ener, atol=1e-12)
