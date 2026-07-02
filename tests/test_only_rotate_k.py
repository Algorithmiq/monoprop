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

from monoprop import MajoranaPropagator, gates_from_monomial_sequence
from monoprop.monomial_data import MonomialOperator, MonomialSequence
from tests.cases import CasesFermionicProblemOrbitalRotations


@pytest.fixture
def serial_mp_kwargs(serial_comm):
    """Common MP constructor arguments — always single-rank."""
    return {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}


def _split_orbital_gates(gates):
    """Split gates into (non-orbital, orbital), where orbital gates are the tail run
    whose generated monomials are all length-2 (free-fermion rotations)."""

    def is_orbital(gate):
        return all(len(term.majorana) == 2 for term in gate.terms)

    for i in range(len(gates)):
        if all(is_orbital(gate) for gate in gates[i:]):
            return gates[:i], gates[i:]
    return gates, []


def test_basic_orbital_rotation(serial_comm):
    n_modes = 4

    operator = MonomialOperator({}, n_modes)
    sequence = MonomialSequence(
        initial_state=[],
        majoranas=[(1, 2)],
        parameters=[np.pi / 4],
        gen_coeffs=[1.0],
        param_inds=[0],
    )
    gates, _ = gates_from_monomial_sequence(sequence)
    kwargs = {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}

    mp_act = MajoranaPropagator(operator, sequence.initial_state, **kwargs)
    initial_state = mp_act.evolved_operator_dict()
    mp_act.propagate(gates, sequence.parameters)
    rotated_state = mp_act.evolved_operator_dict()

    mp_orb = MajoranaPropagator(operator, sequence.initial_state, **kwargs)
    mp_orb.propagate(gates, sequence.parameters, only_rotate_len_k=4)
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
    gates, params_vector = gates_from_monomial_sequence(problem.monomial_circuit)
    parameters = problem.monomial_circuit.parameters
    parameters_by_handle = dict(zip(params_vector, parameters))
    non_orbital_gates, orbital_gates = _split_orbital_gates(gates)

    mp_act = MajoranaPropagator(
        problem.operator, problem.monomial_circuit.initial_state, **serial_mp_kwargs
    )
    mp_act.propagate_build_graph(gates)
    act_ener = mp_act.expectation_value_functional()(parameters)

    mp = MajoranaPropagator(
        problem.operator, problem.monomial_circuit.initial_state, **serial_mp_kwargs
    )

    if inplace:
        mp.propagate(non_orbital_gates, parameters_by_handle)
        mp.propagate(orbital_gates, parameters_by_handle, only_rotate_len_k=4)
        test_expval = mp.expectation_value()
    else:
        mp.propagate_build_graph(non_orbital_gates)
        mp.propagate_build_graph(orbital_gates, only_rotate_len_k=4)
        test_expval = mp.expectation_value_functional()(parameters)
        assert sum(mp.graph_size()) < sum(mp_act.graph_size())

    assert mp.size() < mp_act.size()
    assert np.isclose(test_expval, act_ener, atol=1e-12)
