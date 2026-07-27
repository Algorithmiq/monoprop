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

from monoprop import Circuit, MajoranaPropagator
from monoprop.fermi import MajoranaOperator
from tests.cases import CasesFermionicProblem, FermionicProblem


def _create_mp(
    op,
    initial_state,
    comm,
    upper_atol=None,
    lower_atol=None,
    cutoff=4,
    schrodinger_cutoff=None,
):
    return MajoranaPropagator(
        op,
        initial_state,
        cutoff=cutoff,
        upper_atol=upper_atol,
        lower_atol=lower_atol,
        schrodinger_cutoff=schrodinger_cutoff,
        comm=comm,
    )


def _check_dicts(d1, d2):
    assert len(d1) == len(d2), f"Length mismatch: {len(d1)} vs {len(d2)}"
    for k in d1:
        assert k in d2, f"Key {k} not found in second dictionary"
        assert np.isclose(d1[k], d2[k]), (
            f"Value mismatch for key {k}: {d1[k]} vs {d2[k]}"
        )


def test_coeff_trunc(serial_comm):
    n_modes = 5
    op = MajoranaOperator({(0, 1): 1.0j}, num_modes=n_modes)

    sequence = Circuit.from_dense_arrays(
        initial_state=[],
        majoranas=[(1, 2, 3, 4)],
        parameters=[np.pi / 6],
        gen_coeffs=[1.0],
        param_inds=[0],
        num_modes=n_modes,
    )
    circuit = sequence

    cos_pi6, sin_pi6 = (0.5, -0.8660254037844386)
    act_op = {(0, 1): 1j * cos_pi6, (0, 2, 3, 4): sin_pi6}
    truncated_op = {(0, 1): 1j * cos_pi6}

    def get_test_dict(upper_atol, lower_atol, cutoff):
        mp = _create_mp(
            op, sequence.initial_state, serial_comm, upper_atol, lower_atol, cutoff
        )
        mp.propagate(circuit)
        return mp.evolved_operator()

    test_scenarios = [
        (4, None, None, act_op),
        (1, 0.49, None, act_op),
        (1, 1.0, None, truncated_op),
        (4, None, 0.9, truncated_op),
        (10, None, 0.1, act_op),
    ]

    for cutoff, upper_atol, lower_atol, expected in test_scenarios:
        test_op = get_test_dict(upper_atol, lower_atol, cutoff)
        _check_dicts(test_op, expected)


@parametrize_with_cases("problem", cases=CasesFermionicProblem, has_tag="molecule")
@pytest.mark.parametrize("schrodinger", [True, False])
@pytest.mark.parametrize("upper_atol", [1e-0, 1e-1, 1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("lower_atol", [1e-5, 1e-6])
@pytest.mark.parametrize("cutoff", [4, 6, 8])
def test_coeff_trunc_build_graph_and_inplace_equiv(
    problem: FermionicProblem, schrodinger, upper_atol, lower_atol, cutoff, serial_comm
):
    """Check that build graph with coeffs and inplace version match."""
    n_modes = problem.n_modes
    fermionic_operator = problem.operator
    monomial_circuit = problem.monomial_circuit
    schrodinger_cutoff_val = 2 * n_modes if schrodinger else None
    circuit = monomial_circuit.to_circuit()
    parameters = monomial_circuit.parameters

    mp_inplace = _create_mp(
        fermionic_operator,
        monomial_circuit.initial_state,
        serial_comm,
        upper_atol,
        lower_atol,
        cutoff,
        schrodinger_cutoff_val,
    )
    mp_inplace.propagate(circuit)
    expval_inplace = mp_inplace.expectation_value()

    mp_build = _create_mp(
        fermionic_operator,
        monomial_circuit.initial_state,
        serial_comm,
        upper_atol,
        lower_atol,
        cutoff,
        schrodinger_cutoff_val,
    )
    # Coefficient-informed build: the seed is regenerated internally from parameters.
    mp_build.build_graph(circuit)
    expval_build = mp_build.expectation_value(parameters)

    assert mp_inplace.size() == mp_build.size()
    assert np.isclose(expval_inplace, expval_build, atol=1e-12)


def test_evolution_coeff_trunc_no_atols(serial_comm):
    n_modes = 2
    p = np.pi / 16

    init_op = MajoranaOperator({(0, 1): 1.0j, (0, 2): 1.0j}, num_modes=n_modes)

    sequence = Circuit.from_dense_arrays(
        initial_state=[],
        majoranas=[(1, 2)],
        parameters=[p],
        gen_coeffs=[1.0],
        param_inds=[0],
        num_modes=n_modes,
    )
    circuit = sequence

    cutoff = 4
    final_operator = {
        (0, 1): 1.0j * np.cos(2 * p) - 1.0j * np.sin(2 * p),
        (0, 2): 1.0j * np.cos(2 * p) + 1.0j * np.sin(2 * p),
    }

    mp = MajoranaPropagator(
        init_op,
        sequence.initial_state,
        cutoff=cutoff,
        schrodinger_cutoff=None,
        comm=serial_comm,
    )
    mp.build_graph(circuit)
    assert mp.graph_size()[1] == 1
    assert mp.graph_size()[0] == 0
    assert mp.size() == 2

    test_op = mp.evolved_operator(sequence.parameters)
    _check_dicts(test_op, final_operator)


def test_evolution_coeff_trunc_small_coeffs(serial_comm):
    n_modes = 2
    p = 1e-5
    init_op = {(0, 1): 1.0j, (0, 2): 1e-7j}
    op = MajoranaOperator({(0, 1): 1.0j, (0, 2): 1e-7j}, num_modes=n_modes)

    sequence = Circuit.from_dense_arrays(
        initial_state=[],
        majoranas=[(1, 2)],
        parameters=[p],
        gen_coeffs=[1.0],
        param_inds=[0],
        num_modes=n_modes,
    )
    circuit = sequence

    cutoff = 4
    final_operator = {
        (0, 1): init_op[(0, 1)] * np.cos(2 * p) - init_op[(0, 2)] * np.sin(2 * p),
        (0, 2): init_op[(0, 2)] * np.cos(2 * p) + init_op[(0, 1)] * np.sin(2 * p),
    }

    atol_combinations = [
        (None, None),
        (None, 1e-6),
        (None, 1e-10),
        (1e-4, None),
        (1e-4, 1e-6),
        (1e-4, 1e-10),
        (1e-10, None),
        (1e-10, 1e-10),
    ]

    for upper_atol, lower_atol in atol_combinations:
        mp = _create_mp(
            op, sequence.initial_state, serial_comm, upper_atol, lower_atol, cutoff
        )
        mp.build_graph(circuit)

        assert mp.graph_size()[1] == 1
        assert mp.graph_size()[0] == 0
        assert mp.size() == 2

        test_op = mp.evolved_operator(sequence.parameters)
        _check_dicts(test_op, final_operator)
