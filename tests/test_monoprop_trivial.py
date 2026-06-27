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

from monoprop import MonomialPropagator
from monoprop.fermi_data import FermiCircuit, MajoranaOperator


@pytest.mark.parametrize(
    ("initial_op", "cutoff", "schrodinger_cutoff", "expected"),
    [
        (MajoranaOperator([(0, 1, 2, 4)], [1], 8), 16, None, {(0, 1, 2, 4): 1}),
        (MajoranaOperator([()], [1], 8), 16, None, {(): 1}),
        (
            MajoranaOperator([], [], 1),
            2,
            2,
            {(): 1.0, (0, 1): -1.0j},
        ),  # Schrodinger picture
    ],
)
def test_trivial_evolved_operator_dict(
    initial_op, cutoff, schrodinger_cutoff, expected, serial_comm
):
    """Test trivial evolved operator dict for various initial conditions."""
    kwargs = {"schrodinger_cutoff": schrodinger_cutoff} if schrodinger_cutoff else {}
    quantum_circuit = FermiCircuit(initial_state=[], gates=[])
    mp = MonomialPropagator(
        initial_op, quantum_circuit, cutoff, comm=serial_comm, **kwargs
    )
    result = mp.evolved_operator_dict()
    assert result == expected


def test_trivial_evolved_operator(serial_comm):
    initial_op = MajoranaOperator([(0, 1, 2, 4)], [1], 8)
    quantum_circuit = FermiCircuit(initial_state=[], gates=[])
    mp = MonomialPropagator(initial_op, quantum_circuit, 16, comm=serial_comm)
    op = mp.contract_partially([], [], [], inplace=False)
    assert op == np.array([-1.0])


@pytest.mark.parametrize(
    (
        "init_op",
        "new_op",
        "cutoff",
        "schrodinger_cutoff",
        "expected_new",
        "expval_check",
    ),
    [
        # Regular picture: checks contract_partially (rank-local) → serial_comm
        (
            MajoranaOperator([(0, 1, 2, 4)], [1], 8),
            {(0, 1, 2, 4): 2.0 + 0j},
            16,
            None,
            np.array([-2.0]),
            None,
        ),
        # Schrodinger picture: checks expectation value (allreduced) → but kept here for simplicity
        (
            MajoranaOperator([(0, 3)], [1.0j], 4),
            {(0, 1): 2.0j},
            8,
            8,
            None,
            (0.0, -2.0),
        ),
    ],
)
def test_update_coeffs(
    init_op,
    new_op,
    cutoff,
    schrodinger_cutoff,
    expected_new,
    expval_check,
    serial_comm,
):
    """Test updating coefficients in both regular and Schrodinger pictures."""
    kwargs = {"schrodinger_cutoff": schrodinger_cutoff} if schrodinger_cutoff else {}
    quantum_circuit = FermiCircuit(initial_state=[], gates=[])
    mp = MonomialPropagator(
        init_op, quantum_circuit, cutoff, comm=serial_comm, **kwargs
    )

    if expval_check:
        expval_init = mp.expectation_value()
        assert expval_init == expval_check[0]
        mp.update_coeffs(new_op)
        expval_new = mp.expectation_value()
        assert np.isclose(expval_new, expval_check[1])
    else:
        op_init = mp.contract_partially([], [], [], inplace=False)
        mp.update_coeffs(new_op)
        op_new = mp.contract_partially([], [], [], inplace=False)
        assert np.array_equal(op_new, expected_new)
        assert not np.array_equal(op_init, op_new)
