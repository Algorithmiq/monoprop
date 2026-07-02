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

from monoprop.majorana_data import MajoranaOperator, MajoranaSequence


def test_majorana_sequence_fields():
    majoranas = [np.array([0, 1]), np.array([2, 3])]
    gen_coeffs = [0.5j, -0.5j]
    param_inds = [np.array([0]), np.array([0])]

    mc = MajoranaSequence(
        initial_state=[0, 1],
        majoranas=majoranas,
        parameters=[1.0, 2.0],
        gen_coeffs=gen_coeffs,
        param_inds=param_inds,
    )

    assert mc.initial_state == [0, 1]
    for actual, exp in zip(mc.majoranas, majoranas):
        np.testing.assert_array_equal(actual, exp)
    assert mc.gen_coeffs == gen_coeffs
    for actual, exp in zip(mc.param_inds, param_inds):
        np.testing.assert_array_equal(actual, exp)


def test_majorana_operator_normalizes_terms():
    """Indices are sorted, duplicate monomials summed, tiny terms dropped."""
    op = MajoranaOperator(
        majoranas=[(1, 0), (0, 1), (2, 3)],
        coefficients=[1.0, 2.0, 1e-15],
        num_modes=4,
    )
    assert op.terms == {(0, 1): 3.0}  # (1,0) and (0,1) merge; (2,3) below threshold
    assert len(op) == 1
    assert op.num_modes == 4
    assert not op.is_identity()


def test_majorana_operator_from_dict_round_trips():
    op = MajoranaOperator.from_dict({(0, 1): 1.0j, (2, 3): -0.5}, num_modes=4)
    assert op.terms == {(0, 1): 1.0j, (2, 3): -0.5}
    assert op.get_majorana_operator() is op


def test_majorana_operator_is_identity_when_empty():
    assert MajoranaOperator([], [], num_modes=4).is_identity()


@pytest.mark.parametrize(
    ("majoranas", "coefficients", "num_modes"),
    [
        ([(0, 1)], [1.0], 4),
        ([(i, i + 1) for i in range(0, 16, 2)], [1.0] * 8, 8),
        ([(i, i + 1) for i in range(0, 18, 2)], [float(i) for i in range(0, 18, 2)], 9),
    ],
)
def test_str(majoranas, coefficients, num_modes):
    op = MajoranaOperator(majoranas, coefficients, num_modes)
    r = str(op)
    assert r.startswith("MajoranaOperator(")
    assert f"{len(op.terms)} terms" in r
