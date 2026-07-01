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

from monoprop.monomial_data import Monomial, MonomialOperator, MonomialSequence


def test_monomial_circuit_fields():
    majoranas = [np.array([0, 1]), np.array([2, 3])]
    gen_coeffs = [0.5j, -0.5j]
    param_inds = [np.array([0]), np.array([0])]
    identical_params = np.array([1, 2])

    mc = MonomialSequence(
        majoranas=majoranas,
        parameters=[1.0, 2.0],
        gen_coeffs=gen_coeffs,
        param_inds=param_inds,
        identical_params=identical_params,
    )

    for actual, exp in zip(mc.majoranas, majoranas):
        np.testing.assert_array_equal(actual, exp)
    assert mc.gen_coeffs == gen_coeffs
    for actual, exp in zip(mc.param_inds, param_inds):
        np.testing.assert_array_equal(actual, exp)
    np.testing.assert_array_equal(mc.identical_params, identical_params)


@pytest.mark.parametrize(
    ("terms", "num_modes"),
    [
        ([Monomial(np.array([0, 1]), 1.0)], 4),
        ([Monomial(np.array([i, i + 1]), 1.0) for i in range(0, 16, 2)], 8),
        ([Monomial(np.array([i, i + 1]), float(i)) for i in range(0, 18, 2)], 9),
    ],
)
def test_str(terms, num_modes):
    op = MonomialOperator(terms=terms, num_modes=num_modes)
    r = str(op)
    num_terms = len(op.terms)
    assert r.startswith("MonomialOperator(")
    assert f"{num_terms} terms" in r
