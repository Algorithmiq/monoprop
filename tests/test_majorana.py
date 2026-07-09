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

import pytest

from monoprop import Circuit
from monoprop.majorana import Majorana, MajoranaOperator


class TestMajorana:
    """The Majorana monomial value object: sorting, validation, equality, hashing."""

    def test_indices_are_sorted(self):
        assert Majorana(5, 4, 1).indices == (1, 4, 5)

    def test_empty_is_identity_monomial(self):
        assert Majorana().indices == ()

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Majorana(0, -1)

    def test_repeated_index_raises(self):
        with pytest.raises(ValueError, match="distinct"):
            Majorana(2, 2)

    def test_equal_terms_compare_and_hash_alike(self):
        assert Majorana(4, 5) == Majorana(5, 4)
        assert hash(Majorana(4, 5)) == hash(Majorana(5, 4))
        assert {Majorana(4, 5), Majorana(5, 4)} == {Majorana(4, 5)}

    def test_eq_with_non_majorana_is_false(self):
        assert (Majorana(0, 1) == (0, 1)) is False
        assert Majorana(0, 1) != (0, 1)

    def test_repr(self):
        assert repr(Majorana(4, 5)) == "Majorana(4, 5)"


def test_majorana_operator_validates_raw_tuple_keys():
    """A raw index-tuple key is routed through Majorana, so it gets the same validation."""
    with pytest.raises(ValueError, match="distinct"):
        MajoranaOperator({(0, 0): 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="non-negative"):
        MajoranaOperator({(-1, 0): 1.0}, num_modes=2)


def test_from_dense_arrays_groups_by_param_ind():
    """Consecutive monomials sharing a param_ind group into one gate."""
    circuit = Circuit.from_dense_arrays(
        majoranas=[(0, 1), (2, 3), (0, 3)],
        gen_coeffs=[0.5, -0.5, 1.0],
        param_inds=[0, 0, 1],
        parameters=[1.0, 2.0],
        initial_state=[0, 1],
    )

    assert circuit.initial_state == (0, 1)
    assert circuit.parameters == (1.0, 2.0)
    assert circuit.resolved_mapping == (0, 1)
    assert len(circuit) == 2  # {param 0: two monomials}, {param 1: one monomial}
    assert len(circuit.gates[0].generator.terms) == 2
    assert len(circuit.gates[1].generator.terms) == 1


def test_majorana_operator_normalizes_terms():
    """Indices are sorted and duplicate monomials summed."""
    op = MajoranaOperator(
        {(1, 0): 1.0, (0, 1): 2.0, (2, 3): 1e-15},
        num_modes=4,
    )
    assert op.terms == {(0, 1): 3.0, (2, 3): 1e-15}  # (1,0) and (0,1) merge
    assert len(op) == 2
    assert op.num_modes == 4
    assert not op.is_identity()


def test_majorana_operator_from_dict_round_trips():
    op = MajoranaOperator({(0, 1): 1.0j, (2, 3): -0.5}, num_modes=4)
    assert op.terms == {(0, 1): 1.0j, (2, 3): -0.5}
    assert op.get_majorana_operator() is op


def test_majorana_operator_is_identity_when_empty():
    assert MajoranaOperator({}, num_modes=4).is_identity()


@pytest.mark.parametrize(
    ("majoranas", "coefficients", "num_modes"),
    [
        ([(0, 1)], [1.0], 4),
        ([(i, i + 1) for i in range(0, 16, 2)], [1.0] * 8, 8),
        ([(i, i + 1) for i in range(0, 18, 2)], [float(i) for i in range(0, 18, 2)], 9),
    ],
)
def test_str(majoranas, coefficients, num_modes):
    op = MajoranaOperator(dict(zip(majoranas, coefficients)), num_modes)
    r = str(op)
    assert r.startswith("MajoranaOperator(")
    assert f"{len(op.terms)} terms" in r
