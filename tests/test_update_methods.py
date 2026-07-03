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

"""Tests for update methods in MajoranaPropagator."""

import pytest
from pytest_cases import parametrize_with_cases

from monoprop import (
    Circuit,
    MajoranaGate,
    MajoranaPropagator,
    Term,
    jordan_wigner_basis_change,
)
from monoprop.majorana_data import MajoranaOperator, MajoranaSequence
from tests.cases import CasesFermionicProblem


class TestUpdateMethods:
    """Test class for update methods in MajoranaPropagator."""

    @pytest.fixture
    def mp(self, serial_comm):
        return MajoranaPropagator(
            initial_operator=MajoranaOperator.from_dict(
                terms_dict={(0, 1): 1.0j}, num_modes=4
            ),
            initial_state=[],
            cutoff=4,
            lower_atol=1e-8,
            upper_atol=1e-4,
            comm=serial_comm,
        )

    @pytest.mark.parametrize("valid_value", [1e-10, None])
    def test_update_lower_atol_valid(self, mp, valid_value):
        mp.lower_atol = valid_value
        assert mp.lower_atol == valid_value

    def test_update_lower_atol_invalid(self, mp):
        mp.lower_atol = 1e-10
        with pytest.raises(RuntimeError, match=r"New lower_atol \(0\.001\)"):
            mp.lower_atol = 1e-3

    def test_update_lower_atol_edge_case_equal_to_upper(self, mp):
        mp.lower_atol = 1e-4
        assert mp.lower_atol == 1e-4

    def test_update_lower_atol_edge_case_very_small(self, mp):
        mp.lower_atol = None
        mp.upper_atol = 1e-15
        mp.lower_atol = 1e-16
        assert mp.lower_atol == 1e-16

    @pytest.mark.parametrize("invalid_value", ["invalid", [1e-8]])
    def test_update_lower_atol_type_validation(self, mp, invalid_value):
        with pytest.raises((RuntimeError, TypeError)):
            mp.lower_atol = invalid_value

    @pytest.mark.parametrize("valid_value", [1e-3, None])
    def test_update_upper_atol_valid(self, mp, valid_value):
        mp.upper_atol = valid_value
        assert mp.upper_atol == valid_value

    def test_update_upper_atol_invalid(self, mp):
        mp.upper_atol = 1e-3
        with pytest.raises(
            (ValueError, RuntimeError), match="must be greater than or equal to"
        ):
            mp.upper_atol = 1e-9

    @pytest.mark.parametrize("edge_value", [1e-8, 1.0])
    def test_update_upper_atol_edge_cases(self, mp, edge_value):
        mp.upper_atol = edge_value
        assert mp.upper_atol == edge_value

    @pytest.mark.parametrize("invalid_value", ["invalid", [1e-4]])
    def test_update_upper_atol_type_validation(self, mp, invalid_value):
        with pytest.raises(TypeError):
            mp.upper_atol = invalid_value

    def test_atol_parameter_consistency(self, mp):
        mp.lower_atol = None
        mp.upper_atol = None
        assert mp.lower_atol is None
        assert mp.upper_atol is None

        mp.upper_atol = 1e-5
        mp.lower_atol = 1e-6
        assert mp.lower_atol == 1e-6
        assert mp.upper_atol == 1e-5

        mp.upper_atol = 1e-3
        mp.lower_atol = 1e-4
        assert mp.lower_atol == 1e-4
        assert mp.upper_atol == 1e-3

    def test_update_cutoff_valid(self, mp):
        mp.cutoff = 6
        gate = MajoranaGate((Term((0, 1, 2, 3, 4, 5), 1.0),))
        mp.build_graph(Circuit((gate,)))
        assert mp.size() > 0

    def test_update_cutoff_invalid(self, mp):
        with pytest.raises((ValueError, TypeError)):
            mp.cutoff = -1

    @pytest.mark.parametrize("edge_value", [0, 1000])
    def test_update_cutoff_edge_cases(self, mp, edge_value):
        mp.cutoff = edge_value
        assert mp.cutoff == edge_value

    @pytest.mark.parametrize("invalid_value", [3.14, "invalid"])
    def test_update_cutoff_value_type_validation(self, mp, invalid_value):
        with pytest.raises(TypeError):
            mp.cutoff = invalid_value

    @pytest.mark.parametrize("valid_type", ["support", "length"])
    def test_update_cutoff_type_valid(self, mp, valid_type):
        mp.cutoff_type = valid_type
        assert mp.cutoff_type == valid_type

    def test_update_cutoff_type_invalid(self, mp):
        with pytest.raises((ValueError, RuntimeError)):
            mp.cutoff_type = "invalid_type"

    @pytest.mark.parametrize("invalid_value", [None, 123])
    def test_update_cutoff_type_enum_validation(self, mp, invalid_value):
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            mp.cutoff_type = invalid_value

    @pytest.mark.parametrize("use_none", [False, True])
    def test_update_basis_change_valid(self, mp, use_none):
        if use_none:
            mp.basis_change = None
            assert mp.basis_change is None
        else:
            jw_basis = jordan_wigner_basis_change(mp.num_modes)
            mp.basis_change = jw_basis
            assert mp.basis_change == jw_basis

    def test_update_basis_change_invalid(self, mp):
        with pytest.raises(ValueError, match="must have length 8"):
            mp.basis_change = [[0], [1]]

    @pytest.mark.parametrize(
        "basis_generator",
        [lambda: [[i] for i in range(8)], lambda: [[] for _ in range(8)]],
    )
    def test_update_basis_change_edge_cases_valid(self, mp, basis_generator):
        basis = basis_generator()
        mp.basis_change = basis
        assert mp.basis_change == basis

    @pytest.mark.parametrize(
        ("invalid_basis", "expected_match"),
        [
            ([[0, 1, 2, 3, 4, 5, 6, 7]], "must have length 8, but got 1"),
            ("invalid", "must have length 8, but got 7"),
        ],
    )
    def test_update_basis_change_edge_cases_invalid(
        self, mp, invalid_basis, expected_match
    ):
        with pytest.raises(ValueError, match=expected_match):
            mp.basis_change = invalid_basis

    def test_integration(self, serial_comm):
        sequence = MajoranaSequence(
            initial_state=[],
            majoranas=[(0, 2), (1, 3)],
            gen_coeffs=[0.0, 0.0],
            param_inds=[0, 1],
            parameters=[1.0, 1.0],
        )
        mp = MajoranaPropagator(
            initial_operator=MajoranaOperator.from_dict(
                terms_dict={(0, 1): 1.0j}, num_modes=4
            ),
            initial_state=sequence.initial_state,
            cutoff=4,
            comm=serial_comm,
        )
        updates = {
            "lower_atol": 1e-10,
            "upper_atol": 1e-6,
            "cutoff": 6,
            "cutoff_type": "support",
            "basis_change": jordan_wigner_basis_change(4),
        }
        for attr, value in updates.items():
            setattr(mp, attr, value)

        circuit = sequence.to_circuit()
        mp.propagate(circuit)
        expval = mp.expectation_value()
        assert isinstance(expval, (int, float))

    @pytest.mark.parametrize("order", ["lower_first", "upper_first"])
    def test_parameter_validation_ordering(self, mp, order):
        mp.lower_atol = None
        mp.upper_atol = None
        if order == "lower_first":
            mp.lower_atol = 1e-10
            mp.upper_atol = 1e-6
        else:
            mp.upper_atol = 1e-6
            mp.lower_atol = 1e-10
        assert mp.lower_atol == 1e-10
        assert mp.upper_atol == 1e-6

    def test_reset_to_defaults(self, mp):
        for attr, value in {
            "lower_atol": 1e-12,
            "upper_atol": 1e-2,
            "cutoff": 10,
            "cutoff_type": "length",
            "basis_change": jordan_wigner_basis_change(4),
        }.items():
            setattr(mp, attr, value)
        for attr, value in {
            "lower_atol": None,
            "upper_atol": None,
            "basis_change": None,
            "cutoff_type": "support",
            "cutoff": 4,
        }.items():
            setattr(mp, attr, value)
        assert mp.lower_atol is None
        assert mp.upper_atol is None
        assert mp.basis_change is None
        assert mp.cutoff_type == "support"
        assert mp.cutoff == 4


@parametrize_with_cases("problem", cases=CasesFermionicProblem, has_tag="molecule")
@pytest.mark.parametrize(
    "test_type", ["cutoff", "lower_atol", "upper_atol", "basis_change"]
)
def test_evolutions_after_updates(problem, test_type, serial_comm):
    """Test that evolutions work correctly after parameter updates."""

    circuit = problem.monomial_circuit.to_circuit()

    mp = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=6,
        comm=serial_comm,
    )
    mp.propagate(circuit)
    mp_size = mp.size()

    mp_tes = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=6,
        comm=serial_comm,
    )
    if test_type == "lower_atol":
        mp_tes.lower_atol = 1
    elif test_type == "upper_atol":
        mp_tes.upper_atol = 0.001
    elif test_type == "basis_change":
        mp_tes.basis_change = jordan_wigner_basis_change(problem.n_modes)
    else:
        mp_tes.cutoff = 4

    mp_tes.propagate(circuit)
    assert mp_tes.size() != mp_size


@pytest.mark.parametrize("num_modes", [2, 4, 8, 16])
def test_update_methods_different_modes(num_modes, serial_comm):
    mp = MajoranaPropagator(
        initial_operator=MajoranaOperator.from_dict(
            terms_dict={(0, 1): 1.0j}, num_modes=num_modes
        ),
        initial_state=[],
        cutoff=4,
        comm=serial_comm,
    )
    jw_basis = jordan_wigner_basis_change(num_modes)
    mp.basis_change = jw_basis
    assert mp.basis_change == jw_basis


@pytest.mark.parametrize("num_modes", [4, 8, 16])
def test_basis_change_invalid_dimensions(num_modes, serial_comm):
    mp = MajoranaPropagator(
        initial_operator=MajoranaOperator.from_dict(
            terms_dict={(0, 1): 1.0j}, num_modes=num_modes
        ),
        initial_state=[],
        cutoff=4,
        comm=serial_comm,
    )
    wrong_basis = jordan_wigner_basis_change(num_modes - 2)
    with pytest.raises(ValueError, match=f"must have length {2 * num_modes}"):
        mp.basis_change = wrong_basis
