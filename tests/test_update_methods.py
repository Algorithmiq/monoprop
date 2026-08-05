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

import pytest
from pytest_cases import parametrize_with_cases

from monoprop import (
    Circuit,
    ExpGate,
    MajoranaPropagator,
)
from monoprop.majorana import MajoranaOperator
from tests.cases import CasesFermionicProblem


class TestUpdateMethods:
    @pytest.fixture
    def mp(self, serial_comm):
        return MajoranaPropagator(
            initial_operator=MajoranaOperator({(0, 1): 1.0j}, num_modes=4),
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
        # A weight-6 monomial takes an imaginary Hermitian coefficient (like weight-2).
        gate = ExpGate(MajoranaOperator({(0, 1, 2, 3, 4, 5): 1.0j}, num_modes=4))
        mp.build_graph(Circuit((gate,), initial_state=(), system_size=4))
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

    def test_integration(self, serial_comm):
        sequence = Circuit.from_dense_arrays(
            initial_state=[],
            majoranas=[(0, 2), (1, 3)],
            gen_coeffs=[0.0, 0.0],
            param_inds=[0, 1],
            system_size=4,
            parameters=[1.0, 1.0],
        )
        mp = MajoranaPropagator(
            initial_operator=MajoranaOperator({(0, 1): 1.0j}, num_modes=4),
            initial_state=sequence.initial_state,
            cutoff=4,
            comm=serial_comm,
        )
        updates = {
            "lower_atol": 1e-10,
            "upper_atol": 1e-6,
            "cutoff": 6,
            "cutoff_type": "support",
        }
        for attr, value in updates.items():
            setattr(mp, attr, value)

        circuit = sequence
        mp.propagate(circuit)
        expval = mp.expval()
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
        }.items():
            setattr(mp, attr, value)
        for attr, value in {
            "lower_atol": None,
            "upper_atol": None,
            "cutoff_type": "support",
            "cutoff": 4,
        }.items():
            setattr(mp, attr, value)
        assert mp.lower_atol is None
        assert mp.upper_atol is None
        assert mp.cutoff_type == "support"
        assert mp.cutoff == 4


@parametrize_with_cases("problem", cases=CasesFermionicProblem, has_tag="molecule")
@pytest.mark.parametrize("test_type", ["cutoff", "lower_atol", "upper_atol"])
def test_evolutions_after_updates(problem, test_type, serial_comm):
    circuit = problem.monomial_circuit.to_circuit()

    # upper_atol only rescues over-cutoff partners, of which cutoff 6 produces none here.
    cutoff = 2 if test_type == "upper_atol" else 6

    mp = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=cutoff,
        comm=serial_comm,
    )
    mp.propagate(circuit)
    mp_size = mp.size()

    mp_tes = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=cutoff,
        comm=serial_comm,
    )
    if test_type == "lower_atol":
        mp_tes.lower_atol = 1
    elif test_type == "upper_atol":
        mp_tes.upper_atol = 0.001
    else:
        mp_tes.cutoff = 4

    mp_tes.propagate(circuit)
    assert mp_tes.size() != mp_size
