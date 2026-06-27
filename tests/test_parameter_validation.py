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

"""Test file for parameter validation and error handling in MonomialPropagator."""

from __future__ import annotations

import numpy as np
import pytest

from monoprop import MonomialPropagator
from monoprop.monomial_data import MonomialCircuit, MonomialOperator
from monoprop.monomial_propagator import normalize_parameters


def _validate_all_none_or_all_provided(parameters, parameter_mapping, gen_coeffs):
    all_none = all(x is None for x in (parameters, parameter_mapping, gen_coeffs))
    all_not_none = all(
        x is not None for x in (parameters, parameter_mapping, gen_coeffs)
    )
    if not (all_none or all_not_none):
        raise ValueError(
            "Either all of parameters, parameter_mapping, and gen_coeffs must be None, "
            "or all must be provided."
        )
    return all_none, all_not_none


def _validate_coefficient_lengths(parameter_mapping, gen_coeffs):
    if len(parameter_mapping) != len(gen_coeffs):
        raise ValueError(
            "The length of parameter_mapping and gen_coeffs must be the same."
        )


def _validate_parameters_length(parameters, parameter_mapping):
    if len(parameter_mapping) != 0 and len(parameters) != np.max(parameter_mapping) + 1:
        raise ValueError(
            f"The length of parameters ({len(parameters)}) must be the same as "
            f"max(parameter_mapping)+1 ({np.max(parameter_mapping) + 1})."
        )


def _get_num_params(parameter_mapping):
    return 0 if len(parameter_mapping) == 0 else np.max(parameter_mapping) + 1


class TestValidators:
    """Test the parameter validation helpers."""

    def testnormalize_parameters_all_none(self):
        params, param_mapping, gen_coeffs = normalize_parameters(None, None, None)
        assert params == []
        assert param_mapping == []
        assert gen_coeffs == []

    def testnormalize_parameters_mixed(self):
        normalized_params, normalized_mapping, normalized_coeffs = normalize_parameters(
            [1.0, 2.0], None, [1.0, 2.0]
        )
        assert normalized_params == [1.0, 2.0]
        assert normalized_mapping == []
        assert normalized_coeffs == [1.0, 2.0]

    @pytest.mark.parametrize(
        (
            "params",
            "param_map",
            "gen_coeffs",
            "expected_all_none",
            "expected_all_not_none",
        ),
        [
            (None, None, None, True, False),
            ([1.0], [0], [1.0], False, True),
        ],
    )
    def test_validate_all_none_or_all_provided_success(
        self, params, param_map, gen_coeffs, expected_all_none, expected_all_not_none
    ):
        all_none, all_not_none = _validate_all_none_or_all_provided(
            params, param_map, gen_coeffs
        )
        assert all_none is expected_all_none
        assert all_not_none is expected_all_not_none

    @pytest.mark.parametrize(
        ("params", "param_map", "gen_coeffs"),
        [([1.0], None, [1.0]), (None, [0], [1.0]), ([1.0], [0], None)],
    )
    def test_validate_all_none_or_all_provided_failure(
        self, params, param_map, gen_coeffs
    ):
        with pytest.raises(ValueError, match="Either all of"):
            _validate_all_none_or_all_provided(params, param_map, gen_coeffs)

    @pytest.mark.parametrize(
        ("param_mapping", "gen_coeffs"),
        [([0, 1], [1.0, 2.0]), ([], [])],
    )
    def test_validate_coefficient_lengths_success(self, param_mapping, gen_coeffs):
        _validate_coefficient_lengths(param_mapping, gen_coeffs)

    @pytest.mark.parametrize(
        ("param_mapping", "gen_coeffs"),
        [([0, 1], [1.0]), ([0], [1.0, 2.0])],
    )
    def test_validate_coefficient_lengths_failure(self, param_mapping, gen_coeffs):
        with pytest.raises(ValueError, match="must be the same"):
            _validate_coefficient_lengths(param_mapping, gen_coeffs)

    @pytest.mark.parametrize(
        ("params", "param_mapping"),
        [([1.0, 2.0], [0, 1]), ([5.0], [0, 0]), ([], [])],
    )
    def test_validate_parameters_length_success(self, params, param_mapping):
        _validate_parameters_length(params, param_mapping)

    def test_validate_parameters_length_failure(self):
        with pytest.raises(
            ValueError, match="must be the same as max\\(parameter_mapping\\)\\+1"
        ):
            _validate_parameters_length([1.0], [0, 2])
        with pytest.raises(
            ValueError, match="must be the same as max\\(parameter_mapping\\)\\+1"
        ):
            _validate_parameters_length([1.0, 2.0, 3.0], [0, 1])

    @pytest.mark.parametrize(
        ("param_mapping", "expected"),
        [([], 0), ([0], 1), ([0, 1, 2], 3), ([1, 0, 2], 3), ([5], 6)],
    )
    def test_get_num_params(self, param_mapping, expected):
        assert _get_num_params(param_mapping) == expected


class TestMonomialPropagatorValidation:
    """Test validation and error handling in MonomialPropagator."""

    @pytest.fixture
    def simple_mp(self, serial_comm):
        initial_operator = MonomialOperator.from_dict(
            terms_dict={(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2
        )
        monomial_circuit = MonomialCircuit(
            initial_state=[0, 1],
            majoranas=[],
            parameters=[],
            gen_coeffs=[],
            param_inds=[],
        )
        return MonomialPropagator(
            initial_operator, monomial_circuit, 4, comm=serial_comm
        )

    def test_init_invalid_basis_change_length(self, serial_comm):
        with pytest.raises(
            ValueError, match="Basis change must have length 4, but got 3"
        ):
            MonomialPropagator(
                MonomialOperator.from_dict(terms_dict={(0, 1): 1.0j}, num_modes=2),
                MonomialCircuit(
                    initial_state=[0, 1],
                    majoranas=[],
                    parameters=[],
                    gen_coeffs=[],
                    param_inds=[],
                ),
                4,
                comm=serial_comm,
                basis_change=[[0], [1], [2]],
            )

    def test_init_invalid_tolerances(self, serial_comm):
        with pytest.raises(
            RuntimeError,
            match=r"upper_atol \(0\.1\) must be greater than or equal to lower_atol \(0\.5\)\.",
        ):
            MonomialPropagator(
                MonomialOperator.from_dict(terms_dict={(0, 1): 1.0j}, num_modes=2),
                MonomialCircuit(
                    initial_state=[0, 1],
                    majoranas=[],
                    parameters=[],
                    gen_coeffs=[],
                    param_inds=[],
                ),
                4,
                upper_atol=0.1,
                lower_atol=0.5,
                comm=serial_comm,
            )

    def test_propagate_invalid_partial_parameters(self, simple_mp):
        majoranas = [(0,), (1,)]
        error_msg = "Either all of parameters, parameter_mapping, and gen_coeffs must be None, or all must be provided"
        for params, param_mapping, gen_coeffs in [
            ([1.0], None, None),
            (None, [0], None),
            (None, None, [1.0]),
        ]:
            with pytest.raises(RuntimeError, match=error_msg):
                simple_mp.propagate(
                    majoranas,
                    parameters=params,
                    parameter_mapping=param_mapping,
                    gen_coeffs=gen_coeffs,
                )

    def test_propagate_no_params_graph_mode(self, simple_mp):
        simple_mp.propagate([(0,), (1,)])
        assert simple_mp.graph_layers == 2

    def test_propagate_length_mismatch(self, simple_mp):
        majoranas = [(0,), (1,)]
        with pytest.raises(RuntimeError, match="must be the same"):
            simple_mp.propagate(
                majoranas,
                parameters=[1.0, 2.0],
                parameter_mapping=[0, 1],
                gen_coeffs=[1.0],
            )
        with pytest.raises(RuntimeError, match="The length of parameters"):
            simple_mp.propagate(
                majoranas,
                parameters=[1.0],
                parameter_mapping=[0, 1],
                gen_coeffs=[1.0, 2.0],
            )

    def test_validate_propagation_params(self, simple_mp):
        simple_mp.propagate([(0,), (1,)])
        with pytest.raises(
            RuntimeError,
            match="must be the same as the number of propagated Majoranas 2",
        ):
            simple_mp.expectation_value_functional(
                parameter_mapping=[0], gen_coeffs=[1.0]
            )

    def test_validate_propagation_contraction(self, simple_mp):
        simple_mp.propagate([(0,), (1,)])
        with pytest.raises(RuntimeError, match="must be less than or equal to"):
            simple_mp.contract_partially(
                parameters=[1.0, 2.0, 3.0],
                parameter_mapping=[0, 1, 2],
                gen_coeffs=[1.0, 2.0, 3.0],
            )

    def test_functional_call_validation(self, simple_mp):
        simple_mp.propagate([(0,), (1,)])
        expval_func = simple_mp.expectation_value_functional(
            parameter_mapping=[0, 1],
            gen_coeffs=[1.0, 2.0],
            pare_threshold=None,
        )
        simple_mp.contract_partially(
            parameters=[1.0, 2.0],
            parameter_mapping=[0, 1],
            gen_coeffs=[1.0, 2.0],
            inplace=True,
        )
        with pytest.raises(RuntimeError, match="MP object has been modified"):
            expval_func([1.0, 2.0])

    def test_functional_parameter_length_validation(self, simple_mp):
        simple_mp.propagate([(0,), (1,)])
        expval_func = simple_mp.expectation_value_functional(
            parameter_mapping=[0, 1], gen_coeffs=[1.0, 2.0]
        )
        for params, length in [([1.0, 2.0, 3.0], 3), ([1.0], 1)]:
            with pytest.raises(RuntimeError, match=f"Parameter length {length}"):
                expval_func(params)

    def test_evolved_operator_schrodinger_error(self, serial_comm):
        initial_operator = MonomialOperator.from_dict(
            terms_dict={(0, 1): 1.0j, (1, 0): 1.0j}, num_modes=2
        )
        mp = MonomialPropagator(
            initial_operator,
            MonomialCircuit(
                initial_state=[0, 1],
                majoranas=[],
                parameters=[1.0],
                gen_coeffs=[1.0],
                param_inds=[0],
            ),
            4,
            schrodinger_cutoff=2,
            comm=serial_comm,
        )
        with pytest.raises(
            ValueError, match="Cannot call evolved_operator in Schrodinger picture"
        ):
            mp.evolved_operator(
                evolve_with_coeffs=True,
            )

    def test_gradient_method_validation(self, simple_mp):
        with pytest.raises(RuntimeError, match="must be the same"):
            simple_mp.gradient(
                parameters=[1.0, 2.0], parameter_mapping=[0, 1], gen_coeffs=[1.0]
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_parameter_arrays(self):
        _validate_coefficient_lengths([], [])
        _validate_parameters_length([], [])
        assert _get_num_params([]) == 0

    @pytest.mark.parametrize("array_type", [list, np.array])
    def test_numpy_array_inputs(self, array_type):
        params = array_type([1.0, 2.0])
        param_mapping = array_type([0, 1])
        gen_coeffs = array_type([1.0, 2.0])
        _validate_coefficient_lengths(param_mapping, gen_coeffs)
        _validate_parameters_length(params, param_mapping)
        assert _get_num_params(param_mapping) == 2

    def test_single_element_arrays(self):
        _validate_coefficient_lengths([0], [3.0])
        _validate_parameters_length([5.0], [0])
        assert _get_num_params([0]) == 1
