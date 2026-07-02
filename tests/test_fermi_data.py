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

from monoprop import to_engine_arrays
from monoprop.fermi_data import (
    FermiCircuit,
    FermiEvGate,
    FermiOperator,
    FermiString,
    MajoranaOperator,
)


class TestFermiString:
    def test_valid_creation(self):
        f = FermiString([(0, "+"), (1, "-")])
        assert f.expression == ((0, "+"), (1, "-"))

    def test_empty_expression(self):
        f = FermiString([])
        assert f.expression == ()

    @pytest.mark.parametrize(
        "expression",
        [[(0, "x")], [(0, "+"), (1, "a")], [(0, "x"), (1, "-")]],
        ids=[
            "invalid operator",
            "mixed valid and invalid operators",
            "mixed valid and invalid operators",
        ],
    )
    def test_invalid_operator_raises(self, expression):
        with pytest.raises(ValueError, match="must be '\\+' or '-'"):
            FermiString(expression)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match=r"Invalid index -1: must be non-negative"):
            FermiString([(-1, "+")])

    def test_repr(self):
        f = FermiString([(0, "+"), (2, "-")])
        assert repr(f) == "FermiString(a_0^+ a_2^-)"

    def test_repr_empty(self):
        f = FermiString([])
        assert repr(f) == "FermiString()"


class TestFermiOperator:
    def test_valid_creation(self):
        terms = [FermiString([(0, "+")]), FermiString([(1, "-")])]
        op = FermiOperator(terms, [1.0, -1.0])
        assert op.terms == terms
        assert op.coefficients == [1.0, -1.0]

    def test_num_modes_max_index(self):
        terms = [FermiString([(0, "+"), (3, "-")]), FermiString([(5, "+")])]
        op = FermiOperator(terms, [1.0, 1.0])
        assert op.num_modes == 6

    def test_num_modes_empty_terms(self):
        op = FermiOperator([], [], 10)
        assert op.num_modes == 10

    def test_num_modes_single_term(self):
        op = FermiOperator([FermiString([(2, "+"), (7, "-")])], [1.0])
        assert op.num_modes == 8

    def test_terms_is_copy(self):
        terms = [FermiString([(0, "+")])]
        op = FermiOperator(terms, [1.0])
        terms.append(FermiString([(1, "-")]))
        assert len(op.terms) == 1

    def test_coefficients_is_copy(self):
        coeffs = [1.0]
        op = FermiOperator([FermiString([(0, "+")])], coeffs)
        coeffs.append(2.0)
        assert len(op.coefficients) == 1

    def test_str(self):
        terms = [FermiString([(0, "+")])]
        op = FermiOperator(terms, [2.0])
        assert str(op) == "FermiOperator(1 terms, 1 modes: 2.0*FermiString(a_0^+))"

    def test_num_modes_explicit_override(self):
        terms = [FermiString([(0, "+"), (5, "-")])]
        op = FermiOperator(terms, [1.0], num_modes=12)
        assert op.num_modes == 12

    def test_empty_terms_without_num_modes_raises(self):
        with pytest.raises(
            ValueError,
            match=r"max\(\) (arg is an empty sequence|iterable argument is empty)",
        ):
            FermiOperator([], [])


class TestMajoranaOperator:
    def test_valid_creation_and_repr(self):
        op = MajoranaOperator(
            majoranas=[(0, 1), (2, 3)], coefficients=[1.0, -0.5j], num_modes=2
        )

        assert op.terms == {(0, 1): 1.0, (2, 3): -0.5j}
        assert op.num_modes == 2
        assert (
            str(op) == "MajoranaOperator(2 terms, 2 modes: (1+0j)*[0, 1], -0.5j*[2, 3])"
        )

    def test_get_majorana_operator(self):
        op = MajoranaOperator(majoranas=[(0, 1)], coefficients=[0.25j], num_modes=1)
        mon_op = op.get_majorana_operator()

        assert mon_op.num_modes == 1
        assert len(mon_op.terms) == 1
        assert mon_op.terms[(0, 1)] == 0.25j

    def test_no_duplicate_majoranas(self):
        operator = FermiOperator(
            terms=[
                FermiString([(0, "+"), (1, "-")]),
                FermiString([(1, "+"), (0, "-")]),
            ],
            coefficients=[1.0, -1.0],
            num_modes=2,
        )
        expected_terms = {(0, 2): 0.5, (1, 3): 0.5}
        terms = operator.get_majorana_operator().terms

        assert expected_terms == terms


class TestFermiEvGate:
    def test_creation_from_fermi_operator(self):
        op = FermiOperator([FermiString([(0, "+"), (0, "-")])], [1.0], num_modes=1)

        gate = FermiEvGate(generator=op, parameter=0.75)

        assert gate.parameter == 0.75
        assert gate.generator.num_modes == 1
        assert gate.generator.terms == {(): 0.5, (0, 1): 0.5j}

    def test_creation_from_majorana_operator(self):
        op = MajoranaOperator(majoranas=[(0, 1)], coefficients=[1.0j], num_modes=1)

        gate = FermiEvGate(generator=op, parameter=-2.0)

        assert gate.parameter == -2.0
        assert gate.generator.terms == {(0, 1): 1.0j}


class TestFermiCircuit:
    def test_len(self):
        generator = MajoranaOperator(
            majoranas=[(0, 1)], coefficients=[1.0], num_modes=1
        )
        gates = [
            FermiEvGate(generator=generator, parameter=0.1),
            FermiEvGate(generator=generator, parameter=0.2),
        ]

        circuit = FermiCircuit(initial_state=[0], gates=gates)

        assert len(circuit) == 2

    def test_to_gates(self):
        generator = MajoranaOperator(
            majoranas=[(0, 1), (2, 3)],
            coefficients=[1.0j, -1.0j],
            num_modes=2,
        )
        gate_0 = FermiEvGate(generator=generator, parameter=0.3)
        gate_1 = FermiEvGate(generator=generator, parameter=-0.7)

        circuit = FermiCircuit(initial_state=[0, 1], gates=[gate_0, gate_1])
        gates, parameters, parameter_mapping = circuit.to_gates()

        np.testing.assert_array_equal(circuit.initial_state, np.array([0, 1]))
        np.testing.assert_array_equal(parameters, np.array([0.3, -0.7]))
        # A FermiCircuit lifts one gate per fermi gate with the identity mapping.
        assert parameter_mapping == list(range(len(gates)))

        majoranas, gen_coeffs, per_monomial_mapping = to_engine_arrays(gates)

        np.testing.assert_array_equal(
            np.array(gen_coeffs), np.array([-1.0, 1.0, -1.0, 1.0])
        )
        np.testing.assert_array_equal(
            np.array(per_monomial_mapping), np.array([0, 0, 1, 1])
        )

        np.testing.assert_array_equal(majoranas[0], np.array([0, 1]))
        np.testing.assert_array_equal(majoranas[1], np.array([2, 3]))
        np.testing.assert_array_equal(majoranas[2], np.array([0, 1]))
        np.testing.assert_array_equal(majoranas[3], np.array([2, 3]))

    def test_validate_inputs_duplicate_initial_state_raises(self):
        generator = MajoranaOperator(
            majoranas=[(0, 1)], coefficients=[1.0], num_modes=1
        )
        gate = FermiEvGate(generator=generator, parameter=0.1)

        with pytest.raises(ValueError, match="Duplicate indices in initial state"):
            FermiCircuit(initial_state=[0, 0], gates=[gate])
