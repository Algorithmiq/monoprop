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

from monoprop import Circuit, ExpGate
from monoprop.circuit import expand_monomials
from monoprop.fermi import (
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

    def test_construction_from_another_fermi_string_copies_expression(self):
        original = FermiString([(0, "+"), (1, "-")])
        copy = FermiString(original)
        assert copy.expression == original.expression
        assert copy == original

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
        assert repr(f) == "FermiString(c_0^+ c_2^-)"

    def test_repr_empty(self):
        f = FermiString([])
        assert repr(f) == "FermiString()"

    def test_hash_equal_for_equal_strings(self):
        # Equal FermiStrings must hash equal to be usable as dict keys / set members.
        assert hash(FermiString([(0, "+"), (1, "-")])) == hash(
            FermiString([(0, "+"), (1, "-")])
        )

    def test_hash_distinguishes_different_expressions(self):
        assert hash(FermiString([(0, "+")])) != hash(FermiString([(1, "+")]))


class TestFermiOperator:
    def test_valid_creation(self):
        terms = [FermiString([(0, "+")]), FermiString([(1, "-")])]
        op = FermiOperator(terms, [1.0, -1.0], num_modes=2)
        assert op.terms == terms
        assert op.coefficients == [1.0, -1.0]

    def test_num_modes_max_index(self):
        terms = [FermiString([(0, "+"), (3, "-")]), FermiString([(5, "+")])]
        op = FermiOperator(terms, [1.0, 1.0], num_modes=6)
        assert op.num_modes == 6

    def test_num_modes_empty_terms(self):
        op = FermiOperator([], [], 10)
        assert op.num_modes == 10

    def test_num_modes_single_term(self):
        op = FermiOperator([FermiString([(2, "+"), (7, "-")])], [1.0], num_modes=8)
        assert op.num_modes == 8

    def test_term_index_out_of_bounds_raises(self):
        terms = [FermiString([(0, "+"), (3, "-")])]
        with pytest.raises(
            ValueError, match=r"Fermi term index out of bounds: 3 >= num_modes=2"
        ):
            FermiOperator(terms, [1.0], num_modes=2)

    def test_terms_is_copy(self):
        terms = [FermiString([(0, "+")])]
        op = FermiOperator(terms, [1.0], num_modes=1)
        terms.append(FermiString([(1, "-")]))
        assert len(op.terms) == 1

    def test_coefficients_is_copy(self):
        coeffs = [1.0]
        op = FermiOperator([FermiString([(0, "+")])], coeffs, num_modes=1)
        coeffs.append(2.0)
        assert len(op.coefficients) == 1

    def test_str(self):
        terms = [FermiString([(0, "+")])]
        op = FermiOperator(terms, [2.0], num_modes=1)
        assert str(op) == "FermiOperator(1 terms, 1 modes: 2.0*FermiString(c_0^+))"

    def test_num_modes_explicit_override(self):
        terms = [FermiString([(0, "+"), (5, "-")])]
        op = FermiOperator(terms, [1.0], num_modes=12)
        assert op.num_modes == 12

    def test_empty_terms_without_num_modes_raises(self):
        with pytest.raises(TypeError):
            FermiOperator([], [])  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                FermiOperator(
                    [FermiString([(0, "+")]), FermiString([(1, "-")])],
                    [1.0, 0.5],
                    num_modes=2,
                ),
                FermiOperator(
                    [FermiString([(0, "+")]), FermiString([(1, "-")])],
                    [1.0, 0.5],
                    num_modes=2,
                ),
                True,
                id="same",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(0, "+")])], [1.0 + 1e-9], num_modes=2),
                True,
                id="within_atol",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(0, "+")])], [1.1], num_modes=2),
                False,
                id="outside_atol",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1e-16], num_modes=2),
                FermiOperator([], [], num_modes=2),
                True,
                id="negligible_vs_missing",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(1, "+")])], [1.0], num_modes=2),
                False,
                id="different_terms",
            ),
            pytest.param(
                FermiOperator(
                    [FermiString([(0, "+")]), FermiString([(1, "-")])],
                    [1.0, 0.5],
                    num_modes=2,
                ),
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                False,
                id="different_num_terms",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=3),
                False,
                id="different_num_modes",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+"), (1, "-")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(1, "-"), (0, "+")])], [1.0], num_modes=3),
                False,
                id="same_after_canonicalization",
            ),
        ],
    )
    def test_is_closely_equal(self, left, right, expected):
        assert left.isclose(right) is expected

    def test_isclose_rejects_non_fermi_operator(self):
        op = FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2)
        with pytest.raises(
            TypeError, match="Cannot compare FermiOperator with MajoranaOperator"
        ):
            op.isclose(MajoranaOperator({(0, 1): 1.0}, num_modes=2))

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                True,
                id="equal",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(1, "+")])], [1.0], num_modes=2),
                False,
                id="unequal",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=2),
                FermiOperator([FermiString([(0, "+")])], [1.0], num_modes=3),
                False,
                id="unequal_modes",
            ),
            pytest.param(
                FermiOperator([FermiString([(0, "+")])], [0.0], num_modes=2),
                FermiOperator([], [], num_modes=2),
                False,
                id="unequal_although_same_matrix",
            ),
        ],
    )
    def test_eq_working_and_non_working_examples(self, left, right, expected):
        assert (left == right) is expected


class TestMajoranaOperator:
    def test_valid_creation_and_repr(self):
        op = MajoranaOperator({(0, 1): 1.0, (2, 3): -0.5j}, num_modes=2)

        assert op.terms == {(0, 1): 1.0, (2, 3): -0.5j}
        assert op.num_modes == 2
        assert (
            str(op) == "MajoranaOperator(2 terms, 2 modes: (1+0j)*[0, 1], -0.5j*[2, 3])"
        )

    def test_get_majorana_operator(self):
        op = MajoranaOperator({(0, 1): 0.25j}, num_modes=1)
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
        expected_terms = {(0, 2): 0.5, (0, 3): 0j, (1, 2): 0j, (1, 3): 0.5}
        terms = operator.get_majorana_operator().terms

        assert expected_terms == terms

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                MajoranaOperator({(0, 1): 1.0j, (1,): 0.5}, num_modes=2),
                MajoranaOperator({(0, 1): 1.0j, (1,): 0.5}, num_modes=2),
                True,
                id="same",
            ),
            pytest.param(
                MajoranaOperator({(0, 1): 1.0}, num_modes=2),
                MajoranaOperator({(0, 1): 1.0 + 1e-9}, num_modes=2),
                True,
                id="within_atol",
            ),
            pytest.param(
                MajoranaOperator({(0, 1): 1.0}, num_modes=2),
                MajoranaOperator({(0, 1): 1.1}, num_modes=2),
                False,
                id="outside_atol",
            ),
            pytest.param(
                MajoranaOperator({(0, 1): 1e-16}, num_modes=2),
                MajoranaOperator({}, num_modes=2),
                True,
                id="negligible_vs_missing",
            ),
            pytest.param(
                MajoranaOperator({(0, 1): 1.0}, num_modes=2),
                MajoranaOperator({(0, 2): 1.0}, num_modes=2),
                False,
                id="different_terms",
            ),
            pytest.param(
                MajoranaOperator({(0, 1): 1.0, (1, 2): 0.5}, num_modes=2),
                MajoranaOperator({(0, 1): 1.0}, num_modes=2),
                False,
                id="different_num_terms",
            ),
            pytest.param(
                MajoranaOperator({(0, 1): 1.0}, num_modes=2),
                MajoranaOperator({(0, 1): 1.0}, num_modes=3),
                False,
                id="different_num_qubits",
            ),
        ],
    )
    def test_is_closely_equal(self, left, right, expected):
        assert left.isclose(right) is expected


def _number_op(mode: int = 0, num_modes: int = 1) -> FermiOperator:
    """The number operator n = c^+ c on ``mode``: a valid fermionic generator."""
    return FermiOperator([[(mode, "+"), (mode, "-")]], [1.0], num_modes=num_modes)


class TestExp:
    def test_creation_from_fermi_operator(self):
        gate = ExpGate(generator=_number_op())

        # The stored coefficients are the raw Majorana products (imaginary on the weight-2
        # term); it is _gate_layers that antihermitian-normalizes them later.
        assert gate.family == "majorana"
        assert gate.generator.num_modes == 1
        assert gate.generator.terms == {(): 0.5, (0, 1): 0.5j}

    def test_majorana_operator_is_structural(self):
        # A MajoranaOperator generator is structural: its coefficients are used unnormalized.
        gate = ExpGate(generator=MajoranaOperator({(0, 1): -1.0}, num_modes=1))

        assert gate.family == "majorana"
        assert gate.generator.terms == {(0, 1): -1.0}


class TestCircuit:
    def test_len(self):
        gates = [ExpGate(_number_op()), ExpGate(_number_op())]

        circuit = Circuit(
            gates=gates,
            initial_state=[0],
            system_size=1,
            parameters=[0.1, 0.2],
        )

        assert len(circuit) == 2

    def test_converts_fermi_gates_to_majorana(self):
        hop = FermiOperator(
            [[(0, "+"), (1, "-")], [(1, "+"), (0, "-")]], [1.0, 1.0], num_modes=2
        )
        gate_0 = ExpGate(hop)
        gate_1 = ExpGate(hop)

        circuit = Circuit(
            gates=[gate_0, gate_1],
            initial_state=[0, 1],
            system_size=2,
            parameters=[0.3, -0.7],
        )

        np.testing.assert_array_equal(circuit.initial_state, np.array([0, 1]))
        np.testing.assert_array_equal(list(circuit.parameters), np.array([0.3, -0.7]))
        assert list(circuit.resolved_mapping) == list(range(len(circuit)))
        assert all(g.family == "majorana" for g in circuit.gates)

        majoranas, gen_coeffs, per_monomial_mapping, gate_indices = expand_monomials(
            circuit.gates, circuit.resolved_mapping, circuit.system_size
        )
        # One gate per fermi gate; each generator here has two monomials.
        n_terms = len(gate_0.generator.terms)
        assert gate_indices == [0] * n_terms + [1] * n_terms
        assert per_monomial_mapping == [0] * n_terms + [1] * n_terms
        expected = [tuple(k) for k in circuit.gates[0].generator.terms]
        assert majoranas == expected + expected
        # Antihermitian normalization yielded real structural coefficients (no raise).
        assert all(isinstance(c, float) for c in gen_coeffs)

    def test_drops_identity_generators_and_aligned_parameters(self):
        identity = FermiOperator([], [], num_modes=1)
        circuit = Circuit(
            gates=[ExpGate(_number_op()), ExpGate(identity), ExpGate(_number_op())],
            initial_state=[0],
            system_size=1,
            parameters=[0.1, 0.2, 0.3],
        )
        assert len(circuit) == 2
        assert circuit.parameters == (0.1, 0.3)

    def test_validate_inputs_duplicate_initial_state_raises(self):
        gate = ExpGate(_number_op())
        with pytest.raises(ValueError, match="Duplicate indices in initial state"):
            Circuit(gates=[gate], initial_state=[0, 0], system_size=1)
