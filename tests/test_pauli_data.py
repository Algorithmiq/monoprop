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

"""Unit tests for pauli_data module."""

from __future__ import annotations

import numpy as np
import pytest

from monoprop.pauli_data import (
    PauliEvGate,
    PauliGatesSequence,
    PauliOperator,
    PauliString,
)


class TestPauliString:
    def test_len(self):
        s = PauliString("XYZ")
        assert len(s) == 3

    def test_repr(self):
        s = PauliString("IX")
        assert repr(s) == "PauliString('IX')"

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("XY", "XY", True),
            ("XY", "XZ", False),
        ],
    )
    def test_eq(self, left: str, right: str, expected: bool):  # noqa: FBT001
        assert (PauliString(left) == PauliString(right)) is expected

    @pytest.mark.parametrize("other", ["X", 1, object()])
    def test_eq_non_pauli_string(self, other: object):
        assert (PauliString("X") == other) is False


class TestPauliOperator:
    def test_basic_construction(self):
        op = PauliOperator(["XY", "IZ"], [1.0, 0.5])
        assert op.strings == [PauliString("XY"), PauliString("IZ")]
        assert op.coefficients == [1.0, 0.5]

    def test_num_qubits(self):
        op = PauliOperator(["XYZ"], [1.0])
        assert op.num_qubits == 3

    def test_len(self):
        op = PauliOperator(["X", "Y", "Z"], [1.0, 2.0, 3.0])
        assert len(op) == 3

    def test_single_term(self):
        op = PauliOperator(["XIYZ"], [1 + 2j])
        assert op.num_qubits == 4
        assert len(op) == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            PauliOperator(["XY", "IZ"], [1.0])

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            PauliString("XA")

    def test_inconsistent_string_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            PauliOperator(["X", "XY"], [1.0, 2.0])

    def test_all_valid_pauli_chars(self):
        op = PauliOperator(["XYIZ"], [1.0])
        assert op.strings == [PauliString("XYIZ")]

    def test_str_few_terms(self):
        op = PauliOperator(["XY"], [1.0])
        r = str(op)
        assert "PauliOperator" in r
        assert "XY" in r

    def test_str_many_terms(self):
        strings = ["X"] * 10
        coeffs = [1.0] * 10
        op = PauliOperator(strings, coeffs)
        r = str(op)
        assert "PauliOperator" in r
        # With >8 terms, individual terms should not appear
        assert "10 terms" in r

    def test_from_dict_basic(self):
        op = PauliOperator.from_dict({"XY": 1.0, "IZ": 0.5})
        assert op.strings == [PauliString("XY"), PauliString("IZ")]
        assert op.coefficients == [1.0, 0.5]
        assert op.num_qubits == 2

    def test_from_dict_single_term(self):
        op = PauliOperator.from_dict({"IXYZ": 2.0})
        assert len(op) == 1
        assert op.num_qubits == 4

    def test_from_dict_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            PauliOperator.from_dict({"XA": 1.0})

    def test_from_dict_inconsistent_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            PauliOperator.from_dict({"X": 1.0, "XY": 2.0})

    def test_get_monomial_operator_identity(self):
        op = PauliOperator(["I"], [2.5])

        mon_op = op.get_monomial_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(): pytest.approx(2.5)}

    def test_get_monomial_operator_z(self):
        op = PauliOperator(["Z"], [1.0])

        mon_op = op.get_monomial_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(0, 1): pytest.approx(-1.0j)}

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                PauliOperator(["XY", "IZ"], [1.0, 0.5]),
                PauliOperator(["XY", "IZ"], [1.0, 0.5]),
                True,
                id="same",
            ),
            pytest.param(
                PauliOperator(["XY"], [1.0]),
                PauliOperator(["XY"], [1.0 + 1e-9]),
                True,
                id="within_atol",
            ),
            pytest.param(
                PauliOperator(["XY"], [1.0]),
                PauliOperator(["XY"], [1.1]),
                False,
                id="outside_atol",
            ),
            pytest.param(
                PauliOperator(["XY"], [1.0]),
                PauliOperator(["XZ"], [1.0]),
                False,
                id="different_strings",
            ),
            pytest.param(
                PauliOperator(["XY", "IZ"], [1.0, 0.5]),
                PauliOperator(["XY"], [1.0]),
                False,
                id="different_num_terms",
            ),
            pytest.param(
                PauliOperator(["X"], [1.0]),
                PauliOperator(["XY"], [1.0]),
                False,
                id="different_num_qubits",
            ),
        ],
    )
    def test_is_closely_equal(self, left, right, expected):
        assert left.isclose(right) is expected

    def test_is_closely_equal_type_error(self):
        with pytest.raises(TypeError):
            PauliOperator(["X"], [1.0]).isclose("not an operator")


class TestPauliEvGate:
    def test_basic_construction(self):
        gate = PauliEvGate([0, 1], PauliOperator(["XI", "IX"], [1.0, 1.0]), 0.5)
        assert gate.qubits == [0, 1]
        assert gate.parameter == 0.5

    def test_len(self):
        gate = PauliEvGate(
            [0, 1, 2], PauliOperator(["XII", "IIX", "IXI"], [1.0, 1.0, 1.0]), 1.0
        )
        assert len(gate) == 3

    def test_empty_qubits_raises(self):
        with pytest.raises(ValueError, match=r"At least one qubit"):
            PauliEvGate([], PauliOperator([], []), 1.0)

    def test_mismatched_paulis_raises(self):
        with pytest.raises(ValueError, match="must match"):
            PauliEvGate([0, 1], PauliOperator(["X"], [1.0, 1.0]), 1.0)

    def test_repr(self):
        gate = PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.1)
        r = repr(gate)
        assert "PauliEvGate" in r
        assert "[0]" in r

    @pytest.mark.parametrize(
        ("gate", "other", "expected"),
        [
            pytest.param(
                PauliEvGate([0, 1], PauliOperator(["XI", "IX"], [1.0, 0.5]), 0.3),
                PauliEvGate([0, 1], PauliOperator(["XI", "IX"], [1.0, 0.5]), 0.3),
                True,
                id="same",
            ),
            pytest.param(
                PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.3),
                PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.3 + 1e-9),
                True,
                id="within_atol",
            ),
            pytest.param(
                PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.3),
                PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5),
                False,
                id="different_parameter",
            ),
            pytest.param(
                PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.3),
                PauliEvGate([1], PauliOperator(["X"], [1.0]), 0.3),
                False,
                id="different_qubits",
            ),
            pytest.param(
                PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.3),
                PauliEvGate([0], PauliOperator(["Z"], [1.0]), 0.3),
                False,
                id="different_paulis",
            ),
        ],
    )
    def test_is_closely_equal(self, gate, other, expected):
        assert gate.isclose(other) is expected

    def test_is_closely_equal_type_error(self):
        with pytest.raises(TypeError):
            PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.3).isclose("not a gate")


class TestPauliGatesSequence:
    def _make_gate(self):
        op = PauliOperator(["X"], [1.0])
        return PauliEvGate([0], op, 0.5)

    def test_basic_construction(self):
        gates = [self._make_gate(), self._make_gate()]
        circuit = PauliGatesSequence(gates, num_qubits=1)
        assert len(circuit) == 2
        assert circuit.num_qubits == 1

    def test_empty_gates(self):
        circuit = PauliGatesSequence([], num_qubits=1)
        assert len(circuit) == 0
        assert circuit.num_qubits == 1

    def test_identity_gates_are_filtered(self):
        identity_gate = PauliEvGate([0], PauliOperator(["I"], [1.0]), 0.5)

        circuit = PauliGatesSequence([identity_gate, self._make_gate()], num_qubits=1)

        assert len(circuit) == 1
        assert circuit.gates[0].isclose(self._make_gate())

    def test_num_qubits_preserved_without_gates(self):
        circuit = PauliGatesSequence([], num_qubits=3)
        assert circuit.num_qubits == 3

    def test_repr(self):
        circuit = PauliGatesSequence([self._make_gate()], num_qubits=1)
        r = repr(circuit)
        assert "PauliGatesSequence" in r
        assert "1 gates" in r

    def test_get_monomial_sequence(self):
        op = PauliOperator(["Z"], [2.0])
        gate = PauliEvGate([0], op, 0.7)
        circuit = PauliGatesSequence([gate], num_qubits=1)

        mon_circuit = circuit.get_monomial_sequence()

        assert mon_circuit.parameters == [0.7]
        assert mon_circuit.gen_coeffs == [pytest.approx(2.0)]
        assert mon_circuit.param_inds == [0]
        np.testing.assert_array_equal(mon_circuit.majoranas[0], np.array([0, 1]))

    def test_get_monomial_sequence_empty(self):
        circuit = PauliGatesSequence([], num_qubits=1)

        mon_circuit = circuit.get_monomial_sequence()

        assert mon_circuit.majoranas == []
        assert mon_circuit.parameters == []
        assert len(mon_circuit.gen_coeffs) == 0
        assert mon_circuit.param_inds == []

    @pytest.mark.parametrize(
        ("circuit", "other", "expected"),
        [
            pytest.param(
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    num_qubits=2,
                ),
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    num_qubits=2,
                ),
                True,
                id="same",
            ),
            pytest.param(
                PauliGatesSequence([], num_qubits=2),
                PauliGatesSequence([], num_qubits=3),
                False,
                id="different_num_qubits",
            ),
            pytest.param(
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    num_qubits=1,
                ),
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.7)],
                    num_qubits=1,
                ),
                False,
                id="different_parameter",
            ),
            pytest.param(
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    num_qubits=1,
                ),
                PauliGatesSequence([], num_qubits=1),
                False,
                id="different_num_gates",
            ),
            pytest.param(
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    num_qubits=1,
                ),
                PauliGatesSequence(
                    [PauliEvGate([0], PauliOperator(["Z"], [1.0]), 0.5)],
                    num_qubits=1,
                ),
                False,
                id="different_gate",
            ),
        ],
    )
    def test_is_closely_equal(self, circuit, other, expected):
        assert circuit.isclose(other) is expected

    def test_is_closely_equal_type_error(self):
        with pytest.raises(TypeError):
            PauliGatesSequence([], num_qubits=1).isclose("not a circuit")
