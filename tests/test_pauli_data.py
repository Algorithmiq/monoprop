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

import pytest

from monoprop import PauliPropagator
from monoprop.pauli_data import PauliEvCircuit, PauliEvGate, PauliOperator, PauliString


class TestPauliPropagatorCutoff:
    """PauliPropagator fixes the cutoff to Pauli weight ('support')."""

    def _propagator(self, serial_comm):
        return PauliPropagator(
            PauliOperator(["ZZ"], [1.0]),
            initial_state=[],
            cutoff=4,
            comm=serial_comm,
        )

    def test_cutoff_type_is_support(self, serial_comm):
        assert self._propagator(serial_comm).cutoff_type == "support"

    def test_setting_support_is_allowed(self, serial_comm):
        mp = self._propagator(serial_comm)
        mp.cutoff_type = "support"
        assert mp.cutoff_type == "support"

    def test_setting_length_is_rejected(self, serial_comm):
        mp = self._propagator(serial_comm)
        with pytest.raises(ValueError, match="only supports the 'support' cutoff"):
            mp.cutoff_type = "length"


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

    def test_get_majorana_operator_identity(self):
        op = PauliOperator(["I"], [2.5])

        mon_op = op.get_majorana_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(): pytest.approx(2.5)}

    def test_get_majorana_operator_z(self):
        op = PauliOperator(["Z"], [1.0])

        mon_op = op.get_majorana_operator()

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


class TestPauliEvCircuit:
    def _make_gate(self):
        op = PauliOperator(["X"], [1.0])
        return PauliEvGate([0], op, 0.5)

    def test_basic_construction(self):
        gates = [self._make_gate(), self._make_gate()]
        circuit = PauliEvCircuit(gates, [("Z", 0)], num_qubits=1)
        assert len(circuit) == 2
        assert circuit.initial_state == [("Z", 0)]

    def test_empty_gates(self):
        circuit = PauliEvCircuit([], [("X", 0)], num_qubits=1)
        assert len(circuit) == 0

    def test_empty_initial_state(self):
        circuit = PauliEvCircuit([self._make_gate()], [], num_qubits=1)
        assert circuit.initial_state == []

    def test_all_valid_axes(self):
        circuit = PauliEvCircuit([], [("X", 0), ("Y", 1), ("Z", 2)], num_qubits=3)
        assert len(circuit.initial_state) == 3

    def test_repr(self):
        circuit = PauliEvCircuit([self._make_gate()], [("Z", 0)], num_qubits=1)
        r = repr(circuit)
        assert "PauliEvCircuit" in r
        assert "1 gates" in r

    def test_to_gates(self):
        # The Pauli->Majorana mapping lives in PauliPropagator; the circuit adapter
        # stays in the qubit basis and yields one PauliGate per Pauli evolution gate.
        op = PauliOperator(["Z"], [2.0])
        gate = PauliEvGate([0], op, 0.7)
        pauli_circuit = PauliEvCircuit([gate], [0], num_qubits=1)

        circuit = pauli_circuit.to_circuit()

        assert len(circuit) == 1
        assert circuit.gates[0].qubits == (0,)
        assert circuit.gates[0].paulis is op
        assert len(circuit.parameters) == 1
        assert list(circuit.resolved_mapping) == list(range(len(circuit)))

    def test_to_gates_empty(self):
        pauli_circuit = PauliEvCircuit([], [1], num_qubits=1)

        circuit = pauli_circuit.to_circuit()

        assert circuit.gates == ()
        assert len(circuit.parameters) == 0
        assert list(circuit.resolved_mapping) == list(range(len(circuit)))

    @pytest.mark.parametrize(
        ("circuit", "other", "expected"),
        [
            pytest.param(
                PauliEvCircuit(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    [0],
                    num_qubits=2,
                ),
                PauliEvCircuit(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    [0],
                    num_qubits=2,
                ),
                True,
                id="same",
            ),
            pytest.param(
                PauliEvCircuit([], [0], num_qubits=2),
                PauliEvCircuit([], [0], num_qubits=3),
                False,
                id="different_num_qubits",
            ),
            pytest.param(
                PauliEvCircuit([], [0], num_qubits=2),
                PauliEvCircuit([], [1], num_qubits=2),
                False,
                id="different_initial_state",
            ),
            pytest.param(
                PauliEvCircuit(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    [0],
                    num_qubits=1,
                ),
                PauliEvCircuit([], [0], num_qubits=1),
                False,
                id="different_num_gates",
            ),
            pytest.param(
                PauliEvCircuit(
                    [PauliEvGate([0], PauliOperator(["X"], [1.0]), 0.5)],
                    [0],
                    num_qubits=1,
                ),
                PauliEvCircuit(
                    [PauliEvGate([0], PauliOperator(["Z"], [1.0]), 0.5)],
                    [0],
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
            PauliEvCircuit([], [0], num_qubits=1).isclose("not a circuit")
