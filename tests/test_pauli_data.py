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

from monoprop import MajoranaGate, PauliCircuit, PauliGate, PauliPropagator
from monoprop.majorana_data import MajoranaOperator
from monoprop.pauli_data import PauliOperator, PauliString


class TestPauliPropagatorCutoff:
    """PauliPropagator fixes the cutoff to Pauli weight ('support')."""

    def _propagator(self, serial_comm):
        return PauliPropagator(
            PauliOperator(["ZZ"], [1.0], num_qubits=2),
            initial_state=[],
            cutoff=4,
            comm=serial_comm,
        )

    def test_cutoff_type_is_support(self, serial_comm):
        assert self._propagator(serial_comm).cutoff_type == "support"

    def test_cutoff_type_is_read_only(self, serial_comm):
        mp = self._propagator(serial_comm)
        with pytest.raises(AttributeError):
            mp.cutoff_type = "length"

    def test_basis_change_is_read_only(self, serial_comm):
        mp = self._propagator(serial_comm)
        assert mp.basis_change is not None  # Jordan-Wigner basis set at construction
        with pytest.raises(AttributeError):
            mp.basis_change = None

    def test_num_qubits(self, serial_comm):
        mp = self._propagator(serial_comm)
        assert mp.num_qubits == 2  # "ZZ" operator

    def test_non_hermitian_pauli_gate_rejected(self, serial_comm):
        """A PauliGate with a complex (non-Hermitian) coefficient is rejected."""
        circuit = PauliCircuit(
            (PauliGate((0,), PauliOperator(["X"], [1.0j], num_qubits=1)),),
            parameters=(0.3,),
            num_qubits=2,
        )
        with pytest.raises(ValueError, match="not Hermitian"):
            self._propagator(serial_comm).propagate(circuit)


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
        op = PauliOperator(["XY", "IZ"], [1.0, 0.5], num_qubits=2)
        assert op.strings == [PauliString("XY"), PauliString("IZ")]
        assert op.coefficients == [1.0, 0.5]

    def test_num_qubits(self):
        op = PauliOperator(["XYZ"], [1.0], num_qubits=3)
        assert op.num_qubits == 3

    def test_num_qubits_must_match_string_length(self):
        with pytest.raises(ValueError, match="same length"):
            PauliOperator(["XYZ"], [1.0], num_qubits=2)

    def test_len(self):
        op = PauliOperator(["X", "Y", "Z"], [1.0, 2.0, 3.0], num_qubits=1)
        assert len(op) == 3

    def test_single_term(self):
        op = PauliOperator(["XIYZ"], [1 + 2j], num_qubits=4)
        assert op.num_qubits == 4
        assert len(op) == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            PauliOperator(["XY", "IZ"], [1.0], num_qubits=2)

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            PauliString("XA")

    def test_inconsistent_string_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            PauliOperator(["X", "XY"], [1.0, 2.0], num_qubits=1)

    def test_all_valid_pauli_chars(self):
        op = PauliOperator(["XYIZ"], [1.0], num_qubits=4)
        assert op.strings == [PauliString("XYIZ")]

    def test_str_few_terms(self):
        op = PauliOperator(["XY"], [1.0], num_qubits=2)
        r = str(op)
        assert "PauliOperator" in r
        assert "XY" in r

    def test_str_many_terms(self):
        strings = ["X"] * 10
        coeffs = [1.0] * 10
        op = PauliOperator(strings, coeffs, num_qubits=1)
        r = str(op)
        assert "PauliOperator" in r
        # With >8 terms, individual terms should not appear
        assert "10 terms" in r

    def test_from_dict_basic(self):
        op = PauliOperator.from_dict({"XY": 1.0, "IZ": 0.5}, num_qubits=2)
        assert op.strings == [PauliString("XY"), PauliString("IZ")]
        assert op.coefficients == [1.0, 0.5]
        assert op.num_qubits == 2

    def test_from_dict_num_qubits_optional(self):
        op = PauliOperator.from_dict({"IXYZ": 2.0})  # num_qubits left unspecified
        assert len(op) == 1
        assert op.num_qubits is None

    def test_from_dict_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            PauliOperator.from_dict({"XA": 1.0})

    def test_from_dict_inconsistent_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            PauliOperator.from_dict({"X": 1.0, "XY": 2.0})

    def test_get_majorana_operator_identity(self):
        op = PauliOperator(["I"], [2.5], num_qubits=1)

        mon_op = op.get_majorana_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(): pytest.approx(2.5)}

    def test_get_majorana_operator_z(self):
        op = PauliOperator(["Z"], [1.0], num_qubits=1)

        mon_op = op.get_majorana_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(0, 1): pytest.approx(-1.0j)}

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                PauliOperator(["XY", "IZ"], [1.0, 0.5], num_qubits=2),
                PauliOperator(["XY", "IZ"], [1.0, 0.5], num_qubits=2),
                True,
                id="same",
            ),
            pytest.param(
                PauliOperator(["XY"], [1.0], num_qubits=2),
                PauliOperator(["XY"], [1.0 + 1e-9], num_qubits=2),
                True,
                id="within_atol",
            ),
            pytest.param(
                PauliOperator(["XY"], [1.0], num_qubits=2),
                PauliOperator(["XY"], [1.1], num_qubits=2),
                False,
                id="outside_atol",
            ),
            pytest.param(
                PauliOperator(["XY"], [1.0], num_qubits=2),
                PauliOperator(["XZ"], [1.0], num_qubits=2),
                False,
                id="different_strings",
            ),
            pytest.param(
                PauliOperator(["XY", "IZ"], [1.0, 0.5], num_qubits=2),
                PauliOperator(["XY"], [1.0], num_qubits=2),
                False,
                id="different_num_terms",
            ),
            pytest.param(
                PauliOperator(["X"], [1.0], num_qubits=1),
                PauliOperator(["XY"], [1.0], num_qubits=2),
                False,
                id="different_num_qubits",
            ),
        ],
    )
    def test_is_closely_equal(self, left, right, expected):
        assert left.isclose(right) is expected

    def test_is_closely_equal_type_error(self):
        with pytest.raises(TypeError):
            PauliOperator(["X"], [1.0], num_qubits=1).isclose("not an operator")


class TestPauliCircuit:
    def _make_gate(self):
        return PauliGate((0,), PauliOperator(["X"], [1.0], num_qubits=1))

    def test_basic_construction(self):
        gates = (self._make_gate(), self._make_gate())
        circuit = PauliCircuit(
            gates, parameters=(0.5, 0.5), initial_state=(0,), num_qubits=1
        )
        assert len(circuit) == 2
        assert circuit.initial_state == (0,)
        assert circuit.num_qubits == 1

    def test_empty_gates(self):
        circuit = PauliCircuit((), initial_state=(0,), num_qubits=1)
        assert len(circuit) == 0

    def test_default_mapping_is_identity(self):
        circuit = PauliCircuit((self._make_gate(), self._make_gate()), num_qubits=1)
        assert list(circuit.resolved_mapping) == [0, 1]
        assert circuit.n_parameters == 2

    def test_num_qubits_not_derived_from_gates(self):
        # A gate touching only qubit 0 can still live in a wider register.
        circuit = PauliCircuit((self._make_gate(),), num_qubits=5)
        assert circuit.num_qubits == 5

    def test_rejects_non_pauli_gate(self):
        with pytest.raises(TypeError, match="PauliGate"):
            PauliCircuit(
                (MajoranaGate(MajoranaOperator.from_dict({(0, 1): 1.0})),), num_qubits=1
            )

    def test_pauli_gate_equality(self):
        op = PauliOperator(["X"], [1.0], num_qubits=1)
        assert PauliGate((0,), op) == PauliGate((0,), op)
        assert PauliGate((0,), op) != PauliGate((1,), op)
