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

from monoprop import Circuit, ExpGate, PauliPropagator
from monoprop.majorana import MajoranaOperator
from monoprop.pauli import Pauli, PauliOperator


class TestPauliPropagatorCutoff:
    """PauliPropagator fixes the cutoff to Pauli weight ('support')."""

    def _propagator(self, serial_comm):
        return PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=2),
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

    def test_num_qubits(self, serial_comm):
        mp = self._propagator(serial_comm)
        assert mp.num_qubits == 2  # "ZZ" operator

    def test_non_hermitian_pauli_gate_rejected(self, serial_comm):
        circuit = Circuit(
            (ExpGate(PauliOperator({Pauli("X", 0): 1.0j}, num_qubits=1)),),
            parameters=(0.3,),
        )
        mp = self._propagator(serial_comm)
        with pytest.raises(ValueError, match="not Hermitian"):
            mp.propagate(circuit)

    @pytest.mark.parametrize("schrodinger_cutoff", [3, 4, 5])
    def test_schrodinger_cutoff(self, schrodinger_cutoff, serial_comm):
        """schrodinger_cutoff is a Pauli weight in qubits, matching ``cutoff``.

        PauliPropagator doubles it internally (qubit weight -> gamma-slot popcount).
        """
        mp = PauliPropagator(
            PauliOperator({"ZZ": 1.0}, num_qubits=10),
            initial_state=[],
            cutoff=4,
            schrodinger_cutoff=schrodinger_cutoff,
            comm=serial_comm,
        )
        op = mp.evolved_operator()
        assert max(len(p.qubits) for p in op.terms) == schrodinger_cutoff


class TestPauli:
    def test_default_qubits_are_range(self):
        p = Pauli("XYZ")
        assert p.string == "XYZ"
        assert p.qubits == (0, 1, 2)

    def test_repr(self):
        # Identity letters are dropped, so "IX" is X on qubit 1.
        p = Pauli("IX")
        assert repr(p) == "Pauli('X', (1,))"

    def test_identity_letters_dropped(self):
        assert Pauli("IZ", (0, 1)) == Pauli("Z", 1)
        assert Pauli("Z", 1) == Pauli("Z", (1,))

    def test_canonicalized_by_qubit(self):
        assert Pauli("XY", (1, 0)) == Pauli("YX", (0, 1))

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("XY", "XY", True),
            ("XY", "XZ", False),
        ],
    )
    def test_eq(self, left: str, right: str, expected: bool):  # noqa: FBT001
        assert (Pauli(left) == Pauli(right)) is expected

    @pytest.mark.parametrize("other", ["X", 1, object()])
    def test_eq_non_pauli(self, other: object):
        assert (Pauli("X", 0) == other) is False

    def test_hashable(self):
        assert {Pauli("XY", (0, 1)), Pauli("YX", (1, 0))} == {Pauli("XY", (0, 1))}

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            Pauli("XA")

    def test_string_qubits_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Pauli("XY", (0,))

    def test_duplicate_qubits_raises(self):
        with pytest.raises(ValueError, match="Duplicate qubit indices"):
            Pauli("XY", (0, 0))


class TestPauliOperator:
    def test_basic_construction(self):
        op = PauliOperator({"XY": 1.0, "IZ": 0.5}, num_qubits=2)
        assert set(op.terms) == {Pauli("XY", (0, 1)), Pauli("Z", 1)}
        assert list(op.terms.values()) == [1.0, 0.5]

    def test_num_qubits(self):
        op = PauliOperator({"XYZ": 1.0}, num_qubits=3)
        assert op.num_qubits == 3

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XY": 1.0}, num_qubits=2),
                True,
                id="equal",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XZ": 1.0}, num_qubits=2),
                False,
                id="unequal",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XYI": 1.0}, num_qubits=3),
                False,
                id="unequal_modes",
            ),
            pytest.param(
                PauliOperator({"XY": 0.0}, num_qubits=2),
                PauliOperator({}, num_qubits=2),
                False,
                id="unequal_although_same_matrix",
            ),
        ],
    )
    def test_eq_working_and_non_working_examples(self, left, right, expected):
        assert (left == right) is expected

    def test_qubit_index_must_be_within_num_qubits(self):
        with pytest.raises(ValueError, match="qubit index"):
            PauliOperator({"XYZ": 1.0}, num_qubits=2)

    def test_len(self):
        op = PauliOperator({"X": 1.0, "Y": 2.0, "Z": 3.0}, num_qubits=1)
        assert len(op) == 3

    def test_single_term(self):
        op = PauliOperator({"XIYZ": 1 + 2j}, num_qubits=4)
        assert op.num_qubits == 4
        assert len(op) == 1

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            PauliOperator({"XA": 1.0}, num_qubits=2)

    def test_all_valid_pauli_chars(self):
        op = PauliOperator({"XYIZ": 1.0}, num_qubits=4)
        assert set(op.terms) == {Pauli("XYIZ")}

    def test_get_majorana_operator_requires_num_qubits(self):
        op = PauliOperator._from_terms(["X"], [1.0], num_qubits=None)
        with pytest.raises(ValueError, match="needs num_qubits"):
            op.get_majorana_operator()

    def test_str_few_terms(self):
        op = PauliOperator({"XY": 1.0}, num_qubits=2)
        r = str(op)
        assert "PauliOperator" in r
        assert "XY" in r

    def test_str_many_terms(self):
        op = PauliOperator({Pauli("X", i): 1.0 for i in range(10)}, num_qubits=10)
        r = str(op)
        assert "PauliOperator" in r
        assert "10 terms" in r

    def test_dict_construction(self):
        op = PauliOperator({"XY": 1.0, "IZ": 0.5}, num_qubits=2)
        assert set(op.terms) == {Pauli("XY", (0, 1)), Pauli("Z", 1)}
        assert list(op.terms.values()) == [1.0, 0.5]
        assert op.num_qubits == 2

    def test_num_qubits_required(self):
        with pytest.raises(TypeError):
            PauliOperator({"IXYZ": 2.0})  # type: ignore[call-arg]

    def test_get_majorana_operator_identity(self):
        op = PauliOperator({"I": 2.5}, num_qubits=1)

        mon_op = op.get_majorana_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(): pytest.approx(2.5)}

    def test_get_majorana_operator_z(self):
        op = PauliOperator({"Z": 1.0}, num_qubits=1)

        mon_op = op.get_majorana_operator()

        assert mon_op.num_modes == 1
        assert mon_op.terms == {(0, 1): pytest.approx(-1.0j)}

    @pytest.mark.parametrize(
        ("terms", "expected"),
        [
            pytest.param({}, True, id="empty"),
            pytest.param({Pauli("X", 0): 1.0}, True, id="single_term"),
            pytest.param(
                {Pauli("XX", (0, 1)): 1.0, Pauli("ZZ", (0, 1)): 1.0},
                True,
                id="different_letters_on_two_shared_qubits",
            ),
            pytest.param(
                {Pauli("X", 0): 1.0, Pauli("Z", 0): 1.0},
                False,
                id="different_letters_on_one_shared_qubit",
            ),
        ],
    )
    def test_all_pairwise_commute(self, terms, expected):
        """Pairwise commutation follows the parity of differing shared Pauli letters."""
        assert PauliOperator(terms, num_qubits=2).all_pairwise_commute() is expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(
                PauliOperator({"XY": 1.0, "IZ": 0.5}, num_qubits=2),
                PauliOperator({"XY": 1.0, "IZ": 0.5}, num_qubits=2),
                True,
                id="same",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XY": 1.0 + 1e-9}, num_qubits=2),
                True,
                id="within_atol",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XY": 1.1}, num_qubits=2),
                False,
                id="outside_atol",
            ),
            pytest.param(
                PauliOperator({"XX": 1e-16}, num_qubits=2),
                PauliOperator({}, num_qubits=2),
                True,
                id="negligible_vs_missing",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XZ": 1.0}, num_qubits=2),
                False,
                id="different_strings",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0, "IZ": 0.5}, num_qubits=2),
                PauliOperator({"XY": 1.0}, num_qubits=2),
                False,
                id="different_num_terms",
            ),
            pytest.param(
                PauliOperator({"XY": 1.0}, num_qubits=2),
                PauliOperator({"XYI": 1.0}, num_qubits=3),
                False,
                id="different_num_qubits",
            ),
        ],
    )
    def test_is_closely_equal(self, left, right, expected):
        assert left.isclose(right) is expected

    def test_is_closely_equal_type_error(self):
        op = PauliOperator({"X": 1.0}, num_qubits=1)
        with pytest.raises(TypeError):
            op.isclose("not an operator")


class TestCircuit:
    def _make_gate(self):
        return ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=1))

    def test_basic_construction(self):
        gates = (self._make_gate(), self._make_gate())
        circuit = Circuit(gates, parameters=(0.5, 0.5), initial_state=(0,))
        assert len(circuit) == 2
        assert circuit.initial_state == (0,)

    def test_empty_gates(self):
        circuit = Circuit((), initial_state=(0,))
        assert len(circuit) == 0

    def test_default_mapping_is_identity(self):
        circuit = Circuit((self._make_gate(), self._make_gate()))
        assert list(circuit.resolved_mapping) == [0, 1]
        assert circuit.n_parameters == 2

    def test_rejects_mixed_gate_families(self):
        pauli_gate = ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=1))
        majorana_gate = ExpGate(MajoranaOperator({(0, 1): 1.0}, num_modes=2))
        with pytest.raises(TypeError, match="mix"):
            Circuit((pauli_gate, majorana_gate))

    def test_pauli_gate_equality(self):
        gen = PauliOperator({Pauli("X", 0): 1.0}, num_qubits=2)
        # Two distinct instances built from the same generator must compare equal.
        first = ExpGate(gen)
        second = ExpGate(gen)
        assert first == second
        assert ExpGate(gen) != ExpGate(
            PauliOperator({Pauli("X", 1): 1.0}, num_qubits=2)
        )


def test_pauli_rejects_negative_qubit_index() -> None:
    """A negative qubit index is rejected, mirroring Majorana.

    Unchecked it resolves silently through Python list indexing when the term is widened
    (``Pauli("Z", -1)`` on four qubits becomes Z on qubit 3) and reaches the engine as a huge
    unsigned Majorana slot.
    """
    with pytest.raises(ValueError, match="must be non-negative"):
        Pauli("Z", -1)
    with pytest.raises(ValueError, match="must be non-negative"):
        Pauli("XY", (0, -2))
