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

import importlib
import sys

import cirq
import pytest
from monoprop_cirq import from_cirq_operator, to_cirq_operator

from monoprop.pauli import Pauli, PauliOperator


@pytest.fixture
def cirq_unavailable(monkeypatch: pytest.MonkeyPatch):
    cirq_modules = [
        module_name
        for module_name in sys.modules
        if module_name == "cirq" or module_name.startswith("cirq.")
    ]
    for module_name in cirq_modules:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.delitem(sys.modules, "monoprop_cirq.conversion", raising=False)
    monkeypatch.setitem(sys.modules, "cirq", None)


@pytest.mark.usefixtures("cirq_unavailable")
def test_import_error_raised_without_cirq():
    with pytest.raises(ImportError, match="cirq is required"):
        importlib.import_module("monoprop_cirq.conversion")


class TestFromCirqOperator:
    def test_single_term(self):
        q = cirq.LineQubit.range(2)
        result = from_cirq_operator(cirq.X(q[0]) * cirq.Z(q[1]))
        assert isinstance(result, PauliOperator)
        assert result.num_qubits == 2
        assert result.terms == {Pauli("XZ", (0, 1)): pytest.approx(1.0)}

    def test_multiple_terms(self):
        q = cirq.LineQubit.range(2)
        result = from_cirq_operator(cirq.X(q[0]) * cirq.Z(q[1]) + 0.5 * cirq.Y(q[0]))
        assert result.terms[Pauli("XZ", (0, 1))] == pytest.approx(1.0)
        assert result.terms[Pauli("Y", 0)] == pytest.approx(0.5)

    def test_single_qubit_pauli_operation(self):
        result = from_cirq_operator(cirq.Z(cirq.LineQubit(0)))
        assert result.terms == {Pauli("Z", 0): pytest.approx(1.0)}

    def test_line_qubit_label_is_qubit_index(self):
        """LineQubit(2) stays qubit 2, with qubits 0 and 1 implicitly idle."""
        result = from_cirq_operator(0.75 * cirq.Z(cirq.LineQubit(2)))
        assert result.num_qubits == 3
        assert result.terms == {Pauli("Z", 2): pytest.approx(0.75)}

    def test_atol_filters_small_terms(self):
        q = cirq.LineQubit.range(2)
        result = from_cirq_operator(cirq.X(q[0]) + 1e-10 * cirq.Y(q[1]), atol=1e-8)
        assert result.terms == {Pauli("X", 0): pytest.approx(1.0)}

    def test_complex_coefficient_raises(self):
        with pytest.raises(ValueError, match="complex terms"):
            from_cirq_operator((1.0 + 0.5j) * cirq.X(cirq.LineQubit(0)))

    def test_skip_validation_drops_imaginary_part(self):
        result = from_cirq_operator(
            (1.0 + 0.5j) * cirq.X(cirq.LineQubit(0)), skip_validation=True
        )
        assert result.terms == {Pauli("X", 0): pytest.approx(1.0)}
        assert all(isinstance(c, float) for c in result.terms.values())

    def test_skip_validation_keeps_real_operator_unchanged(self):
        q = cirq.LineQubit.range(2)
        operator = cirq.X(q[0]) * cirq.Z(q[1]) - 0.25 * cirq.Y(q[1])
        assert from_cirq_operator(operator, skip_validation=True).isclose(
            from_cirq_operator(operator)
        )

    def test_identity_term_preserved(self):
        result = from_cirq_operator(cirq.Z(cirq.LineQubit(0)) + 0.5)
        assert result.terms == {
            Pauli("Z", 0): pytest.approx(1.0),
            Pauli(""): pytest.approx(0.5),
        }

    def test_identity_only_operator_needs_explicit_order(self):
        with pytest.raises(ValueError, match="Cannot infer a qubit ordering"):
            from_cirq_operator(cirq.PauliSum.wrap(0.5))
        result = from_cirq_operator(
            cirq.PauliSum.wrap(0.5), qubit_order=cirq.LineQubit.range(2)
        )
        assert result.num_qubits == 2
        assert result.terms == {Pauli(""): pytest.approx(0.5)}

    def test_grid_qubits_need_explicit_order(self):
        a, b = cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)
        with pytest.raises(ValueError, match="Cannot infer a qubit ordering"):
            from_cirq_operator(cirq.X(a) * cirq.Z(b))
        result = from_cirq_operator(cirq.X(a) * cirq.Z(b), qubit_order=[a, b])
        assert result.terms == {Pauli("XZ", (0, 1)): pytest.approx(1.0)}

    def test_qubit_order_default_sorts_touched_qubits(self):
        a, b = cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)
        result = from_cirq_operator(
            cirq.X(a) * cirq.Z(b), qubit_order=cirq.QubitOrder.DEFAULT
        )
        assert result.terms == {Pauli("ZX", (0, 1)): pytest.approx(1.0)}

    def test_qubit_order_missing_a_touched_qubit_raises(self):
        q = cirq.LineQubit.range(2)
        with pytest.raises(ValueError, match="extra qubits"):
            from_cirq_operator(cirq.X(q[0]) * cirq.Z(q[1]), qubit_order=[q[0]])

    def test_negative_line_qubit_needs_explicit_order(self):
        with pytest.raises(ValueError, match="Cannot infer a qubit ordering"):
            from_cirq_operator(cirq.Z(cirq.LineQubit(-1)))

    def test_qudit_rejected(self):
        qutrit = cirq.LineQid(0, dimension=3)
        with pytest.raises(ValueError, match="Only qubits are supported"):
            from_cirq_operator(cirq.PauliSum.wrap(0.5), qubit_order=[qutrit])


class TestToCirqOperator:
    def test_single_term(self):
        q = cirq.LineQubit.range(2)
        result = to_cirq_operator(PauliOperator({"XZ": 1.5}, num_qubits=2))
        assert isinstance(result, cirq.PauliSum)
        assert result == cirq.PauliSum.wrap(1.5 * cirq.X(q[0]) * cirq.Z(q[1]))

    def test_round_trip_matches_pauli_operator(self):
        original = PauliOperator({"XZ": 1.0, "IY": 0.5, "ZZ": -0.25}, num_qubits=2)
        assert from_cirq_operator(to_cirq_operator(original)).isclose(original)

    def test_identity_term_preserved(self):
        original = PauliOperator({"II": 0.5, "ZI": 1.0}, num_qubits=2)
        back = from_cirq_operator(
            to_cirq_operator(original), qubit_order=cirq.LineQubit.range(2)
        )
        assert back.isclose(original)

    def test_custom_qubits(self):
        a, b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
        result = to_cirq_operator(
            PauliOperator({"XY": 1.0}, num_qubits=2), qubits=[b, a]
        )
        assert result == cirq.PauliSum.wrap(cirq.X(b) * cirq.Y(a))

    def test_wrong_qubit_count_raises(self):
        with pytest.raises(ValueError, match="qubits has 3 entries"):
            to_cirq_operator(
                PauliOperator({"X": 1.0}, num_qubits=1), qubits=cirq.LineQubit.range(3)
            )

    def test_duplicate_qubits_raise(self):
        q = cirq.LineQubit(0)
        with pytest.raises(ValueError, match="duplicate"):
            to_cirq_operator(PauliOperator({"XX": 1.0}, num_qubits=2), qubits=[q, q])
