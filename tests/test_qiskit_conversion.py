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

"""Unit tests for qiskit_conversion module."""

from __future__ import annotations

import importlib
import sys

import pytest
from pytest_cases import case, parametrize_with_cases

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    from monoprop.pauli_data import PauliEvCircuit, PauliEvGate, PauliOperator
    from monoprop.qiskit_conversion import (
        from_qiskit_circuit,
        from_qiskit_operator,
        to_qiskit_circuit,
        to_qiskit_operator,
    )

    _qiskit_available = True
except ImportError:
    _qiskit_available = False

requires_qiskit = pytest.mark.skipif(
    not _qiskit_available, reason="qiskit not installed"
)


@pytest.fixture
def qiskit_unavailable(monkeypatch: pytest.MonkeyPatch):
    qiskit_modules = [
        module_name
        for module_name in sys.modules
        if module_name == "qiskit" or module_name.startswith("qiskit.")
    ]
    for module_name in qiskit_modules:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.delitem(sys.modules, "monoprop.qiskit_conversion", raising=False)
    monkeypatch.setitem(sys.modules, "qiskit", None)


@pytest.mark.usefixtures("qiskit_unavailable")
def test_import_error_raised_without_qiskit():
    with pytest.raises(ImportError, match="qiskit is required"):
        importlib.import_module("monoprop.qiskit_conversion")


@requires_qiskit
@pytest.mark.qiskit
class TestFromQiskitOperator:
    def test_single_term(self):
        op = SparsePauliOp.from_list([("XZ", 1.0)])
        result = from_qiskit_operator(op)
        assert isinstance(result, PauliOperator)
        assert len(result) == 1
        assert result.strings[0].string == "ZX"  # qiskit ordering
        assert result.coefficients[0] == pytest.approx(1.0)

    def test_multiple_terms(self):
        op = SparsePauliOp.from_list([("XZ", 1.0), ("IY", 0.5)])
        result = from_qiskit_operator(op)
        assert len(result) == 2
        terms = dict(zip((s.string for s in result.strings), result.coefficients))
        assert terms["ZX"] == pytest.approx(1.0)  # qiskit ordering
        assert terms["YI"] == pytest.approx(0.5)  # qiskit ordering

    def test_atol_filters_small_terms(self):
        op = SparsePauliOp.from_list([("XZ", 1.0), ("IY", 1e-10)])
        result = from_qiskit_operator(op, atol=1e-8)
        assert len(result) == 1
        assert result.strings[0].string == "ZX"  # qiskit ordering

    def test_atol_default_keeps_large_terms(self):
        op = SparsePauliOp.from_list([("XZ", 1.0), ("IY", 0.1)])
        result = from_qiskit_operator(op)
        assert len(result) == 2

    def test_force_real_with_real_coefficients(self):
        op = SparsePauliOp.from_list([("XZ", 2.0 + 0j)])
        result = from_qiskit_operator(op, force_real=True)
        assert result.coefficients[0] == pytest.approx(2.0)

    def test_force_real_raises_for_complex_coefficients(self):
        op = SparsePauliOp.from_list([("XZ", 1.0 + 0.5j)])
        with pytest.raises(ValueError, match="complex terms"):
            from_qiskit_operator(op, force_real=True)

    def test_preserves_coefficient_magnitude(self):
        op = SparsePauliOp.from_list([("IZ", 0.75 + 0j)])
        result = from_qiskit_operator(op)
        assert result.coefficients[0] == pytest.approx(0.75)

    def test_identity_string(self):
        op = SparsePauliOp.from_list([("II", 0.5)])
        result = from_qiskit_operator(op)
        assert result.strings[0].string == "II"

    def test_single_qubit_operator(self):
        op = SparsePauliOp.from_list([("Z", 1.0)])
        result = from_qiskit_operator(op)
        assert result.strings[0].string == "Z"
        assert result.num_qubits == 1


@requires_qiskit
@pytest.mark.qiskit
class TestToQiskitOperator:
    def test_single_term_reverses_string(self):
        result = to_qiskit_operator({"XZ": 1.0})
        labels = list(result.paulis.to_labels(array=True))
        assert "ZX" in labels

    def test_single_term_coefficient(self):
        result = to_qiskit_operator({"XZ": 1.5})
        assert result.coeffs[0] == pytest.approx(1.5)

    def test_returns_sparse_pauli_op(self):
        result = to_qiskit_operator({"X": 1.0})
        assert isinstance(result, SparsePauliOp)

    def test_multiple_terms_reversed(self):
        result = to_qiskit_operator({"XZ": 1.0, "IY": 0.5})
        labels = list(result.paulis.to_labels(array=True))
        assert "ZX" in labels
        assert "YI" in labels

    def test_multiple_terms_coefficients(self):
        result = to_qiskit_operator({"XZ": 1.0, "IY": 0.5})
        label_to_coeff = dict(zip(result.paulis.to_labels(array=True), result.coeffs))
        assert label_to_coeff["ZX"] == pytest.approx(1.0)
        assert label_to_coeff["YI"] == pytest.approx(0.5)

    def test_complex_coefficients(self):
        result = to_qiskit_operator({"XZ": 1.0 + 0.5j})
        assert result.coeffs[0] == pytest.approx(1.0 + 0.5j)

    def test_single_qubit_term(self):
        result = to_qiskit_operator({"Z": 2.0})
        labels = list(result.paulis.to_labels(array=True))
        assert "Z" in labels
        assert result.coeffs[0] == pytest.approx(2.0)

    def test_identity_string_preserved(self):
        result = to_qiskit_operator({"II": 0.5})
        labels = list(result.paulis.to_labels(array=True))
        assert "II" in labels


@requires_qiskit
@pytest.mark.qiskit
class TestToQiskitCircuit:
    def test_single_gate(self):
        circuit = PauliEvCircuit(
            [PauliEvGate([0], PauliOperator(["Z"], [1.0]), 0.7)],
            num_qubits=1,
        )

        result = to_qiskit_circuit(circuit)

        assert isinstance(result, QuantumCircuit)
        assert result.num_qubits == 1
        assert len(result.data) == 1
        assert result.data[0].operation.name == "PauliEvolution"
        assert result.data[0].operation.params[0] == pytest.approx(0.7)


class QiskitCircuitsCases:
    @case(id="single_pauli_evolution_gate")
    def case_single_pauli_evolution_gate(self):
        circuit = QuantumCircuit(1)
        operator = SparsePauliOp.from_list([("Z", 1.0)])
        circuit.append(PauliEvolutionGate(operator, time=0.7), [0])
        expected = PauliEvCircuit(
            gates=[
                PauliEvGate(
                    qubits=[0], paulis=PauliOperator(["Z"], [1.0]), parameter=0.7
                )
            ],
            num_qubits=1,
        )
        return circuit, expected

    def case_multiple_pauli_evolution_gates(self):
        circuit = QuantumCircuit(2)
        operator1 = SparsePauliOp.from_list([("XZ", 1.0)])
        operator2 = SparsePauliOp.from_list([("IY", 0.5)])
        circuit.append(PauliEvolutionGate(operator1, time=0.3), [0, 1])
        circuit.append(PauliEvolutionGate(operator2, time=0.5), [0, 1])
        expected = PauliEvCircuit(
            gates=[
                PauliEvGate(
                    qubits=[0, 1], paulis=PauliOperator(["ZX"], [1.0]), parameter=0.3
                ),
                PauliEvGate(
                    qubits=[0, 1], paulis=PauliOperator(["YI"], [0.5]), parameter=0.5
                ),
            ],
            num_qubits=2,
        )
        return circuit, expected

    def case_rotation_gates_equivalent_to_pauli_evolution(self):
        circuit = QuantumCircuit(1)
        circuit.rx(0.5, 0)
        circuit.ry(0.3, 0)
        circuit.rz(0.7, 0)
        expected = PauliEvCircuit(
            gates=[
                PauliEvGate(
                    qubits=[0], paulis=PauliOperator(["X"], [1.0]), parameter=0.25
                ),
                PauliEvGate(
                    qubits=[0], paulis=PauliOperator(["Y"], [1.0]), parameter=0.15
                ),
                PauliEvGate(
                    qubits=[0], paulis=PauliOperator(["Z"], [1.0]), parameter=0.35
                ),
            ],
            num_qubits=1,
        )
        return circuit, expected

    def case_barrier_ignored(self):
        circuit = QuantumCircuit(1)
        operator = SparsePauliOp.from_list([("Z", 1.0)])
        circuit.append(PauliEvolutionGate(operator, time=0.7), [0])
        circuit.barrier()
        expected = PauliEvCircuit(
            gates=[
                PauliEvGate(
                    qubits=[0], paulis=PauliOperator(["Z"], [1.0]), parameter=0.7
                )
            ],
            num_qubits=1,
        )
        return circuit, expected


@requires_qiskit
@pytest.mark.qiskit
class TestFromQiskitCircuit:
    @parametrize_with_cases("circuit, expected", cases=QiskitCircuitsCases)
    def test_valid_circuits(self, circuit, expected):
        converted_circuit = from_qiskit_circuit(circuit)
        assert converted_circuit.isclose(expected)

    def test_unsupported_gate_raises(self):
        circuit = QuantumCircuit(1)
        operator = SparsePauliOp.from_list([("Z", 1.0)])
        circuit.append(PauliEvolutionGate(operator, time=0.7), [0])
        circuit.h(0)  # Hadamard gate is not supported

        with pytest.raises(ValueError, match="Unsupported gate"):
            from_qiskit_circuit(circuit)
