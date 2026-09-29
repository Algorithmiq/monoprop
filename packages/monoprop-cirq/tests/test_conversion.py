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
from typing import Any, NamedTuple

import cirq
import numpy as np
import pytest
import scipy.linalg
import sympy
from monoprop_cirq import (
    from_cirq_circuit,
    from_cirq_operator,
    to_cirq_circuit,
    to_cirq_operator,
)
from pytest_cases import parametrize_with_cases

from monoprop import Circuit, ExpGate
from monoprop.majorana import MajoranaOperator
from monoprop.pauli import Pauli, PauliOperator


def _assert_pauli_circuits_close(converted, expected) -> None:
    assert converted.initial_state == expected.initial_state
    assert converted.system_size == expected.system_size
    assert len(converted) == len(expected)
    assert list(converted.resolved_mapping) == list(expected.resolved_mapping)
    assert len(converted.parameters) == len(expected.parameters)
    for got, exp in zip(converted.parameters, expected.parameters):
        assert got == pytest.approx(exp)
    for got_gate, exp_gate in zip(converted.gates, expected.gates):
        # The generator's Pauli terms carry the qubit placement, so this also
        # checks the gate acts on the right qubits.
        assert got_gate.generator.isclose(exp_gate.generator)


def _pauli_circuit(
    generators: list[dict[Pauli, float]],
    parameters: list[float],
    num_qubits: int,
    initial_state: tuple[int, ...] = (),
) -> Circuit:
    return Circuit(
        gates=tuple(
            ExpGate(PauliOperator(g, num_qubits=num_qubits)) for g in generators
        ),
        parameters=tuple(parameters),
        initial_state=initial_state,
        system_size=num_qubits,
    )


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


class CirqCircuitCase(NamedTuple):
    """A ``(circuit, kwargs, expected)`` case consumed by ``parametrize_with_cases``."""

    circuit: Any
    kwargs: dict[str, Any]
    expected: Any


class CirqCircuitsCases:
    # Every expected generator is written out by hand from the gate's definition, e.g.
    # CZ**t = exp(i pi t |11><11|) = exp(i t (pi/4) (ZZ - ZI - IZ)) up to phase.
    def case_rx_keeps_radians(self):
        q = cirq.LineQubit(0)
        expected = _pauli_circuit([{Pauli("X", 0): -0.5}], [0.7], 1)
        return CirqCircuitCase(cirq.Circuit(cirq.rx(0.7)(q)), {}, expected)

    def case_x_pow_drops_global_shift(self):
        q = cirq.LineQubit(0)
        gate = cirq.XPowGate(exponent=0.3, global_shift=0.2)
        expected = _pauli_circuit([{Pauli("X", 0): -np.pi / 2}], [0.3], 1)
        return CirqCircuitCase(cirq.Circuit(gate(q)), {}, expected)

    def case_t_gate(self):
        q = cirq.LineQubit(0)
        expected = _pauli_circuit([{Pauli("Z", 0): -np.pi / 2}], [0.25], 1)
        return CirqCircuitCase(cirq.Circuit(cirq.T(q)), {}, expected)

    def case_cz_power_on_reversed_qubits(self):
        q = cirq.LineQubit.range(3)
        generator = {
            Pauli("ZZ", (0, 2)): np.pi / 4,
            Pauli("Z", 0): -np.pi / 4,
            Pauli("Z", 2): -np.pi / 4,
        }
        expected = _pauli_circuit([generator], [0.3], 3)
        return CirqCircuitCase(cirq.Circuit((cirq.CZ**0.3)(q[2], q[0])), {}, expected)

    def case_cnot_power_control_above_target(self):
        q = cirq.LineQubit.range(2)
        generator = {
            Pauli("XZ", (0, 1)): np.pi / 4,
            Pauli("Z", 1): -np.pi / 4,
            Pauli("X", 0): -np.pi / 4,
        }
        expected = _pauli_circuit([generator], [0.5], 2)
        return CirqCircuitCase(cirq.Circuit((cirq.CNOT**0.5)(q[1], q[0])), {}, expected)

    def case_pauli_string_phasor(self):
        q = cirq.LineQubit.range(3)
        phasor = cirq.PauliStringPhasor(
            cirq.X(q[0]) * cirq.Z(q[2]), exponent_neg=0.3, exponent_pos=0.1
        )
        expected = _pauli_circuit([{Pauli("XZ", (0, 2)): -np.pi / 2}], [0.2], 3)
        return CirqCircuitCase(cirq.Circuit(phasor), {}, expected)

    def case_pauli_sum_exponential(self):
        """Expands into one phasor per term: exp(i t c P) is the phasor with relative exponent
        -2 t c / pi."""
        q = cirq.LineQubit.range(3)
        exponential = cirq.PauliSumExponential(
            0.5 * cirq.Z(q[0]) + 0.3 * cirq.Z(q[1]) * cirq.Z(q[2]), exponent=0.7
        )
        expected = _pauli_circuit(
            [{Pauli("Z", 0): -np.pi / 2}, {Pauli("ZZ", (1, 2)): -np.pi / 2}],
            [-2 * 0.7 * 0.5 / np.pi, -2 * 0.7 * 0.3 / np.pi],
            3,
        )
        return CirqCircuitCase(cirq.Circuit(exponential), {}, expected)

    def case_pauli_string_operation(self):
        q = cirq.LineQubit.range(2)
        expected = _pauli_circuit([{Pauli("XY", (0, 1)): -np.pi / 2}], [1.0], 2)
        return CirqCircuitCase(cirq.Circuit(cirq.X(q[0]) * cirq.Y(q[1])), {}, expected)

    def case_identity_and_global_phase_skipped(self):
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.I(q), cirq.global_phase_operation(1j), cirq.rz(0.6)(q)
        )
        expected = _pauli_circuit([{Pauli("Z", 0): -0.5}], [0.6], 1)
        return CirqCircuitCase(circuit, {}, expected)

    def case_tagged_op_inside_subcircuit(self):
        q = cirq.LineQubit(0)
        subcircuit = cirq.FrozenCircuit(cirq.ry(0.4)(q).with_tags("tag"))
        expected = _pauli_circuit([{Pauli("Y", 0): -0.5}], [0.4], 1)
        # use_repetition_ids is explicit because cirq-core 1.5 warns about its changing default.
        subcircuit_op = cirq.CircuitOperation(subcircuit, use_repetition_ids=False)
        return CirqCircuitCase(cirq.Circuit(subcircuit_op), {}, expected)

    def case_resolved_symbol(self):
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.rx(sympy.Symbol("a"))(q))
        expected = _pauli_circuit([{Pauli("X", 0): -0.5}], [0.3], 1)
        return CirqCircuitCase(circuit, {"param_resolver": {"a": 0.3}}, expected)

    def case_constant_term_in_pauli_sum_exponential(self):
        """The constant becomes a qubit-less phasor, a global phase that must not add a gate."""
        q = cirq.LineQubit.range(2)
        exponential = cirq.PauliSumExponential(
            0.5 * cirq.Z(q[0]) + 0.3 * cirq.X(q[1]) + 2.0, exponent=0.7
        )
        expected = _pauli_circuit(
            [{Pauli("Z", 0): -np.pi / 2}, {Pauli("X", 1): -np.pi / 2}],
            [-2 * 0.7 * 0.5 / np.pi, -2 * 0.7 * 0.3 / np.pi],
            2,
        )
        return CirqCircuitCase(cirq.Circuit(exponential), {}, expected)

    def case_symbol_expression(self):
        q = cirq.LineQubit.range(3)
        a = sympy.Symbol("a")
        circuit = cirq.Circuit((cirq.ZZ ** (2 * a))(q[0], q[1]), cirq.rx(a)(q[2]))
        expected = _pauli_circuit(
            [{Pauli("ZZ", (0, 1)): -np.pi / 2}, {Pauli("X", 2): -0.5}], [0.6, 0.3], 3
        )
        return CirqCircuitCase(circuit, {"param_resolver": {"a": 0.3}}, expected)

    def case_grid_qubits_with_explicit_order(self):
        a, b = cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)
        circuit = cirq.Circuit((cirq.ZZ**0.2)(a, b), cirq.rx(0.1)(a))
        expected = _pauli_circuit(
            [{Pauli("ZZ", (0, 1)): -np.pi / 2}, {Pauli("X", 0): -0.5}], [0.2, 0.1], 2
        )
        return CirqCircuitCase(circuit, {"qubit_order": [a, b]}, expected)

    def case_idle_qubit_from_explicit_order(self):
        q = cirq.LineQubit.range(2)
        expected = _pauli_circuit([{Pauli("X", 0): -0.5}], [0.5], 2)
        return CirqCircuitCase(
            cirq.Circuit(cirq.rx(0.5)(q[0])), {"qubit_order": q}, expected
        )


class TestFromCirqCircuit:
    @parametrize_with_cases("circuit, kwargs, expected", cases=CirqCircuitsCases)
    def test_valid_circuits(self, circuit, kwargs, expected):
        _assert_pauli_circuits_close(from_cirq_circuit(circuit, [], **kwargs), expected)

    def test_initial_state_passed_through(self):
        q = cirq.LineQubit.range(3)
        converted = from_cirq_circuit(cirq.Circuit(cirq.rx(0.1)(q[2])), [0, 2])
        assert converted.initial_state == (0, 2)

    def test_unsupported_gate_raises(self):
        gate = cirq.PhasedXZGate(
            x_exponent=0.1, z_exponent=0.2, axis_phase_exponent=0.3
        )
        with pytest.raises(ValueError, match="Unsupported gate"):
            from_cirq_circuit(cirq.Circuit(gate(cirq.LineQubit(0))), [])

    def test_non_commuting_generator_raises(self):
        with pytest.raises(ValueError, match="do not commute"):
            from_cirq_circuit(cirq.Circuit(cirq.H(cirq.LineQubit(0))), [])

    def test_measurement_raises(self):
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.rx(0.1)(q), cirq.measure(q, key="m"))
        with pytest.raises(ValueError, match="drop_terminal_measurements"):
            from_cirq_circuit(circuit, [])
        converted = from_cirq_circuit(cirq.drop_terminal_measurements(circuit), [])
        assert len(converted) == 1

    def test_unresolved_symbol_raises(self):
        circuit = cirq.Circuit(cirq.rx(sympy.Symbol("a"))(cirq.LineQubit(0)))
        with pytest.raises(ValueError, match=r"unresolved parameters \['a'\]"):
            from_cirq_circuit(circuit, [])

    def test_grid_qubits_need_explicit_order(self):
        circuit = cirq.Circuit(cirq.rx(0.1)(cirq.GridQubit(0, 0)))
        with pytest.raises(ValueError, match="Cannot infer a qubit ordering"):
            from_cirq_circuit(circuit, [])

    def test_frozen_circuit_input(self):
        converted = from_cirq_circuit(
            cirq.FrozenCircuit(cirq.rx(0.2)(cirq.LineQubit(0))), []
        )
        _assert_pauli_circuits_close(
            converted, _pauli_circuit([{Pauli("X", 0): -0.5}], [0.2], 1)
        )

    def test_empty_circuit_with_explicit_order(self):
        converted = from_cirq_circuit(
            cirq.Circuit(), [], qubit_order=cirq.LineQubit.range(2)
        )
        assert len(converted) == 0
        assert converted.system_size == 2

    def test_controlled_rotation_raises_with_decompose_hint(self):
        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.rx(0.3).controlled()(q[0], q[1]))
        with pytest.raises(ValueError, match=r"Unsupported gate.*cirq\.decompose"):
            from_cirq_circuit(circuit, [])

    def test_qudit_rejected(self):
        qutrit = cirq.LineQid(0, dimension=3)
        circuit = cirq.Circuit(cirq.XPowGate(dimension=3)(qutrit))
        with pytest.raises(ValueError, match="Only qubits are supported"):
            from_cirq_circuit(circuit, [], qubit_order=[qutrit])


class TestToCirqCircuit:
    def test_matches_exponential_of_each_generator(self):
        """PauliSumExponential is exactly exp(+i theta H), so the unitary matches with no phase."""
        q = cirq.LineQubit.range(3)
        first = {Pauli("Z", 0): 0.5, Pauli("ZZ", (1, 2)): 0.3}
        second = {Pauli("XY", (0, 2)): -1.2}
        circuit = _pauli_circuit([first, second], [0.7, -0.4], 3)
        first_matrix = (0.5 * cirq.Z(q[0]) + 0.3 * cirq.Z(q[1]) * cirq.Z(q[2])).matrix(
            q
        )
        second_matrix = cirq.PauliSum.wrap(-1.2 * cirq.X(q[0]) * cirq.Y(q[2])).matrix(q)
        expected = scipy.linalg.expm(-0.4j * second_matrix) @ scipy.linalg.expm(
            0.7j * first_matrix
        )
        result = to_cirq_circuit(circuit)
        np.testing.assert_allclose(result.unitary(qubit_order=q), expected, atol=1e-12)

    def test_shared_parameter_drives_both_gates(self):
        q = cirq.LineQubit.range(2)
        gates = (
            ExpGate(PauliOperator({Pauli("Z", 0): 1.0}, num_qubits=2), index=0),
            ExpGate(PauliOperator({Pauli("X", 1): 1.0}, num_qubits=2), index=0),
        )
        circuit = Circuit(gates=gates, parameters=(0.4,), system_size=2)
        result = to_cirq_circuit(circuit)
        expected = cirq.Circuit(
            cirq.PauliSumExponential(cirq.Z(q[0]), exponent=0.4),
            cirq.PauliSumExponential(cirq.X(q[1]), exponent=0.4),
        )
        np.testing.assert_allclose(
            result.unitary(qubit_order=q), expected.unitary(qubit_order=q), atol=1e-12
        )

    def test_identity_term_dropped(self):
        circuit = _pauli_circuit([{Pauli(""): 2.0, Pauli("Z", 0): 1.0}], [0.3], 1)
        (op,) = to_cirq_circuit(circuit).all_operations()
        assert op.pauli_string == cirq.PauliString(cirq.Z(cirq.LineQubit(0)))

    def test_custom_qubits(self):
        a, b = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
        circuit = _pauli_circuit([{Pauli("XY", (0, 1)): 1.0}], [0.3], 2)
        result = to_cirq_circuit(circuit, qubits=[b, a])
        (op,) = result.all_operations()
        assert op.pauli_string == cirq.X(b) * cirq.Y(a)

    def test_unbound_circuit_raises(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=1)),),
            system_size=1,
        )
        with pytest.raises(ValueError, match="needs a bound circuit"):
            to_cirq_circuit(circuit)

    def test_rejects_majorana_family(self):
        circuit = Circuit(
            gates=(ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=1)),),
            system_size=1,
            parameters=(0.5,),
        )
        with pytest.raises(TypeError, match="majorana-family gate"):
            to_cirq_circuit(circuit)

    def test_wrong_qubit_count_raises(self):
        circuit = _pauli_circuit([{Pauli("X", 0): 1.0}], [0.5], 2)
        with pytest.raises(ValueError, match="qubits has 3 entries"):
            to_cirq_circuit(circuit, qubits=cirq.LineQubit.range(3))
