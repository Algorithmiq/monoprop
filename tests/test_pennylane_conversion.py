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
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from pytest_cases import parametrize_with_cases

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import pennylane as qml

    from monoprop import Circuit, ExpGate
    from monoprop.majorana import MajoranaOperator
    from monoprop.pauli import Pauli, PauliOperator
    from monoprop.pennylane_conversion import (
        from_pennylane_circuit,
        from_pennylane_operator,
        to_pennylane_circuit,
        to_pennylane_operator,
    )

    _pennylane_available = True
except ImportError:
    _pennylane_available = False


requires_pennylane = pytest.mark.skipif(
    not _pennylane_available, reason="pennylane not installed"
)


def _assert_pauli_circuits_close(converted, expected) -> None:
    assert converted.initial_state == expected.initial_state
    assert len(converted) == len(expected)
    assert list(converted.resolved_mapping) == list(expected.resolved_mapping)
    assert len(converted.parameters) == len(expected.parameters)
    for got, exp in zip(converted.parameters, expected.parameters):
        assert got == pytest.approx(exp)
    for got_gate, exp_gate in zip(converted.gates, expected.gates):
        # The generator's Pauli terms carry the qubit placement, so this also
        # checks the gate acts on the right qubits.
        assert got_gate.generator.isclose(exp_gate.generator)


@pytest.fixture
def pennylane_unavailable(monkeypatch: pytest.MonkeyPatch):
    pennylane_modules = [
        module_name
        for module_name in sys.modules
        if module_name == "pennylane" or module_name.startswith("pennylane.")
    ]
    for module_name in pennylane_modules:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.delitem(sys.modules, "monoprop.pennylane_conversion", raising=False)
    monkeypatch.setitem(sys.modules, "pennylane", None)


@pytest.mark.usefixtures("pennylane_unavailable")
def test_import_error_raised_without_pennylane():
    with pytest.raises(ImportError, match="pennylane is required"):
        importlib.import_module("monoprop.pennylane_conversion")


@requires_pennylane
@pytest.mark.pennylane
class TestFromPennylaneOperator:
    def test_single_term(self):
        op = qml.ops.LinearCombination([1.0], [qml.PauliX(0) @ qml.PauliZ(1)])
        result = from_pennylane_operator(op)
        assert isinstance(result, PauliOperator)
        assert len(result) == 1
        assert result.terms[Pauli("XZ", (0, 1))] == pytest.approx(1.0)  # no reversal

    def test_multiple_terms(self):
        op = qml.ops.LinearCombination(
            [1.0, 0.5], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0)]
        )
        result = from_pennylane_operator(op)
        assert len(result) == 2
        assert result.terms[Pauli("XZ", (0, 1))] == pytest.approx(1.0)
        assert result.terms[Pauli("Y", 0)] == pytest.approx(0.5)

    def test_atol_filters_small_terms(self):
        op = qml.ops.LinearCombination(
            [1.0, 1e-10], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0)]
        )
        result = from_pennylane_operator(op, atol=1e-8)
        assert len(result) == 1
        assert Pauli("XZ", (0, 1)) in result.terms

    def test_atol_default_keeps_large_terms(self):
        op = qml.ops.LinearCombination(
            [1.0, 0.1], [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0)]
        )
        result = from_pennylane_operator(op)
        assert len(result) == 2

    def test_force_real_with_real_coefficients(self):
        op = qml.ops.LinearCombination([2.0 + 0j], [qml.PauliX(0) @ qml.PauliZ(1)])
        result = from_pennylane_operator(op)
        assert result.terms[Pauli("XZ", (0, 1))] == pytest.approx(2.0)

    def test_force_real_raises_for_complex_coefficients(self):
        op = qml.ops.LinearCombination([1.0 + 0.5j], [qml.PauliX(0) @ qml.PauliZ(1)])
        with pytest.raises(ValueError, match="complex terms"):
            from_pennylane_operator(op)

    def test_preserves_coefficient_magnitude(self):
        op = qml.ops.LinearCombination([0.75 + 0j], [qml.PauliZ(1)])
        result = from_pennylane_operator(op)
        # Wire 1 doubles as qubit index 1, in an inferred 2-qubit system (i.e. "IZ").
        assert result.terms[Pauli("Z", 1)] == pytest.approx(0.75)
        assert result.num_qubits == 2

    def test_identity_term_needs_explicit_wires(self):
        """A purely-identity operator has no wires in its pauli_rep to infer a width from."""
        op = qml.ops.LinearCombination([0.5], [qml.Identity(0) @ qml.Identity(1)])
        result = from_pennylane_operator(op, wires=[0, 1])
        assert len(result) == 1
        assert Pauli("II") in result.terms
        assert result.num_qubits == 2

    def test_single_wire_operator(self):
        op = qml.ops.LinearCombination([1.0], [qml.PauliZ(0)])
        result = from_pennylane_operator(op)
        assert Pauli("Z") in result.terms
        assert result.num_qubits == 1

    def test_non_orderable_wires_requires_explicit_wires(self):
        op = qml.ops.LinearCombination([1.0], [qml.PauliX("a") @ qml.PauliZ(0)])
        with pytest.raises(ValueError, match="Cannot infer a qubit ordering"):
            from_pennylane_operator(op)
        result = from_pennylane_operator(op, wires=["a", 0])
        assert result.num_qubits == 2

    def test_explicit_wires_may_include_idle_wires(self):
        op = qml.ops.LinearCombination([1.0], [qml.PauliX(0)])
        result = from_pennylane_operator(op, wires=[0, 1])
        assert result.num_qubits == 2

    def test_rejects_duplicate_explicit_wires(self):
        op = qml.ops.LinearCombination([1.0], [qml.PauliX(0)])
        with pytest.raises(ValueError, match="wires must contain unique labels"):
            from_pennylane_operator(op, wires=[0, 0])


@requires_pennylane
@pytest.mark.pennylane
class TestToPennylaneOperator:
    def test_single_term_no_reversal(self):
        result = to_pennylane_operator(PauliOperator({"XZ": 1.0}, num_qubits=2))
        assert isinstance(result, qml.ops.LinearCombination)
        (word,) = result.pauli_rep.keys()
        assert dict(word) == {0: "X", 1: "Z"}

    def test_single_term_coefficient(self):
        result = to_pennylane_operator(PauliOperator({"XZ": 1.5}, num_qubits=2))
        (coeff,) = result.pauli_rep.values()
        assert coeff == pytest.approx(1.5)

    def test_round_trip_matches_pauli_operator(self):
        original = PauliOperator({"XZ": 1.0, "IY": 0.5}, num_qubits=2)
        converted = to_pennylane_operator(original)
        back = from_pennylane_operator(converted)
        assert back.isclose(original)

    def test_single_qubit_term(self):
        result = to_pennylane_operator(PauliOperator({"Z": 2.0}, num_qubits=1))
        (word,) = result.pauli_rep.keys()
        assert dict(word) == {0: "Z"}

    def test_identity_term_preserved(self):
        original = PauliOperator({"II": 0.5}, num_qubits=2)
        converted = to_pennylane_operator(original)
        back = from_pennylane_operator(converted, wires=[0, 1])
        assert back.isclose(original)

    def test_custom_wires(self):
        result = to_pennylane_operator(
            PauliOperator({"X": 1.0}, num_qubits=1), wires=["q0"]
        )
        assert result.wires == qml.wires.Wires(["q0"])

    def test_rejects_mismatched_wires_length(self):
        with pytest.raises(ValueError, match="wires has 1 entries"):
            to_pennylane_operator(
                PauliOperator({"XZ": 1.0}, num_qubits=2), wires=["q0"]
            )

    def test_rejects_duplicate_wires(self):
        with pytest.raises(ValueError, match="wires must contain unique labels"):
            to_pennylane_operator(
                PauliOperator({"XZ": 1.0}, num_qubits=2), wires=["q0", "q0"]
            )


class PennylaneCircuitCase(NamedTuple):
    """A `(qfunc, args, kwargs, expected)` case consumed by `parametrize_with_cases`."""

    qfunc: Callable[..., None]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expected: Circuit


class PennylaneCircuitsCases:
    # PennyLane's generator() already uses the SAME +i*phi*G sign ExpGate does, so -- unlike the
    # qiskit cases -- the expected monoprop generator carries the SAME sign PennyLane's own
    # -0.5-scaled generator does, never a negated one.
    def case_single_rx_gate(self):
        def qfunc(theta):
            qml.RX(theta, wires=0)

        expected = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("X", 0): -0.5}, num_qubits=1)),),
            system_size=1,
            parameters=(0.7,),
        )
        return PennylaneCircuitCase(qfunc, (0.7,), {}, expected)

    def case_multiple_gates(self):
        def qfunc(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)

        expected = Circuit(
            gates=(
                ExpGate(PauliOperator({Pauli("X", 0): -0.5}, num_qubits=2)),
                ExpGate(PauliOperator({Pauli("Y", 1): -0.5}, num_qubits=2)),
            ),
            initial_state=(),
            system_size=2,
            parameters=(0.3, 0.5),
        )
        return PennylaneCircuitCase(qfunc, (0.3, 0.5), {}, expected)

    def case_pauli_rot_on_non_contiguous_wires(self):
        """Also exercises the implicit idle wire (2) below the highest one touched (3), which is
        counted automatically since wires default to `range(max(touched wire) + 1)`."""

        def qfunc(theta):
            qml.PauliRot(theta, "XY", wires=[3, 1])

        expected = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("XY", (3, 1)): -0.5}, num_qubits=4)),),
            system_size=4,
            parameters=(0.4,),
        )
        return PennylaneCircuitCase(qfunc, (0.4,), {}, expected)

    def case_barrier_ignored(self):
        def qfunc(theta):
            qml.RZ(theta, wires=0)
            qml.Barrier(wires=0)

        expected = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("Z", 0): -0.5}, num_qubits=1)),),
            system_size=1,
            parameters=(0.6,),
        )
        return PennylaneCircuitCase(qfunc, (0.6,), {}, expected)


@requires_pennylane
@pytest.mark.pennylane
class TestFromPennylaneCircuit:
    @parametrize_with_cases(
        "qfunc, args, kwargs, expected", cases=PennylaneCircuitsCases
    )
    def test_valid_circuits(self, qfunc, args, kwargs, expected):
        converted = from_pennylane_circuit(qfunc, [], *args, **kwargs)
        _assert_pauli_circuits_close(converted, expected)

    def test_unsupported_gate_raises(self):
        def qfunc():
            qml.Hadamard(wires=0)

        with pytest.raises(ValueError, match="Unsupported gate"):
            from_pennylane_circuit(qfunc, [])

    def test_multi_parameter_gate_raises(self):
        def qfunc():
            qml.Rot(0.1, 0.2, 0.3, wires=0)

        with pytest.raises(ValueError, match="Unsupported gate"):
            from_pennylane_circuit(qfunc, [])

    def test_non_orderable_wires_requires_explicit_wires(self):
        def qfunc(theta):
            qml.RX(theta, wires="a")
            qml.RY(theta, wires=0)

        with pytest.raises(ValueError, match="Cannot infer a qubit ordering"):
            from_pennylane_circuit(qfunc, [], 0.5)

        converted = from_pennylane_circuit(qfunc, [], 0.5, wires=["a", 0])
        assert converted.system_size == 2

    def test_idle_wire_needs_explicit_wires(self):
        """A qfunc only reveals wires actually touched by a gate."""

        def qfunc(theta):
            qml.RX(theta, wires=0)

        converted = from_pennylane_circuit(qfunc, [], 0.5, wires=(0, 1))
        assert converted.system_size == 2

    def test_qnode_uses_underlying_quantum_function(self):
        device = qml.device("default.qubit", wires=1)

        @qml.qnode(device)
        def qnode(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        converted = from_pennylane_circuit(qnode, [], 0.5)
        assert converted.system_size == 1
        assert converted.parameters == pytest.approx((0.5,))
        assert len(converted.gates) == 1


@requires_pennylane
@pytest.mark.pennylane
class TestToPennylaneCircuit:
    def test_qfunc_replays_bound_parameters_by_default(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("Z", 0): 1.0}, num_qubits=1)),),
            initial_state=(),
            system_size=1,
            parameters=(0.7,),
        )
        qfunc = to_pennylane_circuit(circuit)
        result = qml.tape.make_qscript(qfunc)()
        assert len(result.operations) == 1
        (op,) = result.operations
        assert isinstance(op, qml.ops.Exp)
        assert op.coeff == pytest.approx(1j * 0.7)

    def test_qfunc_accepts_different_angles(self):
        """qfunc is reusable: it isn't frozen to circuit's own bound parameters."""
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("Z", 0): 1.0}, num_qubits=1)),),
            initial_state=(),
            system_size=1,
            parameters=(0.7,),
        )
        qfunc = to_pennylane_circuit(circuit)
        result = qml.tape.make_qscript(qfunc)(0.3)
        (op,) = result.operations
        assert op.coeff == pytest.approx(1j * 0.3)

    def test_qfunc_wrong_angle_count_raises(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("Z", 0): 1.0}, num_qubits=1)),),
            initial_state=(),
            system_size=1,
            parameters=(0.7,),
        )
        qfunc = to_pennylane_circuit(circuit)
        with pytest.raises(ValueError, match="expects 1 angle"):
            qfunc(0.1, 0.2)

    def test_unbound_circuit_needs_explicit_angles(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=1)),),
            initial_state=(),
            system_size=1,
        )
        qfunc = to_pennylane_circuit(circuit)
        with pytest.raises(ValueError, match="expects 1 angle"):
            qfunc()
        result = qml.tape.make_qscript(qfunc)(0.5)
        (op,) = result.operations
        assert op.coeff == pytest.approx(1j * 0.5)

    def test_generator_matches_original_no_reversal(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("XY", (3, 1)): 1.5}, num_qubits=5)),),
            system_size=5,
            parameters=(0.7,),
        )
        qfunc = to_pennylane_circuit(circuit)
        result = qml.tape.make_qscript(qfunc)()
        (op,) = result.operations
        rebuilt = from_pennylane_operator(op.base, wires=range(5))
        assert rebuilt.isclose(circuit.gates[0].generator, atol=0.0, rtol=0.0)

    def test_multi_term_generator_matches_original(self):
        circuit = Circuit(
            gates=(
                ExpGate(
                    PauliOperator(
                        {Pauli("Z", 0): 0.5, Pauli("Z", 1): 0.3}, num_qubits=2
                    )
                ),
            ),
            system_size=2,
            parameters=(0.4,),
        )
        qfunc = to_pennylane_circuit(circuit)
        result = qml.tape.make_qscript(qfunc)()
        (op,) = result.operations
        rebuilt = from_pennylane_operator(op.base, wires=range(2))
        assert rebuilt.isclose(circuit.gates[0].generator, atol=0.0, rtol=0.0)

    def test_rejects_majorana_family(self):
        circuit = Circuit(
            gates=(ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=1)),),
            initial_state=(),
            system_size=1,
            parameters=(0.5,),
        )
        with pytest.raises(TypeError, match="majorana-family gate"):
            to_pennylane_circuit(circuit)

    def test_rejects_mismatched_wires_length(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=2)),),
            initial_state=(),
            system_size=2,
            parameters=(0.5,),
        )
        with pytest.raises(ValueError, match="wires has 3 entries"):
            to_pennylane_circuit(circuit, wires=(0, 1, 2))

    def test_rejects_duplicate_wires(self):
        circuit = Circuit(
            gates=(ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=2)),),
            initial_state=(),
            system_size=2,
            parameters=(0.5,),
        )
        with pytest.raises(ValueError, match="wires must contain unique labels"):
            to_pennylane_circuit(circuit, wires=("q0", "q0"))


@requires_pennylane
@pytest.mark.pennylane
def test_to_pennylane_circuit_generators_match_original() -> None:
    """to_pennylane_circuit's output is not itself valid from_pennylane_circuit input.

    It emits ``qml.exp`` operations, which ``from_pennylane_circuit`` deliberately does not
    accept (see its docstring) -- so, unlike the qiskit module's round trip, this reconstructs
    each gate's operator via ``from_pennylane_operator`` and compares on the monoprop side.
    """
    circuit = Circuit(
        gates=(
            ExpGate(PauliOperator({Pauli("XYZ", (3, 1, 2)): 1.0}, num_qubits=4)),
            ExpGate(
                PauliOperator({Pauli("Z", 0): 0.5, Pauli("Z", 1): 0.5}, num_qubits=4)
            ),
        ),
        initial_state=(),
        system_size=4,
        parameters=[-1.2, 0.3],
    )
    qfunc = to_pennylane_circuit(circuit)
    result = qml.tape.make_qscript(qfunc)()
    for gate, op in zip(circuit.gates, result.operations, strict=True):
        rebuilt = from_pennylane_operator(op.base, wires=range(4))
        assert rebuilt.isclose(gate.generator, atol=0.0, rtol=0.0)
