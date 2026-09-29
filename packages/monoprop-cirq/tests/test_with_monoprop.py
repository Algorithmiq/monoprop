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

import itertools
from typing import TYPE_CHECKING

import cirq
import numpy as np
import pytest
from monoprop_cirq import from_cirq_circuit, from_cirq_operator

from monoprop import Circuit, PauliPropagator

if TYPE_CHECKING:
    from monoprop.pauli import PauliOperator


def _cirq_expectation(
    circuit: cirq.Circuit,
    observable: cirq.PauliSum,
    initial_state: list[int],
    qubits: list[cirq.Qid],
) -> float:
    """Run ``X`` on ``initial_state``, then ``circuit``, on Cirq's own state-vector simulator."""
    prepared = cirq.Circuit(cirq.X(qubits[i]) for i in initial_state) + circuit
    state = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(prepared, qubit_order=qubits)
        .final_state_vector
    )
    qubit_map = {q: i for i, q in enumerate(qubits)}
    return observable.expectation_from_state_vector(state, qubit_map).real


def _monoprop_expectation(
    circuit: Circuit, observable: PauliOperator, cutoff: int
) -> float:
    propagator = PauliPropagator(observable, circuit.initial_state, cutoff=cutoff)
    propagator.propagate(circuit)
    return propagator.expectation_value()


_Q = cirq.LineQubit.range(3)

# One case per supported gate family. Each is compared against Cirq's simulator after a fixed,
# generic prefix, against an observable holding every Pauli string on the three qubits, so a sign,
# scale or qubit-placement error in any single conversion rule changes the expectation value.
_GATE_CASES = {
    "rx": cirq.rx(0.37)(_Q[0]),
    "ry": cirq.ry(-0.8)(_Q[1]),
    "rz": cirq.rz(1.1)(_Q[2]),
    "x_pow_shifted": cirq.XPowGate(exponent=0.3, global_shift=0.2)(_Q[1]),
    "y_pow": (cirq.Y**0.45)(_Q[0]),
    "s": cirq.S(_Q[2]),
    "t": cirq.T(_Q[0]),
    "x": cirq.X(_Q[1]),
    "xx_pow": (cirq.XX**0.3)(_Q[0], _Q[2]),
    "yy_pow": (cirq.YY**-0.6)(_Q[1], _Q[2]),
    "zz_pow": (cirq.ZZ**0.7)(_Q[0], _Q[1]),
    "ms": cirq.ms(0.4)(_Q[1], _Q[0]),
    "cz_pow": (cirq.CZ**-0.45)(_Q[2], _Q[0]),
    "cnot_pow": (cirq.CNOT**0.3)(_Q[2], _Q[1]),
    "iswap_pow": (cirq.ISWAP**0.6)(_Q[0], _Q[1]),
    "swap_pow": (cirq.SWAP**0.35)(_Q[1], _Q[2]),
    "ccz_pow": (cirq.CCZ**0.3)(_Q[0], _Q[1], _Q[2]),
    "ccx_pow": (cirq.CCX**0.7)(_Q[2], _Q[0], _Q[1]),
    "riswap": cirq.riswap(0.5)(_Q[2], _Q[0]),
    "givens": cirq.givens(0.6)(_Q[0], _Q[1]),
    "pauli_string_phasor": cirq.PauliStringPhasor(
        -cirq.X(_Q[0]) * cirq.Y(_Q[2]), exponent_neg=0.3, exponent_pos=-0.2
    ),
    "pauli_sum_exponential": cirq.PauliSumExponential(
        0.5 * cirq.Z(_Q[0]) * cirq.X(_Q[1]) - 0.3 * cirq.Y(_Q[2]), exponent=0.9
    ),
    "pauli_string": cirq.X(_Q[0]) * cirq.Y(_Q[1]) * cirq.Z(_Q[2]),
    # The hint the unsupported-gate error gives: decompose into supported gates first.
    "decomposed_h": cirq.decompose(cirq.H(_Q[1])),
    "decomposed_controlled_rx": cirq.decompose(cirq.rx(0.8).controlled()(_Q[2], _Q[0])),
    "decomposed_phased_xz": cirq.decompose(
        cirq.PhasedXZGate(x_exponent=0.3, z_exponent=0.1, axis_phase_exponent=0.7)(
            _Q[2]
        )
    ),
}


def _dense_observable(qubits: list[cirq.Qid]) -> cirq.PauliSum:
    """Every non-identity Pauli string on ``qubits``, each with its own coefficient."""
    strings = [
        letters
        for letters in itertools.product(
            (cirq.I, cirq.X, cirq.Y, cirq.Z), repeat=len(qubits)
        )
        if any(p != cirq.I for p in letters)
    ]
    return sum(
        (0.1 * (k + 1) * (-1) ** k)
        * cirq.PauliString({q: p for q, p in zip(qubits, letters) if p != cirq.I})
        for k, letters in enumerate(strings)
    )


@pytest.mark.parametrize("gate", _GATE_CASES.values(), ids=_GATE_CASES.keys())
def test_gate_matches_cirq_simulator(gate):
    """A converted gate must reproduce Cirq's own expectation value, not its conjugate.

    A round trip cannot catch a consistently-applied sign or scale bug, so this pins each
    conversion rule against Cirq's simulator instead.
    """
    prefix = cirq.Circuit(
        [cirq.ry(0.3 + 0.2 * i)(q) for i, q in enumerate(_Q)]
        + [cirq.rx(0.5 - 0.3 * i)(q) for i, q in enumerate(_Q)]
    )
    circuit = prefix + cirq.Circuit(gate)
    observable = _dense_observable(_Q)
    initial_state = [1]

    expected = _cirq_expectation(circuit, observable, initial_state, _Q)
    got = _monoprop_expectation(
        from_cirq_circuit(circuit, initial_state, qubit_order=_Q),
        from_cirq_operator(observable, qubit_order=_Q),
        cutoff=len(_Q),
    )
    assert got == pytest.approx(expected, abs=1e-10)


def test_separately_converted_operator_and_circuit_align():
    """Default ordering keys on LineQubit.x, so pieces touching different qubits still line up.

    The observable skips qubit 0; ordering each object by its own sorted qubits would put its
    qubit 1 on the circuit's qubit 0.
    """
    circuit = cirq.Circuit(
        cirq.ry(0.4)(_Q[0]), (cirq.CNOT**0.7)(_Q[0], _Q[1]), cirq.rx(0.9)(_Q[2])
    )
    observable = cirq.Z(_Q[1]) + 0.5 * cirq.X(_Q[1]) * cirq.Y(_Q[2])

    expected = _cirq_expectation(circuit, observable, [], _Q)
    got = _monoprop_expectation(
        from_cirq_circuit(circuit, []), from_cirq_operator(observable), cutoff=len(_Q)
    )
    assert got == pytest.approx(expected, abs=1e-10)
